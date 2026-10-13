"""
Stage 5b: decision-focused Transformer controller (DFT) — limited, validation-only experiment.

Input per (decision, circuit, regime):
  history : the chain's last 20 published epoch states up to the decision epoch
            (log T1, log T2, log sx for 4 qubits; log CZ for 3 pairs) -> (20, 15)
  pred    : model-based predicted log error of the 14 actions from the published state
  context : circuit one-hot, horizon one-hot, regime flag
Target: log true expected error of each action at the execution epoch, learned as a
residual on `pred`. The controller picks argmin(pred + residual).

Models (identical training): transformer (history), mlp_summary (last/mean/std of history),
no_history (pred + context only; control for static, non-drift corrections).
Training: training-period decisions on development chains; early stopping on the last 20 % of
the training period (by decision epoch). Validation decisions are never used for fitting or
stopping. The test period is not used.

    python3 -m experiments.s5b_dft
"""

import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.qem.mitigation import ACTION_NAMES

OUT = "results/stage5b"
L, A = 20, len(ACTION_NAMES)
REG = {"R1_incoherent": 0.0, "R2_coherent": 0.03}
SEEDS = [0, 1, 2]
torch.set_num_threads(8)


class Lookup:
    def __init__(self):
        parts = []
        for idx, arr in [("results/stage5/sim_index.parquet", "results/stage5/sim_arrays.npy"),
                         (f"{OUT}/train_index.parquet", f"{OUT}/train_arrays.npy")]:
            t = pd.read_parquet(idx)
            parts.append((t, np.load(arr)))
        self.d = {}
        for t, a in parts:
            for k, r in zip(zip(t.backend, t.chain, t.epoch, t.circuit, t.quantity, t.delta), t.row):
                self.d.setdefault(k, a[r])

    def get(self, *k):
        return self.d.get(k)


def build():
    from src.qem.benchmarks import all_benchmarks
    circuits = [b["name"] for b in all_benchmarks()]
    em = pd.read_parquet("results/stage5/epoch_states.parquet")
    S = np.log(np.load("results/stage5/epoch_states.npy"))
    em["row"] = np.arange(len(em))
    hist = {k: g.sort_values("epoch")[["epoch", "row"]].to_numpy() for k, g in em.groupby(["backend", "chain"])}
    dec = pd.read_parquet("results/stage5/decisions.parquet")
    dec = dec[(dec.role == "dev") & dec.split.isin(["train", "val"])]
    lk = Lookup()
    H, P, C, Y, meta = [], [], [], [], []
    for r in dec.itertuples():
        h = hist[(r.backend, r.chain)]
        rows = h[h[:, 0] <= r.epoch][-L:, 1]
        seq = S[rows]
        if len(seq) < L:
            seq = np.vstack([np.repeat(seq[:1], L - len(seq), 0), seq])
        for ci, c in enumerate(circuits):
            for reg, d in REG.items():
                p = lk.get(r.backend, r.chain, r.epoch, c, "pred_published", d)
                y = lk.get(r.backend, r.chain, r.exec_epoch, c, "err_true", d)
                if p is None or y is None:
                    continue
                ctx = np.zeros(6 + 3 + 1, np.float32)
                ctx[ci] = 1
                ctx[6 + [1, 4, 7].index(r.k)] = 1
                ctx[9] = d / 0.03
                H.append(seq)
                P.append(np.log(p))
                C.append(ctx)
                Y.append(np.log(y))
                meta.append((r.split, r.k, reg, c, r.backend, r.chain, r.epoch))
    meta = pd.DataFrame(meta, columns=["split", "k", "regime", "circuit", "backend", "chain", "epoch"])
    return (np.asarray(H, np.float32), np.asarray(P, np.float32), np.asarray(C, np.float32),
            np.asarray(Y, np.float32), meta)


class Net(nn.Module):
    def __init__(self, kind, d=32):
        super().__init__()
        self.kind = kind
        if kind == "transformer":
            self.inp = nn.Linear(15, d)
            self.pos = nn.Parameter(torch.zeros(1, L, d))
            layer = nn.TransformerEncoderLayer(d, 4, 2 * d, 0.1, batch_first=True, norm_first=True, activation="gelu")
            self.enc = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
            hdim = 2 * d
        elif kind == "mlp_summary":
            hdim = 45
        else:
            hdim = 0
        self.head = nn.Sequential(nn.Linear(hdim + A + 10, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, A))

    def forward(self, h, p, c):
        if self.kind == "transformer":
            z = self.enc(self.inp(h) + self.pos)
            feats = [torch.cat([z[:, -1], z.mean(1)], 1)]
        elif self.kind == "mlp_summary":
            feats = [torch.cat([h[:, -1], h.mean(1), h.std(1)], 1)]
        else:
            feats = []
        return self.head(torch.cat(feats + [p, c], 1))      # residual on log error


def train(kind, H, P, C, Y, tr, es, seed, epochs=60, patience=8):
    torch.manual_seed(seed)
    np.random.seed(seed)
    mu, sd = H[tr].reshape(-1, 15).mean(0), H[tr].reshape(-1, 15).std(0) + 1e-6
    Hs = (H - mu) / sd
    pm, ps = P[tr].mean(0), P[tr].std(0) + 1e-6
    Ps = (P - pm) / ps
    R = Y - P
    T = lambda x: torch.from_numpy(np.ascontiguousarray(x))
    Ht, Pt, Ct, Rt = T(Hs), T(Ps), T(C), T(R)
    net = Net(kind)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    tri, esi = np.flatnonzero(tr), np.flatnonzero(es)
    best, state, bad = np.inf, None, 0
    for ep in range(epochs):
        net.train()
        perm = np.random.permutation(tri)
        for i in range(0, len(perm), 256):
            b = torch.from_numpy(perm[i:i + 256])
            opt.zero_grad()
            loss = ((net(Ht[b], Pt[b], Ct[b]) - Rt[b]) ** 2).mean()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            b = torch.from_numpy(esi)
            v = ((net(Ht[b], Pt[b], Ct[b]) - Rt[b]) ** 2).mean().item()
        if v < best - 1e-5:
            best, bad, state = v, 0, {k: x.clone() for k, x in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(state)
    net.eval()
    with torch.no_grad():
        corr = np.concatenate([net(Ht[i:i + 4096], Pt[i:i + 4096], Ct[i:i + 4096]).numpy() for i in range(0, len(H), 4096)])
    return P + corr, dict(best_es_mse=best, epochs=ep + 1)


def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    H, P, C, Y, meta = build()
    print(f"built {len(H)} samples in {time.time() - t0:.0f}s:", meta.groupby("split").size().to_dict(), flush=True)
    trn = (meta.split == "train").to_numpy()
    cut = np.quantile(meta.epoch[trn], 0.8)
    es = trn & (meta.epoch.to_numpy() > cut)
    tr = trn & ~es
    val = (meta.split == "val").to_numpy()
    E = np.exp(Y)                                            # true expected errors
    res, logs = {}, {}
    res["persistence"] = E[np.arange(len(E)), P.argmin(1)]
    res["oracle"] = E.min(1)
    for kind in ["no_history", "mlp_summary", "transformer"]:
        preds = []
        for s in SEEDS:
            t1 = time.time()
            pr, info = train(kind, H, P, C, Y, tr, es, s)
            preds.append(pr)
            logs[f"{kind}_seed{s}"] = dict(info, seconds=time.time() - t1)
            print(f"{kind} seed {s}: {info} ({time.time() - t1:.0f}s)", flush=True)
        pr = np.mean(preds, 0)
        res[f"dft_{kind}"] = E[np.arange(len(E)), pr.argmin(1)]
    # best fixed: chosen on the training period (no validation information)
    bf = E[trn].mean(0).argmin()
    res["best_fixed_train"] = E[:, bf]
    df = meta.copy()
    for k, v in res.items():
        df[k] = v
    df.to_parquet(f"{OUT}/dft_results.parquet")
    json.dump(logs, open(f"{OUT}/dft_train_log.json", "w"), indent=2)
    print(f"total {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
