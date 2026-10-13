"""
Stage 4 policies. Every policy picks one action from the shared ACTIONS list per
decision, and only from information published at decision time:

  static     : none, best fixed (chosen on validation)
  controller : argmin of the model-predicted expected error, with the noise model
               built from a state source (published = persistence, window_mean,
               ridge, transformer, or truth = oracle)
  bandit     : contextual LinUCB, Rahman & Nguyen (IEEE JSAC 2026), Eqs. 17–19:
                 UCB_k(x) = x·θ_k + η √(xᵀ A_k⁻¹ x);   k* = argmax_k UCB_k(x) − λ_c C̃_k
                 A_k ← ρ A_k + x xᵀ,  b_k ← ρ b_k + R x  (chosen arm only),  θ_k = A_k⁻¹ b_k
               Reward (cf. Eq. 16) R = −SCALE · (probe error + β · probe variance),
               using the Clifford probe circuit run with PROBE_FRACTION of the budget.
"""

import numpy as np

from src.qem.mitigation import ACTIONS, PROBE_FRACTION, SHOTS, action_cost

REWARD_SCALE = 100.0           # errors ~1e-2: rewards in percent so η is O(1) (our choice; unspecified in the paper)
COST = np.array([action_cost(a) for a in ACTIONS])
COST_NORM = COST / COST.max()  # C̃_k in [0, 1]

# Shot accounting (acceptance criterion 3): every policy spends SHOTS per observable
SHOTS_CONTROLLER = SHOTS
SHOTS_BANDIT_MAIN = int(SHOTS * (1 - PROBE_FRACTION))
SHOTS_BANDIT_PROBE = int(SHOTS * PROBE_FRACTION)
assert SHOTS_BANDIT_MAIN + SHOTS_BANDIT_PROBE == SHOTS_CONTROLLER


def controller_choice(pred):
    """pred: (n_units, n_actions) predicted expected errors -> chosen action index."""
    return np.argmin(pred, axis=1)


class LinUCB:
    def __init__(self, n_arms, dim, eta=1.0, rho=1.0, lam0=1.0, lam_c=0.0):
        self.A = np.stack([lam0 * np.eye(dim) for _ in range(n_arms)])
        self.b = np.zeros((n_arms, dim))
        self.eta, self.rho, self.lam_c = eta, rho, lam_c

    def select(self, x):
        Ainv = np.linalg.inv(self.A)
        theta = np.einsum("kij,kj->ki", Ainv, self.b)
        bonus = np.sqrt(np.einsum("i,kij,j->k", x, Ainv, x))
        return int(np.argmax(theta @ x + self.eta * bonus - self.lam_c * COST_NORM))

    def update(self, k, x, r):
        self.A[k] = self.rho * self.A[k] + np.outer(x, x)
        self.b[k] = self.rho * self.b[k] + r * x

    def pseudo_update_all(self, x, rewards):
        """Full-information pseudo-observation for every arm (warm start from the model)."""
        for k, r in enumerate(rewards):
            self.A[k] += np.outer(x, x)
            self.b[k] += r * x


def run_bandit_stream(X, probe_err, probe_var, eval_err, params, variant="probe", oracle_err=None,
                      warm_pred=None, n_warm=10):
    """
    One stream (chain × circuit × regime), epochs in time order.
    X (T, d) contexts; probe_err / probe_var (T, A); eval_err (T, A) error of the main estimate.
    variant: 'probe' (realistic), 'warm' (model pseudo-rewards for the first n_warm epochs),
             'oracle' (reward = true expected error, no probe shots).
    Returns chosen arms (T,) and their evaluation errors (T,). Feedback from epoch t
    is used only from epoch t+1 onward.
    """
    T, A = probe_err.shape
    ucb = LinUCB(A, X.shape[1], eta=params["eta"], rho=params["rho"], lam0=1.0, lam_c=params["lam_c"])
    if variant == "warm":
        for t in range(min(n_warm, T)):
            w = -REWARD_SCALE * warm_pred[t]
            ucb.pseudo_update_all(X[t], w - (w.mean() if params.get("center", True) else 0.0))
    chosen = np.zeros(T, int)
    rewards, init_buf = [], []
    for t in range(T):
        # Initialisation: play every arm once (standard UCB initialisation), unless warm-started
        if variant != "warm" and t < A:
            k = t
        else:
            k = ucb.select(X[t])
        chosen[t] = k
        if variant == "oracle":
            r = -REWARD_SCALE * oracle_err[t, k]
        else:
            r = -REWARD_SCALE * (probe_err[t, k] + params["beta"] * probe_var[t, k])
        rewards.append(r)
        if params.get("center", True) and not (variant != "warm" and t < A):
            # Centre on the mean of past rewards (causal). After the initial round every
            # arm has data, so the zero-initialised ridge prior no longer favours unseen arms.
            ucb.update(k, X[t], r - np.mean(rewards[:-1]))
        elif params.get("center", True):
            if t == A - 1:
                c0 = np.mean(rewards)
                for kk, xx, rr in init_buf:
                    ucb.update(kk, xx, rr - c0)
                ucb.update(k, X[t], r - c0)
            else:
                init_buf.append((k, X[t], r))
        else:
            ucb.update(k, X[t], r)
    return chosen, eval_err[np.arange(T), chosen]


def sanity_check_bandit(seed=0):
    """Acceptance criterion 5: in a stationary stream where arm 3 is clearly best,
    the bandit converges to arm 3."""
    rng = np.random.default_rng(seed)
    T, A = 300, len(ACTIONS)
    mean = np.full(A, 0.05)
    mean[3] = 0.02
    probe = np.clip(mean + rng.normal(0, 0.01, (T, A)), 0, None)
    X = np.ones((T, 1))
    chosen, _ = run_bandit_stream(X, probe, np.zeros((T, A)), np.tile(mean, (T, 1)),
                                  dict(eta=1.0, rho=1.0, lam_c=0.0, beta=0.0))
    return float((chosen[-100:] == 3).mean())
