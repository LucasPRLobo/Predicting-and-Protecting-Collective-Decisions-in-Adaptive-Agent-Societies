import numpy as np

from .society import validate
from .tables import build_tables
from .frozen import frozen_estimates


def _clip(v):
    return min(max(v, 0.0), 1.0)

def predict(G, x0, p, n_runs=2000, rng=None, a=0.038) -> dict:

    if rng is None:
        rng = np.random.default_rng()

    G, x0 = validate(G, x0)

    tables = build_tables(G)
    estimates = frozen_estimates(G, x0, tables, n_runs, rng)

    pi, leaks, cuts = tables
    h0_val = (1 + pi @ x0) / 2

    # rho: expected number of rewires per game at rate p; the expansion
    # parameter and the certificate of the prediction.
    rho = p * estimates["E_D"]

    # first order winner prediction: the static h0 plus p times the
    # expected voting power leaked to +1 along a frozen game.
    P_plus_raw = h0_val + p * estimates["h1"]

    # split prediction at two orders: first order (the first rewire
    # that cuts, p * f1_none) plus the second order term a * rho^2 / 2
    # (later rewires cut a society the earlier ones carved along its
    # opinion boundary; a = 0.038 measured on sparse societies, N <= 40;
    # a = 0 gives the pure first order prediction).
    P_none_raw = p * estimates["f1_none"] + a * rho**2 / 2

    # the remainder: the three endings are exhaustive.
    P_minus_raw = 1.0 - P_plus_raw - P_none_raw


    if rho <= 1.0:
        certificate = "calibrated"
    elif rho <= 2.0:
        certificate = "degrading"
    else:
        certificate = "out of range"

    P_plus, P_minus, P_none = _clip(P_plus_raw), _clip(P_minus_raw), _clip(P_none_raw)

    return {
        "P_plus": P_plus, "P_minus": P_minus, "P_none": P_none,
        "P_plus_raw": P_plus_raw, "P_minus_raw": P_minus_raw, "P_none_raw": P_none_raw,
        "h0": h0_val,
        "h1": estimates["h1"], "h1_se": estimates["h1_se"],
        "f1_none": estimates["f1_none"], "f1_none_se": estimates["f1_none_se"],
        "E_D": estimates["E_D"], "E_tau": estimates["E_tau"],
        "rho": rho, "certificate": certificate,
        "p": p, "n_runs": n_runs, "a": a,
    }


def influence(G, x0, p, n_runs=1000, rng=None):
    G, x0 = validate(G, x0)

    if rng is None:
        rng = np.random.default_rng()

    N = G.shape[0]
    tables = build_tables(G)
    pi, leaks, cuts = tables

    v = np.zeros(N)
    v_se = np.zeros(N)

    for i in range(N):

        x_plus = x0.copy()
        x_minus = x0.copy()
        
        # Modify the i-th entry of each copy
        x_plus[i] = 1
        x_minus[i] = -1

        est_plus = frozen_estimates(G, x_plus, tables, n_runs, rng)
        est_minus = frozen_estimates(G, x_minus, tables, n_runs, rng)

        v[i] = pi[i] + p * (est_plus['h1'] - est_minus['h1'])
        v_se[i] = p * np.sqrt(est_plus["h1_se"]**2 + est_minus["h1_se"]**2)

    return v, v_se


    