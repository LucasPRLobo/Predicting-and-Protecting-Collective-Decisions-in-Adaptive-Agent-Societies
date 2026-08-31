import numpy as np

from .society import validate
from .tables import build_tables
from .frozen import frozen_estimates


def _clip(v):
    return min(max(v, 0.0), 1.0)

def predict(G, x0, p, n_runs=2000, rng=None, a=0.038) -> dict:
    """Predict the three endings of the adaptive process from the
    starting state, before running it.

    The algorithm (Section 11 of the theory review, Section 5 of the
    paper): (1) the stationary distribution pi of G and the static
    prediction h0 = (1 + pi . x0)/2; (2, 3) the leak and cut tables,
    one entry per single rewire move, computed once per society;
    (4) n_runs frozen games from x0 pricing every rewire opportunity;
    (5) the estimates h1, f1_none, E_D, E_tau; (6) the predictions
    P(+1) = h0 + p h1, P(none) = p f1_none + a (p E_D)^2 / 2, P(-1)
    the remainder; (7) the certificate rho = p E_D.

    Parameters
    ----------
    G, x0 : the society (row stochastic matrix, one closed class) and
        the starting opinions in {-1, +1}.
    p : the rewiring rate.
    n_runs : frozen games for the estimates; the standard errors in
        the output say whether it was enough.
    rng : a numpy Generator; pass a seeded one for reproducibility.
    a : the second order split coefficient per rewire; 0.038 was
        measured on sparse societies at N <= 40 (its derived form is
        the g tilde of the paper); a = 0 gives the pure first order
        prediction. The term is applied only when the society has
        single rewire cuts (f1_none > 0); on societies without cuts
        splitting is a many rewire effect that this constant does not
        describe (dense societies: negligible; rings: underpredicted,
        the derived g tilde is needed). The output field
        second_order_applied says which case applied.

    Returns
    -------
    dict with the three clipped probabilities (P_plus, P_minus,
    P_none), their raw unclipped values (a difference between the two
    means the expansion left its range on this state), the ingredients
    (h0, h1, f1_none with standard errors, E_D, E_tau), the
    certificate rho with its band ("calibrated": rho <= 1, predictions
    within noise on every structure measured; "degrading": rho <= 2,
    off by 15 to 20 percent toward larger effects; "out of range"
    beyond), and the echoed inputs p, n_runs, a.
    """
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
    # The second order term is applied only when the society has single
    # rewire cuts (f1_none > 0): a was measured on such societies, and on
    # dense societies without cuts splitting needs many rewires and the
    # term overpredicts (playground finding, 2026-08-30). Known exception:
    # rings, which split at the second rewire without any single cut; the
    # derived coefficient g tilde of the paper covers them, a does not.
    second_order = estimates["f1_none"] > 0.0
    P_none_raw = p * estimates["f1_none"] + (a * rho**2 / 2 if second_order else 0.0)

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
        "second_order_applied": bool(second_order),
        "p": p, "n_runs": n_runs, "a": a,
    }


def influence(G, x0, p, n_runs=1000, rng=None):
    """The adaptive voting power of every agent at rate p: the
    predicted change of the winning probability when the opinion of
    agent i flips from -1 to +1,

        v_i(p) = pi_i + p * (h1 at x0 with x_i = +1  minus
                             h1 at x0 with x_i = -1).

    At p = 0 this equals pi exactly (static voting power), whatever
    the Monte Carlo noise, because p multiplies the correction. It is
    attribution free (no choice of which agent to credit) and it is
    the quantity validated against simulated opinion flips in the
    paper. Cost: the tables once, then 2 N frozen estimate calls of
    n_runs games each.

    Returns (v, v_se): the influence vector and its Monte Carlo
    standard errors.
    """
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


    