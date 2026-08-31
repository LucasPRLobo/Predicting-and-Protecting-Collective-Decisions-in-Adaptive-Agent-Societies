import numpy as np
from .tables import island_outcomes
from .society import validate


def _move_value(key, x, pi, leaks, cuts):

    x = np.asarray(x)

    if key in leaks:
        return (leaks[key] @ x /2, 0.0)
    
    elif key in cuts:

        P_plus, P_minus, P_none = island_outcomes(cuts[key], x)

        return (P_plus - (1 + pi @ x)/2 , P_none)
    
    else:
        raise KeyError(f"move {key} is in neither the leak nor the cut table; tables and run disagree")
    

def _account_disagreement(i, j, x, G, pi, leaks, cuts):
    C = []
    N = G.shape[0]

    for k in range(N):
        if k != i and x[k] == x[i] and G[i,k] == 0.0:
            C.append(k)

    if len(C) == 0:
        return (0, 0.0, 0.0)
    
    else:
        mean_plus = 0
        mean_none = 0

        for k in C:
           value_plus, value_none = _move_value((i,j,k), x, pi, leaks, cuts)
           mean_plus += value_plus
           mean_none += value_none

        mean_plus = mean_plus/len(C)
        mean_none = mean_none/len(C)

    return (1, mean_plus, mean_none)


def frozen_run(G, x0, tables, rng, max_steps=10_000_000):
    N = G.shape[0]
    x = np.array(x0, dtype=int, copy=True)
    
    cumulative = np.cumsum(G, axis=1)
    cumulative[:, -1] = 1.0

    pi, leaks, cuts = tables

    winner_sum = 0.0
    split_sum = 0.0
    D = 0

    for t in range(max_steps):
        if np.all(x == x[0]):
            return winner_sum, split_sum, D, t
        
        # pick node i uniformly at random
        i = rng.integers(N)

        # Pick j
        j = np.searchsorted(cumulative[i], rng.random())

        if x[i] != x[j]:
            count, mean_plus, mean_none = _account_disagreement(i, int(j), x, G, pi, leaks, cuts)

            D += count
            winner_sum += mean_plus
            split_sum += mean_none

            x[i] = x[j]

    raise RuntimeError(f'No consensus after {max_steps} steps')
            

def frozen_estimates(G, x0, tables, n_runs, rng):
    """
    Returns
    -------
    dict with the frozen run estimates for the state (G, x0):

    h1        first order winner coefficient: the expected total voting
              power leaked to the +1 side over one frozen game, so that
              P(+1) is approximately h0 + p * h1 (Theorem 3). Unbiased.

    h1_se     Monte Carlo standard error of h1 over the n_runs games.

    f1_none   first order split coefficient: the expected total split
              probability created by the cutting rewires over one frozen
              game, so that P(no consensus) is approximately p * f1_none.

    f1_none_se  Monte Carlo standard error of f1_none.

    E_D       expected number of rewire opportunities per game (a 
              disagreement whose agent has somewhere to rewire to). This
              is the count behind the certificate: rho = p * E_D is the
              expected number of rewires per game.

    E_tau     expected number of steps of a frozen game (its consensus
              time). Reported for cost accounting and diagnostics.
    """

    G, x0 = validate(G, x0)

    winner_sums = []
    split_sums = []
    D_sums = 0
    tau = 0


    for _ in range(n_runs):
        winner_cont, split_cont, D_cont, t = frozen_run(G, x0, tables, rng, max_steps=10_000_000)

        winner_sums.append(winner_cont)
        split_sums.append(split_cont)
        D_sums += D_cont
        tau += t

    winner_sums = np.asarray(winner_sums)
    split_sums = np.asarray(split_sums)

    return {
        "h1": winner_sums.mean(), 
        "h1_se": winner_sums.std(ddof=1) / np.sqrt(n_runs),
        "f1_none": split_sums.mean(),
        "f1_none_se": split_sums.std(ddof=1) / np.sqrt(n_runs),
        "E_D": D_sums / n_runs, 
        "E_tau": tau / n_runs,
    }
    



