import numpy as np
from .society import voting_power, closed_classes, _stationary, TOL


def apply_rewire(G, i, j, k):
    G_prime = G.copy()

    if G[i,j] == 0:
        raise ValueError(f'agent {i} (i) does not listen to node {j} (j). Nothing to move')
    if G_prime[i,k] != 0:
        raise ValueError(f'The entry of G ({i, k}) is already connected')
    if i == k:
        raise ValueError(f"The selected target node {k} is the same as the current node i")

    G_prime[i, k] = G_prime[i,j]
    G_prime[i,j] = 0.0

    return G_prime


def rewire_neighbours(G):
    N = G.shape[0]
    legal_rewires = []

    for i in range(N):
        for j in range(N):

            if G[i,j] >0:
                for k in range(N):
                    
                    if k != i and G[i,k] == 0:

                        legal_rewires.append((i,j,k))

    d = (G > 0).sum(axis=1)
    expected = int((d * (N - 1 - d)).sum())
    if len(legal_rewires) != expected:
        raise ArithmeticError(f"enumeration mismatch: {len(legal_rewires)} moves, expected {expected}")
    
    else:
        return legal_rewires
    

def build_tables(G):
    # Compute pi
    pi = voting_power(G)

    # Get G's legal rewires moves:
    rewires = rewire_neighbours(G)

    leaks = dict()
    cuts = dict()

    for (i,j,k) in rewires:
        Gp = apply_rewire(G, i ,j, k)
        Gp_cc = closed_classes(Gp)

        if len(Gp_cc) == 1:
            leak = voting_power(Gp) - pi
            leak[np.abs(leak) < TOL] = 0.0

            if abs(leak.sum()) > TOL:
                raise ArithmeticError(f"leak of move {(i, j, k)} sums to {leak.sum():.2e}")
            leaks[(i, j, k)] = leak 

        else:
            islands = []
            for S in Gp_cc:
                pi_island = np.zeros(G.shape[0])
                pi_island[S] = _stationary(Gp[np.ix_(S,S)])
                islands.append(pi_island)
            cuts[(i,j,k)] = islands

    
    return (pi, leaks, cuts)


def island_outcomes(island, x):
    """
    The three ending probabilities of a society already split into 
    these islands, frozen (each island decides on its own;
    global consensus needs all islands to agree).
    """
    x = np.asarray(x)

    P_plus = 1
    P_minus = 1

    for v in island:
        P_plus *= (1 + v @ x)/2
        P_minus *= (1 - v @ x)/2

    P_none = 1 - (P_plus + P_minus)

    return (P_plus, P_minus, P_none)