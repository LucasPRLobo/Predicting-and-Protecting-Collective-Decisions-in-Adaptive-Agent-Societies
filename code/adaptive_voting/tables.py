import numpy as np

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