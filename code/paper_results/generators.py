import numpy as np
from adaptive_voting import closed_classes

def sparse_random(N, rng, out_degree=(1,2)):
    while True: 
        G = np.zeros((N, N))
        for i in range(N):
            d = int(rng.integers(out_degree[0], out_degree[1] + 1))
            targets = rng.choice([j for j in range(N) if j != i], size=d, replace=False)
            G[i, targets] = 1.0 / d
        if len(closed_classes(G)) == 1:
            return G 

def upward_tree(depth):
    """Complete binary tree with 2^(depth+1) - 1 agents; children of v are
    2v+1 and 2v+2. Every agent listens to its parent (weight 1); the root
    listens to its two children (weight 1/2 each). The closed class is the
    root with its two children; every other agent is voiceless. Single
    rewires can cut this society (a subtree rewired into itself)."""
    N = 2 ** (depth + 1) - 1
    G = np.zeros((N, N))
    for v in range(1, N):
        G[v, (v - 1) // 2] = 1.0
    G[0, 1] = G[0, 2] = 0.5
    assert len(closed_classes(G)) == 1
    return G


def directed_ring(N):
    """Agent i listens only to agent i + 1 (mod N). Strongly connected, one
    closed class, no single rewire cuts it; it splits at the second rewire
    (the case the paper names)."""
    G = np.zeros((N, N))
    for i in range(N):
        G[i, (i + 1) % N] = 1.0
    assert len(closed_classes(G)) == 1
    return G


def voiceless_ring(n_core, n_hangers, rng):
    """A bidirected ring of n_core agents (each listens to both ring
    neighbours, weight 1/2) plus n_hangers agents that each listen to one
    random ring agent (weight 1). The hangers are voiceless."""
    N = n_core + n_hangers
    G = np.zeros((N, N))
    for i in range(n_core):
        G[i, (i + 1) % n_core] = 0.5
        G[i, (i - 1) % n_core] = 0.5
    for h in range(n_core, N):
        G[h, int(rng.integers(n_core))] = 1.0
    assert len(closed_classes(G)) == 1
    return G


def place(G, k, mode, rng):
    """Opinions with exactly k agents at +1 and the rest at -1.
    'scattered': the k agents uniformly at random. 'clustered': a ball
    grown on the undirected support of G from a random seed (close in
    the listening network, whichever direction), topped up at random if
    the ball cannot grow to k."""
    N = G.shape[0]
    x = -np.ones(N, dtype=int)
    if mode == "scattered":
        chosen = list(rng.choice(N, size=k, replace=False))
    elif mode == "clustered":
        A = (G > 0) | (G > 0).T
        seed = int(rng.integers(N))
        chosen, frontier, seen = [seed], [seed], {seed}
        while len(chosen) < k and frontier:
            nxt = []
            for u in frontier:
                for v in np.flatnonzero(A[u]):
                    if int(v) not in seen:
                        seen.add(int(v)); nxt.append(int(v))
            rng.shuffle(nxt)
            for v in nxt:
                if len(chosen) < k:
                    chosen.append(v)
            frontier = nxt
        if len(chosen) < k:                       # disconnected support: top up at random
            rest = [i for i in range(N) if i not in seen]
            chosen += list(rng.choice(rest, size=k - len(chosen), replace=False))
    else:
        raise ValueError(f"unknown placement mode {mode!r}")
    x[chosen] = 1
    return x
