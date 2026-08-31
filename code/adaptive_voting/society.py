"""The state of a society: validation, closed classes, voting power, h0.

Notation of the paper: the state is (G, x). G is an N by N row
stochastic matrix, G[i, j] the share of attention agent i gives to
agent j; x is a vector of opinions in {-1, +1}.

Convention on comparisons: values that are ASSIGNED (a diagonal entry,
an absent edge) are compared exactly to zero; values that are COMPUTED
(row sums, eigenvectors) are compared with tolerance TOL.
"""

import numpy as np
import networkx as nx

TOL = 1e-12


def validate(G, x):
    """Check that (G, x) is a valid state and return the coerced pair.

    G is coerced to a float array and x to an int array; the coerced
    (G, x) is returned so callers use it. Raises ValueError naming the
    violated condition otherwise.
    """
    G = np.asarray(G, dtype=float)
    x = np.asarray(x)

    if G.ndim != 2 or G.shape[0] != G.shape[1]:
        raise ValueError(f"G must be a square matrix; got shape {G.shape}")
    n = G.shape[0]

    row_err = np.abs(G.sum(axis=1) - 1.0).max()
    if row_err > TOL:                       # computed value: tolerance
        raise ValueError(f"G is not row stochastic: worst row sum error {row_err:.2e}")
    if (G < 0).any():
        raise ValueError("G has a negative entry")
    if np.diag(G).any():                    # assigned value: exact zero
        raise ValueError("G has a nonzero diagonal entry (agents do not listen to themselves)")

    if x.ndim != 1 or x.shape[0] != n:
        raise ValueError(f"x must be a vector of length {n}; got shape {x.shape}")
    if not np.isin(x, (-1, 1)).all():
        raise ValueError("x has an entry other than -1 and +1")

    return G, x.astype(int)


def closed_classes(G):
    """Closed communicating classes of the positive-weight digraph of G:
    the strongly connected components with no edge leaving them.
    Returns a list of node sets."""
    
    adj_mat = (G > 0).astype(float)
    di_graph = nx.DiGraph(adj_mat)
    scc = list(nx.strongly_connected_components(di_graph))

    closed = []

    for S in scc:
        members = sorted(int(i) for i in S)
        targets = {int(t) for t in np.flatnonzero(G[members].sum(axis=0))}
        if targets <= set(members):
            closed.append(members)
    return sorted(closed)


def voting_power(G):
    """The stationary distribution pi of G: pi G = pi, entries summing
    to one. Voting power is undefined when G has more than one closed
    class (no unique stationary distribution), so that case raises.
    Agents outside the closed class have pi_i exactly zero: entries
    below TOL are computed values assigned to zero (module docstring
    convention), then the vector is renormalised."""
    n_classes = len(closed_classes(G))
    if n_classes != 1:
        raise ValueError(f"voting_power needs exactly one closed class; found {n_classes}")
    eigen_vals, eigen_vecs = np.linalg.eig(G.T)
    k = int(np.argmin(np.abs(eigen_vals - 1)))
    w = np.real(eigen_vecs[:, k])
    s = w.sum()
    if abs(s) < 1e-9:
        raise ArithmeticError("eigenvector sums to ~0 (degenerate eigenvalue 1)")
    pi = w / s
    pi[np.abs(pi) < TOL] = 0.0       # exact zero for agents outside the closed class
    pi = pi / pi.sum()
    # self-check of the defining property: catches a forgotten
    # transpose or a decaying-eigenvector pick instantly
    if not np.allclose(pi @ G, pi):
        raise ArithmeticError("pi G != pi: transpose or selection bug")
    return pi

def h0(G, x):
    """The static winning probability of +1: (1 + pi . x) / 2
    (Cooper and Rivera, in our encoding)."""
    pi_G = voting_power(G)
    return (1 + pi_G @ np.asarray(x)) / 2