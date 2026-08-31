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
        targets = np.flatnonzero(G[list(S)].sum(axis=0))
        if set(targets) <= S:
            closed.append(S)
    return closed