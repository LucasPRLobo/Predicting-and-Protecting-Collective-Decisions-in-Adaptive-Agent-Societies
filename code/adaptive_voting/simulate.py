import numpy as np

from .society import validate, closed_classes

def one_step(G, x, p, rng):
    """
    One asynchronous step of the adaptive process. MUTATES G and x in
    place; run owns the copies. Returns the event tag: 'agree',
    'copy', 'rewire', or 'skip'.

    The order of the random draws is part of the process definition
    (reproducibility across implementations depends on it): the agent
    i uniformly, its target j by the row weights, the rewire coin with
    probability p, and the new listener k uniformly from the candidate
    set (agents agreeing with i that i does not listen to yet).
    """

    N = G.shape[0]

    # Pick node i with probability 1/N
    i = rng.integers(N)
    # Pick target node j with probability G[i, j]
    row = G[i]
    cum = np.cumsum(row)
    cum[-1] = 1.0
    j = int(np.searchsorted(cum, rng.random()))

    # Agree: nothing happens
    if x[i] == x[j]:
        return "agree"
    
    # Disagree-Rewire (probability p): weight G[i,j] moves to a
    # uniformly chosen candidate; skip if no candidate exists
    if rng.random() < p:
        cand = np.flatnonzero((x == x[i]) & (row == 0.0))
        cand = cand[cand != i]
        if len(cand) == 0:
            return "skip"
        k = int(rng.choice(cand))
        G[i, k] = G[i, j]
        G[i, j] = 0.0
        return "rewire"
    
    # Disagree-Copy (probability 1-p): node i copies the opinion of j
    x[i] = x[j]
    return "copy"
      

def outcome(G, x, classes):
    """
    Referee: decide the game from the current state and the cached
    closed classes. Returns the STRING '+1', '-1', or 'none' (the three
    outcomes), or Python None for undecided (keep playing). The string
    'none' and the value None are different things.
    """

    if np.all(x == x[0]):
        return "+1" if x[0] == 1 else "-1"
    
    if len(classes) < 2:
        return None
    
    class_values = []
    for S in classes:
        opinions = x[list(S)]
        if not np.all(opinions == opinions[0]):
            return None
        class_values.append(opinions[0])

    # the trap: uniform classes with differing values (case A otherwise)
    if len(set(class_values)) > 1:
        return "none"
    
    return None



def run(G0, x0, p, rng, max_steps=10_000_000):
    """Run one game to its ending. Never mutates G0 or x0.

    Returns a record dict: outcome ('+1', '-1', or 'none'), time (steps
    taken before the decision), events (tag counts for agree, copy,
    rewire, skip). Raises RuntimeError if max_steps is reached: on a
    valid society that means something is wrong, and an exception
    cannot be silently averaged into results.

    Three decisions encoded here: (1) the one copy per game lives here
    and nowhere else (one_step mutates its arguments by contract);
    (2) the outcome check precedes the step, so decided starts report
    time 0, the tau convention of the paper; (3) the closed classes
    are recomputed after every rewire; copies never change the graph,
    so the cache is always current.
    """
    G, x = validate(G0, x0)
    G = G.copy()               # validate may return views of the inputs
    x = x.copy()

    events = {"agree": 0, "copy": 0, "rewire": 0, "skip": 0}
    classes = closed_classes(G)

    for t in range(max_steps):
        result = outcome(G, x, classes)
        if result is not None:
            return {"outcome": result, "time": t, "events": events}
        tag = one_step(G, x, p, rng)
        events[tag] += 1
        if tag == "rewire":
            classes = closed_classes(G)

    raise RuntimeError(f"no decision after {max_steps} steps")


def simulate_outcomes(G0, x0, p, n_runs, rng):
    """The empirical twin of predict: run n_runs games of the adaptive
    process from (G0, x0) at rate p and return the measured frequencies
    of the three endings with their standard errors, plus the mean
    decision time in steps. Use it to verify any prediction by brute
    force; the cross check script does exactly that.

    Returns a dict with keys '+1', '-1', 'none' (frequencies),
    '+1_se', '-1_se', 'none_se' (binomial standard errors), and
    'mean_time'.
    """
    plus_count = 0
    minus_count = 0
    none_count = 0
    times = []

    for _ in range(n_runs):
        run_result = run(G0, x0, p, rng)
        run_outcome = run_result['outcome']
        times.append(run_result['time'])

        if run_outcome == "+1":
            plus_count += 1

        elif run_outcome == "-1":
            minus_count += 1 

        elif run_outcome == "none":
            none_count +=1

        else:
            raise AssertionError(f'Impossible outcome {run_outcome}')
        
    out = {}
    for name, count in (("+1", plus_count), ("-1", minus_count), ("none", none_count)):
        f = count / n_runs
        out[name] = f
        out[name + "_se"] = float(np.sqrt(f * (1.0 - f) / n_runs))
    out["mean_time"] = float(np.mean(times))

    return out



def enumerate_branches(G0, x0, p):
    """All one-step branches from state (G, x), analytically.

    Returns a list of (prob, event, key) where key identifies the
    branch: ('noop',) for agree/skip, ('copy', i, j) for a copy,
    ('rewire', i, j, k) for a rewire. Deterministic; no rng. The
    analytical twin of one_step: every rng draw there is a loop or a
    probability factor here.
    """
    branches = []
    N = G0.shape[0]
    for i in range(N):
        row = G0[i]
        for j in range(N):
            if row[j] > 0:
                base = (1 / N) * row[j]
                if x0[i] == x0[j]:
                    branches.append((base, "agree", ("noop",)))
                else:
                    branches.append((base * (1 - p), "copy", ("copy", i, j)))
                    C = np.flatnonzero((x0 == x0[i]) & (row == 0.0))
                    C = C[C != i]
                    if len(C) == 0:
                        branches.append((base * p, "skip", ("noop",)))
                    else:
                        for k in C:
                            branches.append(
                                (base * p / len(C), "rewire",
                                 ("rewire", i, j, int(k)))
                            )
    return branches