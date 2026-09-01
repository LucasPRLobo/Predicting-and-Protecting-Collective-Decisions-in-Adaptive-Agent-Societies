import os, sys, csv, argparse, time
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from adaptive_voting import validate, build_tables, frozen_estimates, predict, simulate_outcomes
from generators import sparse_random, upward_tree, directed_ring, voiceless_ring, place
HERE = os.path.dirname(__file__)
OUT_CSV, OUT_PNG = os.path.join(HERE, "accuracy.csv"), os.path.join(HERE, "accuracy.png")


# ---- Configuration ----

MASTER_SEED = 7
RHOS = (0.25, 0.5, 1.0)
N_FROZEN = 3000
N_GAMES = 3000
PLACEMENTS = [("scattered", 0.25), ("scattered", 0.5), ("scattered", 0.75), ("clustered", 0.5)]

# (family, builder taking the rng, number of instances)
INSTANCES = [
    ("sparse1",    lambda rng: sparse_random(20, rng, (1, 2)), 4),
    ("sparse1_40", lambda rng: sparse_random(40, rng, (1, 2)), 2),
    ("sparse2",    lambda rng: sparse_random(20, rng, (2, 2)), 2),
    ("tree_up",    lambda rng: upward_tree(4),                 1),
    ("ring",       lambda rng: directed_ring(20),              1),
    ("voiceless",  lambda rng: voiceless_ring(8, 6, rng),      1),
]

# ------------------------

def make_states(rng):
    """
    The societies and opinion states of the experiment, deterministic
    given the rng: every instance of every family, four placements each.
    """
    states = []
    for family, build, count in INSTANCES:
        for inst in range(count):
            G = build(rng)
            N = G.shape[0]
            for mode, frac in PLACEMENTS:
                k = max(1, int(round(frac * N)))
                x0 = place(G, k, mode, rng)
                states.append({"family": family, "inst": inst, "mode": mode, "frac": frac, "G": G, "x0": x0}) 
    return states


# ---- CSV Helpers ----
def load_rows():
    if not os.path.exists(OUT_CSV):
        return []
    rows = []
    for r in csv.DictReader(open(OUT_CSV)):
        for k, v in r.items():
            try:
                r[k] = int(v)
            except ValueError:

                try:
                    r[k] = float(v)
                except ValueError:
                    pass
        rows.append(r)

    return rows


def done_keys():
    return {(r["family"], r["inst"], r["mode"], r["frac"], r["rho"]) for r in load_rows()}
        
    
def append_row(row):
    new = not os.path.exists(OUT_CSV) or os.path.getsize(OUT_CSV) == 0
    with open(OUT_CSV, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


# ---- Main Loop ----
def run_experiment(rng, n_frozen, n_games):
    states = make_states(rng)
    done = done_keys()
    t0 = time.time()

    for si, s in enumerate(states):
        keys_needed = [(s["family"], s["inst"], s["mode"], s["frac"], rho) for rho in RHOS]
        if all(k in done for k in keys_needed):
            continue                                    # resumed run: this state is complete
        G, x0 = validate(s["G"], s["x0"])               
        tables = build_tables(G) 
        est = frozen_estimates(G, x0, tables, n_frozen, rng)   # once per state, serves every rho

        if est["E_D"] == 0:
            continue       # no rewire opportunity ever: nothing to test

        for rho in RHOS:
            if (s["family"], s["inst"], s["mode"], s["frac"], rho) in done:
                continue         
            p = min(rho / est["E_D"], 0.9)
            pred = predict(G, x0, p, tables=tables, estimates=est)
            pred1 = predict(G, x0, p, a=0.0, tables=tables, estimates=est)   # first order split only
            sim = simulate_outcomes(G, x0, p, n_games, rng)

            append_row({
                  "family": s["family"], "inst": s["inst"], "mode": s["mode"], "frac": s["frac"],
                  "N": G.shape[0], "rho": rho, "p": p, "certificate": pred["certificate"],
                  "h0": pred["h0"], "h1": pred["h1"], "h1_se": pred["h1_se"],
                  "f1_none": pred["f1_none"], "f1_none_se": pred["f1_none_se"], "E_D": pred["E_D"],
                  "pred_plus": pred["P_plus"], "pred_none": pred["P_none"], "pred_none_first": pred1["P_none"],
                  "meas_plus": sim["+1"], "meas_plus_se": sim["+1_se"],
                  "meas_none": sim["none"], "meas_none_se": sim["none_se"], "n_games": n_games,
            })

        print(f"state {si + 1}/{len(states)} ({s['family']} inst {s['inst']} {s['mode']} {s['frac']}), {time.time() - t0:.0f}s", flush=True)


# ---- Report ----
def report(rows):
    if not rows:
        print("no rows yet"); return
    cols = {0.25: "tab:blue", 0.5: "tab:green", 1.0: "tab:red"}
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    for r in rows:
        c = cols.get(r["rho"], "grey")
        ax[0].plot(r["h0"], r["meas_plus"], "o", mfc="none", mec=c, ms=5, alpha=.5)
        ax[0].errorbar(r["pred_plus"], r["meas_plus"], yerr=r["meas_plus_se"], fmt="o", color=c, ms=4, alpha=.8)
        ax[1].plot(r["pred_none_first"], r["meas_none"], "o", mfc="none", mec=c, ms=5, alpha=.5)
        ax[1].errorbar(r["pred_none"], r["meas_none"], yerr=r["meas_none_se"], fmt="o", color=c, ms=4, alpha=.8)
    top = max(0.05, 1.2 * max(r["meas_none"] for r in rows))
    for a, lim, title, xlabel in ((ax[0], [0, 1], "winner: measured P(+1) against prediction",
                                    "prediction (filled: first order; hollow: static h0)"),
                                (ax[1], [0, top], "split: measured P(none) against prediction",
                                    "prediction (filled: two term; hollow: first order)")):
        a.plot(lim, lim, "k--", lw=1); a.set_xlim(lim); a.set_ylim(lim)
        a.set_title(title); a.set_xlabel(xlabel); a.set_ylabel("measured"); a.grid(alpha=.3)
    for rho, c in cols.items():
        ax[0].plot([], [], "o", color=c, label=f"rho = {rho}")
    ax[0].legend(); fig.tight_layout(); fig.savefig(OUT_PNG, dpi=140)
    print(f"figure written to {OUT_PNG}")


    def se_floor(r):
        return max(r["meas_plus_se"], 0.5 / np.sqrt(r["n_games"]))

    print(f"\n{'rho':>5} {'n':>3} | winner MAE h0 / first | within 2se h0 / first | split meas/first meas/two")
    for rho in RHOS:
        S = [r for r in rows if r["rho"] == rho]
        if not S:
            continue
        mae0 = np.mean([abs(r["meas_plus"] - r["h0"]) for r in S])
        mae1 = np.mean([abs(r["meas_plus"] - r["pred_plus"]) for r in S])
        w0 = np.mean([abs(r["meas_plus"] - r["h0"]) <= 2 * se_floor(r) for r in S])
        w1 = np.mean([abs(r["meas_plus"] - r["pred_plus"]) <= 2 * se_floor(r) for r in S])
        F = [r for r in S if r["f1_none"] > 0]
        num = sum(r["meas_none"] for r in F)
        d1 = sum(r["pred_none_first"] for r in F); d2 = sum(r["pred_none"] for r in F)
        r1 = num / d1 if d1 > 0 else float("nan"); r2 = num / d2 if d2 > 0 else float("nan")
        print(f"{rho:5.2f} {len(S):3d} | {mae0:.4f} / {mae1:.4f} | {w0:.2f} / {w1:.2f} | {r1:.2f} / {r2:.2f}")
    vis = [r for r in rows if abs(r["p"] * r["h1"]) > 2 * se_floor(r)]
    if vis:
        num = sum((r["meas_plus"] - r["h0"]) * r["p"] * r["h1"] / se_floor(r)**2 for r in vis)
        den = sum((r["p"] * r["h1"])**2 / se_floor(r)**2 for r in vis)
        print(f"\nvisible points (|p h1| > 2 se): n = {len(vis)}, MAE h0 {np.mean([abs(r['meas_plus']-r['h0']) for r in vis]):.4f}"
            f" vs first order {np.mean([abs(r['meas_plus']-r['pred_plus']) for r in vis]):.4f}, pooled slope {num/den:.2f}")
        

# ---- Main ----

def main():
      ap = argparse.ArgumentParser(description=__doc__)
      ap.add_argument("--quick", action="store_true", help="ten times fewer frozen games and games (smoke run)")
      ap.add_argument("--report", action="store_true", help="only redraw the figure and the table from the CSV")
      args = ap.parse_args()
      if not args.report:
          div = 10 if args.quick else 1
          rng = np.random.default_rng(MASTER_SEED)
          run_experiment(rng, N_FROZEN // div, N_GAMES // div)
      report(load_rows())
          
        
if __name__ == "__main__":
    main()