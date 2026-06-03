"""
experiments_full.py
-------------------
Full experiment suite matching the sprawozdanie structure:

  Exp 1 – Fixed iteration budget (10 000), insert_node, N = 5..50, 5 seeds
  Exp 2 – Dynamic budget (15·N²),          insert_node, N = 5..50, 5 seeds
  Exp 3 – Neighbourhood function comparison (3 methods), dynamic budget
  Exp 4 – Acceptance criterion comparison (violation vs distance), dynamic budget

Deduplication: identical (n, seed, iters, method, mode) runs are executed only
once and their result is shared across experiments that need it.

Resume: already-completed results are loaded from RESULTS_FILE.cache if it
exists, so an interrupted run can be continued without re-running finished trials.

Outputs:
  experiments_results.json  – raw results per experiment
  experiments_full.png      – multi-panel matplotlib figure
"""

import subprocess, sys, re, json, time, os, math
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

# ── Configuration ─────────────────────────────────────────────────────────────
MAIN_PY      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "main.py")
N_VALUES     = list(range(5, 51, 5))
SEEDS        = [1, 2, 3, 4, 5]
FIXED_ITERS  = 10_000
INITIAL_TEMP = 100.0
COOLING_RATE = 0.995
METHODS      = ["two_opt_swap", "swap_nodes", "insert_node"]
MODES        = ["violation", "distance"]
RESULTS_FILE = "experiments_results.json"
PLOT_FILE    = "experiments_full.png"
MAX_WORKERS  = 4

METHOD_CLR = {"two_opt_swap": "#E74C3C", "swap_nodes": "#9B59B6", "insert_node": "#27AE60"}
MODE_CLR   = {"violation": "#2196F3", "distance": "#FF5722"}
CLR_OK     = "#2ECC71"
CLR_FAIL   = "#E74C3C"


def dyn(n):
    return max(1, 15 * n * n)


# ── Single trial ──────────────────────────────────────────────────────────────

def run_trial(n, seed, iterations, method, acceptance_mode):
    cmd = [sys.executable, MAIN_PY,
           "--num-city",        str(n),
           "--iterations",      str(iterations),
           "--seed",            str(seed),
           "--method",          method,
           "--acceptance-mode", acceptance_mode,
           "--initial-temp",    str(INITIAL_TEMP),
           "--cooling-rate",    str(COOLING_RATE)]
    t0  = time.perf_counter()
    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
    wt  = time.perf_counter() - t0
    sa  = re.search(r"SA Best: Costs=([\d.]+), Feasible=(\w+)", out)
    nn  = re.search(r"NN:\s+([\d.]+)\s+\(Feasible=(\w+)\)", out)
    fi  = re.search(r"Feasible solution found at iteration:\s+(\d+)", out)
    st  = re.search(r"Time for simulated_annealing:\s+([\d.]+)s", out)
    return dict(
        n=n, seed=seed, iterations=iterations, method=method,
        acceptance_mode=acceptance_mode,
        sa_cost=float(sa.group(1)) if sa else None,
        sa_feasible=sa.group(2) == "True" if sa else False,
        nn_cost=float(nn.group(1)) if nn else None,
        feasible_iter=int(fi.group(1)) if fi else None,
        sa_time=float(st.group(1)) if st else wt,
    )


def _worker(args):
    """Module-level picklable worker for ProcessPoolExecutor."""
    key, n, seed, iters, method, mode = args
    return key, run_trial(n, seed, iters, method, mode)


# ── Task planning with deduplication ─────────────────────────────────────────

def plan():
    unique = {}
    exp_keys = {e: [] for e in ("exp1", "exp2", "exp3", "exp4")}

    def reg(exp, n, s, iters, method, mode):
        key = (n, s, iters, method, mode)
        if key not in unique:
            unique[key] = (key, n, s, iters, method, mode)
        exp_keys[exp].append(key)

    for n in N_VALUES:
        for s in SEEDS:
            reg("exp1", n, s, FIXED_ITERS, "insert_node", "violation")
            reg("exp2", n, s, dyn(n),      "insert_node", "violation")
            for m in METHODS:
                reg("exp3", n, s, dyn(n), m, "violation")
            for mo in MODES:
                reg("exp4", n, s, dyn(n), "insert_node", mo)

    return list(unique.values()), exp_keys


# ── Parallel runner with resume ───────────────────────────────────────────────

def run_all(unique_tasks, cache):
    pending = [t for t in unique_tasks if str(t[0]) not in cache]
    total   = len(pending)
    if total == 0:
        print("All tasks already cached.")
        return cache

    print(f"Running {total} trials ({len(unique_tasks)-total} loaded from cache)…")
    done = 0
    t0   = time.perf_counter()

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_worker, t): t for t in pending}
        for fut in as_completed(futures):
            key, r = fut.result()
            cache[str(key)] = r
            done += 1
            eta = (time.perf_counter() - t0) / done * (total - done)
            sym = "✓" if r["sa_feasible"] else "✗"
            print(f"[{done:3d}/{total}] N={r['n']:2d} s={r['seed']} "
                  f"{r['method']:<12s} {r['acceptance_mode']:<10s} "
                  f"{sym} cost={r['sa_cost']:.2f}  ETA {eta:.0f}s", flush=True)
            with open(RESULTS_FILE + ".cache", "w") as f:
                json.dump(cache, f)

    return cache


def build_exp(cache, exp_keys):
    return {exp: [cache[str(k)] for k in keys if str(k) in cache]
            for exp, keys in exp_keys.items()}


# ── Aggregation ───────────────────────────────────────────────────────────────

def sel(lst, **kw):
    out = lst
    for k, v in kw.items():
        out = [r for r in out if r[k] == v]
    return out


def frate(lst, **kw):
    f = sel(lst, **kw)
    return {n: (sum(r["sa_feasible"] for r in f if r["n"] == n) /
                max(1, sum(1 for r in f if r["n"] == n)) * 100)
            for n in N_VALUES}


def mean_n(lst, field, feasible_only=False, **kw):
    f = sel(lst, **kw)
    if feasible_only:
        f = [r for r in f if r["sa_feasible"]]
    out = {}
    for n in N_VALUES:
        vals = [r[field] for r in f if r["n"] == n and r[field] is not None]
        out[n] = float(np.mean(vals)) if vals else None
    return out


# ── Plotting ──────────────────────────────────────────────────────────────────

PT = dict(fontsize=13, fontweight="bold", pad=9, color="#1A1A2E")
METHOD_LBL = {"two_opt_swap": "inverse_swap", "swap_nodes": "swap_nodes", "insert_node": "insert_node"}
XS = N_VALUES


def styled(fig, spec):
    ax = fig.add_subplot(spec)
    ax.set_facecolor("#FFFFFF")
    ax.grid(True, alpha=0.22, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    return ax


def scatter_time(ax, results):
    for r in results:
        ax.scatter(r["n"], r["sa_time"],
                   color=CLR_OK if r["sa_feasible"] else CLR_FAIL,
                   marker="o" if r["sa_feasible"] else "x",
                   s=50, alpha=0.7, zorder=3, linewidths=1.2)


def bar_fr(ax, fr, color, off=0, w=3.5, label=None):
    ax.bar([n + off for n in XS], [fr[n] for n in XS],
           width=w, color=color, alpha=0.78, edgecolor="white", label=label)


FEAS_LEG = [
    Line2D([0],[0], marker="o", color="none", markerfacecolor=CLR_OK, markersize=8, label="Dopuszczalne"),
    Line2D([0],[0], marker="x", color=CLR_FAIL, markersize=8, markeredgewidth=1.8, label="Niedopuszczalne"),
    Line2D([0],[0], color="#333", lw=1.8, linestyle="--", label="Średnia"),
]


def make_plots(exp, path):
    fig = plt.figure(figsize=(20, 28))
    fig.patch.set_facecolor("#F4F6F9")
    gs  = gridspec.GridSpec(5, 2, figure=fig,
                            hspace=0.50, wspace=0.32,
                            top=0.95, bottom=0.04, left=0.07, right=0.97)
    fig.suptitle(
        "TSP-TW – Symulowane wyżarzanie: Pełna seria eksperymentów\n"
        f"N = {XS[0]}..{XS[-1]}, {len(SEEDS)} ziaren na N, "
        f"T₀ = {INITIAL_TEMP}, α = {COOLING_RATE}",
        fontsize=17, fontweight="bold", y=0.975, color="#1A1A2E")

    # ── Exp 1 ─────────────────────────────────────────────────────────────────
    ax = styled(fig, gs[0, 0])
    ax.set_title("Eks. 1: Czas SA vs N  (stała liczba iteracji: 10 000)", **PT)
    ax.set_xlabel("N"); ax.set_ylabel("Czas SA (s)")
    scatter_time(ax, exp["exp1"])
    mt = mean_n(exp["exp1"], "sa_time")
    ax.plot(XS, [mt[n] for n in XS], "#333", lw=1.8, linestyle="--")
    ax.legend(handles=FEAS_LEG, fontsize=10, framealpha=0.9)

    ax = styled(fig, gs[0, 1])
    ax.set_title("Eks. 1: Odsetek rozwiązań dopuszczalnych vs N  (10 000 iteracji)", **PT)
    ax.set_xlabel("N"); ax.set_ylabel("Dopuszczalne (%)"); ax.set_ylim(-5, 120)
    fr1 = frate(exp["exp1"])
    bar_fr(ax, fr1, CLR_OK)
    ax.axhline(50, color="grey", lw=1, linestyle=":", alpha=0.6)
    for n in XS: ax.text(n, fr1[n]+3, f"{fr1[n]:.0f}%", ha="center", fontsize=9)

    # ── Exp 2 ─────────────────────────────────────────────────────────────────
    ax = styled(fig, gs[1, 0])
    ax.set_title("Eks. 2: Czas SA vs N  (dynamiczne 15·N² iteracji)", **PT)
    ax.set_xlabel("N"); ax.set_ylabel("Czas SA (s)")
    scatter_time(ax, exp["exp2"])
    mt2 = mean_n(exp["exp2"], "sa_time")
    ax.plot(XS, [mt2[n] for n in XS], "#333", lw=1.8, linestyle="--")
    ax.legend(handles=FEAS_LEG, fontsize=10, framealpha=0.9)

    ax = styled(fig, gs[1, 1])
    ax.set_title("Eks. 2: Odsetek rozwiązań dopuszczalnych vs N  (dynamiczne 15·N² iteracji)", **PT)
    ax.set_xlabel("N"); ax.set_ylabel("Dopuszczalne (%)"); ax.set_ylim(-5, 120)
    fr2 = frate(exp["exp2"])
    bar_fr(ax, fr2, "#3498DB")
    ax.axhline(50, color="grey", lw=1, linestyle=":", alpha=0.6)
    for n in XS: ax.text(n, fr2[n]+3, f"{fr2[n]:.0f}%", ha="center", fontsize=9)

    # ── Exp 3 ─────────────────────────────────────────────────────────────────
    ax = styled(fig, gs[2, 0])
    ax.set_title("Eks. 3: Czas SA vs N dla różnych metod sąsiedztwa  (dyn. iteracje)", **PT)
    ax.set_xlabel("N"); ax.set_ylabel("Czas SA (s)")
    for m in METHODS:
        for r in sel(exp["exp3"], method=m):
            ax.scatter(r["n"], r["sa_time"], color=METHOD_CLR[m],
                       marker="o" if r["sa_feasible"] else "x",
                       s=35, alpha=0.45, linewidths=1.0)
        mt3 = mean_n(exp["exp3"], "sa_time", method=m)
        ax.plot(XS, [mt3[n] for n in XS], METHOD_CLR[m], lw=2.2, label=METHOD_LBL[m])
    handles = [Line2D([0],[0], color=METHOD_CLR[m], lw=2.2, label=METHOD_LBL[m]) for m in METHODS]
    handles += [
        Line2D([0],[0], marker="o", color="none", markerfacecolor="#666", markersize=7, label="Dopuszczalne"),
        Line2D([0],[0], marker="x", color="#666", markersize=7, markeredgewidth=1.5, label="Niedopuszczalne"),
    ]
    ax.legend(handles=handles, fontsize=10, framealpha=0.9)

    ax = styled(fig, gs[2, 1])
    ax.set_title("Eks. 3: Odsetek rozwiązań dopuszczalnych vs N dla każdej metody", **PT)
    ax.set_xlabel("N"); ax.set_ylabel("Dopuszczalne (%)"); ax.set_ylim(-5, 120)
    w3 = 2.6
    for m, off in zip(METHODS, [-w3, 0, w3]):
        bar_fr(ax, frate(exp["exp3"], method=m), METHOD_CLR[m], off=off, w=w3, label=METHOD_LBL[m])
    ax.axhline(50, color="grey", lw=1, linestyle=":", alpha=0.6)
    ax.legend(fontsize=10, framealpha=0.9)

    # ── Exp 4 ─────────────────────────────────────────────────────────────────
    MODE_LBL = {"violation": "naruszenia", "distance": "odległość"}

    ax = styled(fig, gs[3, 0])
    ax.set_title("Eks. 4: Odsetek rozwiązań dopuszczalnych vs N wg kryterium akceptacji", **PT)
    ax.set_xlabel("N"); ax.set_ylabel("Dopuszczalne (%)"); ax.set_ylim(-5, 120)
    w4 = 1.9
    for mo, off in zip(MODES, [-w4/2 - 0.3, w4/2 + 0.3]):
        bar_fr(ax, frate(exp["exp4"], acceptance_mode=mo),
               MODE_CLR[mo], off=off, w=w4, label=MODE_LBL[mo])
    ax.axhline(50, color="grey", lw=1, linestyle=":", alpha=0.6)
    ax.legend(fontsize=11, framealpha=0.9)

    ax = styled(fig, gs[3, 1])
    ax.set_title("Eks. 4: Średni koszt vs N  (tylko rozwiązania dopuszczalne)", **PT)
    ax.set_xlabel("N"); ax.set_ylabel("Średni koszt")
    for mo in MODES:
        costs = mean_n(exp["exp4"], "sa_cost", feasible_only=True, acceptance_mode=mo)
        vx = [n for n in XS if costs[n] is not None]
        if vx:
            ax.plot(vx, [costs[n] for n in vx], MODE_CLR[mo], lw=2.2,
                    marker="o", markersize=6, label=MODE_LBL[mo])
    ax.legend(fontsize=11, framealpha=0.9)

    # ── Summary table ─────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[4, :])
    ax.axis("off")
    ax.set_title("Podsumowanie: Odsetek rozwiązań dopuszczalnych (%) wg N dla wszystkich eksperymentów", **PT)

    hdr = ["N", "Eksperyment 1\n10k stałe", "Eksperyment 2\ndyn 15N²",
           "Eksperyment 3\ninverse_swap", "Eksperyment 3\nswap_nodes", "Eksperyment 3\ninsert_node",
           "Eksperyment 4\nnaruszenia", "Eksperyment 4\nodległość"]
    hdr_c = ["#2C3E50", CLR_OK, "#3498DB",
              METHOD_CLR["two_opt_swap"], METHOD_CLR["swap_nodes"], METHOD_CLR["insert_node"],
              MODE_CLR["violation"], MODE_CLR["distance"]]

    fr_e1 = frate(exp["exp1"])
    fr_e2 = frate(exp["exp2"])
    fr_3t = frate(exp["exp3"], method="two_opt_swap")
    fr_3s = frate(exp["exp3"], method="swap_nodes")
    fr_3i = frate(exp["exp3"], method="insert_node")
    fr_4v = frate(exp["exp4"], acceptance_mode="violation")
    fr_4d = frate(exp["exp4"], acceptance_mode="distance")

    rows = [[str(n),
             f"{fr_e1[n]:.0f}%", f"{fr_e2[n]:.0f}%",
             f"{fr_3t[n]:.0f}%", f"{fr_3s[n]:.0f}%", f"{fr_3i[n]:.0f}%",
             f"{fr_4v[n]:.0f}%", f"{fr_4d[n]:.0f}%"] for n in XS]

    tbl = ax.table(cellText=rows, colLabels=hdr,
                   cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False); tbl.set_fontsize(11)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#DDDDDD"); cell.set_linewidth(0.5)
        if row == 0:
            cell.set_facecolor(hdr_c[col] if col < len(hdr_c) else "#2C3E50")
            cell.set_text_props(color="white", fontweight="bold", fontsize=10)
        elif col == 0:
            cell.set_facecolor("#F0F0F0"); cell.set_text_props(fontweight="bold")
        else:
            val = float(rows[row-1][col].rstrip("%"))
            a   = val / 100 * 0.4 + 0.05
            hx  = hdr_c[col].lstrip("#")
            ri, gi, bi = int(hx[0:2],16), int(hx[2:4],16), int(hx[4:6],16)
            bg = (1-a)*255
            cell.set_facecolor(((ri*a+bg)/255, (gi*a+bg)/255, (bi*a+bg)/255))

    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Plot saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    unique_tasks, exp_keys = plan()
    total_refs = sum(len(v) for v in exp_keys.values())
    print("=" * 70)
    print("TSP-TW – Full Experiment Suite")
    print(f"  N values     : {N_VALUES}")
    print(f"  Seeds        : {SEEDS}")
    print(f"  Unique runs  : {len(unique_tasks)}  (saves {total_refs-len(unique_tasks)} duplicate runs)")
    print(f"  Workers      : {MAX_WORKERS}")
    print("=" * 70)

    cache_path = RESULTS_FILE + ".cache"
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)
        print(f"Loaded {len(cache)} cached results from {cache_path}")
    else:
        cache = {}

    t0    = time.perf_counter()
    cache = run_all(unique_tasks, cache)
    print(f"\nAll runs finished in {time.perf_counter()-t0:.1f}s")

    exp = build_exp(cache, exp_keys)
    with open(RESULTS_FILE, "w") as f:
        json.dump(exp, f, indent=2)
    print(f"Results saved → {RESULTS_FILE}")

    print("\n── Feasibility summary ──────────────────────────────────────────")
    for label, lst, kw in [
        ("Exp1 (fixed 10k)",    exp["exp1"], {}),
        ("Exp2 (dyn 15N²)",     exp["exp2"], {}),
        ("Exp3 two_opt_swap",   exp["exp3"], {"method": "two_opt_swap"}),
        ("Exp3 swap_nodes",     exp["exp3"], {"method": "swap_nodes"}),
        ("Exp3 insert_node",    exp["exp3"], {"method": "insert_node"}),
        ("Exp4 violation",      exp["exp4"], {"acceptance_mode": "violation"}),
        ("Exp4 distance",       exp["exp4"], {"acceptance_mode": "distance"}),
    ]:
        s = sel(lst, **kw)
        if s:
            f = sum(r["sa_feasible"] for r in s)
            print(f"  {label:<28s}: {f:3d}/{len(s)} ({f/len(s)*100:.0f}%)")

    make_plots(exp, PLOT_FILE)
    print("\nDone.")


if __name__ == "__main__":
    main()