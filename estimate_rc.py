import math
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from collections import deque


# build an RGG on [0,1]^2 and returns (points, adjacency_list)
def construct_rgg(n, r_const, rng):
    points = rng.random((n, 2))
    radius = r_const / math.sqrt(n)

    # use a kd-tree so don't have to check all O(n^2) pairs
    tree = cKDTree(points)
    pairs = tree.query_pairs(radius, output_type="ndarray")

    adj = [[] for _ in range(n)]
    for a, b in pairs:
        adj[int(a)].append(int(b))
        adj[int(b)].append(int(a))

    return points, adj

#BFS from vertex 0, return True if it visits everything
def is_connected(adj, n):
    if n == 0:
        return True

    visited = [False] * n
    visited[0] = True
    queue = deque([0])
    count = 1

    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                count += 1
                queue.append(v)

    return count == n


# main experiment loop where for each (r, n) trials run for random graphs and record what fraction are connected
def run_experiment(n_values, r_values, trials, seed):
    rows = []

    for n in n_values:
        print(f"\n--- n = {n} ---")
        for r in r_values:
            connected_count = 0

            for t in range(trials):
                # each trial gets its own seed so that results are reproducible
                trial_seed = seed + 100000*n + 1000*int(round(r*100)) + t
                rng = np.random.default_rng(trial_seed)

                _, adj = construct_rgg(n, r, rng)
                if is_connected(adj, n):
                    connected_count += 1

            p_conn = connected_count / trials
            rows.append({"n": n, "r": float(r), "P_conn": p_conn, "trials": trials})
            print(f"  r={r:.2f}: P_conn={p_conn:.2f}")

    return pd.DataFrame(rows)

# find r where P_conn crosses target by linear interpolation
def interpolate_threshold(sub_df, target=0.5):
    sub_df = sub_df.sort_values("r")
    rs = sub_df["r"].to_numpy()
    ps = sub_df["P_conn"].to_numpy()

    for i in range(len(rs) - 1):
        if ps[i] <= target <= ps[i+1]:
            if ps[i+1] == ps[i]:
                return (rs[i] + rs[i+1]) / 2.0
            alpha = (target - ps[i]) / (ps[i+1] - ps[i])
            return rs[i] + alpha * (rs[i+1] - rs[i])
    return None  # didn't cross the threshold in this range


# for each n, get the r_c estimate at the 0.5 crossing
def compute_summary(df, n_values):
    rows = []
    for n in n_values:
        sub = df[df["n"] == n]
        rc = interpolate_threshold(sub, 0.5)
        rows.append({"n": n, "r_c": rc})
    return pd.DataFrame(rows)


# plotting

def make_plots(df, summary_df, prefix):
    n_values = sorted(df["n"].unique())
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(n_values)))

    # Plot 1: connectivity probability curves
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, n in enumerate(n_values):
        sub = df[df["n"] == n].sort_values("r")
        ax.plot(sub["r"], sub["P_conn"], "o-", color=colors[i],
                label=f"n={n}", markersize=4, linewidth=1.5)

    ax.axhline(0.5, ls="--", color="red", alpha=0.5, label="P = 0.5")

    # vertical lines at each r_c estimate
    valid = summary_df.dropna(subset=["r_c"])
    for _, row in valid.iterrows():
        idx = n_values.index(int(row["n"]))
        ax.axvline(x=row["r_c"], color=colors[idx], linestyle=":", alpha=0.5)

    ax.set_xlabel(r"$r$ in $R(n) = r/\sqrt{n}$", fontsize=12)
    ax.set_ylabel("P(connected)", fontsize=12)
    ax.set_title("Connectivity phase transition in Random Geometric Graphs", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{prefix}_connectivity.png", dpi=180, bbox_inches="tight")
    plt.close()

    # Plot 2: how r_c changes with n (convergence check)
    if len(valid) > 1:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(valid["n"], valid["r_c"], "s-", color="#378ADD", markersize=8, linewidth=2)
        ax.set_xlabel("n (number of vertices)", fontsize=12)
        ax.set_ylabel(r"$r_c(n)$ estimate", fontsize=12)
        ax.set_title(r"Convergence of $r_c$ estimate across graph sizes", fontsize=13)
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{prefix}_rc_convergence.png", dpi=180, bbox_inches="tight")
        plt.close()

    # Plot 3: heatmap of the full (r, n) space
    fig, ax = plt.subplots(figsize=(12, 5))
    r_vals = sorted(df["r"].unique())

    # build the matrix
    matrix = np.zeros((len(n_values), len(r_vals)))
    for i, n in enumerate(n_values):
        for j, r in enumerate(r_vals):
            row = df[(df["n"] == n) & (np.isclose(df["r"], r))]
            if len(row) > 0:
                matrix[i, j] = row["P_conn"].values[0]

    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1, origin="lower")

    # x-axis: don't label every single r value, just every few
    step = max(1, len(r_vals) // 10)
    ax.set_xticks(range(0, len(r_vals), step))
    ax.set_xticklabels([f"{r_vals[i]:.1f}" for i in range(0, len(r_vals), step)])
    ax.set_yticks(range(len(n_values)))
    ax.set_yticklabels([str(n) for n in n_values])

    ax.set_xlabel(r"$r$ in $R(n) = r/\sqrt{n}$")
    ax.set_ylabel("n")
    ax.set_title("Connectivity probability heatmap")
    plt.colorbar(im, ax=ax, label="P(connected)")

    # draw the P=0.5 contour
    ax.contour(matrix, levels=[0.5], colors="black", linewidths=2)

    plt.tight_layout()
    plt.savefig(f"{prefix}_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate r_c for RGG connectivity")
    parser.add_argument("--n-values", type=int, nargs="+", default=[100, 250, 500, 1000])
    parser.add_argument("--r-min", type=float, default=0.5)
    parser.add_argument("--r-max", type=float, default=2.8)
    parser.add_argument("--r-step", type=float, default=0.1)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-prefix", type=str, default="q4")
    args = parser.parse_args()

    r_values = np.round(np.arange(args.r_min, args.r_max + 1e-9, args.r_step), 2)

    print("=" * 60)
    print("ESTIMATING CRITICAL RADIUS r_c")
    print("=" * 60)
    print(f"n values:  {args.n_values}")
    print(f"r range:   {r_values[0]:.2f} to {r_values[-1]:.2f} (step {args.r_step}, {len(r_values)} values)")
    print(f"Trials:    {args.trials} per (r, n) pair")
    print(f"Seed:      {args.seed}")
    print("=" * 60)

    df = run_experiment(args.n_values, r_values, args.trials, args.seed)
    summary_df = compute_summary(df, args.n_values)

    # save raw data
    df.to_csv(f"{args.output_prefix}_results.csv", index=False)
    summary_df.to_csv(f"{args.output_prefix}_summary.csv", index=False)

    # print results
    print("\n" + "=" * 60)
    print("CRITICAL RADIUS ESTIMATES")
    print("=" * 60)
    print(f"{'n':>8}  {'r_c(n)':>10}")
    print("-" * 22)
    for _, row in summary_df.iterrows():
        n = int(row["n"])
        rc = row["r_c"]
        rc_str = f"{rc:.3f}" if rc is not None else "N/A"
        print(f"{n:>8}  {rc_str:>10}")

    valid = summary_df.dropna(subset=["r_c"])
    if len(valid) > 0:
        rc_vals = valid["r_c"].values
        print(f"\nMean r_c across all n: {np.mean(rc_vals):.3f}")
        print(f"r_c range: {np.min(rc_vals):.3f} to {np.max(rc_vals):.3f}")

    make_plots(df, summary_df, args.output_prefix)
    print(f"\nPlots saved: {args.output_prefix}_connectivity.png, "
          f"{args.output_prefix}_rc_convergence.png, "
          f"{args.output_prefix}_heatmap.png")
    print(f"Data saved:  {args.output_prefix}_results.csv, {args.output_prefix}_summary.csv")