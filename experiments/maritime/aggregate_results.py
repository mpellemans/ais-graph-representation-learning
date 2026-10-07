"""Aggregate maritime experiment results across seeds.

Scans a results directory for {method}_seed{N}/ subdirectories, loads all
*_results.json files, and prints mean +/- std tables grouped by method.
Also generates a LaTeX comparison table for all methods, and (with --hits-k)
the full Hits@1/5/10 breakdown table for the appendix.

Usage:
    python experiments/maritime/aggregate_results.py results/maritime/embeddings
    python experiments/maritime/aggregate_results.py results/maritime/embeddings --latex
    python experiments/maritime/aggregate_results.py results/maritime/embeddings --hits-k
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Canonical display order for the comparison table
METHOD_ORDER = [
    "dgcn", "raw_features", "node2vec", "graphsage", "gae", "hope", "portcity2vec",
    "markov_n1", "markov_n2", "markov_n3", "markov_n5", "markov_n7", "markov_n10",
]
METHOD_DISPLAY = {
    "dgcn": "DGI (DirectedGCN)",
    "raw_features": "Raw Features",
    "node2vec": "Node2Vec",
    "graphsage": "GraphSAGE",
    "gae": "GAE",
    "hope": "HOPE",
    "portcity2vec": "PortCity2Vec",
    "markov_n1": "Markov n=1",
    "markov_n2": "Markov n=2",
    "markov_n3": "Markov n=3",
    "markov_n5": "Markov n=5",
    "markov_n7": "Markov n=7",
    "markov_n10": "Markov n=10",
}


def _fmt(vals, n_seeds):
    """Format mean +/- std (or single value if n_seeds == 1)."""
    if n_seeds > 1:
        return f"{np.mean(vals):.4f} \\pm {np.std(vals):.4f}"
    return f"{vals[0]:.4f}"


def _fmt_plain(vals, n_seeds):
    """Plain (non-LaTeX) mean +/- std."""
    if n_seeds > 1:
        return f"{np.mean(vals):.4f}+/-{np.std(vals):.4f}"
    return f"{vals[0]:.4f}"


def load_seed_dirs(results_dir):
    """Return dict: method -> list of seed result dicts."""
    results_dir = Path(results_dir)
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return {}

    seed_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir()])
    if not seed_dirs:
        print(f"No subdirectories found in {results_dir}")
        return {}

    by_method = {}
    for seed_dir in seed_dirs:
        dir_name = seed_dir.name
        parts = dir_name.rsplit("_seed", 1)
        if len(parts) != 2:
            continue
        method = parts[0]

        seed_results = {"dir": str(seed_dir)}
        for result_file in seed_dir.glob("*_results.json"):
            with open(result_file) as f:
                seed_results[result_file.stem] = json.load(f)

        # Backward compatibility: old-style results.json
        old_results = seed_dir / "results.json"
        if old_results.exists():
            with open(old_results) as f:
                seed_results["results"] = json.load(f)

        if len(seed_results) > 1:
            by_method.setdefault(method, []).append(seed_results)

    return by_method


def print_per_vessel_details(lp_data):
    """Print a per-vessel-type breakdown for each framing that has the data."""
    for framing in ["dot_product", "mlp", "lstm"]:
        per_vt_by_seed = [
            lp[framing]["per_vessel_type"]
            for lp in lp_data
            if framing in lp and "per_vessel_type" in lp[framing] and lp[framing]["per_vessel_type"]
        ]
        if not per_vt_by_seed:
            continue

        # Collect all vessel types present across seeds
        all_vt = sorted({vt for pvt in per_vt_by_seed for vt in pvt})
        n_seeds = len(per_vt_by_seed)

        print(f"\n    Per-vessel-type [{framing}]:")
        print(f"      {'Vessel Type':<24} {'Hits@1':>10} {'MRR':>10} {'Count':>8}")
        print(f"      {'-'*56}")
        for vt in all_vt:
            h1_vals = [pvt[vt]["hits_at_1"] for pvt in per_vt_by_seed if vt in pvt]
            mrr_vals = [pvt[vt]["mrr"] for pvt in per_vt_by_seed if vt in pvt]
            count_vals = [pvt[vt]["count"] for pvt in per_vt_by_seed if vt in pvt]
            if not h1_vals:
                continue
            h1_str = _fmt_plain(h1_vals, n_seeds) if n_seeds > 1 else f"{h1_vals[0]:.4f}"
            mrr_str = _fmt_plain(mrr_vals, n_seeds) if n_seeds > 1 else f"{mrr_vals[0]:.4f}"
            count_str = f"{int(np.mean(count_vals)):,}"
            print(f"      {vt:<24} {h1_str:>10} {mrr_str:>10} {count_str:>8}")


def print_method_details(method, seeds_data, per_vessel=False):
    """Print a detailed per-method summary (training, link prediction, clustering)."""
    n_seeds = len(seeds_data)
    print(f"\n  Method: {METHOD_DISPLAY.get(method, method)} ({n_seeds} seed{'s' if n_seeds > 1 else ''})")

    # --- Training (DGI only) ---
    training_data = [s.get("training_results", {}) for s in seeds_data]
    training_data = [t for t in training_data if t]
    if training_data:
        epochs = [t["timing"]["total_epochs_run"] for t in training_data]
        times = [t["timing"]["training_seconds"] for t in training_data]
        print(f"\n  Training:")
        if n_seeds > 1:
            print(f"    Epochs: {np.mean(epochs):.0f} +/- {np.std(epochs):.0f}")
            print(f"    Time:   {np.mean(times):.1f} +/- {np.std(times):.1f}s")
        else:
            print(f"    Epochs: {epochs[0]}")
            print(f"    Time:   {times[0]:.1f}s")

    # --- Link Prediction ---
    lp_data = [s.get("link_prediction_results", {}) for s in seeds_data]
    lp_data = [lp for lp in lp_data if lp]
    if lp_data:
        print(f"\n  Link Prediction:")
        print(f"    {'Framing':<14} {'Hits@1':>18} {'Hits@5':>18} {'Hits@10':>18} {'MRR':>18}")
        print(f"    {'-'*78}")
        for framing in ["dot_product", "mlp", "lstm"]:
            framing_results = [lp[framing] for lp in lp_data if framing in lp]
            if not framing_results:
                continue
            row = {}
            for metric in ["hits_at_1", "hits_at_5", "hits_at_10", "mrr"]:
                vals = [r[metric] for r in framing_results]
                row[metric] = _fmt_plain(vals, n_seeds)
            print(f"    {framing:<14} {row['hits_at_1']:>18} {row['hits_at_5']:>18} "
                  f"{row['hits_at_10']:>18} {row['mrr']:>18}")
        if per_vessel:
            print_per_vessel_details(lp_data)

    # --- Classification (DGI only) ---
    cls_data = [s.get("classification_results", {}) for s in seeds_data]
    cls_data = [c for c in cls_data if c and "classification" in c]
    if cls_data:
        print(f"\n  Classification (diagnostic):")
        tasks = cls_data[0]["classification"]
        for task_idx, task in enumerate(tasks):
            task_name = task["task"]
            accs = [c["classification"][task_idx]["mean_accuracy"]
                    for c in cls_data if task_idx < len(c["classification"])]
            if n_seeds > 1:
                print(f"    {task_name}: {np.mean(accs):.2f} +/- {np.std(accs):.2f}%")
            else:
                std = cls_data[0]["classification"][task_idx]["std_accuracy"]
                print(f"    {task_name}: {accs[0]:.2f} +/- {std:.2f}% (50 runs)")

    # --- Clustering (DGI only) ---
    analysis_data = [s.get("analysis_results", {}) for s in seeds_data]
    analysis_data = [a for a in analysis_data if a and "clustering" in a]
    if analysis_data:
        print(f"\n  Clustering:")
        km_data = [a["clustering"]["kmeans"] for a in analysis_data]
        print(f"    K-means (k={km_data[0]['n_clusters']}):")
        for metric in ["silhouette", "nmi", "ari", "davies_bouldin"]:
            vals = [k[metric] for k in km_data]
            if n_seeds > 1:
                print(f"      {metric:<16} {np.mean(vals):.4f} +/- {np.std(vals):.4f}")
            else:
                print(f"      {metric:<16} {vals[0]:.4f}")

    # --- Backward compat: old results.json ---
    old_data = [s.get("results", {}) for s in seeds_data]
    old_data = [o for o in old_data if o and "classification" in o]
    if old_data and not cls_data:
        print(f"\n  Classification (from old results.json):")
        for task_idx in range(len(old_data[0]["classification"])):
            task_name = old_data[0]["classification"][task_idx]["task"]
            accs = [r["classification"][task_idx]["mean_accuracy"] for r in old_data]
            if n_seeds > 1:
                print(f"    {task_name}: {np.mean(accs):.2f} +/- {np.std(accs):.2f}%")
            else:
                acc = old_data[0]["classification"][task_idx]
                print(f"    {task_name}: {acc['mean_accuracy']:.2f} +/- {acc['std_accuracy']:.2f}%")


def print_comparison_table(by_method):
    """Print a side-by-side comparison table for all methods."""
    # Gather link prediction results per method
    lp_summary = {}
    for method, seeds_data in by_method.items():
        n_seeds = len(seeds_data)
        lp_data = [s.get("link_prediction_results", {}) for s in seeds_data]
        lp_data = [lp for lp in lp_data if lp]
        if not lp_data:
            continue
        method_lp = {}
        for framing in ["dot_product", "mlp", "lstm"]:
            framing_results = [lp[framing] for lp in lp_data if framing in lp]
            if not framing_results:
                continue
            fr = {}
            for metric in ["hits_at_1", "hits_at_5", "hits_at_10", "mrr"]:
                vals = [r[metric] for r in framing_results]
                fr[metric] = (np.mean(vals), np.std(vals) if n_seeds > 1 else None)
            method_lp[framing] = fr
        lp_summary[method] = (n_seeds, method_lp)

    if not lp_summary:
        return

    # Determine display order
    ordered_methods = [m for m in METHOD_ORDER if m in lp_summary]
    ordered_methods += [m for m in sorted(lp_summary) if m not in METHOD_ORDER]

    print("\n" + "=" * 80)
    print("COMPARISON TABLE — Next-Port Prediction (H@1 / MRR)")
    print("=" * 80)
    header = f"  {'Method':<22} {'Dot H@1':>10} {'Dot MRR':>10} {'MLP H@1':>10} {'MLP MRR':>10} {'LSTM H@1':>10} {'LSTM MRR':>10}"
    print(header)
    print("  " + "-" * 74)

    def _cell(lp, framing, metric):
        if framing not in lp:
            return "  —"
        mean, std = lp[framing][metric]
        if std is not None:
            return f"{mean:.4f}±{std:.4f}"
        return f"{mean:.4f}"

    for method in ordered_methods:
        n_seeds, lp = lp_summary[method]
        name = METHOD_DISPLAY.get(method, method)
        d_h1 = _cell(lp, "dot_product", "hits_at_1")
        d_mrr = _cell(lp, "dot_product", "mrr")
        m_h1 = _cell(lp, "mlp", "hits_at_1")
        m_mrr = _cell(lp, "mlp", "mrr")
        l_h1 = _cell(lp, "lstm", "hits_at_1")
        l_mrr = _cell(lp, "lstm", "mrr")
        n_str = f" (n={n_seeds})" if n_seeds > 1 else ""
        print(f"  {name+n_str:<22} {d_h1:>10} {d_mrr:>10} {m_h1:>10} {m_mrr:>10} {l_h1:>10} {l_mrr:>10}")

    print("=" * 80)


def print_latex_table(by_method):
    """Print a LaTeX-formatted comparison table (H@1 + MRR for all 3 framings)."""
    lp_summary = {}
    for method, seeds_data in by_method.items():
        n_seeds = len(seeds_data)
        lp_data = [s.get("link_prediction_results", {}) for s in seeds_data]
        lp_data = [lp for lp in lp_data if lp]
        if not lp_data:
            continue
        method_lp = {}
        for framing in ["dot_product", "mlp", "lstm"]:
            framing_results = [lp[framing] for lp in lp_data if framing in lp]
            if not framing_results:
                continue
            fr = {}
            for metric in ["hits_at_1", "mrr"]:
                vals = [r[metric] for r in framing_results]
                fr[metric] = (np.mean(vals), np.std(vals) if n_seeds > 1 else None)
            method_lp[framing] = fr
        lp_summary[method] = (n_seeds, method_lp)

    if not lp_summary:
        return

    ordered_methods = [m for m in METHOD_ORDER if m in lp_summary]
    ordered_methods += [m for m in sorted(lp_summary) if m not in METHOD_ORDER]

    print("\n% ---- LaTeX comparison table (copy into paper) ----")
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Next-port prediction results on the maritime test set.")
    print(r"  H@1 and MRR reported for three evaluation framings.")
    print(r"  Best result per column in \textbf{bold}.}")
    print(r"\label{tab:comparison}")
    print(r"\begin{tabular}{l cc cc cc}")
    print(r"\toprule")
    print(r"& \multicolumn{2}{c}{Dot-Product} & \multicolumn{2}{c}{MLP} & \multicolumn{2}{c}{LSTM} \\")
    print(r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}")
    print(r"Method & H@1 & MRR & H@1 & MRR & H@1 & MRR \\")
    print(r"\midrule")

    def _latex_cell(lp, framing, metric):
        if framing not in lp:
            return "—"
        mean, std = lp[framing][metric]
        pct = mean * 100 if metric == "hits_at_1" else mean
        fmt = f"{pct:.1f}" if metric == "hits_at_1" else f"{pct:.3f}"
        if std is not None:
            std_scaled = std * 100 if metric == "hits_at_1" else std
            fmt_std = f"{std_scaled:.1f}" if metric == "hits_at_1" else f"{std_scaled:.3f}"
            return f"${fmt}\\scriptstyle{{\\pm{fmt_std}}}$"
        return f"${fmt}$"

    # Find the best per column for bold
    best = {}
    for framing in ["dot_product", "mlp", "lstm"]:
        for metric in ["hits_at_1", "mrr"]:
            key = (framing, metric)
            best_val = -1
            for method in ordered_methods:
                _, lp = lp_summary[method]
                if framing in lp and metric in lp[framing]:
                    val = lp[framing][metric][0]
                    if val > best_val:
                        best_val = val
                        best[key] = method

    for method in ordered_methods:
        name = METHOD_DISPLAY.get(method, method)
        _, lp = lp_summary[method]
        cells = []
        for framing in ["dot_product", "mlp", "lstm"]:
            for metric in ["hits_at_1", "mrr"]:
                cell = _latex_cell(lp, framing, metric)
                if best.get((framing, metric)) == method:
                    # Wrap innermost value in bold — simple approach: wrap whole cell
                    cell = r"\textbf{" + cell + "}"
                cells.append(cell)
        row = f"{name} & " + " & ".join(cells) + r" \\"
        print(row)

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


def print_hits_k_table(by_method):
    """Print the full Hits@1/5/10 breakdown per method and framing (plain + LaTeX)."""
    framings = ["dot_product", "mlp", "lstm"]
    metrics = ["hits_at_1", "hits_at_5", "hits_at_10"]
    latex_display = dict(METHOD_DISPLAY)
    for n in [1, 2, 3, 5, 7, 10]:
        latex_display[f"markov_n{n}"] = f"Markov ($n{{=}}{n}$)"

    method_stats = {}
    n_by_method = {}
    for method in METHOD_ORDER:
        if method not in by_method:
            continue
        lp_data = [s.get("link_prediction_results", {}) for s in by_method[method]]
        lp_data = [lp for lp in lp_data if lp]
        if not lp_data:
            continue
        s = {}
        for framing in framings:
            framing_present = [lp[framing] for lp in lp_data if framing in lp]
            if not framing_present:
                continue
            for metric in metrics:
                vals = [r[metric] for r in framing_present if metric in r]
                if vals:
                    s[(framing, metric)] = (float(np.mean(vals)), float(np.std(vals)), len(vals))
        if s:
            method_stats[method] = s
            n_by_method[method] = len(lp_data)

    if not method_stats:
        return

    print("\n" + "=" * 80)
    print("FULL HITS@K BREAKDOWN — H@1 / H@5 / H@10 per framing (mean +/- std)")
    print("=" * 80)
    for method, s in method_stats.items():
        n = n_by_method[method]
        for framing in framings:
            h_vals = [s.get((framing, m)) for m in metrics]
            if h_vals[0] is None:
                continue
            line = f"  {METHOD_DISPLAY.get(method, method):<22}  {framing:<12}"
            for label, val in zip(["H@1", "H@5", "H@10"], h_vals):
                if val is None:
                    line += f"  {label}=—"
                else:
                    m, sd, _ = val
                    if n > 1:
                        line += f"  {label}={m*100:6.2f}±{sd*100:.2f}"
                    else:
                        line += f"  {label}={m*100:6.2f}"
            line += f"  (n={n})"
            print(line)

    # LaTeX table: one row per method, columns H@1 H@5 H@10 per framing
    def fmt_pct(mean, std, n_seeds):
        if n_seeds > 1:
            return f"${mean*100:.2f} \\pm {std*100:.2f}$"
        return f"${mean*100:.2f}$"

    print("\n% ---- LaTeX full Hits@K table (copy into paper) ----")
    print(r"\begin{table*}[htb]")
    print(r"\centering")
    print(r"\small")
    print(r"\caption{Full Hits@$K$ results ($K\in\{1,5,10\}$) on the maritime test set"
          r" under the three next-port predictors. Best per column among the learned"
          r" embeddings in \textbf{bold}.}")
    print(r"\label{tab:full_hits_k}")
    print(r"\begin{tabular}{l ccc ccc ccc}")
    print(r"\toprule")
    print(r" & \multicolumn{3}{c}{\textbf{Dot-Product}} & \multicolumn{3}{c}{\textbf{MLP}} & \multicolumn{3}{c}{\textbf{LSTM}} \\")
    print(r"\cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10}")
    print(r"\textbf{Method} & H@$1$ & H@$5$ & H@$10$ & H@$1$ & H@$5$ & H@$10$ & H@$1$ & H@$5$ & H@$10$ \\")
    print(r"\midrule")

    # Best per column among the embedding methods only (exclude markov for bold)
    embedding_methods = [m for m in method_stats if not m.startswith("markov_")]
    best = {}
    for framing in framings:
        for metric in metrics:
            key = (framing, metric)
            best_val = -1
            for m in embedding_methods:
                v = method_stats[m].get(key)
                if v is not None and v[0] > best_val:
                    best_val = v[0]
                    best[key] = m

    last_embedding = embedding_methods[-1] if embedding_methods else None
    for method in METHOD_ORDER:
        if method not in method_stats:
            continue
        cells = []
        for framing in framings:
            for metric in metrics:
                v = method_stats[method].get((framing, metric))
                if v is None:
                    cells.append("--")
                    continue
                m, sd, ns = v
                cell = fmt_pct(m, sd, ns)
                if best.get((framing, metric)) == method:
                    cell = r"\textbf{" + cell + "}"
                cells.append(cell)
        row = latex_display.get(method, method) + " & " + " & ".join(cells) + r" \\"
        print(row)
        if method == last_embedding:
            print(r"\midrule")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table*}")


def aggregate_results(results_dir, latex=False, per_vessel=False, save=False,
                      hits_k=False):
    """Read all result JSON files and print mean+/-std tables grouped by method."""
    import io, sys

    by_method = load_seed_dirs(results_dir)
    if not by_method:
        return

    # Capture all output so we can optionally tee to a file
    buffer = io.StringIO() if save else None

    class _Tee:
        def write(self, s):
            sys.__stdout__.write(s)
            buffer.write(s)
        def flush(self):
            sys.__stdout__.flush()

    if save:
        sys.stdout = _Tee()

    try:
        print("\n" + "=" * 80)
        print("AGGREGATED RESULTS (mean +/- std)")
        print("=" * 80)

        for method in METHOD_ORDER:
            if method in by_method:
                print_method_details(method, by_method[method], per_vessel=per_vessel)

        # Any methods not in canonical order
        for method, seeds_data in sorted(by_method.items()):
            if method not in METHOD_ORDER:
                print_method_details(method, seeds_data, per_vessel=per_vessel)

        print("\n" + "=" * 80)

        # Comparison table (always shown)
        print_comparison_table(by_method)

        # LaTeX table (optional)
        if latex:
            print_latex_table(by_method)

        # Full Hits@K breakdown (optional)
        if hits_k:
            print_hits_k_table(by_method)
    finally:
        if save:
            sys.stdout = sys.__stdout__
            save_path = Path(results_dir) / "aggregate_results.txt"
            save_path.write_text(buffer.getvalue(), encoding="utf-8")
            print(f"\nSaved to {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate maritime experiment results across seeds"
    )
    parser.add_argument(
        "results_dir", type=str,
        help="Path to results directory (e.g. results/maritime/embeddings)"
    )
    parser.add_argument(
        "--latex", action="store_true",
        help="Also print LaTeX-formatted comparison table"
    )
    parser.add_argument(
        "--per-vessel", action="store_true",
        help="Print per-vessel-type Hits@1 and MRR breakdown for each method"
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save output to results_dir/aggregate_results.txt"
    )
    parser.add_argument(
        "--hits-k", action="store_true",
        help="Also print the full Hits@1/5/10 breakdown table (plain + LaTeX)"
    )
    args = parser.parse_args()

    aggregate_results(args.results_dir, latex=args.latex, per_vessel=args.per_vessel,
                      save=args.save, hits_k=args.hits_k)


if __name__ == "__main__":
    main()
