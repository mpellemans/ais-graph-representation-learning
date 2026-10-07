"""Evaluate the dot-product framing over saved embeddings (fair retrieval setting).

The dot-product framing is sensitive to two retrieval confounds: (a) embedding-norm
domination when raw inner products are used, and (b) self-inclusion, where the
origin port itself sits in the candidate set and can rank first even though
self-transitions do not occur in the data. The fair setting removes both:
cosine similarity (L2-normalized embeddings) with the origin masked from its own
candidate ranking. The paper reports the fair setting; ``--settings both`` also
scores the raw self-included variant for comparison.

This script re-scores existing frozen embeddings
(``<results-dir>/{method}_seed{N}/embeddings.npy``) on the full test set without
retraining, aggregates over seeds, and prints the consolidated dot-product column
(plain table plus LaTeX cells) for the main results table. Deterministic
embeddings (HOPE, raw features) are seed-independent for the parameter-free
dot product and are evaluated once.

Usage:
    python experiments/maritime/evaluate_dot_product.py \
        --results-dir results/maritime/embeddings \
        --cache data/maritime/processed \
        --save results/maritime/dot_product_evaluation.txt

    # Section 4.5 ablation tables (single-seed, seed 42):
    python experiments/maritime/evaluate_dot_product.py --ablations
"""
import argparse
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.evaluation.link_prediction import (  # noqa: E402
    DotProductLinkPredictor, _build_warm_port_mapping, load_transitions,
)

# (key, display name, deterministic?) in main-table order.
METHODS = [
    ("dgcn", "DGI (DirectedGCN)", False),
    ("raw_features", "Raw features", True),
    ("node2vec", "Node2Vec", False),
    ("graphsage", "GraphSAGE", False),
    ("gae", "GAE", False),
    ("hope", "HOPE", True),
    ("portcity2vec", "PortCity2Vec", False),
]

# Diagnostic-ablation embeddings (Section 4.5). These are seed-42 single runs, so
# the ablation re-score is restricted to seed 42 for self-consistency within
# those tables.
ABLATION_METHODS = [
    ("dgcn", "DGI  K=inf (DirectedGCN)", False),    # sparsification K=inf baseline AND DirectedGCN row
    ("dgcn_k20", "DGI  K=20 (sparsified)", False),  # sparsification table
    ("dgcn_k10", "DGI  K=10 (sparsified)", False),  # sparsification table
    ("ecdgcn", "ECDGCN", False),                    # encoder-comparison table
    ("hope", "HOPE (reference)", True),             # encoder-comparison reference row
]
SEEDS = [0, 1, 2, 3, 4, 42]
SETTINGS = {
    "old": dict(normalize=False, exclude_self=False),   # raw inner product, self included
    "fair": dict(normalize=True, exclude_self=True),    # cosine + self-excluded
}


def eval_one(emb, test, warm_idx, g2w, setting, device):
    dp = DotProductLinkPredictor()
    r = dp.evaluate(emb, test, warm_idx, g2w, device=device, **SETTINGS[setting])
    return {"hits_at_1": r.hits_at_1, "hits_at_5": r.hits_at_5,
            "hits_at_10": r.hits_at_10, "mrr": r.mrr}


def aggregate(per_seed):
    """per_seed: list of metric dicts -> mean/std dict."""
    out = {}
    for k in ["hits_at_1", "hits_at_5", "hits_at_10", "mrr"]:
        vals = np.array([d[k] for d in per_seed])
        out[k] = (float(vals.mean()), float(vals.std()))
    out["n"] = len(per_seed)
    return out


def run(results_dir, cache, settings, sample, seed, save=None, methods=None,
        ablations=False):
    rng = np.random.default_rng(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    method_filter = set(methods) if methods else None
    method_list = ABLATION_METHODS if ablations else METHODS
    # Ablation tables are single-seed; pin every row to seed 42.
    seed_override = [42] if ablations else None

    train = load_transitions(cache, "train")
    val = load_transitions(cache, "val")
    test = load_transitions(cache, "test")
    warm_idx, g2w = _build_warm_port_mapping(train, val, test)

    if sample and sample < len(test):
        test = test[rng.choice(len(test), size=sample, replace=False)]

    setting_keys = ["old", "fair"] if settings == "both" else [settings]

    buffer = io.StringIO() if save else None

    class _Tee:
        def write(self, s):
            sys.__stdout__.write(s); buffer.write(s)

        def flush(self):
            sys.__stdout__.flush()

    if save:
        sys.stdout = _Tee()

    results = {}  # method -> setting -> aggregate
    try:
        print("#" * 96)
        print("# DOT-PRODUCT EVALUATION (fair setting = cosine + self-excluded)")
        print("#" * 96)
        print(f"\nCandidates W = {len(warm_idx)} | test transitions = {len(test):,}"
              f"{' (sampled)' if sample else ' (full)'} | device = {device}")

        for key, disp, deterministic in method_list:
            if method_filter and key not in method_filter:
                continue
            seeds = seed_override or ([42] if deterministic else SEEDS)
            per_setting = {s: [] for s in setting_keys}
            present = []
            for sd in seeds:
                emb_path = Path(results_dir) / f"{key}_seed{sd}" / "embeddings.npy"
                if not emb_path.exists():
                    continue
                present.append(sd)
                emb = np.load(emb_path)
                for s in setting_keys:
                    per_setting[s].append(eval_one(emb, test, warm_idx, g2w, s, device))
            if not present:
                print(f"\n[skip] {disp}: no embeddings found")
                continue
            results[key] = {s: aggregate(per_setting[s]) for s in setting_keys}
            note = " (deterministic, n=1)" if deterministic else f" (n={len(present)}, seeds={present})"
            print(f"\n{disp}{note}")
            for s in setting_keys:
                a = results[key][s]
                print(f"   {s:>5}: H@1={a['hits_at_1'][0]*100:6.3f}±{a['hits_at_1'][1]*100:.3f}%  "
                      f"H@5={a['hits_at_5'][0]*100:6.3f}%  H@10={a['hits_at_10'][0]*100:6.3f}%  "
                      f"MRR={a['mrr'][0]:.4f}±{a['mrr'][1]:.4f}")

        # Consolidated fair dot-product column: plain table + LaTeX cells.
        if "fair" in setting_keys:
            tbl = "ablation tables (Section 4.5)" if ablations else "main results table"
            print("\n" + "=" * 96)
            print(f"Consolidated fair dot-product column for the {tbl}")
            print("=" * 96)
            print(f"  {'Method':<26} {'H@1 %':>14} {'H@5 %':>10} {'H@10 %':>10} {'MRR':>16}  n")
            print("  " + "-" * 82)
            for key, disp, det in method_list:
                if key not in results:
                    continue
                a = results[key]["fair"]
                h1 = f"{a['hits_at_1'][0]*100:.2f}±{a['hits_at_1'][1]*100:.2f}"
                h5 = f"{a['hits_at_5'][0]*100:.2f}"
                h10 = f"{a['hits_at_10'][0]*100:.2f}"
                mrr = f"{a['mrr'][0]:.3f}±{a['mrr'][1]:.3f}"
                print(f"  {disp:<26} {h1:>14} {h5:>10} {h10:>10} {mrr:>16}  {a['n']}")

            print("\n" + "=" * 96)
            print(f"LaTeX — fair dot-product cells (H@1 %% and MRR) for the {tbl}")
            print("=" * 96)
            for key, disp, det in method_list:
                if key not in results:
                    continue
                a = results[key]["fair"]
                h1m, h1s = a["hits_at_1"][0] * 100, a["hits_at_1"][1] * 100
                mm, ms = a["mrr"]
                if det:
                    print(f"  {disp:<26} & ${h1m:.2f}$ & ${mm:.3f}$  % deterministic")
                else:
                    print(f"  {disp:<26} & ${h1m:.2f} \\pm {h1s:.2f}$ & "
                          f"${mm:.3f} \\pm {ms:.3f}$")
    finally:
        if save:
            sys.stdout = sys.__stdout__
            p = Path(save)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(buffer.getvalue(), encoding="utf-8")
            # also dump machine-readable JSON next to the report
            json_path = p.with_suffix(".json")
            json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(f"\nSaved report to {p}\nSaved JSON to {json_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default="results/maritime/embeddings")
    ap.add_argument("--cache", default="data/maritime/processed")
    ap.add_argument("--settings", choices=["old", "fair", "both"], default="fair")
    ap.add_argument("--sample", type=int, default=0, help="0 = full test set")
    ap.add_argument("--seed", type=int, default=0, help="sampling seed (if --sample)")
    ap.add_argument("--save", default=None)
    ap.add_argument("--methods", default=None,
                    help="comma-separated method keys to restrict to (e.g. node2vec)")
    ap.add_argument("--ablations", action="store_true",
                    help="re-score the Section 4.5 ablation embeddings "
                         "(K=inf/K=20/K=10, ECDGCN, HOPE ref), seed 42")
    args = ap.parse_args()
    methods = args.methods.split(",") if args.methods else None
    run(args.results_dir, args.cache, args.settings, args.sample, args.seed,
        save=args.save, methods=methods, ablations=args.ablations)


if __name__ == "__main__":
    main()
