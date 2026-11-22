"""
Experiments: evaluation, cross-validation and visualization
============================================================
Runs the full experiment grid (n-gram size x smoothing technique), computes
held-out perplexity, runs k-fold cross-validation, saves the result tables
as CSV and produces the comparison plots.

Usage
-----
    python -m ngram_smoothing.experiments                 # full run (all 2,000 reviews)
    python -m ngram_smoothing.experiments --reviews 200   # quick run on a sample
    python -m ngram_smoothing.experiments --help
"""

import argparse
import statistics as stats
import time
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe: figures are written to disk, never shown
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import k_fold_split, load_and_preprocess_sentences, train_test_split
from .model import NGramLanguageModel

N_VALUES = [1, 2, 3, 4]
METHODS = ["laplace", "good_turing", "kneser_ney"]
METHOD_LABELS = {
    "laplace": "Laplace (Add-One)",
    "good_turing": "Good-Turing",
    "kneser_ney": "Kneser-Ney (interpolated)",
}
COLORS = {"laplace": "#4C72B0", "good_turing": "#DD8452", "kneser_ney": "#55A868"}
MARKERS = {"laplace": "o", "good_turing": "s", "kneser_ney": "^"}
K_FOLDS = 5


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------
def run_holdout_evaluation(train, test, n_values=N_VALUES):
    """Single 80/20 train/test perplexity for every (n, method)."""
    rows = []
    for n in n_values:
        t0 = time.time()
        lm = NGramLanguageModel(n).fit(train)
        fit_time = time.time() - t0
        for method in METHODS:
            pp = lm.perplexity(test, method)
            rows.append({"n": n, "method": method, "perplexity": pp, "fit_time_s": fit_time})
    return pd.DataFrame(rows)


def run_cross_validation(sentences, k=K_FOLDS, n_values=N_VALUES):
    """k-fold cross-validation, averaging perplexity per (n, method)."""
    fold_results = {(n, m): [] for n in n_values for m in METHODS}

    for fold_idx, (train, test) in enumerate(k_fold_split(sentences, k=k)):
        for n in n_values:
            lm = NGramLanguageModel(n).fit(train)
            for method in METHODS:
                fold_results[(n, method)].append(lm.perplexity(test, method))
        print(f"  fold {fold_idx + 1}/{k} done")

    rows = []
    for (n, method), pps in fold_results.items():
        rows.append({
            "n": n,
            "method": method,
            "mean_perplexity": stats.mean(pps),
            "std_perplexity": stats.pstdev(pps),
            "fold_perplexities": pps,
        })
    return pd.DataFrame(rows)


def ngram_statistics(train, max_order=max(N_VALUES)):
    """
    Sparsity statistics per order: distinct n-gram types, share of types seen
    exactly once (hapax legomena), and the size of the full event space (V+1)^n,
    where the +1 is the <s> marker.
    Counted from the order-``max_order`` model's padded training data.
    """
    lm = NGramLanguageModel(max_order).fit(train)
    vocab_size = len(lm.vocab) + 1  # +1 for the <s> marker, which appears in the counts
    rows = []
    for k in range(1, max_order + 1):
        counts = lm.counts[k]
        types = len(counts)
        singletons = sum(1 for c in counts.values() if c == 1)
        rows.append({
            "n": k,
            "distinct_types": types,
            "singleton_share": singletons / types,
            "possible_types": vocab_size ** k,
            "observed_share_of_possible": types / (vocab_size ** k),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------
def plot_holdout_results(df_holdout, out_path):
    """Line plot: perplexity vs n, one line per smoothing method (log scale)."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for method in METHODS:
        sub = df_holdout[df_holdout["method"] == method].sort_values("n")
        ax.plot(sub["n"], sub["perplexity"], marker=MARKERS[method],
                label=METHOD_LABELS[method], color=COLORS[method], linewidth=2, markersize=8)

    ax.set_yscale("log")
    ax.set_xlabel("N-gram size (n)")
    ax.set_ylabel("Held-out Perplexity (log scale)")
    ax.set_title("Perplexity vs. N-Gram Size by Smoothing Technique\n"
                 "(80/20 train/test split, Movie Review Dataset)")
    ax.set_xticks(sorted(df_holdout["n"].unique()))
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_cv_results(df_cv, out_path, k=K_FOLDS):
    """Grouped bar chart: mean CV perplexity with std-dev error bars."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    width = 0.25
    for i, method in enumerate(METHODS):
        sub = df_cv[df_cv["method"] == method].sort_values("n")
        x = [n + (i - 1) * width for n in sub["n"]]
        ax.bar(x, sub["mean_perplexity"], width=width, label=METHOD_LABELS[method],
               yerr=sub["std_perplexity"], capsize=4, color=COLORS[method])

    ax.set_yscale("log")
    ax.set_xlabel("N-gram size (n)")
    ax.set_ylabel("Mean Perplexity across folds (log scale)")
    ax.set_title(f"{k}-Fold Cross-Validation: Mean Perplexity ± Std Dev")
    ax.set_xticks(sorted(df_cv["n"].unique()))
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_fit_time(df_holdout, out_path):
    """Bar chart: training time vs n (model complexity growth)."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sub = df_holdout.drop_duplicates("n").sort_values("n")
    ax.bar([str(n) for n in sub["n"]], sub["fit_time_s"], color="#8172B2")
    ax.set_xlabel("N-gram size (n)")
    ax.set_ylabel("Training time (seconds)")
    ax.set_title("Model Fitting Time vs. N-Gram Size")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_heatmap(df_cv, out_path, k=K_FOLDS):
    """Heatmap of mean CV perplexity (rows = n, columns = method), log-coloured."""
    pivot = df_cv.pivot(index="n", columns="method", values="mean_perplexity")[METHODS]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    im = ax.imshow(np.log10(pivot.values), cmap="RdYlGn_r", aspect="auto")

    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"n = {n}" for n in pivot.index])
    best = pivot.min(axis=1)
    for i, n in enumerate(pivot.index):
        for j, m in enumerate(METHODS):
            val = pivot.loc[n, m]
            weight = "bold" if val == best[n] else "normal"
            r, g, b, _ = im.cmap(im.norm(np.log10(val)))
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            ax.text(j, i, f"{val:,.0f}", ha="center", va="center", fontsize=12,
                    fontweight=weight, color="white" if luminance < 0.5 else "black")

    ax.set_title(f"Mean {k}-fold perplexity (lower is better; bold = best per row)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("log10(perplexity)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_sparsity(df_stats, out_path):
    """Why smoothing matters: distinct n-gram types and share of singletons per order."""
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.bar(df_stats["n"], df_stats["distinct_types"], color="#8172B2", width=0.55,
            label="Distinct n-gram types seen in training")
    ax1.set_yscale("log")
    ax1.set_ylim(1e4, 3e6)
    ax1.set_xlabel("N-gram size (n)")
    ax1.set_ylabel("Distinct n-gram types (log scale)")
    ax1.set_xticks(df_stats["n"])
    ax1.grid(True, axis="y", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(df_stats["n"], df_stats["singleton_share"] * 100, color="#C44E52",
             marker="o", linewidth=2.2, markersize=8, label="% of types seen exactly once")
    ax2.set_ylabel("% of n-gram types that occur exactly once")
    ax2.set_ylim(0, 110)
    for n, share in zip(df_stats["n"], df_stats["singleton_share"]):
        ax2.annotate(f"{share * 100:.0f}%", (n, share * 100), textcoords="offset points",
                     xytext=(0, 9), ha="center", color="#C44E52", fontweight="bold")

    handles = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax1.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax1.legend(handles, labels, loc="upper left")
    ax1.set_title("Data sparsity grows with n-gram order")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------
# Command line entry point
# ----------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(
        prog="python -m ngram_smoothing.experiments",
        description="Compare Laplace, Good-Turing and Kneser-Ney smoothing for n = 1..4.",
    )
    p.add_argument("--reviews", type=int, default=None,
                   help="use a random sample of this many reviews (default: all 2,000)")
    p.add_argument("--folds", type=int, default=K_FOLDS, help="number of CV folds (default: 5)")
    p.add_argument("--results-dir", default="results", help="where to write CSV tables")
    p.add_argument("--figures-dir", default="assets", help="where to write PNG figures")
    p.add_argument("--skip-cv", action="store_true", help="skip cross-validation (much faster)")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    results_dir, figures_dir = Path(args.results_dir), Path(args.figures_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("Loading and preprocessing the Movie Review Dataset...")
    sentences = load_and_preprocess_sentences(n_reviews=args.reviews)
    print(f"  {len(sentences)} sentences total")

    train, test = train_test_split(sentences, test_ratio=0.2)
    print(f"  train={len(train)} sentences, test={len(test)} sentences")

    print("\nRunning held-out evaluation...")
    df_holdout = run_holdout_evaluation(train, test)
    df_holdout.to_csv(results_dir / "results_holdout.csv", index=False)
    print(df_holdout[["n", "method", "perplexity"]].to_string(index=False))

    df_cv = None
    if not args.skip_cv:
        print(f"\nRunning {args.folds}-fold cross-validation...")
        df_cv = run_cross_validation(sentences, k=args.folds)
        df_cv.drop(columns=["fold_perplexities"]).to_csv(results_dir / "results_cv.csv", index=False)
        print(df_cv[["n", "method", "mean_perplexity", "std_perplexity"]].to_string(index=False))

    print("\nComputing sparsity statistics...")
    df_stats = ngram_statistics(train)
    df_stats.to_csv(results_dir / "ngram_statistics.csv", index=False)
    print(df_stats[["n", "distinct_types", "singleton_share"]].to_string(index=False))

    print("\nGenerating plots...")
    plot_holdout_results(df_holdout, figures_dir / "perplexity_vs_n.png")
    plot_fit_time(df_holdout, figures_dir / "fit_time.png")
    plot_sparsity(df_stats, figures_dir / "sparsity.png")
    if df_cv is not None:
        plot_cv_results(df_cv, figures_dir / "cv_perplexity.png", k=args.folds)
        plot_heatmap(df_cv, figures_dir / "cv_heatmap.png", k=args.folds)
    print(f"Done. Tables -> {results_dir}/   Figures -> {figures_dir}/")


if __name__ == "__main__":
    main()
