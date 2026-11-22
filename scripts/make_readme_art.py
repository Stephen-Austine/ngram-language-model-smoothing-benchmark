"""
Generate the README artwork from the project's real results.

    python scripts/make_readme_art.py

Run ``python -m ngram_smoothing.experiments`` first: this script reads
``results/results_cv.csv`` and re-fits models to produce its figures:

  assets/banner.png                 README header (1600x420)
  assets/social-preview.png         GitHub social preview card (1280x640)
  assets/pipeline.png               architecture / data-flow diagram
  assets/terminal_quickstart.png    terminal card with the quickstart output
  assets/terminal_generation.png    terminal card with sampled sentences per order
  assets/frequency_vs_versatility.png  why Kneser-Ney's continuation count matters
"""

import argparse
import subprocess
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from ngram_smoothing import NGramLanguageModel, load_and_preprocess_sentences, train_test_split
from ngram_smoothing.experiments import COLORS, METHOD_LABELS, METHODS

BG_TOP, BG_BOTTOM = "#0b1220", "#1b2640"
INK, MUTED, ACCENT = "#f1f5f9", "#94a3b8", "#55A868"


# ----------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------
def gradient_background(fig):
    ax = fig.add_axes([0, 0, 1, 1])
    grad = np.linspace(0, 1, 256).reshape(-1, 1)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("bg", [BG_TOP, BG_BOTTOM])
    ax.imshow(grad, aspect="auto", cmap=cmap, extent=[0, 1, 0, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return ax


def rounded_box(ax, x, y, w, h, fc, ec="none", radius=0.02, lw=1.5, alpha=1.0, z=2):
    box = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                         fc=fc, ec=ec, lw=lw, alpha=alpha, zorder=z, transform=ax.transData)
    ax.add_patch(box)
    return box


def mini_chart(fig, rect, df_cv, font=11):
    """Stylised version of the main result: mean CV perplexity vs n, log scale."""
    ax = fig.add_axes(rect)
    ax.set_facecolor("none")
    markers = {"laplace": "o", "good_turing": "s", "kneser_ney": "^"}
    for method in METHODS:
        sub = df_cv[df_cv["method"] == method].sort_values("n")
        ax.plot(sub["n"], sub["mean_perplexity"], color=COLORS[method], marker=markers[method],
                lw=3, ms=9, label=METHOD_LABELS[method])
    ax.set_yscale("log")
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlabel("n-gram size", color=MUTED, fontsize=font)
    ax.set_ylabel("perplexity (log)", color=MUTED, fontsize=font)
    ax.tick_params(colors=MUTED, labelsize=font - 1)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#475569")
    ax.grid(True, which="major", color="#334155", alpha=0.6, lw=0.7)
    leg = ax.legend(frameon=False, fontsize=font - 1, loc="upper left")
    for t in leg.get_texts():
        t.set_color(INK)
    return ax


# ----------------------------------------------------------------------
# Banner + social preview
# ----------------------------------------------------------------------
def make_banner(df_cv, out):
    fig = plt.figure(figsize=(16, 4.2), dpi=100)
    ax = gradient_background(fig)
    ax.text(0.04, 0.70, "N-Gram Language Model", color=INK, fontsize=44, fontweight="bold", va="center")
    ax.text(0.04, 0.50, "Smoothing Benchmark", color=ACCENT, fontsize=44, fontweight="bold", va="center")
    ax.text(0.04, 0.31, "Laplace  ·  Good-Turing  ·  Kneser-Ney", color=INK, fontsize=19, va="center")
    ax.text(0.04, 0.19, "Built from scratch on raw counts and benchmarked with perplexity\n"
                        "and 5-fold cross-validation on 65,161 movie-review sentences.",
            color=MUTED, fontsize=13, va="top", linespacing=1.5)
    mini_chart(fig, [0.60, 0.20, 0.36, 0.62], df_cv)
    fig.savefig(out, dpi=100)
    plt.close(fig)


def make_social_preview(df_cv, out):
    fig = plt.figure(figsize=(12.8, 6.4), dpi=100)
    ax = gradient_background(fig)
    ax.text(0.06, 0.86, "N-Gram Language Model Smoothing Benchmark", color=INK, fontsize=32,
            fontweight="bold", va="center")
    ax.text(0.06, 0.755, "Laplace  ·  Good-Turing  ·  Kneser-Ney", color=ACCENT, fontsize=20,
            fontweight="bold", va="center")
    ax.text(0.06, 0.675, "Built from scratch: perplexity, cross-validation and text generation.",
            color=MUTED, fontsize=13.5, va="center")
    mini_chart(fig, [0.10, 0.12, 0.82, 0.44], df_cv, font=13)
    fig.savefig(out, dpi=100)
    plt.close(fig)


# ----------------------------------------------------------------------
# Pipeline diagram
# ----------------------------------------------------------------------
def make_pipeline(out):
    fig, ax = plt.subplots(figsize=(15, 6.2), dpi=110)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    def node(x, y, w, h, title, body, fc, ec):
        rounded_box(ax, x, y, w, h, fc=fc, ec=ec, radius=0.12, lw=2)
        ax.text(x + w / 2, y + h - 0.32, title, ha="center", va="center", fontsize=12.5,
                fontweight="bold", color="#0f172a", zorder=5)
        ax.text(x + w / 2, y + h / 2 - 0.22, body, ha="center", va="center", fontsize=10,
                color="#334155", zorder=5, linespacing=1.4)

    def arrow(x1, y1, x2, y2, color="#64748b", rad=0.0):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=18,
                                     lw=2, color=color, connectionstyle=f"arc3,rad={rad}", zorder=1))

    # Row 1: data
    node(0.3, 3.9, 3.1, 1.9, "1  Corpus", "NLTK Movie Reviews\n2,000 reviews\n(Pang & Lee, 2004)", "#e0f2fe", "#38bdf8")
    node(4.1, 3.9, 3.3, 1.9, "2  Preprocess", "split sentences / words\nlowercase · strip punctuation\n65,161 clean sentences", "#e0f2fe", "#38bdf8")
    node(8.1, 3.9, 3.1, 1.9, "3  Split", "80 / 20 hold-out\n52,129 train · 13,032 test\n+ 5-fold cross-validation", "#e0f2fe", "#38bdf8")
    node(11.9, 3.9, 2.8, 1.9, "4  Count", "pad with <s> … </s>\ncount every order 1 … n\n(one pass over train)", "#ede9fe", "#a78bfa")
    arrow(3.4, 4.85, 4.1, 4.85)
    arrow(7.4, 4.85, 8.1, 4.85)
    arrow(11.2, 4.85, 11.9, 4.85)

    # Row 2: smoothing methods
    ax.text(7.5, 3.45, "5  Smooth: three ways to turn counts into probabilities", ha="center",
            fontsize=12, fontweight="bold", color="#0f172a")
    mx = [(0.9, "Laplace (Add-One)", "(C + 1) / (C_ctx + V)", COLORS["laplace"]),
          (5.4, "Good-Turing", "c* = (c+1) · N(c+1) / N(c)", COLORS["good_turing"]),
          (9.9, "Kneser-Ney (interpolated)", "discount + back off to\ncontinuation counts", COLORS["kneser_ney"])]
    for x, title, body, col in mx:
        rounded_box(ax, x, 1.75, 4.2, 1.35, fc="white", ec=col, radius=0.12, lw=3)
        ax.text(x + 2.1, 2.78, title, ha="center", va="center", fontsize=12, fontweight="bold", color=col, zorder=5)
        ax.text(x + 2.1, 2.2, body, ha="center", va="center", fontsize=10.5, color="#334155", zorder=5,
                family="monospace", linespacing=1.4)
    arrow(13.3, 3.9, 12.0, 3.12, rad=-0.15)
    arrow(13.3, 3.9, 7.5, 3.12, rad=-0.05, color="#cbd5e1")
    arrow(13.3, 3.9, 3.0, 3.12, rad=0.0, color="#cbd5e1")

    # Row 3: outputs
    node(2.2, 0.1, 4.6, 1.15, "6  Evaluate", "perplexity  PP = 2^( -1/N · Σ log2 P )", "#dcfce7", "#4ade80")
    node(8.2, 0.1, 4.6, 1.15, "7  Report", "CSV tables · plots · generated text", "#dcfce7", "#4ade80")
    for x, _, _, col in mx:
        arrow(x + 2.1, 1.75, 4.5 if x < 8 else 5.5, 1.27, color="#94a3b8", rad=0.0)
    arrow(6.8, 0.68, 8.2, 0.68)

    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ----------------------------------------------------------------------
# Terminal cards
# ----------------------------------------------------------------------
def terminal_card(command, body, out, width=11.5, title="bash"):
    lines = body.rstrip("\n").split("\n")
    height = 1.0 + 0.30 * (len(lines) + 2)
    fig = plt.figure(figsize=(width, height), dpi=120)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")
    rounded_box(ax, 0.1, 0.1, width - 0.2, height - 0.2, fc="#0d1117", ec="#30363d", radius=0.18, lw=1.5)
    for i, c in enumerate(["#ff5f56", "#ffbd2e", "#27c93f"]):
        ax.add_patch(plt.Circle((0.5 + i * 0.32, height - 0.5), 0.1, color=c, zorder=5))
    ax.text(width / 2, height - 0.5, title, color="#8b949e", ha="center", va="center", fontsize=10, zorder=5)
    y = height - 1.05
    ax.text(0.45, y, "$", color="#3fb950", family="monospace", fontsize=11, va="center", zorder=5)
    ax.text(0.8, y, command, color="#e6edf3", family="monospace", fontsize=11, va="center", zorder=5)
    for line in lines:
        y -= 0.30
        ax.text(0.45, y, line, color="#c9d1d9", family="monospace", fontsize=10.5, va="center", zorder=5)
    fig.savefig(out, facecolor="none", transparent=True)
    plt.close(fig)


def run_example(script):
    """Run an example script and return its real stdout."""
    return subprocess.run([sys.executable, script], capture_output=True, text=True, check=True).stdout


def wrap_output(text, width=92):
    out = []
    for line in text.rstrip("\n").split("\n"):
        out.extend(textwrap.wrap(line, width, subsequent_indent="      ") or [""])
    return "\n".join(out)


def make_terminal_quickstart(out):
    body = wrap_output(run_example("examples/quickstart.py"))
    terminal_card("python examples/quickstart.py", body, out, width=11.5)


def make_terminal_generation(out):
    body = wrap_output(run_example("examples/generate_text.py"))
    terminal_card("python examples/generate_text.py", body, out, width=11.5)


# ----------------------------------------------------------------------
# Frequency vs. versatility (Kneser-Ney intuition) on the real training data
# ----------------------------------------------------------------------
def make_frequency_vs_versatility(train, out):
    lm = NGramLanguageModel(2).fit(train)
    unigrams = {w[0]: c for w, c in lm.counts[1].items() if w[0] not in ("<s>", "</s>")}
    cont = lm._kn_continuation_counts
    words = [w for w, c in unigrams.items() if c >= 40]
    freq = np.array([unigrams[w] for w in words], dtype=float)
    versatility = np.array([cont[w] for w in words], dtype=float)
    ratio = versatility / freq  # share of occurrences that are a *new* left context

    fig, ax = plt.subplots(figsize=(9, 6), dpi=130)
    ax.scatter(freq, versatility, s=14, color="#8172B2", alpha=0.45, edgecolor="none")
    order = np.argsort(ratio)
    low = [words[i] for i in order[:8]]       # least versatile relative to frequency
    high = [words[i] for i in order[-3:]]     # most versatile relative to frequency
    labelled = {"t": (8, 6), "ve": (8, 6), "york": (8, 6), "damme": (10, -3)}
    labelled_high = {"nor": (-8, 8), "etc": (-8, 8)}
    for w in low:
        i = words.index(w)
        ax.scatter([freq[i]], [versatility[i]], s=45, color="#C44E52", zorder=5)
        if w in labelled:
            ax.annotate(f"{w}  ({int(freq[i]):,} → {int(versatility[i])})", (freq[i], versatility[i]),
                        textcoords="offset points", xytext=labelled[w], fontsize=10,
                        color="#C44E52", fontweight="bold", va="center")
    for w in high:
        i = words.index(w)
        ax.scatter([freq[i]], [versatility[i]], s=45, color="#55A868", zorder=5)
        if w in labelled_high:
            ax.annotate(f"{w}  ({int(freq[i]):,} → {int(versatility[i])})", (freq[i], versatility[i]),
                        textcoords="offset points", xytext=labelled_high[w], fontsize=10,
                        color="#55A868", fontweight="bold", ha="right")
    lim = [30, freq.max() * 1.5]
    ax.plot(lim, lim, ls="--", color="#94a3b8", lw=1)
    ax.text(lim[1] * 0.55, lim[1] * 0.40, "every occurrence follows\na brand-new word", color="#64748b",
            fontsize=9, ha="right", rotation=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim)
    ax.set_ylim(0.7, lim[1])
    ax.set_xlabel("Raw frequency  (how often the word occurs)")
    ax.set_ylabel("Continuation count  (how many DIFFERENT words precede it)")
    ax.set_title("Frequent is not the same as versatile\n"
                 "labels show: word  (raw frequency → distinct preceding words)")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return low, high, {w: (unigrams[w], cont[w]) for w in low + high}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default="assets")
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    assets = Path(args.assets)
    assets.mkdir(exist_ok=True)

    df_cv = pd.read_csv(Path(args.results) / "results_cv.csv")
    make_banner(df_cv, assets / "banner.png")
    make_social_preview(df_cv, assets / "social-preview.png")
    make_pipeline(assets / "pipeline.png")

    sentences = load_and_preprocess_sentences(n_reviews=None)
    train, _ = train_test_split(sentences, test_ratio=0.2)
    make_terminal_generation(assets / "terminal_generation.png")
    low, high, detail = make_frequency_vs_versatility(train, assets / "frequency_vs_versatility.png")
    make_terminal_quickstart(assets / "terminal_quickstart.png")
    print("low-versatility words :", {w: detail[w] for w in low})
    print("high-versatility words:", {w: detail[w] for w in high})
    print("art written to", assets)


if __name__ == "__main__":
    main()
