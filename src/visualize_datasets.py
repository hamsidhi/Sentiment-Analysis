"""
Universal visualizer for sentiment/text datasets.

Generates, for each dataset:
  - 01_overview.png
  - 02_sentiment.png  (if sentiment available)
  - 03_samples.png
  - 04_stats.png
  - report.txt (multi‑section detailed text report)

Works both for:
  - data/processed/sentiment_data.csv       -> ALL_DATASETS_COMBINED
  - each CSV in data/raw/                   -> per‑dataset reports
"""

import os
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# ---------------------------------------------------------------------------
# Utility: robust CSV loader (handles twitter & any future file)
# ---------------------------------------------------------------------------

def load_csv_robust(path: Path) -> pd.DataFrame:
    """
    Try multiple encodings so that CSVs with non‑UTF‑8 encodings (e.g. twitter)
    still load correctly. Universal for any future dataset.
    """
    encodings = ["utf-8", "latin1", "iso-8859-1", "cp1252"]
    last_error: Optional[Exception] = None

    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except UnicodeDecodeError as exc:
            last_error = exc
            continue

    # If all encodings fail, re‑raise last error with context
    raise UnicodeDecodeError(
        f"Could not decode {path} with encodings {encodings}. "
        f"Last error: {last_error}"
    )


# ---------------------------------------------------------------------------
# Core visualization
# ---------------------------------------------------------------------------

def safe_visualization(df: pd.DataFrame, dataset_name: str) -> Path:
    """
    Create a standard set of visualizations for any dataset that has at least
    a 'text' column and optionally a 'sentiment' column.
    """
    output_dir = Path("reports") / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Creating charts for {dataset_name} ({len(df):,} rows)")

    # ---------- BASIC CLEANING ----------
    df = df.copy()

    if "text" not in df.columns:
        raise ValueError(f"Dataset '{dataset_name}' does not contain a 'text' column.")

    df["text"] = df["text"].astype(str).fillna("").str.strip()
    df = df[df["text"].str.len() > 0]

    if df.empty:
        print(f"[WARN] Dataset '{dataset_name}' has no non‑empty text rows, skipping.")
        return output_dir

    df["text_length"] = df["text"].str.len()
    has_sentiment = "sentiment" in df.columns

    # ---------- 1. OVERVIEW ----------
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"{dataset_name} Dataset Analysis", fontsize=16, fontweight="bold")

    # 1A. Text length distribution
    bins = min(60, max(10, len(df) // 2000))
    axes[0, 0].hist(
        df["text_length"],
        bins=bins,
        edgecolor="black",
        alpha=0.7,
        color="#4ecdc4",
    )
    axes[0, 0].set_title("Text Length Distribution")
    axes[0, 0].set_xlabel("Characters")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].grid(True, alpha=0.3)

    # 1B. Sentiment pie chart
    if has_sentiment:
        try:
            sentiment_counts = df["sentiment"].value_counts()
            labels = [str(x) for x in sentiment_counts.index]
            colors = plt.cm.Set3(np.linspace(0, 1, len(sentiment_counts)))
            axes[0, 1].pie(
                sentiment_counts.values,
                labels=labels,
                autopct="%1.1f%%",
                colors=colors,
                startangle=90,
            )
            axes[0, 1].set_title("Sentiment Distribution")
        except Exception as exc:
            axes[0, 1].text(
                0.5,
                0.5,
                f"Could not plot sentiment pie:\n{exc}",
                ha="center",
                va="center",
                fontsize=9,
                transform=axes[0, 1].transAxes,
            )
            axes[0, 1].set_title("Sentiment Distribution (error)")
    else:
        axes[0, 1].text(
            0.5,
            0.5,
            "No 'sentiment' column\nin this dataset.",
            ha="center",
            va="center",
            fontsize=11,
            transform=axes[0, 1].transAxes,
        )
        axes[0, 1].set_title("Sentiment")

    # 1C. Text length stats
    length_mean = df["text_length"].mean()
    length_median = df["text_length"].median()
    length_max = df["text_length"].max()

    axes[1, 0].bar(
        ["Mean", "Median", "Max"],
        [length_mean, length_median, length_max],
        color=["#4ecdc4", "#ff6b6b", "#45b7d1"],
    )
    axes[1, 0].set_title("Text Length Stats")
    axes[1, 0].set_ylabel("Characters")

    # 1D. Text length percentiles
    percentiles = [10, 25, 50, 75, 90]
    perc_values = [df["text_length"].quantile(p / 100.0) for p in percentiles]
    axes[1, 1].bar(
        [f"P{p}" for p in percentiles],
        perc_values,
        color="#a0c4ff",
        alpha=0.9,
    )
    axes[1, 1].set_title("Text Length Percentiles")
    axes[1, 1].set_ylabel("Characters")

    plt.tight_layout()
    plt.savefig(output_dir / "01_overview.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ---------- 2. SENTIMENT ANALYSIS ----------
    if has_sentiment and df["sentiment"].nunique() > 1:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            f"{dataset_name} - Sentiment Analysis", fontsize=16, fontweight="bold"
        )

        # 2A. Sentiment bar chart
        sentiment_counts = df["sentiment"].value_counts().head(10)
        colors = plt.cm.Set2(np.linspace(0, 1, len(sentiment_counts)))
        sentiment_counts.plot(kind="bar", ax=axes[0, 0], color=colors)
        axes[0, 0].set_title("Sentiment Distribution")
        axes[0, 0].set_xlabel("Sentiment")
        axes[0, 0].set_ylabel("Count")
        axes[0, 0].tick_params(axis="x", rotation=45)

        # 2B. Text length by sentiment (violin)
        try:
            sns.violinplot(
                data=df,
                x="sentiment",
                y="text_length",
                hue="sentiment",
                legend=False,
                ax=axes[0, 1],
            )
            axes[0, 1].set_title("Text Length by Sentiment")
        except Exception as exc:
            axes[0, 1].text(
                0.5,
                0.5,
                f"Could not plot violin:\n{exc}",
                ha="center",
                va="center",
                fontsize=9,
                transform=axes[0, 1].transAxes,
            )
            axes[0, 1].set_title("Text Length by Sentiment (error)")

        # 2C. Word count distribution
        df["word_count"] = df["text"].str.split().str.len()
        axes[1, 0].hist(
            df["word_count"],
            bins=min(60, max(10, len(df) // 2000)),
            edgecolor="black",
            alpha=0.7,
            color="#ff9999",
        )
        axes[1, 0].set_title("Word Count Distribution")
        axes[1, 0].set_xlabel("Words")
        axes[1, 0].set_ylabel("Frequency")

        # 2D. Sentiment ratio pie
        ratio = df["sentiment"].value_counts(normalize=True)
        labels = [str(x) for x in ratio.index]
        colors = plt.cm.Pastel1(np.linspace(0, 1, len(ratio)))
        axes[1, 1].pie(
            ratio.values,
            labels=labels,
            autopct="%1.1f%%",
            colors=colors,
            startangle=90,
        )
        axes[1, 1].set_title("Sentiment Ratio")

        plt.tight_layout()
        plt.savefig(output_dir / "02_sentiment.png", dpi=300, bbox_inches="tight")
        plt.close()

    # ---------- 3. TEXT SAMPLES ----------
    fig, axes = plt.subplots(4, 3, figsize=(20, 16))
    fig.suptitle(f"{dataset_name} - Text Samples", fontsize=16, fontweight="bold")

    sample_rows = df.head(12)
    for i, (_, row) in enumerate(sample_rows.iterrows()):
        ax = axes[i // 3, i % 3]
        full_text = str(row["text"])
        preview = full_text[:200] + ("..." if len(full_text) > 200 else "")
        sentiment_label = row["sentiment"] if has_sentiment else "N/A"

        ax.text(
            0.05,
            0.95,
            preview,
            va="top",
            fontsize=10,
            wrap=True,
            transform=ax.transAxes,
        )
        ax.set_title(
            f"#{i + 1} | sentiment={sentiment_label} | len={len(full_text)}",
            fontsize=9,
            pad=8,
        )
        ax.axis("off")

    # Hide unused axes
    for j in range(len(sample_rows), 12):
        axes[j // 3, j % 3].axis("off")

    plt.tight_layout()
    plt.savefig(output_dir / "03_samples.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ---------- 4. SUMMARY TABLE ----------
    stats = {
        "Total Samples": f"{len(df):,}",
        "Avg Length": f"{length_mean:.0f} chars",
        "Median Length": f"{length_median:.0f} chars",
        "Longest Text": f"{length_max:,} chars",
        "Shortest Text": f"{df['text_length'].min()} chars",
    }

    if has_sentiment:
        value_counts = df["sentiment"].value_counts(normalize=True)
        if set([0, 1]).issubset(set(value_counts.index)):
            pos_ratio = value_counts.get(1, 0.0) * 100
            stats["Positive % (label 1)"] = f"{pos_ratio:.1f}%"
        stats["Unique Labels"] = df["sentiment"].nunique()

    summary_df = pd.DataFrame(list(stats.items()), columns=["Metric", "Value"])

    fig, ax = plt.subplots(figsize=(12, 0.6 * len(summary_df) + 1))
    ax.axis("tight")
    ax.axis("off")
    table = ax.table(
        cellText=summary_df.values,
        colLabels=summary_df.columns,
        cellLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)

    for row_idx in range(len(summary_df) + 1):
        table[(row_idx, 0)].set_facecolor("#e3f2fd")

    ax.set_title(
        f"{dataset_name} - Summary Statistics",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    plt.savefig(output_dir / "04_stats.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ---------- 5. DETAILED TEXT REPORT ----------
    report_path = output_dir / "report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"{dataset_name.upper()} DATASET REPORT\n")
        f.write("=" * 70 + "\n\n")

        # Section 1: Overview
        f.write("1. OVERVIEW\n")
        f.write(f"   Total samples           : {len(df):,}\n")
        f.write(f"   Columns                 : {list(df.columns)}\n")
        f.write("\n")

        # Section 2: Text length statistics
        f.write("2. TEXT LENGTH (in characters)\n")
        f.write(
            f"   Mean length             : {length_mean:.2f}\n"
            f"   Median length           : {length_median:.2f}\n"
            f"   Standard deviation      : {df['text_length'].std():.2f}\n"
            f"   Minimum length          : {df['text_length'].min()}\n"
            f"   Maximum length          : {length_max}\n"
        )
        for p in [10, 25, 50, 75, 90]:
            f.write(
                f"   {p:>2}th percentile        : "
                f"{df['text_length'].quantile(p/100.0):.2f}\n"
            )
        f.write("\n")

        # Section 3: Sentiment statistics
        if has_sentiment:
            f.write("3. SENTIMENT STATISTICS\n")
            sent_counts = df["sentiment"].value_counts()
            sent_ratio = df["sentiment"].value_counts(normalize=True)
            for label in sent_counts.index:
                f.write(
                    f"   Label '{label}': "
                    f"count={sent_counts[label]:,}, "
                    f"fraction={sent_ratio[label]:.3f}\n"
                )
            f.write(f"   Number of unique labels : {df['sentiment'].nunique()}\n")
            f.write("\n")
        else:
            f.write("3. SENTIMENT STATISTICS\n")
            f.write("   No 'sentiment' column available in this dataset.\n\n")

        # Section 4: Example rows
        f.write("4. EXAMPLE ROWS (first 5)\n")
        for idx, row in df.head(5).iterrows():
            txt = str(row["text"])
            preview = txt[:200] + ("..." if len(txt) > 200 else "")
            sentiment_label = row["sentiment"] if has_sentiment else "N/A"
            f.write(f"   Row index {idx} | sentiment={sentiment_label}\n")
            f.write(f"      {preview}\n")
        f.write("\n")

        f.write("End of report.\n")

    print(f"[OK] Saved charts and report to {output_dir}")
    return output_dir


# ---------------------------------------------------------------------------
# Main entry: unified + per‑dataset
# ---------------------------------------------------------------------------

def main() -> None:
    Path("reports").mkdir(exist_ok=True)

    # 1) Unified / processed dataset
    unified_path = Path("data/processed/sentiment_data.csv")
    if unified_path.exists():
        print("[INFO] Analyzing unified dataset...")
        df_unified = load_csv_robust(unified_path)
        safe_visualization(df_unified, "ALL_DATASETS_COMBINED")
    else:
        print("[WARN] Unified file data/processed/sentiment_data.csv not found.")

    # 2) Individual raw datasets
    raw_folder = Path("data/raw")
    if raw_folder.exists():
        print("\n[INFO] Analyzing individual raw datasets...")
        for file in sorted(raw_folder.iterdir()):
            if file.suffix.lower() == ".csv":
                dataset_name = file.stem
                print(f"[INFO] Loading raw dataset: {file}")
                df_raw = load_csv_robust(file)

                # Ensure there is a 'text' column for universality
                if "text" not in df_raw.columns:
                    candidates = [
                        c
                        for c in df_raw.columns
                        if any(
                            kw in c.lower()
                            for kw in ["review", "text", "tweet", "comment", "content"]
                        )
                    ]
                    if candidates:
                        df_raw = df_raw.rename(columns={candidates[0]: "text"})
                    else:
                        last_col = df_raw.columns[-1]
                        df_raw = df_raw.rename(columns={last_col: "text"})

                safe_visualization(df_raw, dataset_name)
    else:
        print("[WARN] Raw folder data/raw/ not found.")

    print("\n[INFO] Visualization complete. Check the 'reports/' folder.")


if __name__ == "__main__":
    plt.style.use("default")
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    main()
