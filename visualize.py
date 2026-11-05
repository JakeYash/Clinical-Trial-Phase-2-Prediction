# Data Visualization for Phase 2 Clinical Trials
# Reads pipeline output and saves descriptive statistics and charts.
# This file does NOT perform sentiment analysis or train a model.

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Save plots to files instead of opening a window.
import matplotlib.pyplot as plt
import seaborn as sns


# ---------- Settings ----------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "processed" / "trials_dataset.csv"
FILTERED_PATH = BASE_DIR / "data" / "processed" / "phase2_trials_filtered.csv"
ENRICHED_PATH = BASE_DIR / "data" / "raw" / "phase2_trials_enriched.jsonl"
ABSTRACTS_PATH = BASE_DIR / "data" / "raw" / "pubmed_abstracts.jsonl"
OUTPUT_FOLDER = BASE_DIR / "outputs"

MIN_TRIALS = 5  # Lower to 1 or 2 for a small --max-trials 50 test.
TOP_N = 15

sns.set_theme(style="whitegrid")


# ---------- Helpers ----------
def load_data():
    if not DATA_PATH.exists():
        print("ERROR: Dataset not found:", DATA_PATH)
        print("Run pipeline.py first to create trials_dataset.csv.")
        return None

    df = pd.read_csv(DATA_PATH, low_memory=False)
    if df.empty:
        print("ERROR: The dataset is empty.")
        return None

    number_columns = [
        "label", "enrollment_count", "number_of_arms",
        "pubmed_mention_count", "pubmed_with_abstract_count"
    ]
    for col in number_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def save_chart(name):
    plt.tight_layout()
    path = OUTPUT_FOLDER / (name + ".png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved chart:", path.name)


def existing_columns(df, columns):
    found = []
    for col in columns:
        if col in df.columns:
            found.append(col)
        else:
            print("WARNING: Missing column, skipping:", col)
    return found


def nonnegative_values(df, column):
    # Article counts and enrollment should never be negative.
    return df[column].dropna().loc[lambda values: values >= 0]


# ---------- Overall dataset statistics ----------
def print_summary(df):
    summary = {
        "number_of_trials": len(df),
        "number_of_columns": len(df.columns),
        "fully_duplicate_rows": int(df.duplicated().sum()),
    }

    if "nct_id" in df.columns:
        summary["unique_nct_ids"] = int(df["nct_id"].nunique())
        summary["duplicate_nct_ids"] = int(df["nct_id"].duplicated().sum())

    if "label" in df.columns:
        labels = df["label"]
        valid = labels[labels.isin([0, 1])]
        summary["successful_trials"] = int((valid == 1).sum())
        summary["failed_trials"] = int((valid == 0).sum())
        summary["missing_or_invalid_labels"] = len(df) - len(valid)
        if len(valid) > 0:
            summary["success_rate_percent"] = round(valid.mean() * 100, 2)

    if "pubmed_mention_count" in df.columns:
        counts = nonnegative_values(df, "pubmed_mention_count")
        if len(counts) > 0:
            summary["total_pubmed_mentions_across_trials"] = int(counts.sum())
            summary["average_articles_per_trial"] = round(counts.mean(), 2)
            summary["median_articles_per_trial"] = float(counts.median())
            summary["maximum_articles_for_one_trial"] = int(counts.max())
            summary["trials_with_zero_articles"] = int((counts == 0).sum())
            summary["percent_with_zero_articles"] = round((counts == 0).mean() * 100, 2)
            summary["trials_at_200_article_cap"] = int((counts >= 200).sum())

    if "pubmed_with_abstract_count" in df.columns:
        counts = nonnegative_values(df, "pubmed_with_abstract_count")
        if len(counts) > 0:
            summary["trials_with_at_least_one_abstract"] = int((counts > 0).sum())

    if "enrollment_count" in df.columns:
        enrollment = nonnegative_values(df, "enrollment_count")
        if len(enrollment) > 0:
            summary["mean_enrollment"] = round(enrollment.mean(), 2)
            summary["median_enrollment"] = float(enrollment.median())
            summary["maximum_enrollment"] = float(enrollment.max())

    print("\n========== DATASET SUMMARY ==========")
    for name, value in summary.items():
        print(f"{name}: {value}")

    pd.DataFrame(list(summary.items()), columns=["statistic", "value"]).to_csv(
        OUTPUT_FOLDER / "dataset_summary.csv", index=False
    )
    print("Saved: dataset_summary.csv")

    # All columns, not just the few used in charts.
    missing = pd.DataFrame({
        "column": df.columns,
        "missing_count": df.isna().sum().values,
        "missing_percent": (df.isna().mean() * 100).round(2).values,
    }).sort_values("missing_percent", ascending=False)
    missing.to_csv(OUTPUT_FOLDER / "missing_values_summary.csv", index=False)
    print("Saved: missing_values_summary.csv")


def print_filtering_stats(df):
    # Optional: earlier-stage files show data lost BEFORE the final CSV.
    # These counts are comparable only if all files are from the same run.
    if not ENRICHED_PATH.exists():
        print("Enriched trial file not found; skipping pre-filter date check.")
        return

    total = 0
    missing_dates = 0
    with open(ENRICHED_PATH, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            try:
                trial = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            if not str(trial.get("start_date") or "").strip():
                missing_dates += 1

    print("\n========== FILTERING CHECK ==========")
    print("Trials in CT.gov enriched file:", total)
    print("Enriched trials missing start_date:", missing_dates)
    if FILTERED_PATH.exists():
        filtered_count = len(pd.read_csv(FILTERED_PATH, usecols=["nct_id"]))
        print("Trials after filtering:", filtered_count)
    print("Trials in final dataset:", len(df))
    print("Note: filtering counts assume these files came from the same pipeline run.")


# ---------- Chart 1: Success versus failure ----------
def plot_class_balance(df):
    if "label" not in df.columns:
        return

    labels = df["label"]
    counts = [(labels == 0).sum(), (labels == 1).sum()]
    total = sum(counts)
    if total == 0:
        return

    names = ["Failure (0)", "Success (1)"]
    plt.figure(figsize=(7, 4))
    plt.bar(names, counts, color=["#d95f5f", "#5fa86b"])
    for i, count in enumerate(counts):
        plt.text(i, count, f"{count} ({count / total * 100:.1f}%)",
                 ha="center", va="bottom")
    plt.title("Phase 2 Trial Outcomes")
    plt.ylabel("Number of trials")
    plt.ylim(0, max(counts) * 1.15 + 1)
    save_chart("01_class_balance")


# ---------- Chart 2: Missing data ----------
def plot_missingness(df):
    columns = [
        "start_date", "primary_completion_date", "completion_date",
        "enrollment_count", "number_of_arms", "lead_sponsor_name",
        "condition", "drugs", "pubmed_mention_count",
        "pubmed_with_abstract_count", "why_stopped"
    ]
    columns = existing_columns(df, columns)
    if not columns:
        return

    # Sample up to 400 trials to make a large heatmap readable.
    sample = df[columns].sample(n=min(400, len(df)), random_state=42)
    plt.figure(figsize=(11, 6))
    sns.heatmap(sample.isna(), cbar=False, yticklabels=False, cmap="Greys")
    plt.title("Missing Values (dark = missing; up to 400 sampled trials)")
    plt.xticks(rotation=45, ha="right")
    save_chart("02_missingness_heatmap")

    percent_missing = (df[columns].isna().mean() * 100).sort_values()
    plt.figure(figsize=(9, 5))
    plt.barh(percent_missing.index, percent_missing.values)
    plt.title("Missing Values by Column (All Trials)")
    plt.xlabel("Percent missing")
    plt.xlim(0, 100)
    save_chart("02_missingness_percent")


# ---------- Chart 3: PubMed article distributions ----------
def plot_pubmed_counts(df):
    columns = existing_columns(
        df, ["pubmed_mention_count", "pubmed_with_abstract_count"]
    )
    if not columns:
        return

    fig, axes = plt.subplots(1, len(columns), figsize=(6 * len(columns), 4))
    if len(columns) == 1:
        axes = [axes]

    for ax, col in zip(axes, columns):
        values = nonnegative_values(df, col)
        if values.empty:
            continue
        sns.histplot(np.log1p(values), bins=30, ax=ax, color="#4c72b0")
        ax.set_title(col)
        ax.set_xlabel("log(1 + article count)")
        ax.set_ylabel("Number of trials")
        print(f"{col}: {(values == 0).mean() * 100:.1f}% with zero articles")

    save_chart("03_pubmed_counts")


# ---------- Chart 4: Trial size and number of arms ----------
def plot_trial_size(df):
    columns = existing_columns(df, ["enrollment_count", "number_of_arms"])
    if not columns:
        return

    fig, axes = plt.subplots(1, len(columns), figsize=(6 * len(columns), 4))
    if len(columns) == 1:
        axes = [axes]

    for ax, col in zip(axes, columns):
        values = nonnegative_values(df, col)
        if values.empty:
            continue
        if col == "enrollment_count":
            sns.histplot(np.log1p(values), bins=30, ax=ax, color="#4c72b0")
            ax.set_title("Enrollment Distribution")
            ax.set_xlabel("log(1 + enrollment)")
        else:
            arm_counts = values.astype(int).value_counts().sort_index()
            ax.bar(arm_counts.index.astype(str), arm_counts.values)
            ax.set_title("Number of Trial Arms")
            ax.set_xlabel("Number of arms")
            ax.tick_params(axis="x", rotation=45)
        ax.set_ylabel("Number of trials")

    save_chart("04_trial_size")


# ---------- Chart 5: Distributions split by success or failure ----------
def plot_by_label(df):
    if "label" not in df.columns:
        return
    columns = existing_columns(df, ["enrollment_count", "pubmed_mention_count"])
    if not columns:
        return

    fig, axes = plt.subplots(2, len(columns), figsize=(6 * len(columns), 9),
                             squeeze=False)
    for i, col in enumerate(columns):
        temp = df[df["label"].isin([0, 1])].copy()
        temp = temp[temp[col].notna() & (temp[col] >= 0)]
        if temp.empty:
            continue
        temp["log_value"] = np.log1p(temp[col])
        temp["outcome"] = temp["label"].map({0: "Failure", 1: "Success"})

        sns.boxplot(data=temp, x="outcome", y="log_value", ax=axes[0, i],
                    order=["Failure", "Success"], color="#4c72b0")
        axes[0, i].set_title("Box Plot: " + col)
        axes[0, i].set_ylabel("log(1 + value)")
        axes[0, i].set_xlabel("")

        sns.violinplot(data=temp, x="outcome", y="log_value", ax=axes[1, i],
                       order=["Failure", "Success"], cut=0, color="#5fa86b")
        axes[1, i].set_title("Violin Plot: " + col)
        axes[1, i].set_ylabel("log(1 + value)")
        axes[1, i].set_xlabel("Trial outcome")

    save_chart("05_by_label")


# ---------- Chart 6: Top sponsors and conditions ----------
def plot_top_categories(df, column, filename):
    if column not in df.columns or "label" not in df.columns:
        print("Skipping category chart (column missing):", column)
        return

    temp = df[df["label"].isin([0, 1])].dropna(subset=[column]).copy()
    if temp.empty:
        return

    stats = temp.groupby(column)["label"].agg(["count", "mean"])
    stats = stats[stats["count"] >= MIN_TRIALS]
    stats = stats.sort_values("count", ascending=False).head(TOP_N)
    if stats.empty:
        print(f"No groups in {column} have at least {MIN_TRIALS} trials.")
        return

    stats = stats.iloc[::-1]
    display_names = [str(name)[:50] for name in stats.index]
    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, len(stats) * 0.45)))

    axes[0].barh(display_names, stats["count"])
    axes[0].set_title("Most Common: " + column)
    axes[0].set_xlabel("Number of trials")

    axes[1].barh(display_names, stats["mean"] * 100)
    axes[1].set_title("Success Rate: " + column)
    axes[1].set_xlabel("Success rate (%)")
    axes[1].set_xlim(0, 100)

    save_chart(filename)


# ---------- Charts 7 & 8: PubMed coverage buckets ----------
def plot_pubmed_coverage(df):
    if "pubmed_mention_count" not in df.columns:
        return

    counts = df["pubmed_mention_count"]
    temp = df[counts.notna() & (counts >= 0)].copy()
    if temp.empty:
        return

    # The pipeline caps retrieved PubMed records at 200 per drug/trial.
    bins = [-0.5, 0.5, 5.5, 20.5, 50.5, 199.5, np.inf]
    labels = ["0", "1-5", "6-20", "21-50", "51-199", "200+"]
    temp["article_bucket"] = pd.cut(
        temp["pubmed_mention_count"], bins=bins, labels=labels
    )
    temp["valid_label"] = temp["label"].where(temp["label"].isin([0, 1])) \
        if "label" in temp.columns else np.nan

    stats = temp.groupby("article_bucket", observed=False).agg(
        number_of_trials=("article_bucket", "size"),
        trials_with_labels=("valid_label", "count"),
        success_rate=("valid_label", "mean")
    )
    stats["success_rate_percent"] = (stats["success_rate"] * 100).round(2)
    stats.drop(columns="success_rate").to_csv(OUTPUT_FOLDER / "pubmed_article_buckets.csv")
    print("Saved: pubmed_article_buckets.csv")

    plt.figure(figsize=(9, 4))
    plt.bar(labels, stats["number_of_trials"])
    plt.title("PubMed Articles per Trial: Coverage Buckets")
    plt.xlabel("Number of PubMed articles")
    plt.ylabel("Number of trials")
    save_chart("07_pubmed_article_buckets")

    enough = stats[stats["trials_with_labels"] >= MIN_TRIALS]
    if enough.empty:
        print("Not enough labeled trials per bucket to show success rates.")
        return

    plt.figure(figsize=(9, 4))
    plt.bar(enough.index.astype(str), enough["success_rate_percent"])
    plt.title("Trial Success Rate by Pre-Trial PubMed Article Volume")
    plt.xlabel("Number of PubMed articles")
    plt.ylabel("Success rate (%)")
    plt.ylim(0, 100)
    for i, (_, row) in enumerate(enough.iterrows()):
        plt.text(i, row["success_rate_percent"] + 2,
                 f"n={int(row['trials_with_labels'])}", ha="center")
    save_chart("08_pubmed_success_by_volume")


# ---------- Chart 9: Unusually large PubMed article counts ----------
def plot_pubmed_outliers(df):
    if "pubmed_mention_count" not in df.columns:
        return

    values = nonnegative_values(df, "pubmed_mention_count")
    if values.empty or values.max() == 0:
        return

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1
    upper_limit = q3 + 1.5 * iqr
    print(f"PubMed IQR outlier cutoff: greater than {upper_limit:.1f} articles")
    print("Trials above this cutoff:", int((values > upper_limit).sum()))
    print("Remember: values of 200 may reflect the retrieval cap.")

    # Save the highest-count trials so the group can investigate them.
    top_indices = values.nlargest(TOP_N).index
    output_cols = existing_columns(
        df, ["nct_id", "pubmed_mention_count", "label", "condition", "lead_sponsor_name"]
    )
    top = df.loc[top_indices, output_cols].copy()
    top["above_iqr_cutoff"] = top["pubmed_mention_count"] > upper_limit
    top.to_csv(OUTPUT_FOLDER / "pubmed_high_count_trials.csv", index=False)
    print("Saved: pubmed_high_count_trials.csv")

    names = (top["nct_id"].astype(str) if "nct_id" in top.columns
             else pd.Series(top.index.astype(str), index=top.index))
    plt.figure(figsize=(10, 6))
    plt.barh(names.iloc[::-1], top["pubmed_mention_count"].iloc[::-1])
    plt.title(f"Top {len(top)} Trials by Number of PubMed Articles")
    plt.xlabel("Number of articles retrieved")
    save_chart("09_high_pubmed_counts")


# ---------- Chart 10: Raw PubMed abstract file ----------
def plot_abstracts():
    if not ABSTRACTS_PATH.exists():
        print("Abstracts file not found, skipping:", ABSTRACTS_PATH)
        return

    # Read line-by-line to avoid loading a potentially huge JSONL file into RAM.
    per_trial = Counter()
    word_lengths = []
    records = 0
    with open(ABSTRACTS_PATH, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            records += 1
            if record.get("nct_id"):
                per_trial[record["nct_id"]] += 1
            abstract_text = record.get("abstract") or ""
            word_lengths.append(len(abstract_text.split())
                                if isinstance(abstract_text, str) else 0)

    print("Total PubMed JSONL records:", records)
    if records == 0:
        print("The PubMed abstracts file is empty; skipping abstract plots.")
        return

    print("PubMed records with empty abstracts:", sum(x == 0 for x in word_lengths))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    if per_trial:
        sns.histplot(list(per_trial.values()), bins=30, ax=axes[0], color="#4c72b0")
    axes[0].set_title("PubMed Records per Represented Trial")
    axes[0].set_xlabel("Number of records (excludes trials with zero records)")

    sns.histplot(word_lengths, bins=30, ax=axes[1], color="#4c72b0")
    axes[1].set_title("PubMed Abstract Lengths")
    axes[1].set_xlabel("Number of words")
    save_chart("10_abstracts")


# ---------- Run everything ----------
def main():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    df = load_data()
    if df is None:
        return

    print_summary(df)
    print_filtering_stats(df)

    plot_class_balance(df)
    plot_missingness(df)
    plot_pubmed_counts(df)
    plot_trial_size(df)
    plot_by_label(df)
    plot_top_categories(df, "lead_sponsor_name", "06_top_sponsors")

    # Accept either sponsor-class name, depending on how the data was flattened.
    if "sponsors/lead_sponsor/agency_class" in df.columns:
        sponsor_class = "sponsors/lead_sponsor/agency_class"
    else:
        sponsor_class = "lead_sponsor_class"
    plot_top_categories(df, sponsor_class, "06_sponsor_class")
    plot_top_categories(df, "condition", "06_top_conditions")

    plot_pubmed_coverage(df)
    plot_pubmed_outliers(df)
    plot_abstracts()
    print("\nDone! Check the outputs folder for charts and summary CSVs.")


if __name__ == "__main__":
    main()
