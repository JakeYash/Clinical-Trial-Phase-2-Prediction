"""Create leakage-safe, model-ready feature matrices.

This script reads the trial-level CSV produced by ``pipeline.py`` and uses
TrialBench's existing ``top_split`` column to keep the original train/test
assignment.  Preprocessing is fitted on the training rows only, then applied
unchanged to the test rows.

Run from the repository root:

    python3 feature_engineering.py

Generated files are written under ``data/processed/model_ready`` and are
ignored by Git because they can be recreated from the data pipeline.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT_DIR / "data" / "processed" / "trials_dataset.csv"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "data" / "processed" / "model_ready"
DEFAULT_SENTIMENT = ROOT_DIR / "data" / "processed" / "pubmed_sentiment_features.csv"

TARGET_COLUMN = "label"
SPLIT_COLUMN = "top_split"

# These fields are known only during/after the trial or directly reveal its
# result. They must never be given to a model that predicts pre-trial success.
LEAKAGE_COLUMNS = {
    "overall_status",
    "why_stopped",
    "has_results",
    "primary_completion_date",
    "completion_date",
    "fdaaa801_violation",
}

# Identifiers and raw/high-cardinality text are kept out of the structured
# baseline. Separate NLP/chemistry models would be needed to represent them.
NON_PREDICTOR_COLUMNS = {
    "nct_id",
    "pubmed_pmids",
    "brief_summary/textblock",
    "brief_title_top",
    "brief_title",
    "criteria",
    "intervention/description",
    "keyword",
    "location/facility/address/city",
    "diseases",
    "condition_browse/mesh_term",
    "condition",
    "icdcodes",
    "drugs",
    "intervention_browse/mesh_term",
    "intervention_name",
    "smiless",
    "lead_sponsor_name",
    "countries",
}

# Prefer the newer CT.gov fields where the source dataset contains duplicates.
DUPLICATE_COLUMNS = {
    "enrollment",
    "n_arms",
    "eligibility/gender",
    "eligibility/healthy_volunteers",
    "eligibility/minimum_age",
    "eligibility/maximum_age",
    "study_design_info/allocation",
    "study_design_info/intervention_model",
    "study_design_info/primary_purpose",
    "sponsors/lead_sponsor/agency_class",
    "oversight_info/is_fda_regulated_drug",
    "phase_top",
    "study_type_top",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build normalized and encoded train/test feature matrices."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--sentiment",
        type=Path,
        default=DEFAULT_SENTIMENT,
        help="Per-trial output of score_sentiment.py; skipped if absent.",
    )
    return parser.parse_args()


def merge_sentiment_features(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Left-join ``score_sentiment.py`` output onto the trial table by nct_id.

    The scorer writes one row per trial to a separate CSV, so it has to be
    joined here for the sentiment columns to reach the model.  Trials missing
    from that file keep NaN values (imputed later from training medians).
    """
    sentiment = pd.read_csv(path, low_memory=False)
    if "nct_id" not in sentiment.columns:
        raise ValueError(f"{path}: missing nct_id column")
    if sentiment["nct_id"].duplicated().any():
        raise ValueError(f"{path}: nct_id must be unique (one row per trial)")
    new_columns = [
        column
        for column in sentiment.columns
        if column == "nct_id" or column not in df.columns
    ]
    return df.merge(sentiment[new_columns], on="nct_id", how="left")


def add_relative_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Replace the safe pre-trial start date with model-readable values.

    Absolute timestamps are not fed to the model.  A fixed reference date is
    used so the transformation is identical for training and future data.
    Completion dates are deliberately not converted because they are leakage.
    """
    result = df.copy()
    if "start_date" in result.columns:
        start_date = pd.to_datetime(result["start_date"], errors="coerce")
        reference_date = pd.Timestamp("2000-01-01")
        result["start_days_since_2000"] = (
            start_date - reference_date
        ).dt.days.astype("float64")
        result["start_month"] = start_date.dt.month.astype("float64")
        result = result.drop(columns=["start_date"])
    return result


def _age_to_years(value) -> float:
    """Convert CT.gov age strings such as '18 Years' or '6 Months' to years."""
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    if not text or text in {"none", "nan", "n/a"}:
        return np.nan
    pieces = text.split()
    try:
        amount = float(pieces[0])
    except (ValueError, IndexError):
        return np.nan
    unit = pieces[1] if len(pieces) > 1 else "years"
    if unit.startswith("day"):
        return amount / 365.25
    if unit.startswith("week"):
        return amount / 52.18
    if unit.startswith("month"):
        return amount / 12.0
    return amount


def add_age_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn minimum/maximum age text into comparable numeric years."""
    result = df.copy()
    for source, output in (
        ("min_age", "min_age_years"),
        ("max_age", "max_age_years"),
    ):
        if source in result.columns:
            result[output] = result[source].map(_age_to_years)
            result = result.drop(columns=[source])
    return result


def _sentiment_columns(df: pd.DataFrame) -> list[str]:
    """Find numeric sentiment outputs added by Task 1 without fixed names."""
    keywords = ("sentiment", "positive", "negative", "neutral", "polarity")
    return [
        column
        for column in df.select_dtypes(include=["number", "bool"]).columns
        if any(keyword in column.lower() for keyword in keywords)
    ]


def select_predictors(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Remove targets, leakage, identifiers, raw text, and duplicate fields."""
    excluded = (
        {TARGET_COLUMN, SPLIT_COLUMN}
        | LEAKAGE_COLUMNS
        | NON_PREDICTOR_COLUMNS
        | DUPLICATE_COLUMNS
    )
    predictors = df.drop(
        columns=[column for column in excluded if column in df.columns]
    ).copy()

    return predictors, []


def build_preprocessor(
    X_train: pd.DataFrame,
) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric_columns = X_train.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()
    categorical_columns = [
        column for column in X_train.columns if column not in numeric_columns
    ]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value="missing"),
            ),
            (
                "one_hot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=5,
                    max_categories=50,
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor, numeric_columns, categorical_columns


def create_feature_matrices(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, dict, object]:
    required = {TARGET_COLUMN, SPLIT_COLUMN}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    labels = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    valid_labels = labels.isin([0, 1])
    if not valid_labels.all():
        df = df.loc[valid_labels].copy()
        labels = labels.loc[valid_labels]

    split = df[SPLIT_COLUMN].astype(str).str.strip().str.lower()
    train_mask = split.eq("train")
    test_mask = split.eq("test")
    if not train_mask.any() or not test_mask.any():
        raise ValueError(
            "top_split must contain at least one 'train' and one 'test' row."
        )

    engineered = add_age_features(add_relative_date_features(df))
    predictors, _ = select_predictors(engineered)

    # Feature selection is based only on training data so the test set cannot
    # influence which fields are retained.
    train_predictors = predictors.loc[train_mask]
    constant_columns = [
        column
        for column in train_predictors.columns
        if train_predictors[column].nunique(dropna=True) <= 1
    ]
    predictors = predictors.drop(columns=constant_columns)
    sentiment_columns = _sentiment_columns(predictors)

    X_train_raw = predictors.loc[train_mask].reset_index(drop=True)
    X_test_raw = predictors.loc[test_mask].reset_index(drop=True)
    y_train = labels.loc[train_mask].astype(int).reset_index(drop=True)
    y_test = labels.loc[test_mask].astype(int).reset_index(drop=True)

    # CSV inference can create mixed Python types (for example True and
    # "False") in a single categorical column. OneHotEncoder requires one
    # consistent type, so convert every nonmissing category to text first.
    numeric_columns = X_train_raw.select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()
    categorical_columns = [
        column for column in X_train_raw.columns if column not in numeric_columns
    ]
    for column in categorical_columns:
        X_train_raw[column] = X_train_raw[column].map(
            lambda value: np.nan if pd.isna(value) else str(value)
        )
        X_test_raw[column] = X_test_raw[column].map(
            lambda value: np.nan if pd.isna(value) else str(value)
        )

    preprocessor, numeric_columns, categorical_columns = build_preprocessor(
        X_train_raw
    )
    X_train_array = preprocessor.fit_transform(X_train_raw)
    X_test_array = preprocessor.transform(X_test_raw)
    feature_names = preprocessor.get_feature_names_out().tolist()

    X_train = pd.DataFrame(X_train_array, columns=feature_names)
    X_test = pd.DataFrame(X_test_array, columns=feature_names)

    if X_train.isna().any().any() or X_test.isna().any().any():
        raise ValueError("Feature engineering produced missing values.")
    if not np.isfinite(X_train.to_numpy()).all():
        raise ValueError("Training matrix contains infinite values.")
    if not np.isfinite(X_test.to_numpy()).all():
        raise ValueError("Test matrix contains infinite values.")

    summary = {
        "input_rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "input_predictor_columns": int(len(predictors.columns)),
        "output_feature_columns": int(len(feature_names)),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "sentiment_columns_detected": sentiment_columns,
        "constant_or_empty_columns_removed": constant_columns,
        "leakage_columns_excluded": sorted(
            LEAKAGE_COLUMNS.intersection(df.columns)
        ),
    }
    return X_train, X_test, y_train, y_test, summary, preprocessor


def run(
    input_path: Path,
    output_dir: Path,
    sentiment_path: Path | None = None,
) -> None:
    if not input_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {input_path}\nRun pipeline.py first."
        )

    df = pd.read_csv(input_path, low_memory=False)
    if sentiment_path is not None and sentiment_path.exists():
        df = merge_sentiment_features(df, sentiment_path)
        print(f"Merged sentiment features from {sentiment_path}")
    else:
        print(
            "No sentiment features file found; run score_sentiment.py to "
            "include PubMed sentiment."
        )
    X_train, X_test, y_train, y_test, summary, preprocessor = (
        create_feature_matrices(df)
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(output_dir / "X_train.csv", index=False)
    X_test.to_csv(output_dir / "X_test.csv", index=False)
    y_train.rename(TARGET_COLUMN).to_csv(output_dir / "y_train.csv", index=False)
    y_test.rename(TARGET_COLUMN).to_csv(output_dir / "y_test.csv", index=False)
    with (output_dir / "preprocessor.pkl").open("wb") as file:
        pickle.dump(preprocessor, file)
    with (output_dir / "feature_engineering_summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2)

    print("Feature engineering complete.")
    print(f"Training matrix: {X_train.shape}")
    print(f"Testing matrix:  {X_test.shape}")
    print(f"Sentiment columns detected: {summary['sentiment_columns_detected']}")
    print(f"Files saved in: {output_dir}")


if __name__ == "__main__":
    arguments = _parse_args()
    run(arguments.input, arguments.output_dir, arguments.sentiment)
