"""Home Credit: fixed experiments, development CV, and one final holdout check.

Run from the repository root: python -m src.train --help
No downloads, Kaggle uploads, or credentials are handled by this module.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from lightgbm import LGBMClassifier
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ID = "SK_ID_CURR"
LABEL = "TARGET"
SOURCE_URL = "https://www.kaggle.com/competitions/home-credit-default-risk/data"


@dataclass(frozen=True)
class Experiment:
    name: str
    model: str
    engineered: bool = False
    history: bool = False
    drop_external: bool = False


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_applications(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = [data_dir / "application_train.csv", data_dir / "application_test.csv"]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path}. Download the competition data and extract it to {data_dir}."
            )
    train, test = (pd.read_csv(path, low_memory=False) for path in paths)
    for name, frame in [("train", train), ("test", test)]:
        if ID not in frame or frame[ID].isna().any() or frame[ID].duplicated().any():
            raise ValueError(f"{name}: {ID} must be present, non-null, and unique.")
    if LABEL not in train or LABEL in test:
        raise ValueError("Only application_train.csv must contain TARGET.")
    if train[LABEL].isna().any() or set(train[LABEL].unique()) != {0, 1}:
        raise ValueError("TARGET must contain both 0 and 1, without missing values.")
    if set(train.columns) - {LABEL} != set(test.columns):
        raise ValueError("Train and test predictor columns do not match.")
    if set(train[ID]).intersection(test[ID]):
        raise ValueError("Train and test applicant IDs overlap.")
    return train, test


def clean_application(frame: pd.DataFrame) -> pd.DataFrame:
    """Only deterministic row-wise cleaning; no population statistics are fitted."""
    result = frame.drop(columns=[ID, LABEL], errors="ignore").copy()
    if "DAYS_EMPLOYED" in result:
        sentinel = result["DAYS_EMPLOYED"].eq(365243)
        result["DAYS_EMPLOYED_SENTINEL"] = sentinel.astype("int8")
        result.loc[sentinel, "DAYS_EMPLOYED"] = np.nan
    if "CODE_GENDER" in result:
        result["CODE_GENDER"] = result["CODE_GENDER"].replace("XNA", np.nan)
    for column in result.select_dtypes(include=["object", "string", "category"]):
        # Keep object/np.nan for consistent SimpleImputer behavior across pandas versions.
        result[column] = result[column].astype(object).where(result[column].notna(), np.nan)
    return result.replace([np.inf, -np.inf], np.nan)


def make_features(
    frame: pd.DataFrame, experiment: Experiment, history: pd.DataFrame | None = None
) -> pd.DataFrame:
    result = clean_application(frame)
    if experiment.drop_external:
        external = [column for column in result if column.startswith("EXT_SOURCE_")]
        result = result.drop(columns=external)
    if experiment.engineered:
        ratios = {
            "CREDIT_INCOME_RATIO": ("AMT_CREDIT", "AMT_INCOME_TOTAL"),
            "ANNUITY_INCOME_RATIO": ("AMT_ANNUITY", "AMT_INCOME_TOTAL"),
            "ANNUITY_CREDIT_RATIO": ("AMT_ANNUITY", "AMT_CREDIT"),
            "CREDIT_GOODS_RATIO": ("AMT_CREDIT", "AMT_GOODS_PRICE"),
            "EMPLOYED_LIFE_RATIO": ("DAYS_EMPLOYED", "DAYS_BIRTH"),
        }
        for name, (numerator, denominator) in ratios.items():
            if numerator in result and denominator in result:
                result[name] = result[numerator] / result[denominator].replace(0, np.nan)
        if "DAYS_BIRTH" in result:
            result["AGE_YEARS"] = -result["DAYS_BIRTH"] / 365.25
        external = [c for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"] if c in result]
        if external:
            result["EXT_SOURCE_MEAN"] = result[external].mean(axis=1)
            result["EXT_SOURCE_STD"] = result[external].std(axis=1)
            result["EXT_SOURCE_MISSING_COUNT"] = result[external].isna().sum(axis=1)
    if experiment.history:
        if history is None:
            raise ValueError("History experiment requires bureau.csv aggregates.")
        # Reindex rather than a raw one-to-many join: exactly one output row per applicant.
        aligned = history.reindex(frame[ID].to_numpy()).reset_index(drop=True)
        aligned.index = result.index
        aligned["BUREAU_RECORD_COUNT"] = aligned["BUREAU_RECORD_COUNT"].fillna(0)
        result = pd.concat([result, aligned], axis=1)
    return result.replace([np.inf, -np.inf], np.nan)


def aggregate_bureau(path: Path, applicant_ids: set[int]) -> pd.DataFrame:
    """Use applicant-level past records, without targets or cross-applicant fitting."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; omit --with-bureau or download this file.")
    wanted = {
        ID, "SK_ID_BUREAU", "CREDIT_ACTIVE", "DAYS_CREDIT", "CREDIT_DAY_OVERDUE",
        "AMT_CREDIT_SUM", "AMT_CREDIT_SUM_DEBT", "AMT_CREDIT_SUM_OVERDUE",
    }
    bureau = pd.read_csv(path, usecols=lambda column: column in wanted, low_memory=False)
    if ID not in bureau or bureau[ID].isna().any():
        raise ValueError("bureau.csv must contain non-null SK_ID_CURR.")
    if "SK_ID_BUREAU" in bureau and (
        bureau["SK_ID_BUREAU"].isna().any() or bureau["SK_ID_BUREAU"].duplicated().any()
    ):
        raise ValueError("bureau.csv has missing or duplicated loan IDs; investigate before aggregating.")
    bureau = bureau.loc[bureau[ID].isin(applicant_ids)].copy()
    grouped = bureau.groupby(ID, sort=False)
    result = grouped.size().rename("BUREAU_RECORD_COUNT").to_frame()
    stats = {
        "DAYS_CREDIT": ["mean", "min", "max"],
        "CREDIT_DAY_OVERDUE": ["mean", "max"],
        "AMT_CREDIT_SUM": ["mean", "max"],
        "AMT_CREDIT_SUM_DEBT": ["mean", "max"],
        "AMT_CREDIT_SUM_OVERDUE": ["mean", "max"],
    }
    available = {column: functions for column, functions in stats.items() if column in bureau}
    if available:
        numeric = grouped.agg(available)
        numeric.columns = [f"BUREAU_{column}_{stat.upper()}" for column, stat in numeric.columns]
        result = result.join(numeric, validate="one_to_one")
    if "CREDIT_ACTIVE" in bureau:
        known = bureau["CREDIT_ACTIVE"].notna()
        bureau["IS_ACTIVE"] = bureau["CREDIT_ACTIVE"].eq("Active").astype(float).where(known)
        result = result.join(bureau.groupby(ID)["IS_ACTIVE"].mean().rename("BUREAU_ACTIVE_FRACTION"))
    return result


def build_pipeline(features: pd.DataFrame, model: str, seed: int, threads: int, trees: int) -> Pipeline:
    numeric_columns = features.select_dtypes(include="number").columns.tolist()
    categorical_columns = [column for column in features if column not in numeric_columns]
    numeric_steps = [("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True))]
    if model == "logistic":
        numeric_steps.append(("scaler", StandardScaler()))
    transforms = []
    if numeric_columns:
        transforms.append(("numeric", Pipeline(numeric_steps), numeric_columns))
    if categorical_columns:
        categorical = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__", keep_empty_features=True)),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float32)),
        ])
        transforms.append(("categorical", categorical, categorical_columns))
    preprocessor = ColumnTransformer(transforms, sparse_threshold=1.0)
    if model == "dummy":
        estimator = DummyClassifier(strategy="prior")
    elif model == "logistic":
        estimator = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=seed)
    elif model == "lightgbm":
        estimator = LGBMClassifier(
            n_estimators=trees, learning_rate=0.05, num_leaves=31,
            min_child_samples=40, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=seed, n_jobs=threads, verbosity=-1,
            deterministic=True, force_col_wise=True,
        )
    else:
        raise ValueError(f"Unknown model {model}")
    return Pipeline([("preprocessor", preprocessor), ("model", estimator)])


def metric_values(y: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "average_precision": float(average_precision_score(y, probabilities)),
        "brier_score": float(brier_score_loss(y, probabilities)),
    }


def bootstrap_auc_interval(y: np.ndarray, probabilities: np.ndarray, seed: int, draws: int = 1000) -> list[float]:
    """Stratified bootstrap for fixed predictions; excludes model training uncertainty."""
    rng = np.random.default_rng(seed)
    positive, negative = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    scores = []
    for _ in range(draws):
        indices = np.concatenate([rng.choice(positive, len(positive)), rng.choice(negative, len(negative))])
        scores.append(roc_auc_score(y[indices], probabilities[indices]))
    return np.quantile(scores, [0.025, 0.975]).tolist()


def plot_results(summary: pd.DataFrame, holdout_y: np.ndarray, holdout_p: np.ndarray, out: Path) -> None:
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    ordered = summary.sort_values("cv_auc_mean")
    fig, ax = plt.subplots(figsize=(9, max(4, len(ordered) * 0.55)))
    ax.errorbar(ordered["cv_auc_mean"], np.arange(len(ordered)), xerr=ordered["cv_auc_std"],
                fmt="o", color="#3264A3", capsize=4)
    ax.set_yticks(np.arange(len(ordered)), ordered["experiment"])
    ax.set(xlim=(0.45, 1.0), xlabel="Development CV ROC-AUC (mean ± fold SD)",
           title="Fixed experiments on the same development folds")
    ax.axvline(0.5, color="#555555", linestyle="--", linewidth=1)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "cv_comparison.png", dpi=160)
    plt.close(fig)

    fpr, tpr, _ = roc_curve(holdout_y, holdout_p)
    precision, recall, _ = precision_recall_curve(holdout_y, holdout_p)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(fpr, tpr, color="#3264A3", label=f"AUC = {roc_auc_score(holdout_y, holdout_p):.4f}")
    axes[0].plot([0, 1], [0, 1], "--", color="#555555", label="Random ranking")
    axes[0].set(xlim=(0, 1), ylim=(0, 1), xlabel="False positive rate", ylabel="True positive rate", title="Selected model: holdout ROC")
    axes[1].plot(recall, precision, color="#3264A3", label=f"AP = {average_precision_score(holdout_y, holdout_p):.4f}")
    axes[1].axhline(holdout_y.mean(), linestyle="--", color="#555555", label=f"Prevalence = {holdout_y.mean():.3f}")
    axes[1].set(xlim=(0, 1), ylim=(0, 1), xlabel="Recall", ylabel="Precision", title="Selected model: holdout precision-recall")
    for ax in axes:
        ax.legend(loc="best")
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "holdout_curves.png", dpi=160)
    plt.close(fig)

    observed, predicted = calibration_curve(holdout_y, holdout_p, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(predicted, observed, "o-", color="#3264A3", label="10 quantile bins")
    ax.plot([0, 1], [0, 1], "--", color="#555555", label="Perfect calibration")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability", ylabel="Observed default fraction", title="Selected model: holdout calibration")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "holdout_calibration.png", dpi=160)
    plt.close(fig)


def run_experiments(
    data_dir: Path = Path("data/raw"), output_dir: Path = Path("outputs/run"),
    folds: int = 5, seed: int = 42, max_rows: int | None = None,
    with_bureau: bool = False, make_submission: bool = False,
    threads: int = 4, trees: int = 400,
) -> pd.DataFrame:
    if folds < 2 or trees < 1 or threads < 1 or (max_rows is not None and max_rows < 20):
        raise ValueError("Use folds >= 2, trees/threads >= 1, and max_rows >= 20.")
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    # A new directory avoids mixing scores/predictions from different runs.
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"{output_dir} is not empty. Choose a new --output-dir.")
    train, test = load_applications(data_dir)
    full_rows = len(train)
    if max_rows is not None and max_rows < len(train):
        sampled, _ = train_test_split(np.arange(len(train)), train_size=max_rows, stratify=train[LABEL], random_state=seed)
        train = train.iloc[np.sort(sampled)].reset_index(drop=True)
    else:
        train = train.reset_index(drop=True)
    y = train[LABEL].to_numpy(dtype=int)
    dev_indices, holdout_indices = train_test_split(np.arange(len(train)), test_size=0.2, stratify=y, random_state=seed)
    dev_y = y[dev_indices]
    if np.bincount(dev_y, minlength=2).min() < folds:
        raise ValueError("Not enough examples per class for development folds; use more rows/fewer folds.")
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    splits = list(splitter.split(dev_indices, dev_y))
    output_dir.mkdir(parents=True, exist_ok=True)
    train.drop(columns=[LABEL]).isna().mean().sort_values(ascending=False).rename("missing_fraction").to_csv(output_dir / "missingness.csv")
    history = None
    if with_bureau:
        history = aggregate_bureau(data_dir / "bureau.csv", set(train[ID]).union(test[ID]))
    experiments = [
        Experiment("dummy_prior", "dummy"),
        Experiment("logistic_raw", "logistic"),
        Experiment("logistic_engineered", "logistic", engineered=True),
        Experiment("lightgbm_raw", "lightgbm"),
        Experiment("lightgbm_engineered", "lightgbm", engineered=True),
        Experiment("lightgbm_no_external", "lightgbm", engineered=True, drop_external=True),
    ]
    if with_bureau:
        experiments.append(Experiment("lightgbm_with_bureau", "lightgbm", engineered=True, history=True))
    if not any(column.startswith("EXT_SOURCE_") for column in train):
        experiments = [e for e in experiments if not e.drop_external]

    assignments = np.full(len(train), -1, dtype=int)
    for fold, (_, valid_local) in enumerate(splits, 1):
        assignments[dev_indices[valid_local]] = fold
    pd.DataFrame({ID: train[ID], "split": np.where(assignments == -1, "holdout", "development"),
                  "development_fold": assignments}).to_csv(output_dir / "split_assignments.csv", index=False)
    oof_table = pd.DataFrame({ID: train.iloc[dev_indices][ID].to_numpy(), LABEL: dev_y})
    fold_rows, summary_rows = [], []
    for experiment in experiments:
        features = make_features(train, experiment, history)
        oof = np.full(len(dev_indices), np.nan)
        start = perf_counter()
        current = []
        for fold, (fit_local, valid_local) in enumerate(splits, 1):
            fit_indices, valid_indices = dev_indices[fit_local], dev_indices[valid_local]
            pipeline = build_pipeline(features, experiment.model, seed + fold, threads, trees)
            fold_start = perf_counter()
            with warnings.catch_warnings():
                # Non-converged logistic scores must not silently become reported results.
                warnings.simplefilter("error", ConvergenceWarning)
                pipeline.fit(features.iloc[fit_indices], y[fit_indices])
            p = pipeline.predict_proba(features.iloc[valid_indices])[:, 1]
            oof[valid_local] = p
            estimator = pipeline.named_steps["model"]
            iterations = getattr(estimator, "n_iter_", None)
            row = {
                "experiment": experiment.name, "fold": fold,
                **metric_values(y[valid_indices], p),
                "fit_and_predict_seconds": perf_counter() - fold_start,
                "iterations": int(np.max(iterations)) if iterations is not None else None,
            }
            fold_rows.append(row)
            current.append(row)
            print(f"{experiment.name}: fold {fold}/{folds}, AUC={row['roc_auc']:.5f}", flush=True)
        if not np.isfinite(oof).all():
            raise RuntimeError("Missing or non-finite out-of-fold probabilities.")
        oof_table[experiment.name] = oof
        aucs = [r["roc_auc"] for r in current]
        summary_rows.append({
            "experiment": experiment.name, "n_features": features.shape[1],
            "cv_auc_mean": np.mean(aucs), "cv_auc_std": np.std(aucs, ddof=1),
            "oof_auc": roc_auc_score(dev_y, oof),
            "oof_average_precision": average_precision_score(dev_y, oof),
            "oof_brier_score": brier_score_loss(dev_y, oof),
            "seconds": perf_counter() - start,
        })
        # Preserve progress even if a later model fails.
        pd.DataFrame(fold_rows).to_csv(output_dir / "cv_folds.csv", index=False)
        oof_table.to_csv(output_dir / "development_oof.csv", index=False)
        pd.DataFrame(summary_rows).to_csv(output_dir / "cv_progress.csv", index=False)
    summary = pd.DataFrame(summary_rows).sort_values("cv_auc_mean", ascending=False).reset_index(drop=True)
    summary.to_csv(output_dir / "cv_summary.csv", index=False)
    selected = next(e for e in experiments if e.name == summary.iloc[0]["experiment"])
    features = make_features(train, selected, history)
    model = build_pipeline(features, selected.model, seed, threads, trees)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(features.iloc[dev_indices], dev_y)
    joblib.dump(model, output_dir / "holdout_model.joblib", compress=3)
    holdout_y = y[holdout_indices]
    holdout_p = model.predict_proba(features.iloc[holdout_indices])[:, 1]
    holdout_metrics = {
        "selected_experiment": selected.name, "n": len(holdout_indices),
        "default_fraction": float(holdout_y.mean()), **metric_values(holdout_y, holdout_p),
        "auc_fixed_prediction_bootstrap_95_interval": bootstrap_auc_interval(holdout_y, holdout_p, seed),
        "interval_note": "Stratified application bootstrap, 1000 draws; conditional on this fitted model. Does not include training/model-selection uncertainty.",
    }
    (output_dir / "holdout_metrics.json").write_text(json.dumps(holdout_metrics, indent=2), encoding="utf-8")
    pd.DataFrame({ID: train.iloc[holdout_indices][ID].to_numpy(), LABEL: holdout_y,
                  "probability": holdout_p}).to_csv(output_dir / "holdout_predictions.csv", index=False)
    if selected.model == "lightgbm":
        names = model.named_steps["preprocessor"].get_feature_names_out()
        gains = model.named_steps["model"].booster_.feature_importance(importance_type="gain")
        importance = pd.DataFrame({"encoded_feature": names, "training_gain": gains}).sort_values("training_gain", ascending=False)
        importance.to_csv(output_dir / "training_gain_importance.csv", index=False)
    plot_results(summary, holdout_y, holdout_p, output_dir)

    if make_submission:
        # Fit the selected fixed configuration on all rows used in this run.
        # max_rows runs remain sampled submissions, clearly marked in metadata.
        final_model = build_pipeline(features, selected.model, seed, threads, trees)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            final_model.fit(features, y)
        joblib.dump(final_model, output_dir / "final_model.joblib", compress=3)
        test_features = make_features(test, selected, history)
        probabilities = final_model.predict_proba(test_features)[:, 1]
        if not np.isfinite(probabilities).all() or not ((probabilities >= 0) & (probabilities <= 1)).all():
            raise RuntimeError("Invalid submission probabilities.")
        submission = pd.DataFrame({ID: test[ID].to_numpy(), LABEL: probabilities})
        sample_path = data_dir / "sample_submission.csv"
        if sample_path.is_file():
            sample = pd.read_csv(sample_path)
            if sample[ID].duplicated().any() or set(sample[ID]) != set(test[ID]):
                raise ValueError("sample_submission.csv IDs do not match test IDs.")
            submission = submission.set_index(ID).loc[sample[ID]].reset_index()
        submission.to_csv(output_dir / "submission.csv", index=False)
    files = [data_dir / "application_train.csv", data_dir / "application_test.csv"]
    if with_bureau:
        files.append(data_dir / "bureau.csv")
    sample_path = data_dir / "sample_submission.csv"
    if sample_path.is_file():
        files.append(sample_path)
    metadata = {
        "source": SOURCE_URL, "data_directory": str(data_dir.resolve()),
        "input_sha256": {path.name: file_hash(path) for path in files},
        "full_train_rows": full_rows, "used_train_rows": len(train), "test_rows": len(test),
        "sampled_run": len(train) != full_rows, "seed": seed, "folds": folds,
        "holdout_fraction": 0.2, "threads": threads, "trees": trees,
        "selected_experiment": asdict(selected), "experiments": [asdict(e) for e in experiments],
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scikit-learn", "lightgbm", "matplotlib"]},
        "code_sha256": file_hash(Path(__file__)),
        "model_parameters": model.named_steps["model"].get_params(),
        "selection_rule": "Highest mean development CV ROC-AUC; stable experiment order breaks ties. Holdout checked only after selection.",
        "validation_limit": "Stratified random applicant split; does not establish future-time generalization or lending policy effects.",
    }
    manifest_path = data_dir.parent / "source_manifest.json"
    if manifest_path.is_file():
        metadata["download_provenance"] = json.loads(manifest_path.read_text())
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Selected: {selected.name}; final holdout AUC={holdout_metrics['roc_auc']:.5f}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/run"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--with-bureau", action="store_true")
    parser.add_argument("--make-submission", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--trees", type=int, default=400)
    args = parser.parse_args()
    print(run_experiments(**vars(args)).to_string(index=False))


if __name__ == "__main__":
    main()
