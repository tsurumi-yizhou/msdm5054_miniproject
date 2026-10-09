"""Independently verify saved predictions, metrics, splits and fitted preprocessing.

Run after training: python -m src.check_results
ROC-AUC uses a rank statistic; AP uses threshold-group cumulative counts.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from .train import ID, LABEL, Experiment, aggregate_bureau, file_hash, load_applications, make_features


def independent_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    positives = y == 1
    npos, nneg = int(positives.sum()), int((~positives).sum())
    ranks = pd.Series(p).rank(method='average').to_numpy()
    auc = (ranks[positives].sum() - npos * (npos + 1) / 2) / (npos * nneg)
    order = np.argsort(-p, kind='mergesort')
    sorted_p, sorted_y = p[order], y[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_p)), len(p) - 1]
    cumulative_positive = np.cumsum(sorted_y)[ends]
    precision = cumulative_positive / (ends + 1)
    positive_increment = np.diff(np.r_[0, cumulative_positive])
    ap = np.sum(positive_increment / npos * precision)
    return {'roc_auc': float(auc), 'average_precision': float(ap),
            'brier_score': float(np.mean((p-y) ** 2))}


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, float_precision='round_trip')


def check(root: Path, run: Path) -> dict:
    meta = json.loads((run/'run_metadata.json').read_text())
    plan = json.loads((root/'outputs/experiment_plan.json').read_text())
    assert not meta['sampled_run'] and meta['full_train_rows'] == meta['used_train_rows']
    assert meta['code_sha256'] == plan['training_code_sha256'] == file_hash(root/'src/train.py')
    assert plan['data_source_manifest_sha256'] == file_hash(root/'data/source_manifest.json')
    for name, expected in meta['input_sha256'].items():
        assert file_hash(root/'data/raw'/name) == expected
    train, test = load_applications(root/'data/raw')
    y = train[LABEL].to_numpy()
    indices = np.arange(len(train))
    dev, hold = train_test_split(indices, test_size=0.2, stratify=y, random_state=meta['seed'])
    assignments = read_csv(run/'split_assignments.csv').set_index(ID)
    assert len(assignments) == len(train) and assignments.index.is_unique
    assert set(assignments.index) == set(train[ID])
    assert set(assignments.index[assignments['split'].eq('holdout')]) == set(train.iloc[hold][ID])
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for number, (_, valid) in enumerate(folds.split(dev, y[dev]), 1):
        assert (assignments.loc[train.iloc[dev[valid]][ID], 'development_fold'] == number).all()
    oof = read_csv(run/'development_oof.csv')
    assert oof[ID].is_unique and set(oof[ID]) == set(train.iloc[dev][ID])
    assert set(oof[ID]).isdisjoint(set(train.iloc[hold][ID]))
    assert np.array_equal(oof[LABEL], train.set_index(ID).loc[oof[ID], LABEL])
    summary = read_csv(run/'cv_summary.csv')
    per_fold = read_csv(run/'cv_folds.csv')
    fold_id = assignments.loc[oof[ID], 'development_fold'].to_numpy()
    checks = []
    for row in summary.itertuples():
        probability = oof[row.experiment].to_numpy()
        assert np.isfinite(probability).all() and ((probability >= 0) & (probability <= 1)).all()
        actual = independent_metrics(oof[LABEL].to_numpy(), probability)
        assert abs(actual['roc_auc'] - row.oof_auc) < 1e-12
        assert abs(actual['average_precision'] - row.oof_average_precision) < 1e-12
        assert abs(actual['brier_score'] - row.oof_brier_score) < 1e-12
        scores = []
        for number in range(1, 6):
            selection = fold_id == number
            values = independent_metrics(oof[LABEL].to_numpy()[selection], probability[selection])
            saved = per_fold.loc[per_fold.experiment.eq(row.experiment) & per_fold.fold.eq(number)].iloc[0]
            for metric in values:
                assert abs(values[metric] - saved[metric]) < 1e-12
            scores.append(values['roc_auc'])
        assert abs(np.mean(scores) - row.cv_auc_mean) < 1e-12
        assert abs(np.std(scores, ddof=1) - row.cv_auc_std) < 1e-12
        checks.append({'experiment': row.experiment, 'independent_oof_metrics': actual, 'fold_metrics_match': True})
    assert summary.iloc[0].experiment == meta['selected_experiment']['name']
    assert per_fold.loc[per_fold.experiment.str.startswith('logistic'), 'iterations'].lt(2000).all()
    holdout = read_csv(run/'holdout_predictions.csv')
    assert holdout[ID].is_unique and set(holdout[ID]) == set(train.iloc[hold][ID])
    assert np.array_equal(holdout[LABEL], train.set_index(ID).loc[holdout[ID], LABEL])
    metrics = json.loads((run/'holdout_metrics.json').read_text())
    recomputed = independent_metrics(holdout[LABEL].to_numpy(), holdout.probability.to_numpy())
    for metric, value in recomputed.items():
        assert abs(value - metrics[metric]) < 1e-12
    selected = Experiment(**meta['selected_experiment'])
    history = aggregate_bureau(root/'data/raw/bureau.csv', set(train[ID]).union(test[ID])) if selected.history else None
    features = make_features(train, selected, history)
    model = joblib.load(run/'holdout_model.joblib')
    reloaded = model.predict_proba(features.iloc[hold])[:, 1]
    assert np.allclose(reloaded, holdout.probability, rtol=0, atol=1e-12)
    numeric_columns = features.select_dtypes('number').columns
    expected_medians = features.iloc[dev][numeric_columns].median().fillna(0).to_numpy()
    imputer = model.named_steps['preprocessor'].named_transformers_['numeric'].named_steps['imputer']
    assert np.allclose(imputer.statistics_, expected_medians, rtol=0, atol=1e-12)
    submission = read_csv(run/'submission.csv')
    sample = read_csv(root/'data/raw/sample_submission.csv')
    assert submission.columns.tolist() == [ID, LABEL]
    assert submission[ID].tolist() == sample[ID].tolist()
    assert submission[ID].is_unique and len(submission) == len(test)
    assert submission[LABEL].notna().all() and submission[LABEL].between(0, 1).all()
    final_model = joblib.load(run/'final_model.joblib')
    final_p = final_model.predict_proba(make_features(test, selected, history))[:, 1]
    reordered = pd.Series(final_p, index=test[ID]).loc[sample[ID]].to_numpy()
    assert np.allclose(reordered, submission[LABEL], rtol=0, atol=1e-12)
    report = {
        'status': 'passed', 'independent_metric_method': 'Mann-Whitney rank ROC-AUC, threshold-group AP, mean squared probability error',
        'experiments_checked': checks, 'folds_checked': len(per_fold),
        'train_rows': len(train), 'development_rows': len(dev), 'holdout_rows': len(hold),
        'holdout_independent_metrics': recomputed,
        'fixed_seed_splits_verified': True, 'oof_labels_and_fold_membership_verified': True,
        'model_reload_predictions_verified': True, 'development_only_imputation_verified': True,
        'submission_ids_probabilities_and_model_verified': True,
        'frozen_plan_input_and_code_hashes_verified': True,
        'logistic_convergence_verified': True,
        'holdout_model_sha256': file_hash(run/'holdout_model.joblib'),
        'final_model_sha256': file_hash(run/'final_model.joblib'),
        'scope_limit': 'Local random application validation; no Kaggle hidden-label score and no claim of prospective or causal validation.',
    }
    (run/'validation_checks.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['status', 'folds_checked', 'development_rows', 'holdout_rows', 'holdout_independent_metrics']}, indent=2))
    return report


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    check(root, root/'outputs/real_20260930')
