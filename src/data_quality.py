"""Audit full downloaded Home Credit inputs before model fitting.

Run: python -m src.data_quality
The second-copy check reads the independently downloaded files under /tmp.
No raw application records are printed.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .train import ID, LABEL, Experiment, aggregate_bureau, clean_application, file_hash, load_applications, make_features


def uncompressed_hash(path: Path, expected_name: str) -> str:
    digest = hashlib.sha256()
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            entries = [entry for entry in archive.infolist() if not entry.is_dir()]
            if len(entries) != 1 or Path(entries[0].filename).name != expected_name:
                raise ValueError(f"Unexpected independent-copy archive: {path}")
            with archive.open(entries[0]) as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
    else:
        return file_hash(path)
    return digest.hexdigest()


def audit(root: Path) -> dict:
    raw = root / "data/raw"
    output = root / "outputs/data_quality"
    output.mkdir(parents=True, exist_ok=True)
    train, test = load_applications(raw)
    manifest = json.loads((root / "data/source_manifest.json").read_text())
    for name, details in manifest["files"].items():
        assert (raw/name).stat().st_size == details["bytes"], f"Size mismatch: {name}"
        assert file_hash(raw/name) == details["sha256"], f"Hash mismatch: {name}"
    crosscheck = {}
    for name, second_path, second_name in [
        ("application_train.csv", Path('/tmp/home-credit-independent-application_train.csv'), "HC_application_train.csv"),
        ("bureau.csv", Path('/tmp/home-credit-independent-bureau.csv'), "HC_bureau.csv"),
    ]:
        if not second_path.is_file():
            raise FileNotFoundError(f"Independent copy missing: {second_path}")
        digest = uncompressed_hash(second_path, second_name)
        expected = manifest["files"][name]["sha256"]
        assert digest == expected, f"Independent public copies disagree: {name}"
        crosscheck[name] = {
            "sha256": digest, "exact_byte_match": True,
            "independent_source": "https://www.kaggle.com/datasets/anggundwilestari/home-credit",
            "independent_dataset_version": 4,
        }
    dictionary = pd.read_csv(raw/"HomeCredit_columns_description.csv", encoding="latin1")
    definitions = dictionary.loc[dictionary['Table'].str.contains('application_', na=False)]
    documented = set(definitions['Row'])
    undocumented = sorted(set(train.columns) - documented)
    assert not undocumented, f"Application fields absent from dictionary: {undocumented}"
    meanings = {
        field: str(definitions.loc[definitions['Row'].eq(field), 'Description'].iloc[0])
        for field in [ID, LABEL, "DAYS_EMPLOYED", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    }
    bureau = pd.read_csv(raw/'bureau.csv', low_memory=False)
    assert bureau['SK_ID_BUREAU'].notna().all() and bureau['SK_ID_BUREAU'].is_unique
    assert bureau[ID].notna().all()
    applications = set(train[ID]).union(test[ID])
    orphan_count = int((~bureau[ID].isin(applications)).sum())
    assert orphan_count == 0, 'Bureau contains IDs outside applications'
    future_credit = int(bureau['DAYS_CREDIT'].gt(0).sum())
    assert future_credit == 0, 'Bureau loan-start dates after the current application require investigation'
    sample = pd.read_csv(raw/'sample_submission.csv')
    assert sample[ID].is_unique and set(sample[ID]) == set(test[ID])
    numeric = train.select_dtypes('number')
    infinity_count = int(np.isinf(numeric.to_numpy()).sum())
    assert infinity_count == 0
    invalid_scores = {c: int((train[c].notna() & ~train[c].between(0, 1)).sum())
                      for c in ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']}
    assert not any(invalid_scores.values())
    monetary = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE']
    negative_amounts = {c: int(train[c].lt(0).sum()) for c in monetary}
    assert not any(negative_amounts.values())
    unknown = {}
    for column in train.select_dtypes(include=['object', 'string']):
        unseen = test[column].notna() & ~test[column].isin(train[column].dropna().unique())
        unknown[column] = int(unseen.sum())
    cleaned = clean_application(train)
    assert ID not in cleaned and LABEL not in cleaned and len(cleaned) == len(train)
    engineered = make_features(train, Experiment('audit', 'lightgbm', engineered=True))
    assert not np.isinf(engineered.select_dtypes('number').to_numpy()).any()
    no_external = make_features(train, Experiment('audit', 'lightgbm', engineered=True, drop_external=True))
    assert not any(column.startswith('EXT_SOURCE_') for column in no_external)
    history = aggregate_bureau(raw/'bureau.csv', applications)
    joined = make_features(train, Experiment('audit', 'lightgbm', engineered=True, history=True), history)
    assert len(joined) == len(train) and joined.index.equals(train.index)
    assert history.index.is_unique
    missingness = pd.DataFrame({
        'train_missing_fraction': train.drop(columns=[LABEL]).isna().mean(),
        'test_missing_fraction': test.isna().mean(),
    }).sort_values('train_missing_fraction', ascending=False)
    missingness.to_csv(output/'missingness_train_test.csv')
    label_counts = train[LABEL].value_counts().sort_index()
    report = {
        'status': 'passed_with_documented_limitations',
        'unit': 'loan application; SK_ID_CURR is the loan/application ID, not a verified unique natural-person ID',
        'source_manifest_sha256': file_hash(root/'data/source_manifest.json'),
        'source_identity': crosscheck,
        'source_limit': 'Exact identity established between two public Kaggle copies. An authenticated competition hash was unavailable; this is not a claim of direct official download.',
        'shapes': {'train': list(train.shape), 'test': list(test.shape), 'bureau': list(bureau.shape)},
        'label_counts': {str(k): int(v) for k, v in label_counts.items()},
        'payment_difficulty_fraction': float(train[LABEL].mean()),
        'dictionary_definitions': meanings,
        'duplicate_application_ids': {'train': int(train[ID].duplicated().sum()), 'test': int(test[ID].duplicated().sum())},
        'exact_duplicate_train_rows': int(train.duplicated().sum()),
        'train_test_id_overlap': len(set(train[ID]).intersection(test[ID])),
        'bureau_orphan_records': orphan_count,
        'bureau_future_loan_start_records': future_credit,
        'bureau_loan_id_duplicates': int(bureau['SK_ID_BUREAU'].duplicated().sum()),
        'train_applications_with_bureau': int(train[ID].isin(history.index).sum()),
        'test_applications_with_bureau': int(test[ID].isin(history.index).sum()),
        'employment_sentinel_count': int(train['DAYS_EMPLOYED'].eq(365243).sum()),
        'gender_XNA_count': int(train['CODE_GENDER'].eq('XNA').sum()),
        'numeric_infinite_values': infinity_count,
        'negative_amount_counts': negative_amounts,
        'zero_amount_counts': {c: int(train[c].eq(0).sum()) for c in monetary},
        'invalid_external_score_counts': invalid_scores,
        'unseen_test_category_row_counts': unknown,
        'all_missing_train_columns': train.columns[train.isna().all()].tolist(),
        'top_missing_train_columns': missingness.head(12)['train_missing_fraction'].to_dict(),
        'train_memory_bytes': int(train.memory_usage(deep=True).sum()),
        'required_label_key_schema_checks': 'passed',
        'fold_preprocessing_design': 'Median imputation, scaling and one-hot fitting happen only inside each development training fold.',
        'limitations': [
            'Historical competition snapshot, not current lending population.',
            'Actual application calendar dates and unique natural-person identifiers are unavailable; random application splits do not establish future-time performance.',
            'Missingness and special values are retained/encoded, not silently removed.',
            'TARGET denotes payment difficulty per hidden X/Y criteria; it is not a separately validated bankruptcy or fixed-horizon default event.',
        ],
    }
    (output/'data_quality.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({key: report[key] for key in ['status', 'shapes', 'label_counts', 'payment_difficulty_fraction', 'employment_sentinel_count', 'gender_XNA_count', 'train_applications_with_bureau']}, indent=2))
    return report


if __name__ == '__main__':
    audit(Path(__file__).resolve().parents[1])
