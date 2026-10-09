# Home Credit Project 1

## Run locally
Use Python 3.11 or newer. From this project folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --user --name home-credit --display-name "Home Credit"
```

On macOS, LightGBM also needs OpenMP. Install it with `brew install libomp` if an import reports that libomp.dylib is missing. Restart the notebook kernel after installation.

Open the notebooks in VS Code with the Jupyter extension and select the Home Credit kernel. Run cells from top to bottom. The working directory should be the project folder or its notebooks folder. For the JupyterLab interface, install jupyterlab and run `jupyter lab` from this folder.

## Files
- `notebooks/home_credit_results.ipynb`: reads the saved full-data results and redraws figures. No training is needed. It uses `outputs/real_20260930` and `outputs/data_quality`.
- `notebooks/home_credit.ipynb`: runs new experiments using `src/train.py`. Its default is a quick check with 30,000 labelled rows, 3 folds and 150 trees, without bureau features. These scores are not the report scores.
- `src/`: training, data download, data checks and result checks.
- `data/raw/`: application_train.csv, application_test.csv, bureau.csv and sample_submission.csv. The input data is already included in this copy.
- `outputs/`: saved predictions, metrics and figures. Keep the saved full-data results when sharing this project.

## Full training
In the training notebook, set MAX_ROWS=None, FOLDS=5, TREES=400, WITH_BUREAU=True and MAKE_SUBMISSION=True. Each run uses a new output folder to avoid replacing previous results. A new full run can take longer and its results can differ with package versions. Update report tables together if using a new run.

The original report uses the saved full-data run, with holdout AUC 0.772647. Reported Kaggle scores are Public 0.77088 and Private 0.76731 from late submissions; they are separate from local validation scores.

## Local check on 8 October 2026
Both notebooks were executed from top to bottom with Python 3.14.7. The result notebook completed all 10 code cells and saved 7 inline figures. The quick experiment completed all 9 code cells and saved 5 inline figures. Its holdout AUC was 0.720803 on 6,000 held-out rows. The original full-data metrics were not replaced. Exact tested package versions are in requirements-tested.txt. Kernel shutdown printed a sandbox process-inspection permission warning; no notebook cell failed.
