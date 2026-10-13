# Fraud Detection using AI and Machine Learning

Reproducible research pipeline for imbalanced fraud classification on IEEE-CIS and PaySim. This revision prioritises train-fitted preprocessing, temporal evaluation, validation-only threshold selection, leakage-aware feature exclusions, and transparent cost sensitivity.

## Integrity statement
This branch adds reusable code and tests, not new verified model results. No revised scores are claimed. Run the experiments locally with the source datasets to generate metrics; raw datasets are not included here.

## Setup
Python 3.10+ recommended.

    python -m venv .venv
    python -m pip install --upgrade pip
    pip install -r requirements.txt

Datasets:
- IEEE-CIS: https://www.kaggle.com/competitions/ieee-fraud-detection/data
- PaySim: https://www.kaggle.com/datasets/ealaxi/paysim1

Keep data local; do not commit CSVs.

## Run
IEEE-CIS (identity merge optional, validated many-to-one):
    
    python fraud_detection.py --dataset ieee --csv data/train_transaction.csv --identity-csv data/train_identity.csv --output-dir outputs/ieee

PaySim:

    python fraud_detection.py --dataset paysim --csv data/PS_20174392719_1491204439457_log.csv --output-dir outputs/paysim

Optional scenario-cost assumptions:

    python fraud_detection.py --dataset paysim --csv data/paysim.csv --fp-cost 10 --fn-cost 100 --output-dir outputs/paysim

## Methodology
- Sort by TransactionDT (IEEE-CIS) or step (PaySim), then split rows chronologically into 60% train, 20% validation and 20% test. This is a simple temporal holdout, not a guarantee of production generalisation.
- Fit median imputation and one-hot categorical encoding on training data only. Unknown categories are ignored safely; categorical codes are not independently assigned to train and validation.
- Exclude transaction IDs and time-order columns from predictors by default.
- PaySim excludes nameOrig, nameDest, isFlaggedFraud, newbalanceOrig and newbalanceDest by default. The balances are post-transaction fields; isFlaggedFraud is an existing simulator flag. The --include-post-event option is diagnostic only and unsuitable for a pre-authorisation model.
- Fit the model on training data, choose F1 and minimum-precision thresholds on validation predictions, freeze both, and evaluate them on the later test set.
- Report average precision (PR ranking metric), ROC-AUC and threshold-specific precision/recall/F1 plus confusion counts. Threshold metrics in each row are calculated from the same predictions.
- If the precision constraint is not achieved on validation, the output explicitly says so. Validation precision is not a guarantee of test precision.

## Cost sensitivity, not ROI
The calculation is false positives × assumed FP cost + false negatives × assumed FN cost. Costs are user-supplied scenario assumptions, not measured operational cost, profit, savings or ROI. Document the basis and sensitivity range before making business claims.

## Tests
    python -m pytest -q

Tests use synthetic data and do not require the full datasets. Passing unit tests does not establish model performance. The XGBoost baseline is not a completed hyperparameter search. PaySim is synthetic and IEEE-CIS is an anonymised competition dataset; neither establishes generalisation to a real institution or future production period.

## Outputs
The run creates metrics.json, test_predictions.csv and model.joblib in the selected output directory. Do not deploy without independent validation, monitoring, privacy review and operational assessment.
