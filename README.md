# Fraud Detection Using AI and Machine Learning
### A Cross-Domain Study of Class Imbalance Strategies and Operational Feasibility

**MSc Business Analytics Dissertation** · University of Surrey · January 2026  
**Author:** Zeeshan Azeem Khalid (Student ID: 6905195)  
**Supervisor:** Chris Turner, Lecturer in Business Analytics

---

## Key Results

| Dataset | Model | ROC-AUC | PR-AUC | F1 | Precision@P70 | Recall@P70 |
|---------|-------|---------|--------|----|---------------|------------|
| IEEE-CIS | XGBoost | 0.9034 | 0.5513 | 0.5390 | 0.6385 | 0.4663 |
| **PaySim** | **XGBoost** | **0.9998** | **0.9506** | **0.8089** | **0.7000** | **0.9367** |

Bootstrap 95% CI (XGBoost, IEEE-CIS): PR-AUC [0.538, 0.565] · ROC-AUC [0.898, 0.908]

**Core finding:** Cost-sensitive learning (XGBoost `scale_pos_weight`) outperforms SMOTE in operational settings — delivering higher fraud capture at precision-constrained thresholds with better generalisation and more stable score distributions. SMOTE inflates apparent metrics but underperforms once precision constraints and investigation capacity are applied.

---

## Overview

This dissertation investigates AI/ML approaches to transaction-level fraud detection across two benchmark datasets — the high-dimensional anonymised IEEE-CIS dataset and the behaviourally realistic PaySim simulator. The study evaluates how class imbalance handling strategies influence model performance, stability, and operational feasibility under real-world precision constraints.

**Research Questions:**
- **RQ1:** How do SMOTE vs cost-sensitive learning compare under operational (precision-constrained) conditions?
- **RQ2:** Do models trained on IEEE-CIS transfer reliably to PaySim without adaptation?
- **RQ3:** How useful are SHAP explanations on anonymised fraud data for operational decision-making?

---

## Repository Structure

```
.
├── notebooks/
│   └── fraud_detection_pipeline.ipynb   # Full pipeline (IEEE-CIS + PaySim + SHAP)
├── figures/                              # All dissertation figures (PNG, 300 dpi)
│   ├── ieee_xgb_roc.png
│   ├── ieee_xgb_pr.png
│   ├── Figure_6.1_feature_importance_top20_xgb.png
│   ├── Figure_6.2_shap_summary_xgb.png
│   ├── executive_summary_panel.png
│   └── ...  (33 figures total)
├── reports/                              # Metrics CSVs + LaTeX tables
│   ├── ieee_metrics.csv                  # All models × thresholds (IEEE-CIS)
│   ├── paysim_metrics.csv                # Confirmed PaySim XGBoost results
│   ├── cross_dataset_comparison.csv      # Corrected cross-domain comparison
│   ├── ieee_bootstrap_ci.csv             # Bootstrap CI (n=250 resamples)
│   ├── ieee_business_metrics.csv         # Business cost model (TP/FP/FN costs)
│   ├── ieee_paired_bootstrap_diffs.csv   # Paired significance tests
│   ├── reproducibility_results_3runs.csv # 3-seed stability check
│   └── ...  (26 report files total)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Methodology

| Component | Detail |
|-----------|--------|
| Framework | CRISP-DM |
| IEEE-CIS sample | Stratified 15% (memory-constrained; fraud rate preserved) |
| Imputation | Sentinel value −999 (preserves informative missingness) |
| Encoding | Frequency encoding for high-cardinality categoricals |
| Time features | Cyclic sin/cos encoding of hour-of-day and day-of-week |
| Models | Logistic Regression · Random Forest · XGBoost |
| Imbalance strategies | Baseline · SMOTE · Cost-sensitive (scale_pos_weight / class_weight) |
| Primary metric | PR-AUC (extreme class imbalance) |
| Operational threshold | P ≥ 70% precision-constrained |
| Statistical validation | Bootstrap CI (n=250) · Paired tests · 3-seed reproducibility |

---

## Datasets

**Not included** — download from Kaggle and place in `data/` (or update `DATA_DIR` in Cell 1).

| Dataset | Source | Rows | Fraud Rate |
|---------|--------|------|------------|
| IEEE-CIS Fraud Detection | [Kaggle](https://www.kaggle.com/c/ieee-fraud-detection) | ~590k | 3.5% |
| PaySim | [Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1) | 6.36M | 0.13% |
| Google Drive (dissertation copies) | [Drive folder](https://drive.google.com/drive/folders/1YQTSBRXcNG28IMenowAqXdZqHSNqiAlR?usp=drive_link) | — | — |

---

## Quick Start (Google Colab)

```python
# 1. Open notebooks/fraud_detection_pipeline.ipynb in Colab
# 2. Mount Drive and set DATA_DIR to your dataset folder (Cell 1)
# 3. Run all cells — outputs write to figures/ and reports/
```

All randomness uses `RANDOM_STATE = 42`. Reproducibility across seeds 42–44 confirmed (F1 variance < 0.01).

---

## Post-Submission Bug Fix

The original **Cell 28** (cross-dataset comparison table) showed **NaN** for PaySim's F1 / Precision / Recall columns. Root cause: `paysim_metrics.csv` stores columns as `f1@f1thr` / `precision@p70thr` / `recall@p70thr`, but the join expected `f1` / `precision` / `recall`. The mismatch produced silent NaN values.

**The dissertation narrative was unaffected** — PaySim metrics were cited correctly throughout the text and figures. Cell 28 has been corrected in this repository, and a new **"Confirmed Results"** cell documents the verified numbers.

---

## Citation

```bibtex
@mastersthesis{khalid2026fraud,
  author  = {Khalid, Zeeshan Azeem},
  title   = {Fraud Detection Using AI and Machine Learning: A Cross-Domain Study
             of Class Imbalance Strategies and Operational Feasibility},
  school  = {University of Surrey},
  year    = {2026},
  month   = {January},
  type    = {MSc Dissertation},
}
```
