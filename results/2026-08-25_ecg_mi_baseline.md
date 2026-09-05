# Experiment record — ECG baseline (MI vs non-MI)

- **Date:** 2026-08-25
- **Notebook:** [`notebooks/01_first_ecg_model.ipynb`](../notebooks/01_first_ecg_model.ipynb)
- **Dataset:** PTB-XL v1.0.3 (PhysioNet), official patient-wise folds — train 1–8, validation 9, test 10
- **Target:** binary, presence vs absence of an MI diagnostic label (non-MI includes all other diagnoses, not a healthy-control group)
- **Model:** 1D CNN (3 conv blocks: 32→64→128 channels, BatchNorm + ReLU + MaxPool, global average pool, linear head)
- **Training:** 10 epochs, batch size 64, Adam, lr=1e-3, `BCEWithLogitsLoss` with positive-class weighting, seed=42
- **Compute:** trained on CPU (Google Colab; GPU was unavailable for this run)

## Validation AUROC by epoch

| Epoch | val_AUROC |
| --- | --- |
| 1 | 0.8884 |
| 2 | 0.8879 |
| 3 | 0.9067 |
| 4 | 0.9136 |
| 5 | 0.9071 |
| 6 | 0.9143 |
| 7 | 0.9147 |
| 8 | 0.9170 |
| 9 | (not recorded) |
| 10 | (not recorded) |

## Held-out test set results

| Metric | Value |
| --- | --- |
| Test loss | 0.5369 |
| AUROC | **0.9207** |
| Average precision | 0.8138 |
| F1 | 0.7218 |
| Sensitivity (recall, MI) | 0.8564 |
| Specificity | 0.8277 |

Confusion matrix (rows = actual, columns = predicted; 0 = non-MI, 1 = MI):

| | Predicted non-MI | Predicted MI |
| --- | --- | --- |
| **Actual non-MI** | 1364 | 284 |
| **Actual MI** | 79 | 471 |

Classification report:

```
              precision    recall  f1-score   support

      non-MI       0.95      0.83      0.88      1648
          MI       0.62      0.86      0.72       550

    accuracy                           0.83      2198
   macro avg       0.78      0.84      0.80      2198
weighted avg       0.86      0.83      0.84      2198
```

## Artifacts

- Trained checkpoint: [`results/checkpoints/cardio_nexus_ecg_mi_baseline.pt`](checkpoints/cardio_nexus_ecg_mi_baseline.pt) (245 KB)
- Attribution figure: [`results/explainability/`](explainability/)

## Interpretation and limitations

This file is a metric record only. The written analysis of this run — what it means, how it compares to the 8 August baseline, and where it falls short — is maintained in the research documentation repository, which owns interpretation:

- [Experiment 001 — ECG MI versus non-MI baseline](https://github.com/Cardiac-Nexus-Lab/nexus-research-docs/blob/main/experiments/001_ecg_mi_baseline.md)
- [Experiment 002 — Integrated Gradients explainability](https://github.com/Cardiac-Nexus-Lab/nexus-research-docs/blob/main/experiments/002_ecg_explainability_integrated_gradients.md)

These metrics describe the held-out PTB-XL fold 10 only. They do not establish clinical safety, generalization to other populations, or readiness for diagnostic use.
