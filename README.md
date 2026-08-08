# Cardiac Nexus AI Engine

Research and engineering workspace for ECG signal processing, model development, and evaluation within the Cardiac Nexus project.

## Overview

The repository contains the computational components used to study cardiovascular patterns in 12-lead electrocardiogram (ECG) recordings. Development begins with a reproducible ECG baseline and will progress through signal preprocessing, model comparison, interpretability, and inference services.

The current experiment uses the public PTB-XL dataset to classify recordings according to the presence or absence of a myocardial infarction (MI) diagnostic label.

## Current experiment

| Component | Description |
| --- | --- |
| Dataset | PTB-XL, version 1.0.3 |
| Input | 10-second, 12-lead ECG recordings at 100 Hz |
| Target | MI versus non-MI |
| Baseline | One-dimensional convolutional neural network |
| Training split | PTB-XL folds 1–8 |
| Validation split | PTB-XL fold 9 |
| Test split | PTB-XL fold 10 |
| Metrics | AUROC, average precision, F1-score, sensitivity, specificity, and confusion matrix |

The non-MI category includes recordings without an MI diagnostic label and should not be interpreted as a healthy-control category.

## Repository structure

```text
.
├── notebooks/
│   └── 01_first_ecg_model.ipynb
└── README.md
```

The notebook contains the initial end-to-end experiment, including dataset preparation, label construction, signal normalization, model training, and evaluation.

## Running the experiment

The recommended environment for the first experiment is Google Colab with a GPU runtime.

1. Open the [first ECG experiment in Google Colab](https://colab.research.google.com/github/Cardiac-Nexus-Lab/nexus-ai-engine/blob/main/notebooks/01_first_ecg_model.ipynb).
2. Connect a runtime. A GPU is recommended for training.
3. Run the notebook from top to bottom.
4. Review the validation and test metrics and retain the outputs with the experiment record.

The notebook downloads the PTB-XL archive into the temporary Colab workspace. The dataset is not stored in this repository.

## Data and privacy

Patient-identifiable information, hospital records, credentials, and private datasets must not be committed to this repository. Public datasets should be downloaded using the documented source and handled according to their applicable terms.

PTB-XL source: [PhysioNet PTB-XL v1.0.3](https://physionet.org/content/ptb-xl/1.0.3/)

## Project status

The repository is currently at the baseline experimentation stage. Results from the first run will be reviewed before additional architectures, modalities, or deployment components are introduced.

This work is intended for research evaluation. Model outputs must not be used as a standalone basis for medical decisions.

## Planned development

- Establish a reproducible ECG baseline.
- Compare alternative signal-processing and model configurations.
- Add signal-level interpretation methods.
- Document experiment results and limitations.
- Define an inference interface for later integration with the Web Portal.
