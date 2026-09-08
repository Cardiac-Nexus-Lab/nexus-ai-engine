# Cardiac Nexus AI Engine

Research and engineering workspace for ECG signal processing, model development, and evaluation within the Cardiac Nexus project.

This repository holds code and artifacts: training and evaluation code, model checkpoints, raw metric records, and generated figures. The written interpretation of each experiment lives in [nexus-research-docs](https://github.com/Cardiac-Nexus-Lab/nexus-research-docs), which owns that record.

## What is built

Everything here processes 12-lead ECG. MRI, tabular EHR, and multimodal fusion are planned but not started.

| Component | State |
| --- | --- |
| Multi-label classifier over five diagnostic superclasses | Test macro AUROC 0.911 |
| Probability calibration | Mean expected calibration error 0.091 to 0.015 |
| Attribution with sanity checking | Passes model-randomization test |
| ECG printout rendering and photographic distortion | Working |
| Trace digitization from a photographed printout | In training |

Models are split into an encoder returning a 128-dimensional embedding and a task head, so the trained ECG encoder can later become one branch of a multimodal model without being rewritten.

## Results

Best configuration: `xresnet1d18` with augmentation, weight decay, and a one-cycle learning rate schedule.

| Class | AUROC (95% CI) | Average precision (95% CI) | Test positives |
| --- | --- | --- | ---: |
| NORM | 0.941 (0.932–0.950) | 0.912 (0.894–0.930) | 963 |
| MI | 0.921 (0.907–0.933) | 0.822 (0.791–0.850) | 550 |
| STTC | 0.930 (0.918–0.942) | 0.814 (0.778–0.848) | 521 |
| CD | 0.924 (0.909–0.938) | 0.843 (0.815–0.872) | 496 |
| HYP | 0.837 (0.813–0.860) | 0.474 (0.415–0.539) | 262 |

Published reference for this task is 0.928 macro AUROC (Strodthoff et al., IEEE JBHI 2021, using the much larger xresnet1d101).

HYP is weak where it matters: an AUROC of 0.837 alongside an average precision of 0.474 means the model ranks reasonably but finds a minority of true cases at the default threshold.

## Repository structure

```text
.
├── src/cardiac_nexus/
│   ├── data.py            PTB-XL download, decoding, and caching
│   ├── models.py          ECG encoder and classifier head
│   ├── xresnet1d.py       1D XResNet architectures
│   ├── augment.py         training-time ECG augmentation
│   ├── training.py        training loop, metrics, epoch selection
│   ├── evaluate.py        bootstrap intervals and calibration
│   ├── explain.py         Integrated Gradients and sanity checks
│   └── ecg_image/         printout rendering, distortion, digitization
├── scripts/               command-line entry points
├── notebooks/             the original Colab experiments
└── results/               checkpoints, metrics, figures
```

## Running it

Requires Python 3.11 or later. Training uses a GPU when one is available, including Apple silicon via the MPS backend, which measured roughly twelve times faster than the CPU path on an M1.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/fetch_data.py        # download and cache PTB-XL (~0.5 GB retained)
python scripts/train_ecg.py --arch xresnet1d18 --epochs 25 --augment \
    --weight-decay 1e-4 --one-cycle --learning-rate 3e-3
python scripts/evaluate_ecg.py --checkpoint results/local/ecg_multilabel.pt
python scripts/explain_ecg.py --checkpoint results/local/ecg_multilabel.pt
```

`fetch_data.py` resumes if interrupted. It downloads the published archive once and extracts only the 100 Hz records, so 1.71 GB downloaded becomes about 0.5 GB on disk. The dataset is not committed.

The original Colab notebooks in `notebooks/` still run and remain the record of how the first experiments were performed, but local training is now the primary path.

## Local ECG review app

The repository also includes a polished local Streamlit interface for reviewing a
de-identified, 12-lead ECG CSV with the included research checkpoint and an
Integrated Gradients attribution overlay.

```bash
python3 -m pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit (normally `http://localhost:8501`). The
CSV needs at least 12 numeric ECG lead columns; it is resampled to the model's
1000-sample input. The app is a research interface, not a diagnostic device.

## Data and privacy

Patient-identifiable information, hospital records, credentials, and private datasets must not be committed to this repository. Public datasets should be downloaded using the documented source and handled according to their applicable terms.

PTB-XL source: [PhysioNet PTB-XL v1.0.3](https://physionet.org/content/ptb-xl/1.0.3/)

## Limitations

- Evaluated on PTB-XL fold 10 only. No external dataset has been tested, so generalisation beyond this cohort, its equipment, and its labelling conventions is unestablished.
- Results come from a single random seed; seed-to-seed variation is unquantified.
- Attribution maps describe what this model responded to. Published work finds such methods disagree with one another and can survive weight randomization, so they are reported as exploratory and accompanied by the sanity check rather than presented as evidence.
- Digitization is trained on rendered printouts, not photographs of real ones. Performance on genuine clinical paper is untested.
- This work is intended for research evaluation. Model outputs must not be used as a standalone basis for medical decisions.

## Planned development

- External validation on an independent public dataset.
- Deliberate per-class operating thresholds rather than a default of 0.5.
- Tabular EHR as a second modality, then multimodal fusion. Fusion requires paired records, meaning the same patient across modalities, which no public dataset currently provides.
- An inference interface for the [Web Portal](https://github.com/Cardiac-Nexus-Lab/nexus-web-portal).
