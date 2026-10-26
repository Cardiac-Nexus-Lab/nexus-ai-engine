# Cardiac Nexus AI Engine

Research and engineering workspace for ECG signal processing, model development, and evaluation within the Cardiac Nexus project.

This repository holds code and artifacts: training and evaluation code, model checkpoints, raw metric records, and generated figures. The written interpretation of each experiment lives in [nexus-research-docs](https://github.com/Cardiac-Nexus-Lab/nexus-research-docs), which owns that record.

## What is built

The ECG, cardiac MRI and health-record models are each trained and tested on their own data, and combined into one probability by Bayes' rule. No public dataset has all three for the same patients, so the combination itself cannot be trained or validated.

| Component | State |
| --- | --- |
| Multi-label classifier over five diagnostic superclasses | Test macro AUROC 0.911 |
| Probability calibration | Mean expected calibration error 0.091 to 0.015 |
| Attribution with sanity checking | Passes model-randomization test on the best model: similarity to randomized weights +0.004 over 10 examples |
| ECG printout rendering and photographic distortion | Working |
| Trace digitization from a printout image | 0.940 waveform correlation on flat pages (ceiling 0.964); angled photographs unresolved |
| Cardiac MRI segmentation (ACDC) | Test LV Dice 0.956 (ED); RV 0.933; myocardium 0.870 |
| Pooled MRI segmentation (ACDC, M&Ms, M&Ms-2) | M&Ms-2 test RV Dice 0.433 → 0.908 (ED); Kaggle external LV EF MAE 5.65 points |
| Ejection fraction from the predicted masks | LV r 0.991 against expert masks, limits of agreement -6.6 to +5.8 points |
| MRI diagnosis from measurements | 45/50 test patients from the ACDC-only model's masks; 43/50 from the pooled model's (served) |
| Health-record model, coronary artery disease (UCI, 4 centres) | Internal test AUROC 0.85 (184 patients); 0.73–0.87 at a centre left out of training |
| Three-way combination (health record, ECG, MRI) | Bayes' rule over the three validated models; the combination is not validated |
| MRI explainability with checks | Uncertainty detects errors at AUROC 0.861; Grad-CAM depends on learned weights on average |
| Web application serving these models | Live in [nexus-web-portal](https://github.com/Cardiac-Nexus-Lab/nexus-web-portal) |

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

## MRI results

ACDC, 50 held-out test patients (10 per diagnosis), 2.5D U-Net with 2.0M parameters. 95% bootstrap intervals over patients.

| Structure | Dice end-diastole | Dice end-systole |
| --- | --- | --- |
| LV | 0.956 (0.950–0.962) | 0.914 (0.898–0.929) |
| RV | 0.933 (0.921–0.944) | 0.880 (0.860–0.899) |
| MYO | 0.870 (0.853–0.885) | 0.896 (0.884–0.905) |

Diagnosis uses ten measurements from the segmentation rather than raw pixels, building on an earlier end-to-end experiment by Sahana NS: 45/50 from the model's masks (95% Wilson interval 79%–96%), against 46/50 from expert masks. Four of five errors confuse infarction with dilated cardiomyopathy, which cine MRI alone cannot reliably separate.

Full record: [Experiment 005](https://github.com/Cardiac-Nexus-Lab/nexus-research-docs/blob/main/experiments/005_cardiac_mri_segmentation_and_diagnosis.md).

```bash
python scripts/train_mri.py --epochs 120 --base-width 16
python scripts/evaluate_mri.py
python scripts/diagnose_mri.py --evaluate-test
python scripts/explain_mri.py --split test
```

### Pooled model: ACDC, M&Ms and M&Ms-2

The ACDC model was fine-tuned for 12 epochs on 390 training patients from three cohorts (80 ACDC, 150 M&Ms, 160 M&Ms-2), with 94 validation patients and 346 held-out test patients. Dice at end-diastole, ACDC-only model → pooled model:

| Test cohort | LV | MYO | RV | RV HD95 (mm) | LV EF MAE (points) |
| --- | --- | --- | --- | --- | --- |
| ACDC (50) | 0.956 → 0.954 | 0.870 → 0.861 | 0.933 → 0.928 | 3.9 → 4.2 | 2.4 → 3.0 |
| M&Ms (136) | 0.928 → 0.937 | 0.810 → 0.828 | 0.861 → 0.892 | 9.9 → 6.4 | 4.6 → 4.1 |
| M&Ms-2 (160) | 0.828 → 0.944 | 0.700 → 0.833 | 0.433 → 0.908 | 42.2 → 8.3 | 9.5 → 3.6 |

M&Ms and M&Ms-2 test patients are held out, but their cohorts are in training, so this is not external validation. Keeping the largest connected component of each structure lowers RV HD95 on M&Ms-2 (8.3 → 7.1 mm at end-diastole) and barely changes Dice. No nnU-Net baseline has been run on these splits.

External check on the Kaggle Data Science Bowl cardiac cohort (439 test patients, clinician LV volumes, no masks), frustum volumes: pooled model LV EF MAE 5.65 points (r 0.80, bias +4.1) against 6.95 (r 0.75, bias +5.9) for the ACDC-only model. Every stored stack has a 10 mm slice spacing in its header, but 51 patients list a different original spacing, about 12 of them impossible values (18–340 mm); without the 51, EF MAE is 5.3 points.

```bash
python scripts/train_mri_pooled.py                       # label checks, training, per-cohort test
python scripts/train_mri_pooled.py --evaluate-only
python scripts/validate_mri_kaggle.py --bf16 --checkpoints results/mri_segmentation_pooled/mri_segmenter.pt results/mri_segmentation/mri_segmenter.pt
python scripts/fusion_likelihood_ratios.py               # MRI likelihood ratios used by the web portal
```

Raw records: `results/mri_segmentation_pooled/` and `results/mri_segmentation_pooled/kaggle/`.

## Health record and the deep learning fusion

The health-record model estimates angiographic coronary artery disease (more than 50% narrowing of a major
vessel) from the seven findings the web form asks for: age, sex, chest-pain type, resting blood pressure,
cholesterol, maximum heart rate on exercise and exercise angina. Data: UCI Heart Disease, 920 patients referred
for angiography at Cleveland, Budapest, Zurich and Basel, and VA Long Beach (CC BY 4.0). Values recorded as 0
(all Swiss cholesterol, many at VA) are treated as missing.

Missingness marks the centre in this collection (Switzerland has no cholesterol and 93% disease), so the model is
not given it: training gaps take the median, and a field left blank in use is averaged over the training patients'
values instead. With missing-value flags the model had learned "blank means disease": a 58-year-old man with
nothing else entered scored 90%.

| Test | Patients | AUROC (95% CI) |
| --- | --- | --- |
| Internal, held out before model selection | 184 | 0.850 (0.79–0.90) |
| Cleveland, left out of training | 303 | 0.859 |
| Hungary, left out of training | 294 | 0.870 |
| Switzerland, left out of training | 123 | 0.788 |
| VA Long Beach, left out of training | 200 | 0.731 |

Probabilities are calibrated on the internal test but shift with a centre's disease rate (Switzerland: 93%
observed, 69% predicted; Hungary: 36% observed, 52% predicted), and every patient was referred for angiography, so a probability applies to such
patients, not to the general population.

The stand-alone model above is a baseline. In the app the health record is read by a neural network branch of
the fusion network.

### Fusion network: ECG + MRI + health record

`src/cardiac_nexus/fusion.py`. Each modality keeps its trained network as a branch: the ECG classifier's
128-number embedding, the pooled MRI segmenter's deepest feature map (256 channels, averaged over the heart's
slices, at end-diastole and end-systole) with the eight measurements of its outlines, and a small network over the
seven form fields (encoded as ten numbers; a blank field takes the training mean). Each is projected to 128
numbers; two transformer attention layers (four heads) let the three inform each other; a learned gate weighs
them. The output is hybrid: the branches' own log-odds summed, plus the attention layers' correction. Any modality
can be missing (training drops each whole modality with probability 0.3). 401,001 parameters.

The target is the one outcome all three datasets label: heart disease present (an ECG that is not normal, an MRI
that is not NOR, UCI num > 0). No public dataset has all three for the same people, so training and testing use
virtual patients that join an ECG, an MRI and a health record of different people with the same label, each
source keeping its own patient-wise split: PTB-XL fold 9 (which the ECG network never trained on) for training and
validation and fold 10 for testing; the MRI segmenter's splits of ACDC, M&Ms and M&Ms-2 (550 labelled patients);
UCI's 736/184 split. AdamW, learning rate 3e-4, weight decay 1e-3, batch 64, input dropout 0.2/0.5/0.1, epoch
4 chosen on validation; one calibration and a threshold per input combination set on validation.

5,000 virtual test patients, half with heart disease:

| Inputs | AUROC (95% CI) | Precision | Recall | F1 | Brier | ECE |
| --- | --- | --- | --- | --- | --- | --- |
| ECG + MRI + health record | 0.973 (0.970–0.977) | 0.904 | 0.928 | 0.916 | 0.065 | 0.028 |
| ECG + MRI | 0.957 (0.952–0.962) | 0.835 | 0.930 | 0.880 | 0.084 | 0.038 |
| ECG + health record | 0.966 (0.961–0.970) | 0.888 | 0.916 | 0.902 | 0.073 | 0.043 |
| MRI + health record | 0.906 (0.898–0.914) | 0.790 | 0.842 | 0.815 | 0.133 | 0.058 |
| ECG | 0.943 (0.937–0.949) | 0.890 | 0.871 | 0.880 | 0.095 | 0.050 |
| MRI | 0.850 (0.839–0.861) | 0.824 | 0.663 | 0.735 | 0.174 | 0.097 |
| health record | 0.823 (0.811–0.835) | 0.747 | 0.815 | 0.779 | 0.171 | 0.063 |
| Mean of the three single readings (baseline) | 0.973 | 0.902 | 0.929 | 0.916 | 0.102 | 0.185 |

Every modality added helps. The mean of the single readings ranks as well, as expected when a virtual patient's
modalities are independent given the label (their summed log-odds are then the best possible combination), but it
is badly calibrated; the network's probabilities are not. The network cannot learn how modalities interact within
one person from virtual patients; that needs paired data (MIMIC-IV with MIMIC-IV-ECG, UK Biobank).

Explanations (`fusion.predict`, `fusion.explain_in_words`): each modality's own reading, how far the result moves
when it is left out, the attention weights, Integrated Gradients over the health-record fields, and sentences for
patients and for clinicians built only from those numbers.

Run history: `results/fusion_dl/run1` (no input dropout, one threshold for all combinations: over-trusted the MRI and
called everyone ill without it) and `run2` (input dropout and per-combination thresholds, plain output: below the
baseline) are kept; the hybrid output is run 3.

```bash
python scripts/train_ehr.py             # stand-alone health-record baseline
python scripts/fusion_embeddings.py     # branch inputs from the deployed ECG and MRI networks (data/fusion)
python scripts/train_fusion.py          # train, calibrate and test the fusion network
```

Raw records: `results/ehr/ehr_report.json` and `results/fusion_dl/fusion_report.json`.

## Web application

The trained models are served by the backend in
[nexus-web-portal](https://github.com/Cardiac-Nexus-Lab/nexus-web-portal), which
loads them directly from this repository's `results/` and `src/`. Clone the two
repositories side by side. The files it uses:

| File | Model |
| --- | --- |
| `results/local_xresnet18_full/ecg_multilabel.pt` and `evaluation_report.json` | ECG classifier and its calibration |
| `results/digitizer/trace_localizer.pt` | ECG printout digitizer |
| `results/mri_segmentation_pooled/mri_segmenter.pt` | MRI segmenter (pooled; falls back to `results/mri_segmentation/` in older checkouts) |
| `results/mri_diagnosis/diagnosis_model.joblib` | MRI diagnosis classifier |
| `results/ehr/ehr_model.joblib` and `ehr_report.json` | Health-record model (coronary artery disease) |
| `results/fusion/likelihood_ratios.json` and `three_way.json` | Settings for combining the three models |

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

MRI sources: resampled copies of ACDC, M&Ms, M&Ms-2 and the Kaggle Data Science Bowl cardiac cohort, downloaded from the `viennh2012/cardiac_cine_*` mirrors on Hugging Face. The licence of ACDC, M&Ms and M&Ms-2 has been reported by the project as CC BY-NC-SA 4.0 (non-commercial, share-alike); this has not been checked against the original challenge pages, and the mirrors' own terms are unconfirmed. Confirm both before any use beyond research. None of this data is committed.

## Limitations

- Evaluated on PTB-XL fold 10 only. No external dataset has been tested, so generalisation beyond this cohort, its equipment, and its labelling conventions is unestablished.
- Results come from a single random seed; seed-to-seed variation is unquantified.
- Attribution maps describe what this model responded to. Published work finds such methods disagree with one another and can survive weight randomization, so they are reported as exploratory and accompanied by the sanity check rather than presented as evidence.
- Digitization is trained on rendered printouts, not photographs of real ones. It works on flat pages and scans but not on pages photographed at an angle, and performance on genuine clinical paper is untested.
- MRI segmentation is trained and tested on three cohorts; the only external check (Kaggle) has volumes but no masks, so external Dice is unmeasured. All MRI data are resampled mirror copies rather than the original releases, the model is 2.5D and was trained on CPU without full convergence, and no nnU-Net baseline has been run.
- This work is intended for research evaluation. Model outputs must not be used as a standalone basis for medical decisions.

## Planned development

- External validation on an independent public dataset.
- Deliberate per-class operating thresholds rather than a default of 0.5.
- Tabular EHR as a second modality, then multimodal fusion. Fusion requires paired records, meaning the same patient across modalities. Such cohorts exist: UK Biobank pairs 12-lead ECG with cardiac MRI at the same imaging visit, and MIMIC-IV-ECG pairs ECG with hospital records. Both are gated by application, cost, and storage on a scale this project cannot currently meet, so the constraint here is access rather than existence.
