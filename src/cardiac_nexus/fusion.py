"""Multimodal deep learning fusion of ECG, cardiac MRI and the health record.

Each modality keeps its own trained network as a branch:

* ECG: the xresnet1d18 classifier's 128-number embedding of the 12-lead, 10 s
  recording (the layer before its five-finding output);
* MRI: the pooled 2.5D U-Net's deepest feature map (256 channels), averaged over
  the slices where it finds the heart, at end-diastole and end-systole, plus the
  eight measurements its outlines give (volumes, ejection fractions, mass);
* health record: the seven form fields, encoded as ten numbers.

The fusion network projects each to a 128-number token, lets the three tokens
attend to each other (two transformer layers, four heads), weighs them with a
learned gate and predicts one output: whether heart disease is present. Every
modality is optional: training drops whole modalities at random, so a patient
with only an ECG, or only a form, still gets a result, and the same network gives
each modality's own reading by switching the others off.

Target. "Heart disease present" is the one outcome all three training datasets
label: an ECG that is not normal (PTB-XL), an MRI that is not normal (ACDC, M&Ms,
M&Ms-2: dilated, hypertrophic, infarcted or right-ventricular disease), and
angiographic coronary disease (UCI Heart Disease, num > 0).

No public dataset has all three for the same patients, so the network is trained
and tested on virtual patients: an ECG, an MRI and a health record drawn from
patients with the same label, each source keeping its own patient-wise splits.
The network therefore learns how much each modality's evidence is worth, not how
the three interact within one person, and its test results are on virtual
patients, not a clinical validation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .mri_diagnosis import POOLED_FEATURES, pooled_features

MODALITIES = ("ecg", "mri", "clinical")
MODALITY_NAMES = {"ecg": "ECG", "mri": "cardiac MRI", "clinical": "health record"}
WIDTH = 128
MRI_FEATURES = list(POOLED_FEATURES)

# Health-record encoding. Numbers are standardized with the training patients' mean
# and spread; categories are one-hot. A blank field takes the training patients'
# mean for its columns, during training (fields are blanked at random) and in use.
CLINICAL_NUMERIC = ["age", "trestbps", "chol", "thalach"]
CLINICAL_FIELDS = CLINICAL_NUMERIC + ["sex", "cp", "exang"]
CLINICAL_COLUMNS = CLINICAL_NUMERIC + ["sex_male", "cp_typical", "cp_atypical", "cp_non_anginal",
                                       "cp_asymptomatic", "exang_yes"]
FIELD_COLUMNS = {field: [c for c in CLINICAL_COLUMNS if c == field or c.startswith(field + "_")]
                 for field in CLINICAL_FIELDS}
FIELD_NAMES = {"age": "age", "trestbps": "resting blood pressure", "chol": "cholesterol",
               "thalach": "maximum heart rate", "sex": "sex", "cp": "chest-pain type", "exang": "exercise angina"}


def encode_clinical(row: dict, stats: dict) -> tuple[np.ndarray, list[str]]:
    """One patient's fields (UCI coding: sex 1 male, cp 1-4, exang 1 yes) as the clinical branch's input.

    Returns the vector and the fields that were blank."""
    vector = np.array(stats["column_mean"], dtype=np.float32)
    blank = []
    for field in CLINICAL_FIELDS:
        value = row.get(field)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            blank.append(field)
            continue
        if field in CLINICAL_NUMERIC:
            i = CLINICAL_COLUMNS.index(field)
            vector[i] = (float(value) - stats["numeric_mean"][field]) / stats["numeric_std"][field]
        elif field == "sex":
            vector[CLINICAL_COLUMNS.index("sex_male")] = float(value == 1)
        elif field == "exang":
            vector[CLINICAL_COLUMNS.index("exang_yes")] = float(value == 1)
        else:  # cp: 1 typical, 2 atypical, 3 non-anginal, 4 asymptomatic
            for k, column in enumerate(FIELD_COLUMNS["cp"], start=1):
                vector[CLINICAL_COLUMNS.index(column)] = float(int(value) == k)
    return vector, blank


def clinical_stats(table) -> dict:
    """Encoding statistics from the training patients (a pandas frame in UCI coding)."""
    numeric_mean = {f: float(table[f].mean()) for f in CLINICAL_NUMERIC}
    numeric_std = {f: float(table[f].std()) for f in CLINICAL_NUMERIC}
    stats = {"numeric_mean": numeric_mean, "numeric_std": numeric_std,
             "column_mean": [0.0] * len(CLINICAL_COLUMNS)}
    rows = [encode_clinical(r, stats)[0] for r in table[CLINICAL_FIELDS].to_dict("records")]
    observed = np.array(rows)
    present = ~table[CLINICAL_FIELDS].isna().to_numpy()
    means = []
    for i, column in enumerate(CLINICAL_COLUMNS):
        field = next(f for f, cols in FIELD_COLUMNS.items() if column in cols)
        keep = present[:, CLINICAL_FIELDS.index(field)]
        means.append(float(observed[keep, i].mean()))
    stats["column_mean"] = means
    return stats


def blank_fields(vector: np.ndarray, fields: list[str], stats: dict) -> np.ndarray:
    out = vector.copy()
    for field in fields:
        for column in FIELD_COLUMNS[field]:
            i = CLINICAL_COLUMNS.index(column)
            out[i] = stats["column_mean"][i]
    return out


# --- branch inputs --------------------------------------------------------------

def normalize_volume(volume: np.ndarray) -> np.ndarray:
    """Per-volume intensity standardization, as the segmenter was trained (and as the portal does)."""
    low, high = np.percentile(volume, (0.5, 99.5))
    clipped = np.clip(volume, low, high)
    return ((clipped - clipped.mean()) / (clipped.std() + 1e-6)).astype(np.float32)


@torch.no_grad()
def mri_volume_features(segmenter, normalized: np.ndarray, device=None, batch: int = 16):
    """Predicted masks [slices, H, W] and the deepest feature map averaged over the heart's slices (256)."""
    device = device or next(segmenter.parameters()).device
    n = len(normalized)
    stacks = np.stack([np.stack([normalized[max(i - 1, 0)], normalized[i], normalized[min(i + 1, n - 1)]])
                       for i in range(n)])
    masks, pooled = [], []
    for start in range(0, n, batch):
        x = torch.from_numpy(stacks[start:start + batch]).to(device)
        _, skips = segmenter.encoder.forward_features(x)
        y = skips[-1]
        pooled.append(F.adaptive_avg_pool2d(y, 1).flatten(1).cpu().numpy())
        for block, skip in zip(segmenter.decoder, reversed(skips[:-1])):
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = block(torch.cat([y, skip], dim=1))
        masks.append(segmenter.head(y).argmax(dim=1).cpu().numpy().astype(np.uint8))
    masks, pooled = np.concatenate(masks), np.concatenate(pooled)
    heart = (masks > 0).reshape(n, -1).any(axis=1)
    return masks, pooled[heart].mean(axis=0) if heart.any() else pooled.mean(axis=0)


def mri_branch_input(segmenter, ed_volume: np.ndarray, es_volume: np.ndarray, spacing, device=None):
    """MRI branch input from raw end-diastolic and end-systolic volumes [slices, H, W] at 1 mm in-plane.

    Returns (features [512], measurements dict, ed masks, es masks)."""
    ed_masks, ed_pooled = mri_volume_features(segmenter, normalize_volume(ed_volume), device)
    es_masks, es_pooled = mri_volume_features(segmenter, normalize_volume(es_volume), device)
    return np.concatenate([ed_pooled, es_pooled]), pooled_features(ed_masks, es_masks, spacing), ed_masks, es_masks


def mri_vector(features: np.ndarray, measurements: dict, stats: dict) -> np.ndarray:
    """Feature map summary plus standardized measurements; an undefined measurement takes the training median."""
    values = np.array([measurements.get(k, np.nan) for k in MRI_FEATURES], dtype=np.float32)
    values = np.where(np.isfinite(values), values, np.asarray(stats["measure_median"], dtype=np.float32))
    values = (values - np.asarray(stats["measure_mean"])) / np.asarray(stats["measure_std"])
    return np.concatenate([features.astype(np.float32), values.astype(np.float32)])


@torch.no_grad()
def ecg_embedding(classifier, signals: np.ndarray, batch: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """[N, 12, 1000] standardized signals -> embeddings [N, 128] and raw logits [N, 5]."""
    classifier.eval()
    embeddings, logits = [], []
    for start in range(0, len(signals), batch):
        x = torch.from_numpy(np.asarray(signals[start:start + batch], dtype=np.float32))
        e = classifier.encoder(x)
        embeddings.append(e.numpy())
        logits.append(classifier.head(e).numpy())
    return np.concatenate(embeddings), np.concatenate(logits)


# --- the fusion network -----------------------------------------------------------

class FusionNet(nn.Module):
    """Three branch projections, cross-modal attention, a gate over modalities, one output."""

    def __init__(self, ecg_dim: int = 128, mri_dim: int = 256 * 2 + len(MRI_FEATURES),
                 clinical_dim: int = len(CLINICAL_COLUMNS), width: int = WIDTH, heads: int = 4,
                 layers: int = 2, dropout: float = 0.3, input_dropout: dict | None = None):
        super().__init__()
        # Dropout on each branch's input features. The MRI pool is a few hundred patients,
        # each reused across thousands of virtual patients, so without it the network
        # memorizes their feature vectors and over-trusts the MRI on new patients.
        input_dropout = input_dropout or {"ecg": 0.0, "mri": 0.0, "clinical": 0.0}

        def branch(dim: int, hidden: int, p_in: float) -> nn.Module:
            return nn.Sequential(nn.LayerNorm(dim), nn.Dropout(p_in), nn.Linear(dim, hidden), nn.GELU(),
                                 nn.Dropout(0.2), nn.Linear(hidden, width))

        self.branches = nn.ModuleDict({"ecg": branch(ecg_dim, width, input_dropout["ecg"]),
                                       "mri": branch(mri_dim, width, input_dropout["mri"]),
                                       "clinical": branch(clinical_dim, 64, input_dropout["clinical"])})
        self.modality_tag = nn.Parameter(torch.randn(len(MODALITIES), width) * 0.02)
        layer = nn.TransformerEncoderLayer(width, heads, dim_feedforward=2 * width, dropout=0.1,
                                           batch_first=True, norm_first=True)
        self.attention = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.gate = nn.Linear(width, 1)
        self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 64), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(64, 1))
        # Each branch's own reading, from its token before fusion: a training aid that
        # keeps every branch informative on its own.
        self.solo = nn.ModuleDict({m: nn.Linear(width, 1) for m in MODALITIES})

    def forward(self, ecg: torch.Tensor, mri: torch.Tensor, clinical: torch.Tensor, present: torch.Tensor):
        """present: [batch, 3] booleans, in MODALITIES order; at least one per row.

        Returns the fused logit [batch], gate weights [batch, 3] and solo logits [batch, 3]."""
        inputs = {"ecg": ecg, "mri": mri, "clinical": clinical}
        tokens = torch.stack([self.branches[m](inputs[m]) for m in MODALITIES], dim=1) + self.modality_tag
        solo = torch.cat([self.solo[m](tokens[:, i]) for i, m in enumerate(MODALITIES)], dim=1)
        mixed = self.attention(tokens, src_key_padding_mask=~present)
        scores = self.gate(mixed).squeeze(-1).masked_fill(~present, float("-inf"))
        weights = torch.softmax(scores, dim=1)
        fused = (weights.unsqueeze(-1) * mixed).sum(dim=1)
        return self.head(fused).squeeze(-1), weights, solo


@dataclass
class FusionInputs:
    ecg: np.ndarray | None          # [128]
    mri: np.ndarray | None          # [520]
    clinical: np.ndarray | None     # [10]

    def tensors(self, ecg_dim: int = 128, mri_dim: int = 520, clinical_dim: int = len(CLINICAL_COLUMNS)):
        def t(x, dim):
            return torch.zeros(1, dim) if x is None else torch.from_numpy(np.asarray(x, np.float32))[None]
        present = torch.tensor([[self.ecg is not None, self.mri is not None, self.clinical is not None]])
        return t(self.ecg, ecg_dim), t(self.mri, mri_dim), t(self.clinical, clinical_dim), present


def load_fusion(path, device=None) -> tuple[FusionNet, dict]:
    checkpoint = torch.load(path, map_location=device or "cpu", weights_only=True)
    model = FusionNet(**checkpoint["model_args"])
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.eval(), checkpoint


# --- prediction with its explanation --------------------------------------------------

def _probability(logit: torch.Tensor, calibration: dict) -> float:
    return float(torch.sigmoid(logit * calibration["scale"] + calibration["bias"]))


@torch.no_grad()
def predict(model: FusionNet, inputs: FusionInputs, calibration: dict, clinical_stats_: dict | None = None,
            blank: list[str] | None = None) -> dict:
    """Fused probability, each modality's weight and effect, and the health-record fields that moved it.

    A modality's effect is how far the fused probability moves when that modality is
    switched off, which the network was trained to handle.
    """
    ecg, mri, clinical, present = inputs.tensors()
    if not present.any():
        raise ValueError("at least one modality is needed")
    logit, weights, solo = model(ecg, mri, clinical, present)
    result = {"probability": _probability(logit[0], calibration), "modalities": {}}
    for i, m in enumerate(MODALITIES):
        if not present[0, i]:
            continue
        entry = {"weight": float(weights[0, i]), "alone": _probability(solo[0, i], calibration)}
        if present[0].sum() > 1:
            without = present.clone()
            without[0, i] = False
            entry["without_it"] = _probability(model(ecg, mri, clinical, without)[0][0], calibration)
            entry["effect"] = result["probability"] - entry["without_it"]
        result["modalities"][m] = entry
    if inputs.clinical is not None and clinical_stats_ is not None:
        result["clinical_fields"] = clinical_attributions(model, ecg, mri, clinical, present,
                                                          clinical_stats_, blank or [], calibration)
    return result


def clinical_attributions(model, ecg, mri, clinical, present, stats, blank, calibration, steps: int = 32) -> list[dict]:
    """Integrated Gradients over the health-record input, from a blank form (every field at the training mean).

    Returns, per filled-in field, its share of the move from a blank form's probability,
    in probability points, largest first."""
    baseline = torch.tensor(stats["column_mean"], dtype=torch.float32)[None]
    alphas = torch.linspace(0, 1, steps)[:, None]
    path = (baseline + alphas * (clinical - baseline)).requires_grad_(True)
    with torch.enable_grad():
        logits = model(ecg.expand(steps, -1), mri.expand(steps, -1), path, present.expand(steps, -1))[0]
        probabilities = torch.sigmoid(logits * calibration["scale"] + calibration["bias"])
        gradients, = torch.autograd.grad(probabilities.sum(), path)
    attribution = ((clinical - baseline) * gradients.mean(dim=0, keepdim=True))[0].numpy()
    rows = []
    for field in CLINICAL_FIELDS:
        if field in blank:
            continue
        points = float(sum(attribution[CLINICAL_COLUMNS.index(c)] for c in FIELD_COLUMNS[field])) * 100
        rows.append({"field": field, "name": FIELD_NAMES[field], "points": points})
    return sorted(rows, key=lambda r: -abs(r["points"]))


def _pct(p: float) -> str:
    if p >= 0.995:
        return "above 99%"
    if p < 0.005:
        return "below 1%"
    return f"{100 * p:.0f}%"


def explain_in_words(result: dict, findings: dict | None = None, audience: str = "patient") -> list[str]:
    """The fused result as plain sentences, built only from the numbers in `result`.

    `findings` may hold short phrases each branch's own model produced (for example
    {"ecg": "signs of a previous heart attack", "mri": "a weak, enlarged left ventricle"}).
    audience "patient" avoids technical terms; "clinician" adds weights and effects.
    """
    findings = findings or {}
    p = result["probability"]
    level = "high" if p >= 0.7 else "moderate" if p >= 0.4 else "low"
    used = [MODALITY_NAMES[m] for m in MODALITIES if m in result["modalities"]]
    missing = [MODALITY_NAMES[m] for m in MODALITIES if m not in result["modalities"]]
    opening = (f"Looking at the {', '.join(used[:-1])} and {used[-1]} together" if len(used) > 1
               else f"From the {used[0]} alone")
    lines = [f"{opening}, the model puts the chance of heart disease at {_pct(p)}, which is {level}."]
    ranked = sorted(result["modalities"].items(), key=lambda kv: -abs(kv[1].get("effect", 0.0)))
    for m, entry in ranked:
        name = MODALITY_NAMES[m]
        what = f" ({findings[m]})" if m in findings else ""
        if "effect" in entry:
            points = 100 * entry["effect"]
            direction = "raised" if points > 0.5 else "lowered" if points < -0.5 else "barely changed"
            amount = f" by {abs(points):.0f} points" if direction != "barely changed" else ""
            sentence = f"The {name}{what} {direction} it{amount}"
            sentence += f" (from {_pct(entry['without_it'])} without it)." if direction != "barely changed" else "."
        else:
            sentence = f"This rests on the {name}{what} alone."
        if audience == "clinician":
            sentence += f" Fusion weight {entry['weight']:.2f}; on its own the {name} gives {_pct(entry['alone'])}."
        lines.append(sentence)
    fields = [f for f in result.get("clinical_fields", []) if abs(f["points"]) >= 1]
    if fields:
        parts = [f"{f['name']} ({'+' if f['points'] > 0 else '−'}{abs(f['points']):.0f})" for f in fields[:3]]
        lines.append("In the health record, the values that mattered most were " + ", ".join(parts) +
                     " points, compared with a blank form.")
    if missing:
        lines.append(f"No {' or '.join(missing)} was given, so it could not add to this estimate.")
    lines.append("The model was trained on ECGs, scans and health records of different people with the same "
                 "diagnosis, because no public dataset has all three for one person. Treat this as a research "
                 "estimate, not a diagnosis, and discuss it with a doctor.")
    return lines
