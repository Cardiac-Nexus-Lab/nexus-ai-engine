"""Rebuild the fusion network on the current ECG and MRI models, after either is retrained.

The fusion network reads the ECG classifier's embedding and the MRI segmenter's
features, so it is only valid on the exact models it was trained on: the portal
compares their SHA-256 with those recorded in results/fusion_dl/fusion_model.pt
and leaves the combined result out when they differ. After retraining a branch:

1. replace its checkpoint (results/local_xresnet18_full/ecg_multilabel.pt or
   results/mri_segmentation_pooled/mri_segmenter.pt) and evaluate it as usual;
2. run this script, which recomputes the branch inputs and retrains and tests the
   fusion network (scripts/fusion_embeddings.py, then scripts/train_fusion.py);
3. commit results/fusion_dl and restart the portal backend.

Retraining the health-record side needs only step 2's second half, since the
fusion network has its own health-record branch.

Usage:
    python scripts/rebuild_fusion.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fusion_embeddings  # noqa: E402
import train_fusion  # noqa: E402


if __name__ == "__main__":
    fusion_embeddings.main()
    train_fusion.main()
