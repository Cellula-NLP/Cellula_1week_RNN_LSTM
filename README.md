<div align="center">

# Toxic Comment Classification

**BiRNN vs BiLSTM with attention pooling, calibrated thresholds, and hierarchical gating.**

A leakage-free pipeline for multi-label toxicity detection on the Jigsaw Wikipedia dataset.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/M7mdNassar/Cellula_1week_RNN_LSTM/blob/main/notebook.ipynb)
[![Python](https://img.shields.io/badge/python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

</div>

---

## Overview

Six binary labels, 160k comments, extreme class imbalance (threat: 0.3 %, identity_hate: 0.7 %). This repo trains and compares two recurrent architectures under an identical, deliberately conservative pipeline:

- **Stacked BiRNN** and **Stacked BiLSTM**, both with attention + max + mean pooling heads
- **GloVe-300d** initialization, **focal loss** with capped `pos_weight`
- **Platt calibration** on a dedicated split, **per-class threshold tuning** on validation
- **Hierarchical gating** justified by EDA (severe_toxic never co-occurs without toxic)
- **Strict split discipline** — no threshold or calibration choice ever touches the test set

## Results

| Metric | BiRNN | BiLSTM |
|---|---|---|
| Macro-F1 | 0.6182 | **0.6368** |
| Micro-F1 | 0.7475 | **0.7559** |
| Macro-AUC | 0.9376 | **0.9571** |

The LSTM wins on threshold-tuned F1; the RNN wins on ranking quality. Section 6 of the notebook unpacks that trade-off per class.

## Pipeline

```
Raw text
   │
   ▼  deobfuscate → strip markup → lemmatize
Cleaned text
   │
   ▼  split (train / val / calib / test, multilabel-stratified)
Splits
   │
   ▼  GloVe-300d  +  focal loss  +  pos_weight cap
BiRNN / BiLSTM  ── attention ⊕ max ⊕ mean ──▶ classifier
   │
   ▼  Platt on calib → thresholds on val → gate on val
Test evaluation
```

## Quickstart

**On Colab** — click the badge at the top. The notebook downloads GloVe on first run.

**Locally**

```bash
git https://github.com/Cellula-NLP/Cellula_1week_RNN_LSTM.git
cd Cellula_1week_RNN_LSTM
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

You will also need the Jigsaw `train.csv` placed next to the notebook.

## Repository layout

```
.
├── notebook.ipynb        # full pipeline: EDA → training → evaluation → comparison
├── preprocessing.py      # cleaning, tokenization, vocabulary (importable)
├── requirements.txt
└── README.md
```

## Dataset

[Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) — `train.csv`, 159,571 comments, 6 binary labels.

## License

MIT — see [`LICENSE`](./LICENSE).
