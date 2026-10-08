# NSL-KDD Intrusion Detection Benchmark

A comparative machine-learning study for **network intrusion detection** using the NSL-KDD dataset.

The recovered project evaluates multiple modeling families rather than relying on a single classifier:

- Gaussian Naive Bayes;
- CNN;
- LSTM;
- tabular Transformer / SAINT-style model;
- self-supervised / contrastive autoencoder + SVM;
- federated MLP experiments;
- class-level evaluation across `normal`, `dos`, `probe`, `r2l` and `u2r`.

The original benchmark also compares accuracy, AUC-ROC, weighted F1, false-positive rate, training time and per-class precision/recall.

## Why this repository exists

The original work was spread across several Colab notebooks. This repository groups the related NSL-KDD experiments into one coherent project instead of publishing several academic notebooks with names such as "TFF" or "teste".

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── src/
│   ├── federated_mlp.py
│   ├── contrastive_autoencoder.py
│   ├── saint_tabular_transformer.py
│   ├── lstm.py
│   ├── gaussian_nb.py
│   ├── cnn.py
│   └── analysis.py
└── data/
    └── README.md
```

## Authorship note

The recovered notebooks do **not** contain a reliable explicit author block. Contributor names should be added here before public release if this project was collaborative. The repository intentionally does not guess authorship.

## Reproducibility

The source files are cleaned exports of the recovered Colab experiments. Colab-only `!pip` cells were removed in favor of `requirements.txt`; dataset paths may still need to be adapted locally.

No license is included by default.
