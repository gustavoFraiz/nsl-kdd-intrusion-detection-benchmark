"""Recovered benchmark result table and visualization helpers."""

import pandas as pd
import matplotlib.pyplot as plt

RESULTS = {
    'Naive Bayes': {
        'Accuracy': 0.4670,
        'AUC-ROC': 0.6979,
        'Weighted F1': 0.4779,
        'False Positive Rate (%)': 33.77,
        'Training Time (s)': 23,
    },
    'LSTM': {
        'Accuracy': 0.8021,
        'AUC-ROC': 0.8899,
        'Weighted F1': 0.7895,
        'False Positive Rate (%)': 4.06,
        'Training Time (s)': 60,
    },
    'CNN': {
        'Accuracy': 0.7911,
        'AUC-ROC': 0.9439,
        'Weighted F1': 0.7740,
        'False Positive Rate (%)': 4.83,
        'Training Time (s)': 60,
    },
    'TabTransformer': {
        'Accuracy': 0.7792,
        'AUC-ROC': 0.9338,
        'Weighted F1': 0.7526,
        'False Positive Rate (%)': 7.94,
        'Training Time (s)': 210.5,
    },
    'Autoencoder+SVM': {
        'Accuracy': 0.7692,
        'AUC-ROC': 0.9171,
        'Weighted F1': 0.7318,
        'False Positive Rate (%)': 3.71,
        'Training Time (s)': 450.2,
    },
    'Federated MLP (SMOTE)': {
        'Accuracy': 0.7653,
        'AUC-ROC': 0.8885,
        'Weighted F1': 0.7516,
        'False Positive Rate (%)': 8.16,
        'Training Time (s)': 240,
    },
}

F1_PER_CLASS = {
    'CNN': {'dos': 0.8969, 'normal': 0.8249, 'probe': 0.7179, 'r2l': 0.3408, 'u2r': 0.4025},
    'GaussianNB': {'dos': 0.5630, 'normal': 0.5768, 'probe': 0.1806, 'r2l': 0.1843, 'u2r': 0.0717},
    'LSTM': {'dos': 0.8835, 'normal': 0.8324, 'probe': 0.8127, 'r2l': 0.3971, 'u2r': 0.1599},
    'TabTransformer': {'dos': 0.8810, 'normal': 0.8011, 'probe': 0.7617, 'r2l': 0.2626, 'u2r': 0.2189},
    'AE+SVM': {'dos': 0.8818, 'normal': 0.7905, 'probe': 0.7561, 'r2l': 0.1408, 'u2r': 0.1096},
    'Federated MLP (SMOTE)': {'dos': 0.8609, 'normal': 0.8073, 'probe': 0.6915, 'r2l': 0.3460, 'u2r': 0.1619},
}


def results_frame():
    return pd.DataFrame(RESULTS).T


def plot_overall_metrics():
    df = results_frame()[['Accuracy', 'AUC-ROC', 'Weighted F1']]
    ax = df.plot(kind='bar', figsize=(14, 7), rot=20)
    ax.set_ylim(0, 1.05)
    ax.set_title('NSL-KDD Model Comparison')
    ax.set_ylabel('Score')
    plt.tight_layout()
    return ax
