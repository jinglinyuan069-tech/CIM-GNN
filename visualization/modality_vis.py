import numpy as np
import matplotlib.pyplot as plt


def collect_modality_weights(alpha_list):

    weights = []

    for alpha in alpha_list:

        if hasattr(alpha, "detach"):
            alpha = alpha.detach().cpu().numpy()

        weights.append(alpha)

    weights = np.concatenate(weights, axis=1)

    return weights


def average_modality_weights(weights):

    mean_weights = weights.mean(axis=1)

    return mean_weights


def plot_modality_bar(weights, labels=None, title="Modality Contribution"):

    if labels is None:
        labels = [f"M{i}" for i in range(len(weights))]

    x = np.arange(len(weights))

    plt.figure(figsize=(5,4))

    plt.bar(x, weights)

    plt.xticks(x, labels)

    plt.ylabel("Weight")

    plt.title(title)

    plt.tight_layout()

    plt.show()


def plot_modality_distribution(weights, labels=None):

    if labels is None:
        labels = [f"M{i}" for i in range(weights.shape[0])]

    plt.figure(figsize=(6,4))

    data = [weights[i] for i in range(weights.shape[0])]

    plt.boxplot(data, labels=labels)

    plt.ylabel("Weight")

    plt.title("Modality Weight Distribution")

    plt.tight_layout()

    plt.show()


def plot_modality_heatmap(weights, title="Modality Weight Heatmap"):

    plt.figure(figsize=(6,4))

    plt.imshow(weights, aspect="auto", cmap="hot")

    plt.colorbar()

    plt.xlabel("Subject")

    plt.ylabel("Modality")

    plt.title(title)

    plt.tight_layout()

    plt.show()