import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def plot_connectivity_matrix(matrix, title=None, cmap="hot"):

    plt.figure(figsize=(6, 5))

    plt.imshow(matrix, cmap=cmap)

    plt.colorbar()

    if title is not None:
        plt.title(title)

    plt.xlabel("ROI")
    plt.ylabel("ROI")

    plt.tight_layout()
    plt.show()


def edge_mask_to_matrix(edge_index, edge_mask, num_nodes):

    matrix = np.zeros((num_nodes, num_nodes))

    edge_index = edge_index.cpu().numpy()
    edge_mask = edge_mask.detach().cpu().numpy()

    for i in range(edge_index.shape[1]):

        src = edge_index[0, i]
        dst = edge_index[1, i]

        matrix[src, dst] = edge_mask[i]

    matrix = (matrix + matrix.T) / 2

    return matrix


def plot_brain_graph(adj, threshold=0.0):

    adj = np.array(adj)

    adj[adj < threshold] = 0

    G = nx.from_numpy_array(adj)

    pos = nx.spring_layout(G)

    plt.figure(figsize=(6, 6))

    weights = [G[u][v]["weight"] for u, v in G.edges()]

    nx.draw(
        G,
        pos,
        node_size=50,
        edge_color=weights,
        edge_cmap=plt.cm.hot,
        with_labels=False
    )

    plt.title("Brain Connectivity Graph")
    plt.show()


def plot_node_importance(scores):

    scores = np.array(scores)

    plt.figure(figsize=(6, 3))

    plt.bar(np.arange(len(scores)), scores)

    plt.xlabel("ROI index")
    plt.ylabel("Importance")

    plt.title("Node Importance")

    plt.tight_layout()
    plt.show()