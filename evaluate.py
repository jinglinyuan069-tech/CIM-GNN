import torch
import numpy as np

from configs.config import Config

from data.loader import build_dataset, create_loader

from models.model import MultimodalBrainGNN

from evaluation.evaluator import Evaluator

from utils.seed import set_seed


def load_data():

    data = np.load("data/hcpd_data.npz", allow_pickle=True)

    fc = data["fc"]
    sc = data["sc"]
    feat = data["feat"]
    y = data["y"]

    return fc, sc, feat, y


def build_loader(fc, sc, feat, y, config):

    fc_graphs, sc_graphs, targets = build_dataset(
        fc,
        sc,
        feat,
        y
    )

    loader = create_loader(
        fc_graphs,
        sc_graphs,
        targets,
        batch_size=config.batch_size,
        shuffle=False
    )

    return loader


def main():

    set_seed(42)

    config = Config()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    fc, sc, feat, y = load_data()

    test_loader = build_loader(
        fc,
        sc,
        feat,
        y,
        config
    )

    model = MultimodalBrainGNN(config)

    model.load_state_dict(
        torch.load(
            "checkpoints/best_model.pt",
            map_location=device
        )
    )

    model.to(device)

    evaluator = Evaluator(
        model,
        device,
        task=config.task
    )

    metrics = evaluator.evaluate(test_loader)

    print("===== Evaluation Results =====")

    for k, v in metrics.items():

        print(f"{k}: {v:.4f}")


if __name__ == "__main__":

    main()