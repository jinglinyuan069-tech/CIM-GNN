import os
import torch
import numpy as np

from configs.config import Config

from data.loader import build_dataset
from data.loader import create_loader

from models.model import MultimodalBrainGNN

from training.trainer import Trainer

from evaluation.evaluator import Evaluator

from utils.seed import set_seed
from utils.logger import Logger, log_config


def load_data(path):

    data = np.load(path, allow_pickle=True)

    fc = data["fc"]
    sc = data["sc"]
    feat = data["feat"]
    y = data["y"]

    return fc, sc, feat, y


def split_dataset(fc, sc, feat, y, train_ratio=0.8):

    n = len(y)

    idx = np.random.permutation(n)

    train_n = int(n * train_ratio)

    train_idx = idx[:train_n]
    val_idx = idx[train_n:]

    train_data = (
        fc[train_idx],
        sc[train_idx],
        feat[train_idx],
        y[train_idx],
    )

    val_data = (
        fc[val_idx],
        sc[val_idx],
        feat[val_idx],
        y[val_idx],
    )

    return train_data, val_data


def build_loader(fc, sc, feat, y, config, shuffle=True):

    fc_graphs, sc_graphs, targets = build_dataset(fc, sc, feat, y)

    loader = create_loader(
        fc_graphs,
        sc_graphs,
        targets,
        batch_size=config.batch_size,
        shuffle=shuffle
    )

    return loader


def main():

    set_seed(42)

    config = Config()

    logger = Logger("logs", "train")

    log_config(logger, config)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    fc, sc, feat, y = load_data("data/hcpd_data.npz")

    train_data, val_data = split_dataset(fc, sc, feat, y)

    train_loader = build_loader(*train_data, config)

    val_loader = build_loader(*val_data, config, shuffle=False)

    model = MultimodalBrainGNN(config)

    trainer = Trainer(model, config)

    evaluator = Evaluator(
        model,
        device,
        task=config.task
    )

    os.makedirs("checkpoints", exist_ok=True)

    best_score = float("inf")

    for epoch in range(config.epochs):

        train_loss = trainer.train_epoch(train_loader)

        metrics = evaluator.evaluate(val_loader)

        if config.task == "regression":
            score = metrics["RMSE"]
        else:
            score = -metrics["Accuracy"]

        logger.log(
            f"Epoch {epoch:03d} | "
            f"TrainLoss {train_loss:.4f} | "
            f"Metrics {metrics}"
        )

        if score < best_score:

            best_score = score

            torch.save(
                model.state_dict(),
                "checkpoints/best_model.pt"
            )

            logger.log("Best model updated")

    logger.log("Training finished")

    logger.close()


if __name__ == "__main__":

    main()