import torch
import torch.nn.functional as F


def prediction_loss(pred, target, task="regression"):

    if task == "regression":

        return F.mse_loss(pred.squeeze(), target)

    if task == "classification":

        return F.binary_cross_entropy_with_logits(
            pred.squeeze(), target.float()
        )

    raise ValueError("Unknown task")


def consistency_loss(h_fc, h_sc):

    return F.mse_loss(h_fc, h_sc)


def subgraph_sparsity_loss(mask):

    return torch.mean(mask)


def total_loss(
    pred,
    target,
    h_fc,
    h_sc,
    edge_mask,
    config
):

    loss_pred = prediction_loss(
        pred,
        target,
        config.task
    )

    loss_cm = consistency_loss(h_fc, h_sc)

    loss_sg = subgraph_sparsity_loss(edge_mask)

    loss = (
        loss_pred
        + config.lambda_cm * loss_cm
        + config.lambda_sg * loss_sg
    )

    losses = {
        "total": loss,
        "pred": loss_pred,
        "cm": loss_cm,
        "sg": loss_sg
    }

    return loss, losses