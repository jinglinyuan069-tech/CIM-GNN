import torch
import torch.nn.functional as F


def node_consistency_loss(h_fc, h_sc):

    return F.mse_loss(h_fc, h_sc)


def normalized_consistency_loss(h_fc, h_sc):

    h_fc = F.normalize(h_fc, dim=-1)
    h_sc = F.normalize(h_sc, dim=-1)

    return F.mse_loss(h_fc, h_sc)


def cosine_consistency_loss(h_fc, h_sc):

    h_fc = F.normalize(h_fc, dim=-1)
    h_sc = F.normalize(h_sc, dim=-1)

    cos = (h_fc * h_sc).sum(dim=-1)

    loss = 1 - cos.mean()

    return loss


def consistency_loss(h_fc, h_sc, mode="mse"):

    if mode == "mse":
        return node_consistency_loss(h_fc, h_sc)

    if mode == "cos":
        return cosine_consistency_loss(h_fc, h_sc)

    if mode == "norm":
        return normalized_consistency_loss(h_fc, h_sc)

    raise ValueError("Unknown consistency mode")