import torch
import torch.nn.functional as F


def nll_loss(output, target, weight=None):
    return F.nll_loss(output, target, weight=weight)

def BCEWithLogitsLoss(output, target, pos_neg_ratio=None):
    return F.binary_cross_entropy_with_logits(output, target.type_as(output), pos_weight=torch.tensor(pos_neg_ratio))

def cross_entropy_loss(output, target):
    return F.cross_entropy(output, target)