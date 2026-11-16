import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score


def move_to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, list):
        return [move_to_device(item, device) for item in data]
    if isinstance(data, tuple):
        return tuple(move_to_device(item, device) for item in data)
    if isinstance(data, dict):
        return {key: move_to_device(value, device) for key, value in data.items()}
    return data


def f_latency(predictions, labels, delays, penalty=0.037):
    f1 = f1_score(labels, predictions, average="binary", zero_division=0)
    true_positive_delays = delays[(predictions == 1) & (labels == 1)]
    if true_positive_delays.size == 0:
        return 0.0
    delay_penalties = -1.0 + 2.0 / (
        1.0 + np.exp(-penalty * true_positive_delays)
    )
    speed = 1.0 - float(np.median(delay_penalties))
    return f1 * speed


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        targets = targets.float()
        binary_cross_entropy = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        probabilities = torch.sigmoid(logits)
        correct_class_probability = (
            probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
        )
        alpha_weight = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        loss = (
            alpha_weight
            * (1.0 - correct_class_probability).pow(self.gamma)
            * binary_cross_entropy
        )
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
