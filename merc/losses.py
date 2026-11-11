"""Focal loss (Lin et al., 2017) used by every run in this project."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=1.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, log_prob, target):
        """log_prob: (N, C) log-softmax output; target: (N,) labels."""
        log_pt = log_prob.gather(1, target.view(-1, 1))
        weight = (1 - log_pt.detach().exp()) ** self.gamma
        if self.alpha is not None:
            weight = weight * self.alpha[target].view(-1, 1)
        return (-weight * log_pt).mean()


def distillation_loss(student_log_prob, teacher_prob, weight, temperature=2.0):
    """Soft-target term between a student and a (cached) teacher distribution."""
    log_q = torch.log_softmax(student_log_prob / temperature, -1)
    log_p = torch.log(teacher_prob.clamp_min(1e-8) ** (1.0 / temperature))
    log_p = log_p - torch.logsumexp(log_p, -1, keepdim=True)
    return weight * temperature * temperature * F.kl_div(
        log_q, log_p, log_target=True, reduction='batchmean')
