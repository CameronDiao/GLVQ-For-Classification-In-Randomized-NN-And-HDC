"""ProtoTorch losses."""

import torch

from .activations import get_activation
from .losses import glvq_loss, kglvq_loss


class GLVQLoss(torch.nn.Module):
    def __init__(self, margin=0.0, squashing='identity', beta=10, **kwargs):
        super().__init__(**kwargs)
        self.margin = margin
        self.squashing = get_activation(squashing)
        self.beta = torch.tensor(beta)

    def forward(self, outputs, targets):
        distances, plabels = outputs
        mu = glvq_loss(distances, targets, plabels)
        batch_loss = self.squashing(mu + self.margin, beta=self.beta)
        return torch.sum(batch_loss, dim=0)

class KGLVQLoss(torch.nn.Module):
    def __init__(self, margin=0.0, squashing='identity', beta=10, **kwargs):
        super().__init__(**kwargs)
        self.margin = margin
        self.squashing = get_activation(squashing)
        self.beta = torch.tensor(beta)

    def forward(self, outputs, targets):
        distances, plabels = outputs
        mu = kglvq_loss(distances, targets, plabels)
        batch_loss = self.squashing(mu + self.margin, beta=self.beta)
        return torch.sum(batch_loss, dim=0)