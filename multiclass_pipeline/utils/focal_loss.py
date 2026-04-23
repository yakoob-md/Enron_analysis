# multiclass_pipeline/utils/focal_loss.py
# Focal Loss for multiclass (softmax-based) classification
# Reduces over-confidence on easy majority-class examples

import torch
import torch.nn as nn
import torch.nn.functional as F


class MulticlassFocalLoss(nn.Module):
    """
    Focal loss for multiclass classification.
    
    FL(pt) = -alpha_t * (1 - pt)^gamma * log(pt)
    
    gamma=2.0  → focus on hard examples
    alpha=None → no per-class weighting (use class_weights for that)
    
    To combine with class weights, pass weight= to CrossEntropyLoss instead
    and use this on top.
    """
    def __init__(self, gamma=2.0, weight=None, reduction='mean'):
        super().__init__()
        self.gamma     = gamma
        self.weight    = weight      # per-class weights tensor (optional)
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        logits  : (batch, num_classes) — raw logits (before softmax)
        targets : (batch,) — integer class indices
        """
        # Standard cross-entropy gives log(pt)
        log_pt = F.log_softmax(logits, dim=-1)
        log_pt = log_pt.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt     = log_pt.exp()

        # Focal weight
        focal_weight = (1 - pt) ** self.gamma

        # Per-class alpha weighting
        if self.weight is not None:
            alpha_t = self.weight[targets]
            focal_weight = alpha_t * focal_weight

        loss = -focal_weight * log_pt

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss