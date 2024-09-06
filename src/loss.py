import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import torch


class FocalLoss(nn.Module):
    def __init__(self, num_classes, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.num_classes = num_classes
        self.reduction = reduction

        if alpha is None:
            self.alpha = torch.ones(num_classes, dtype=torch.float)
        else:
            self.alpha = torch.tensor(alpha, dtype=torch.float)

    def forward(self, inputs, targets):
        weights = self.alpha.to(inputs.device)
        targets_one_hot = F.one_hot(targets, self.num_classes).float()

        probs = F.softmax(inputs, dim=1)
        pt = (probs * targets_one_hot).sum(dim=1)

        focal_weight = torch.pow(1 - pt, self.gamma)

        loss = -weights[targets] * focal_weight * torch.log(pt + 1e-8)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class ClassBalancedFocalLoss(nn.Module):
    def __init__(self, num_classes, samples_per_class, beta=0.999, gamma=2.0, alpha=None, reduction='mean'):
        super(ClassBalancedFocalLoss, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class
        self.reduction = reduction

        assert len(samples_per_class) == num_classes, \
            f"Expected samples_per_class to have length {num_classes}, but got {len(samples_per_class)}."

        effective_num = 1.0 - np.power(self.beta, self.samples_per_class)
        self.class_weights = (1.0 - self.beta) / np.array(effective_num)
        self.class_weights = self.class_weights / np.sum(self.class_weights) * self.num_classes

        if alpha is None:
            self.alpha = torch.tensor(self.class_weights, dtype=torch.float)
        else:
            self.alpha = torch.tensor(alpha, dtype=torch.float)

    def forward(self, inputs, targets):
        weights = self.alpha.to(inputs.device)
        targets_one_hot = F.one_hot(targets, self.num_classes).float()

        probs = F.softmax(inputs, dim=1)
        pt = (probs * targets_one_hot).sum(dim=1)  

        focal_weight = torch.pow(1 - pt, self.gamma)

        class_balanced_weight = weights[targets]

        loss = -class_balanced_weight * focal_weight * torch.log(pt + 1e-8)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
        

class ClassBalancedCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes, samples_per_class, beta=0.999, reduction='mean'):
        super(ClassBalancedCrossEntropyLoss, self).__init__()
        self.beta = beta
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class
        self.reduction = reduction

        assert len(samples_per_class) == num_classes, \
            f"Expected samples_per_class to have length {num_classes}, but got {len(samples_per_class)}."

        effective_num = 1.0 - np.power(self.beta, self.samples_per_class)
        self.class_weights = (1.0 - self.beta) / np.array(effective_num)
        self.class_weights = self.class_weights / np.sum(self.class_weights) * self.num_classes

    def forward(self, inputs, targets):
        weights = torch.tensor(self.class_weights, dtype=torch.float).to(inputs.device)

        loss = F.cross_entropy(inputs, targets, weight=weights, reduction=self.reduction)
        return loss


class LossFactory:
    def __init__(self, cfg, samples_per_class, num_classes):
        self.loss_type = cfg['loss_type']  # 'cross_entropy', 'class_balanced_cross_entropy', 'class_balanced_focal', 'focal_loss'
        self.beta = cfg.get('beta', 0.999)
        self.gamma = cfg.get('gamma', 2.0)
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class

    def get_loss(self):
        if self.loss_type == 'CE_loss':
            return nn.CrossEntropyLoss()
        elif self.loss_type == 'class_balanced_CE_loss':
            return ClassBalancedCrossEntropyLoss(
                num_classes=self.num_classes,
                samples_per_class=self.samples_per_class,
                beta=self.beta
            )
        elif self.loss_type == 'class_balanced_Focal_loss':
            return ClassBalancedFocalLoss(
                num_classes=self.num_classes,
                samples_per_class=self.samples_per_class,
                beta=self.beta,
                gamma=self.gamma
            )
        elif self.loss_type == 'Focal_loss':
            return FocalLoss(
                num_classes=self.num_classes,
                gamma=self.gamma
            )
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")