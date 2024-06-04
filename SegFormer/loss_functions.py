import torch.nn as nn
import torch

class JaccardLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(JaccardLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = torch.softmax(preds, dim=1)[:, 1]  # Use the softmax output for the lesion class
        targets = targets.float()
        intersection = (preds * targets).sum(dim=(1, 2))
        union = preds.sum(dim=(1, 2)) + targets.sum(dim=(1, 2)) - intersection
        jaccard = (intersection + self.smooth) / (union + self.smooth)
        return 1 - jaccard.mean()

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class CombinedLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss(smooth)
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, preds, targets):
        dice = self.dice_loss(preds, targets)
        ce = self.ce_loss(preds, targets)
        return dice + ce

def dice_coefficient(preds, targets, smooth=1e-6):
    # Convert predictions to binary format
    preds = preds.argmax(dim=1).float()
    targets = targets.float()

    intersection = (preds * targets).sum(dim=(1, 2))
    union = preds.sum(dim=(1, 2)) + targets.sum(dim=(1, 2))

    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean()

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = torch.softmax(preds, dim=1)[:, 1]  # Use the softmax output for the lesion class
        targets = targets.float()
        intersection = (preds * targets).sum(dim=(1, 2))
        union = preds.sum(dim=(1, 2)) + targets.sum(dim=(1, 2))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()