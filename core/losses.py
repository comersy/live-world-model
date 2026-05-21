"""
core / losses.py

Shared loss functions for training. Kept in one place so offline and any other
trainer use exactly the same definitions.
"""


def mse_loss(pred, target):
    return ((pred - target) ** 2).mean()


def motion_weighted_loss(pred, target, prev, motion_weight):
    """
    MSE weighted by where real motion happens between prev and target.

        weight = 1 + motion_weight * |target - prev|
        loss   = mean(weight * (pred - target)^2) / mean(weight)

    Moving pixels dominate the loss; the static background barely counts. Use a
    large motion_weight to force the model to focus on the moving region.
    Set motion_weight = 0 to recover plain MSE.
    """
    motion = (target - prev).abs()
    weight = 1.0 + motion_weight * motion
    sq = (pred - target) ** 2
    return (weight * sq).mean() / weight.mean()