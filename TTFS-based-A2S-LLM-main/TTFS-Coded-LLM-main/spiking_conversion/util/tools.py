import torch

from typing import Any, Sequence

__all__ = ["AverageMeter", "accuracy"]


class AverageMeter:
    """Computes and stores the average and current value."""

    sum: Any
    count: int

    def __init__(self):
        self.sum = None
        self.count = 0

    def update(self, val, num: int):
        if self.count == 0:
            self.sum = val * num
        else:
            self.sum += val * num
        self.count += num

    @property
    def mean(self):
        return None if self.count == 0 else self.sum / self.count


def accuracy(
    output: torch.Tensor,
    target: torch.Tensor,
    topk: Sequence[int] = (1,),
) -> list[float]:
    """
    Compute top-k accuracies for multiple `k`s at once.

    :param output: prediction made by model, shape `[batch_size, num_classes]`
    :type output: torch.Tensor
    :param target: ground-truth label, shape `[batch_size]`
    :type target: torch.Tensor
    :param topk: a list of `k`s, for each of them we should compute the top-`k` accuracy
    :type topk: torch.Tensor
    :returns: a list of accuracies, each in range [0, 1]
    :rtype: list[float]
    """
    maxk = max(topk)
    batch_size = target.shape[0]

    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)  # pred: [batch_size, maxk]
    correct = pred == target.reshape(-1, 1)

    res = []
    for k in topk:
        correct_k = correct[:, :k].sum().item()
        res.append(correct_k / batch_size)
    return res


def checked_clip(x: torch.Tensor) -> torch.Tensor:
    assert torch.all((x >= 0) & (x <= 1))
    return x
