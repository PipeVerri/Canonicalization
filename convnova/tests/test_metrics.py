from types import SimpleNamespace

import pytest
import torch

from src.tasks.metrics import mcc as batch_mcc
from src.tasks.tasks import BaseTask


def _mcc_task():
    return BaseTask(
        dataset=SimpleNamespace(d_output=2),
        loss="cross_entropy",
        metrics=["mcc"],
    )


def test_mcc_accumulates_confusion_matrix_across_batches():
    task = _mcc_task()
    batches = [
        (torch.tensor([[2.0, 0.0], [2.0, 0.0]]), torch.tensor([0, 0])),
        (torch.tensor([[0.0, 2.0], [0.0, 2.0]]), torch.tensor([1, 1])),
    ]

    # Each single-class batch has MCC 0, while the combined predictions are perfect.
    assert [batch_mcc(logits, y) for logits, y in batches] == [0.0, 0.0]

    metric = task.val_torchmetrics["mcc"]
    for logits, y in batches:
        metric.update(logits, y)

    assert task.metric_names == []
    assert task.torchmetric_names == ["mcc"]
    assert metric.compute().item() == pytest.approx(1.0)


def test_mcc_ignores_padded_labels():
    metric = _mcc_task().val_torchmetrics["mcc"]
    batches = [
        (torch.tensor([[2.0, 0.0], [0.0, 2.0]]), torch.tensor([0, -100])),
        (torch.tensor([[0.0, 2.0], [2.0, 0.0]]), torch.tensor([1, -100])),
    ]

    for logits, y in batches:
        metric.update(logits, y)

    assert metric.compute().item() == pytest.approx(1.0)
