from types import SimpleNamespace
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from sklearn.metrics import roc_auc_score

from src.tasks.metrics import auroc as batch_auroc
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


def _auroc_task():
    return BaseTask(
        dataset=SimpleNamespace(d_output=2),
        loss="binary_cross_entropy",
        metrics=["auroc"],
    )


def test_auroc_uses_cross_batch_rankings_not_average_batch_scores():
    task = _auroc_task()
    batches = [
        (torch.tensor([[0.2, 0.2], [0.3, 0.3]]), torch.tensor([[0, 0], [1, 1]])),
        (torch.tensor([[-0.2, -0.2], [-0.1, -0.1]]), torch.tensor([[0, 0], [1, 1]])),
    ]
    assert [batch_auroc(logits, y) for logits, y in batches] == [1.0, 1.0]

    metric = task.val_torchmetrics["auroc"]
    for logits, y in batches:
        assert task.metrics(logits, y) == {}
        metric.update(logits, y)

    assert task.metric_names == []
    assert task.torchmetric_names == ["auroc"]
    assert metric.compute().item() == pytest.approx(0.75)


def test_auroc_retains_single_class_batches():
    metric = _auroc_task().val_torchmetrics["auroc"]
    batches = [
        (torch.tensor([[-2.0, -1.0]]), torch.tensor([[0, 0]])),
        (torch.tensor([[2.0, 1.0]]), torch.tensor([[1, 1]])),
    ]
    assert [batch_auroc(logits, y) for logits, y in batches] == [None, None]

    for logits, y in batches:
        metric.update(logits, y)

    assert metric.compute().item() == pytest.approx(1.0)


def test_auroc_normalizes_logits_consistently_across_batches():
    metric = _auroc_task().val_torchmetrics["auroc"]
    batches = [
        (torch.tensor([[0.2, 0.2], [0.3, 0.3]]), torch.tensor([[0, 0], [1, 1]])),
        (torch.tensor([[-0.2, -0.2], [-0.1, -0.1]]), torch.tensor([[1, 1], [1, 1]])),
    ]
    for logits, y in batches:
        metric.update(logits, y)

    assert metric.compute().item() == pytest.approx(1 / 3)


def test_auroc_matches_full_dataset_sklearn_macro_average():
    task = BaseTask(
        dataset=SimpleNamespace(d_output=7),
        loss="binary_cross_entropy",
        metrics=["auroc"],
    )
    generator = torch.Generator().manual_seed(42)
    logits = torch.randn(23, 7, generator=generator)
    labels = torch.randint(0, 2, (23, 7), generator=generator)
    metric = task.val_torchmetrics["auroc"]
    for x, y in zip(logits.split(5), labels.split(5)):
        # Sequence decoders may leave an extra singleton output dimension.
        metric.update(x.unsqueeze(1), y.float())

    expected = roc_auc_score(labels.numpy(), logits.sigmoid().numpy(), average="macro")
    assert metric.compute().item() == pytest.approx(expected)


def test_auroc_excludes_labels_missing_a_class_over_the_epoch():
    metric = _auroc_task().val_torchmetrics["auroc"]
    metric.update(torch.tensor([[-1.0, 2.0], [1.0, 3.0]]), torch.tensor([[0, 1], [1, 1]]))

    with pytest.warns(UserWarning, match="No negative samples"):
        assert metric.compute().item() == pytest.approx(1.0)


def test_auroc_is_nan_if_no_label_has_both_classes():
    metric = _auroc_task().val_torchmetrics["auroc"]
    metric.update(torch.tensor([[-1.0, 2.0], [1.0, 3.0]]), torch.tensor([[0, 1], [0, 1]]))

    with pytest.warns(UserWarning):
        assert torch.isnan(metric.compute())


def test_auroc_reset_separates_epochs():
    metric = _auroc_task().val_torchmetrics["auroc"]
    logits = torch.tensor([[-1.0, -2.0], [1.0, 2.0]])
    labels = torch.tensor([[0, 0], [1, 1]])
    metric.update(logits, labels)
    assert metric.compute().item() == pytest.approx(1.0)

    metric.reset()
    metric.update(-logits, labels)
    assert metric.compute().item() == pytest.approx(0.0)


def _distributed_auroc_worker(rank, rendezvous):
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        metric = _auroc_task().val_torchmetrics["auroc"]
        # Neither rank has both classes locally, but both labels are valid globally.
        logits = torch.full((2, 2), 2.0 if rank else -2.0)
        metric.update(logits, torch.full((2, 2), rank))
        assert metric.compute().item() == pytest.approx(1.0)

        metric.reset()
        logits = torch.tensor([[0.2, 0.2], [0.3, 0.3]]) - rank * 0.4
        metric.update(logits, torch.tensor([[0, 0], [1, 1]]))
        assert metric.compute().item() == pytest.approx(0.75)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="Gloo is unavailable")
def test_auroc_aggregates_across_distributed_ranks(tmp_path):
    mp.spawn(_distributed_auroc_worker, args=(str(tmp_path / "auroc-rendezvous"),), nprocs=2)
