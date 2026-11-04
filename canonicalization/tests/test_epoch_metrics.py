from types import SimpleNamespace

import pytorch_lightning as pl
import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from src.tasks.tasks import BaseTask
from train import SequenceLightningModule


class MetricLoggingModule(SequenceLightningModule):
    """Use the real metric logging paths with synthetic logits and no backbone."""

    def __init__(self, compute_train_metrics=True):
        pl.LightningModule.__init__(self)
        self.config = OmegaConf.create({"train": {"compute_train_torchmetrics": compute_train_metrics}})
        self.save_hyperparameters(OmegaConf.create({
            "train": {"state": {"mode": None, "n_context": 0, "n_context_eval": 0}},
        }))
        self.task = BaseTask(
            dataset=SimpleNamespace(d_output=2),
            loss="binary_cross_entropy",
            metrics=["auroc"],
        )
        self.metrics = self.task.metrics
        self.loss = self.loss_val = self.task.loss
        self.train_torchmetrics = self.task.train_torchmetrics
        self.val_torchmetrics = self.task.val_torchmetrics
        self.test_torchmetrics = self.task.test_torchmetrics
        self.val_loader_names = ["val/0", "test/0"]
        self.test_loader_names = ["final/val/0", "final/test/0"]
        self.offset = torch.nn.Parameter(torch.zeros(()))

    def setup(self, stage=None):
        pass

    def forward(self, batch):
        logits, target = batch
        return logits + self.offset * 0, target, {}

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


def _loaders():
    logits = torch.tensor([[0.2, 0.2], [0.3, 0.3], [-0.2, -0.2], [-0.1, -0.1]])
    target = torch.tensor([[0, 0], [1, 1], [0, 0], [1, 1]])
    return [
        DataLoader(TensorDataset(x, target), batch_size=2)
        for x in (logits, -logits)
    ]


def _trainer(tmp_path, **kwargs):
    return pl.Trainer(
        default_root_dir=tmp_path,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        **kwargs,
    )


def test_auroc_logging_isolates_loaders_and_resets_between_validation_epochs(tmp_path):
    model = MetricLoggingModule()
    loaders = _loaders()
    # A lazily-created collection must not inherit the training metric's state.
    model.train_torchmetrics["auroc"].update(*next(iter(loaders[1])))
    trainer = _trainer(tmp_path)

    trainer.validate(model, dataloaders=loaders, verbose=False)
    assert trainer.callback_metrics["val/0/auroc"].item() == pytest.approx(0.75)
    assert trainer.callback_metrics["test/0/auroc"].item() == pytest.approx(0.25)

    trainer.validate(model, dataloaders=loaders[::-1], verbose=False)
    assert trainer.callback_metrics["val/0/auroc"].item() == pytest.approx(0.25)
    assert trainer.callback_metrics["test/0/auroc"].item() == pytest.approx(0.75)


@pytest.mark.parametrize("compute_train_metrics", [True, False])
def test_auroc_training_respects_metric_opt_out(tmp_path, compute_train_metrics):
    model = MetricLoggingModule(compute_train_metrics=compute_train_metrics)
    trainer = _trainer(tmp_path, max_epochs=2, num_sanity_val_steps=0)
    loaders = _loaders()

    trainer.fit(model, train_dataloaders=loaders[0], val_dataloaders=loaders)

    assert ("train/auroc" in trainer.callback_metrics) is compute_train_metrics
    if compute_train_metrics:
        assert trainer.callback_metrics["train/auroc"].item() == pytest.approx(0.75)
    assert trainer.callback_metrics["val/0/auroc"].item() == pytest.approx(0.75)
    assert trainer.callback_metrics["test/0/auroc"].item() == pytest.approx(0.25)
    assert not model.train_torchmetrics["auroc"].update_called


def test_auroc_standalone_test_uses_epoch_metrics(tmp_path):
    model = MetricLoggingModule()
    trainer = _trainer(tmp_path)

    trainer.test(model, dataloaders=_loaders(), verbose=False)

    assert trainer.callback_metrics["final/val/0/auroc"].item() == pytest.approx(0.75)
    assert trainer.callback_metrics["final/test/0/auroc"].item() == pytest.approx(0.25)
