import pytorch_lightning as pl
import pytest
import torch
from omegaconf import OmegaConf

import train as training


class CheckpointModule(training.SequenceLightningModule):
    """Exercise the real state loader without constructing datasets or ConvNova."""

    def __init__(self, config):
        pl.LightningModule.__init__(self)
        self.save_hyperparameters(config, logger=False)
        self.model = torch.nn.Linear(1, 1, bias=False)
        self.decoder = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.model.weight.fill_(1)
            self.decoder.weight.fill_(2)


def _config(test=False):
    return OmegaConf.create({
        "train": {
            "seed": None,
            "pretrained_model_path": None,
            "pretrained_model_state_hook": {"_name_": "load_backbone"},
            "validate_at_start": False,
            "ckpt": None,
            "test": test,
        },
    })


@pytest.mark.parametrize("test", [False, True])
@pytest.mark.parametrize("strict", [False, True])
def test_post_fit_evaluation_restores_best_backbone_and_decoder(monkeypatch, test, strict):
    config = _config(test)
    model = CheckpointModule(config)

    class Trainer:
        def fit(self, module):
            assert module.hparams.train.pretrained_model_state_hook._name_ == "load_backbone"
            # The best epoch and the final epoch have different weights in both parts.
            self.best_state = {
                "model.weight": torch.full_like(module.model.weight, 10),
                "decoder.weight": torch.full_like(module.decoder.weight, 20),
            }
            with torch.no_grad():
                module.model.weight.fill_(30)
                module.decoder.weight.fill_(40)

        def _evaluate(self, module, ckpt_path, stage):
            assert ckpt_path == "best"
            module.load_state_dict(self.best_state, strict=strict)
            assert module.model.weight.item() == 10
            assert module.decoder.weight.item() == 20
            return stage

        def validate(self, module, ckpt_path):
            return self._evaluate(module, ckpt_path, "validate")

        def test(self, module, ckpt_path):
            return self._evaluate(module, ckpt_path, "test")

    monkeypatch.setattr(training, "SequenceLightningModule", lambda config: model)
    monkeypatch.setattr(training, "create_trainer", lambda config: Trainer())

    assert training.train(config) == ("test" if test else "validate")
    # The temporary override must not change a configuration reused by another run.
    assert model.hparams.train.pretrained_model_state_hook._name_ == "load_backbone"
    assert config.train.pretrained_model_state_hook._name_ == "load_backbone"


def test_pretraining_transfer_still_keeps_the_existing_downstream_decoder():
    model = CheckpointModule(_config())
    pretrained_state = {
        "model.weight": torch.full_like(model.model.weight, 10),
        "decoder.weight": torch.full_like(model.decoder.weight, 20),
    }

    model.load_state_dict(pretrained_state, strict=False)

    assert model.model.weight.item() == 10
    assert model.decoder.weight.item() == 2
