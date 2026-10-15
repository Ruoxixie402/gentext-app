"""Verify reusable classroom helpers independently of full training."""

import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset

from helper_lib.data_loader import get_data_loader
from helper_lib.model import get_model
from helper_lib.trainer import train_model
from helper_lib.evaluator import evaluate_model
from helper_lib.utils import save_model, load_model


def test_evaluation_weights_partial_batch_and_disables_gradients():
    inputs = torch.tensor([[3., 0.], [0., 2.], [1., 0.]])
    labels = torch.tensor([0, 1, 1])
    loader = get_data_loader("unused", batch_size=2, train=False,
                             dataset=TensorDataset(inputs, labels))
    loss, accuracy = evaluate_model(nn.Identity(), loader, nn.CrossEntropyLoss())
    assert loss == pytest.approx(nn.functional.cross_entropy(inputs, labels).item())
    assert accuracy == pytest.approx(2 / 3)


def test_training_resets_mode_after_validation_and_updates_weights():
    torch.manual_seed(42)
    model = nn.Linear(2, 2)
    before = model.weight.detach().clone()
    dataset = TensorDataset(torch.ones(5, 2), torch.zeros(5, dtype=torch.long))
    loader = get_data_loader("unused", batch_size=2, dataset=dataset)
    epochs = []
    modes = []
    handle = model.register_forward_pre_hook(lambda module, inputs: modes.append(module.training))

    def callback(epoch, loss):
        epochs.append((epoch, loss))
        model.eval()

    result = train_model(model, loader, nn.CrossEntropyLoss(),
                         torch.optim.SGD(model.parameters(), lr=0.1),
                         epochs=2, on_epoch_end=callback)
    handle.remove()
    assert result is model
    assert modes == [True] * 6
    assert [epoch for epoch, loss in epochs] == [1, 2]
    assert epochs[-1][1] < epochs[0][1]
    assert not torch.equal(before, model.weight)


def test_checkpoint_roundtrip(tmp_path):
    torch.set_num_threads(2)
    model = get_model("CNN").eval()
    inputs = torch.randn(1, 3, 64, 64)
    path = tmp_path / "model.pt"
    save_model(model, path, epoch=1)
    loaded = load_model(path)
    with torch.inference_mode():
        torch.testing.assert_close(model(inputs), loaded(inputs), rtol=0, atol=0)
    assert loaded.training is False


def test_unknown_model_is_explicitly_rejected():
    with pytest.raises(ValueError, match="Unsupported model"):
        get_model("VAE")


def test_empty_evaluation_is_rejected():
    dataset = TensorDataset(torch.empty(0, 2), torch.empty(0, dtype=torch.long))
    loader = get_data_loader("unused", train=False, dataset=dataset)
    with pytest.raises(ValueError, match="empty"):
        evaluate_model(nn.Identity(), loader, nn.CrossEntropyLoss())
