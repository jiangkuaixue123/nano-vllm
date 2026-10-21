import json

import pytest
import torch
from safetensors.torch import save_file

from nanovllm.qwen38.loader import load_weights


def native_model():
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.embed_tokens = torch.nn.Embedding(5, 3, dtype=torch.bfloat16)
    model.lm_head = torch.nn.Linear(3, 5, bias=False, dtype=torch.bfloat16)
    for parameter in model.parameters():
        parameter.data.fill_(-7)
    return model


def checkpoint(root, tensors=None, index=None):
    if tensors is None:
        tensors = {
            "model.language_model.embed_tokens.weight": torch.arange(15).reshape(5, 3).bfloat16(),
            "lm_head.weight": torch.ones(5, 3, dtype=torch.bfloat16),
            "model.visual.patch_embed.proj.weight": torch.zeros(2),
            "mtp.fc.weight": torch.zeros(2),
        }
    save_file(tensors, root / "model.safetensors")
    if index is None:
        index = {key: "model.safetensors" for key in tensors}
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": index}))
    return tensors, index


def test_load_complete_text_and_report_explicit_exclusions(tmp_path):
    tensors, _ = checkpoint(tmp_path)
    model = native_model()
    report = load_weights(model, tmp_path)
    assert report == {"loaded_parameters": 2, "excluded_tensors": {"vision": 1, "mtp": 1},
                      "scope": "text_only", "dtype": "bfloat16"}
    assert torch.equal(model.model.embed_tokens.weight, tensors["model.language_model.embed_tokens.weight"])
    assert torch.equal(model.lm_head.weight, tensors["lm_head.weight"])


@pytest.mark.parametrize("corruption", ["missing_parameter", "unknown_namespace", "wrong_shape",
                                        "wrong_dtype", "unindexed_tensor", "missing_tensor"])
def test_bad_checkpoint_rejected_before_parameter_mutation(tmp_path, corruption):
    tensors, index = checkpoint(tmp_path)
    if corruption == "missing_parameter":
        del tensors["lm_head.weight"]
        del index["lm_head.weight"]
    elif corruption == "unknown_namespace":
        tensors["arbitrary.weight"] = torch.zeros(1)
        index["arbitrary.weight"] = "model.safetensors"
    elif corruption == "wrong_shape":
        tensors["lm_head.weight"] = torch.ones(1, dtype=torch.bfloat16)
    elif corruption == "wrong_dtype":
        tensors["lm_head.weight"] = tensors["lm_head.weight"].float()
    elif corruption == "unindexed_tensor":
        tensors["model.visual.extra"] = torch.zeros(1)
    elif corruption == "missing_tensor":
        del tensors["mtp.fc.weight"]
    checkpoint(tmp_path, tensors, index)
    model = native_model()
    with pytest.raises(ValueError):
        load_weights(model, tmp_path)
    assert all(torch.all(parameter == -7) for parameter in model.parameters())


def test_duplicate_index_key_is_rejected(tmp_path):
    checkpoint(tmp_path)
    (tmp_path / "model.safetensors.index.json").write_text(
        '{"weight_map":{"lm_head.weight":"a.safetensors","lm_head.weight":"b.safetensors"}}'
    )
    with pytest.raises(ValueError, match="Duplicate"):
        load_weights(native_model(), tmp_path)


def test_path_escape_rejected(tmp_path):
    _, index = checkpoint(tmp_path)
    index["lm_head.weight"] = "../outside.safetensors"
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": index}))
    with pytest.raises(ValueError, match="canonical"):
        load_weights(native_model(), tmp_path)


def test_symlink_escape_rejected(tmp_path):
    root = tmp_path / "inside"
    root.mkdir()
    checkpoint(root)
    (root / "model.safetensors").rename(tmp_path / "outside.safetensors")
    (root / "model.safetensors").symlink_to(tmp_path / "outside.safetensors")
    with pytest.raises(ValueError, match="escapes"):
        load_weights(native_model(), root)


def test_wrong_destination_dtype_rejected(tmp_path):
    checkpoint(tmp_path)
    with pytest.raises(ValueError, match="BF16"):
        load_weights(native_model().float(), tmp_path)
