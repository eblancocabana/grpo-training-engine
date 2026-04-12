import types
from unittest.mock import patch

import torch


def test_load_4bit_engine_uses_model_config_knobs_without_quantization():
    from src.core import model_loader

    fake_tokenizer = types.SimpleNamespace(
        pad_token=None,
        eos_token="<eos>",
        padding_side="right",
    )
    fake_model = types.SimpleNamespace(
        device="cpu",
        parameters=lambda: [types.SimpleNamespace(requires_grad=True)],
    )

    with patch.object(
        model_loader, "BitsAndBytesConfig", autospec=True
    ) as bnb_cls, patch.object(
        model_loader.AutoTokenizer, "from_pretrained", return_value=fake_tokenizer
    ) as tok_loader, patch.object(
        model_loader.AutoModelForCausalLM,
        "from_pretrained",
        return_value=fake_model,
    ) as model_loader_fn:
        model, tokenizer = model_loader.load_4bit_engine(
            "custom-model",
            load_in_4bit=False,
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_quant_type="fp4",
            bnb_4bit_use_double_quant=False,
            attn_implementation="eager",
            device_map="cpu",
        )

    assert model is fake_model
    assert tokenizer is fake_tokenizer
    tok_loader.assert_called_once_with("custom-model")
    bnb_cls.assert_not_called()
    _, kwargs = model_loader_fn.call_args
    assert kwargs["torch_dtype"] is torch.float16
    assert kwargs["attn_implementation"] == "eager"
    assert kwargs["device_map"] == "cpu"
    assert "quantization_config" not in kwargs
    assert fake_tokenizer.pad_token == fake_tokenizer.eos_token
    assert fake_tokenizer.padding_side == "left"


def test_load_4bit_engine_builds_quantization_config_from_model_settings():
    from src.core import model_loader

    fake_tokenizer = types.SimpleNamespace(
        pad_token="<pad>",
        eos_token="<eos>",
        padding_side="right",
    )
    fake_model = types.SimpleNamespace(
        device="cuda:0",
        parameters=lambda: [types.SimpleNamespace(requires_grad=True)],
    )

    with patch.object(
        model_loader, "BitsAndBytesConfig", autospec=True, return_value="bnb-config"
    ) as bnb_cls, patch.object(
        model_loader.AutoTokenizer, "from_pretrained", return_value=fake_tokenizer
    ), patch.object(
        model_loader.AutoModelForCausalLM,
        "from_pretrained",
        return_value=fake_model,
    ) as model_loader_fn:
        model_loader.load_4bit_engine(
            "custom-model",
            load_in_4bit=True,
            bnb_4bit_compute_dtype="bf16",
            bnb_4bit_quant_type="fp4",
            bnb_4bit_use_double_quant=False,
            attn_implementation="sdpa",
            device_map="balanced",
        )

    bnb_cls.assert_called_once_with(
        load_in_4bit=True,
        bnb_4bit_quant_type="fp4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )
    _, kwargs = model_loader_fn.call_args
    assert kwargs["quantization_config"] == "bnb-config"
    assert kwargs["torch_dtype"] is torch.bfloat16
    assert kwargs["device_map"] == "balanced"
    assert kwargs["attn_implementation"] == "sdpa"
