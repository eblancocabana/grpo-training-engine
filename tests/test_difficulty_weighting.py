import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

from src.data.gsm8k_loader import _make_sent_cache_key, create_grpo_dataloader
from src.grpo.algorithm import GRPOTrainer
from src.utils.config import Config, SENTConfig


def _configure_mock_tokenizer() -> Mock:
    tokenizer = Mock()
    tokenizer.name_or_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    tokenizer.chat_template = "<chat>"
    tokenizer.apply_chat_template.return_value = "formatted"
    tokenizer.return_value = {"input_ids": [1, 2], "attention_mask": [1, 1]}
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 1
    tokenizer.padding_side = "right"
    return tokenizer


def test_difficulty_weighting_defaults_round_trip():
    config = Config()

    assert config.grpo.difficulty_weighting_mode == "off"
    assert config.grpo.difficulty_weighting_min_weight == pytest.approx(1.0)
    assert config.grpo.difficulty_weighting_max_weight == pytest.approx(1.2)

    round_tripped = Config.from_dict(config.to_dict())
    assert round_tripped.grpo.difficulty_weighting_mode == "off"
    assert round_tripped.grpo.difficulty_weighting_min_weight == pytest.approx(1.0)
    assert round_tripped.grpo.difficulty_weighting_max_weight == pytest.approx(1.2)


def test_compute_grpo_loss_applies_sample_weights():
    trainer = GRPOTrainer(group_size=1, use_triton_kernels=False)
    policy_logits = torch.tensor(
        [
            [[2.0, 0.0]],
            [[2.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    old_policy_logits = policy_logits.clone()
    target_ids = torch.tensor([[0], [0]], dtype=torch.long)
    advantages = torch.tensor([1.0, 1.0], dtype=torch.float32)

    loss_unweighted, _ = trainer.compute_grpo_loss(
        policy_logits=policy_logits,
        advantages=advantages,
        old_policy_logits=old_policy_logits,
        target_ids=target_ids,
    )
    loss_weighted, _ = trainer.compute_grpo_loss(
        policy_logits=policy_logits,
        advantages=advantages,
        old_policy_logits=old_policy_logits,
        target_ids=target_ids,
        sample_weights=torch.tensor([1.0, 1.2], dtype=torch.float32),
    )

    assert loss_unweighted.item() == pytest.approx(-2.0, rel=1e-5)
    assert loss_weighted.item() == pytest.approx(-2.2, rel=1e-5)


def test_sent_rank_fraction_is_only_exposed_when_sent_order_is_active(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    tokenizer = _configure_mock_tokenizer()
    sent_config = SENTConfig()
    cache_path = tmp_path / "cache.json"
    cache_data = {
        "metadata": {
            "status": "complete",
            "config_hash": "abc",
            "sent_cache_key": _make_sent_cache_key(
                sent_config,
                tokenizer=tokenizer,
                max_prompt_length=2,
                model_id=tokenizer.name_or_path,
            ),
        },
        "indices": [0, 1, 2, 3],
        "entropies": [0.1, 0.2, 0.3, 0.4],
        "clusters": [[], [], [], []],
    }
    cache_path.write_text(json.dumps(cache_data), encoding="utf-8")

    fake_dataset = [
        {"question": f"Q{i}?", "answer": f"#### {i}"} for i in range(4)
    ]
    monkeypatch.setattr(
        "src.data.gsm8k_loader.load_dataset",
        lambda *args, **kwargs: fake_dataset,
    )

    sent_loader = create_grpo_dataloader(
        tokenizer=tokenizer,
        use_sent=True,
        batch_size=2,
        cache_path=str(cache_path),
        max_prompt_length=2,
        sent_config=sent_config,
        shuffle=False,
    )
    sent_batch = next(iter(sent_loader))
    assert sent_batch["sent_rank"].tolist() == [0, 1]
    torch.testing.assert_close(
        sent_batch["sent_rank_fraction"],
        torch.tensor([0.0, 1.0 / 3.0], dtype=torch.float32),
    )
    assert sent_batch["sent_rank_total"] == 4

    plain_loader = create_grpo_dataloader(
        tokenizer=tokenizer,
        use_sent=False,
        batch_size=2,
        max_prompt_length=2,
        shuffle=False,
    )
    plain_batch = next(iter(plain_loader))
    assert "sent_rank" not in plain_batch
    assert "sent_rank_fraction" not in plain_batch
    assert "sent_rank_total" not in plain_batch
