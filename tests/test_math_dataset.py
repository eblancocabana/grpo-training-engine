import json
import os
import tempfile
from unittest.mock import Mock

import pytest

from src.data import gsm8k_loader
from src.data.math_dataset import (
    DATASET_SPECS,
    FILTERED_DATASET_DIRS,
    GRPOMathDataset,
    MathDatasetError,
    deterministic_split_indices,
    extract_answer,
    load_math_split_rows,
    normalize_math_row,
    safe_dataset_cache_stem,
    supported_dataset_names,
)
from src.data.sent_calculator import make_sent_metadata
from src.utils.config import SENTConfig, get_8gb_vram_config


def _fake_tokenizer():
    tokenizer = Mock()
    tokenizer.name_or_path = "tok"
    tokenizer.chat_template = "<chat>"
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 0
    tokenizer.padding_side = "right"
    tokenizer.apply_chat_template.side_effect = (
        lambda messages, tokenize=False, add_generation_prompt=True: messages[0]["content"]
    )
    tokenizer.side_effect = lambda text, **kwargs: {
        "input_ids": [1, 2],
        "attention_mask": [1, 1],
    }
    return tokenizer


def _rows(n=20):
    return [
        {
            "id": f"row-{i}",
            "problem": f"What is {i}+1?",
            "answer": str(i + 1),
            "solution": f"Compute it. \\boxed{{{i + 1}}}",
        }
        for i in range(n)
    ]


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _filtered_row(split, raw_index):
    return {
        "question": f"Q {split} {raw_index}",
        "answer": str(raw_index),
        "solution": f"S {raw_index}",
        "original_id": f"{split}-{raw_index}",
        "raw_index": raw_index,
        "split": split,
        "dataset_name": "dapo-open-rs-lenfilter-640",
        "source": "dapo-math-17k",
        "original_dataset_name": "dapo-math-17k",
        "length_filter_num_samples": 4,
        "length_filter_keep_min_finished": 4,
        "length_filter_response_lengths": [10, 11, 12, 13],
        "length_filter_finish_reasons": ["stop", "stop", "stop", "stop"],
        "length_filter_finished_count": 4,
        "length_filter_kept": True,
        "length_filter_generation_config": {},
    }


def test_deterministic_splits_for_new_datasets_are_stable_and_disjoint():
    for dataset_name in ("open-rs", "dapo-math-17k", "open-deepscaler"):
        assert dataset_name in DATASET_SPECS
        first = deterministic_split_indices(100, seed=42)
        second = deterministic_split_indices(100, seed=42)
        other_seed = deterministic_split_indices(100, seed=7)

        assert first == second
        assert first != other_seed
        assert set(first["train"]).isdisjoint(first["validation"])
        assert set(first["train"]).isdisjoint(first["test"])
        assert set(first["validation"]).isdisjoint(first["test"])
        assert len(first["train"]) + len(first["validation"]) + len(first["test"]) == 100


def test_dataset_registry_exposes_mixed_lenfilter():
    assert "dapo-open-rs-lenfilter-640" in supported_dataset_names()


def test_dataset_registry_exposes_open_rs_lenfilter():
    assert "open-rs-lenfilter-640" in supported_dataset_names()


def test_safe_dataset_cache_stems_for_filtered_dataset_names():
    assert (
        safe_dataset_cache_stem("dapo-open-rs-lenfilter-640")
        == "dapo_open_rs_lenfilter_640"
    )
    assert safe_dataset_cache_stem("open-rs-lenfilter-640") == "open_rs_lenfilter_640"


def test_filtered_local_jsonl_splits_load_without_resplitting(monkeypatch, tmp_path):
    dataset_dir = tmp_path / "dapo-open-rs_lenfilter_640"
    dataset_dir.mkdir()
    rows_by_split = {
        "train": [_filtered_row("train", 101), _filtered_row("train", 102)],
        "validation": [_filtered_row("validation", 201)],
        "test": [_filtered_row("test", 301)],
    }
    for split, rows in rows_by_split.items():
        _write_jsonl(dataset_dir / f"{split}.jsonl", rows)

    monkeypatch.setitem(
        FILTERED_DATASET_DIRS,
        "dapo-open-rs-lenfilter-640",
        str(dataset_dir),
    )

    train_rows, train_metadata = load_math_split_rows(
        "dapo-open-rs-lenfilter-640",
        "train",
        split_seed=999,
    )
    validation_rows, _ = load_math_split_rows(
        "dapo-open-rs-lenfilter-640",
        "validation",
        split_seed=999,
    )
    test_rows, _ = load_math_split_rows(
        "dapo-open-rs-lenfilter-640",
        "test",
        split_seed=999,
    )

    assert [row["raw_index"] for row in train_rows] == [101, 102]
    assert [row["raw_index"] for row in validation_rows] == [201]
    assert [row["raw_index"] for row in test_rows] == [301]
    assert train_metadata["filtered_local"] is True
    assert train_metadata["split_sizes"] == {"train": 2, "validation": 1, "test": 1}


@pytest.mark.parametrize(
    "row, expected_question, expected_answer",
    [
        ({"question": "Q?", "answer": "#### 42"}, "Q?", "42"),
        ({"problem": "P?", "gold_parsed": "13"}, "P?", "13"),
        ({"query": "R?", "reward_model": {"ground_truth": "7"}}, "R?", "7"),
        ({"instruction": "I?", "extra_info": {"answer": "5"}}, "I?", "5"),
        ({"prompt": [{"role": "user", "content": "Chat?"}], "solution": "\\boxed{9}"}, "Chat?", "9"),
    ],
)
def test_dataset_normalization_from_mocked_rows(row, expected_question, expected_answer):
    normalized = normalize_math_row(
        row,
        dataset_name="open-rs",
        split="train",
        raw_index=3,
    )

    assert normalized["question"] == expected_question
    assert normalized["answer"] == expected_answer
    assert normalized["dataset_name"] == "open-rs"
    assert normalized["split"] == "train"
    assert normalized["raw_index"] == 3


def test_answer_extraction_fails_loudly_when_missing():
    with pytest.raises(MathDatasetError, match="Could not extract"):
        extract_answer({"question": "No answer here"})


def test_strict_filtering_drops_invalid_rows_and_counts(monkeypatch):
    monkeypatch.setattr(
        "src.data.math_dataset.load_dataset",
        lambda *args, **kwargs: {
            "train": [
                {"question": "valid", "answer": "1"},
                {"question": "invalid"},
                {"question": "valid2", "answer": "2"},
            ]
        },
    )

    rows, metadata = load_math_split_rows(
        "open-rs",
        "train",
        split_seed=42,
        strict_filter_invalid=True,
    )

    assert metadata["invalid_rows_dropped"] >= 0
    assert all(row["answer"] for row in rows)


def test_train_validation_test_separation_with_mocked_hf_train_only(monkeypatch):
    raw_rows = _rows(30)
    monkeypatch.setattr(
        "src.data.math_dataset.load_dataset",
        lambda *args, **kwargs: {"train": raw_rows},
    )

    split_to_raw_indices = {}
    for split in ("train", "validation", "test"):
        rows, metadata = load_math_split_rows("open-rs", split, split_seed=42)
        split_to_raw_indices[split] = {row["raw_index"] for row in rows}
        assert metadata["split_sizes"][split] == len(rows)

    assert split_to_raw_indices["train"].isdisjoint(split_to_raw_indices["validation"])
    assert split_to_raw_indices["train"].isdisjoint(split_to_raw_indices["test"])
    assert split_to_raw_indices["validation"].isdisjoint(split_to_raw_indices["test"])


def test_sent_cache_metadata_compatibility_rejects_dataset_mismatch():
    config = get_8gb_vram_config()
    config.training.dataset_name = "open-rs"
    config.training.max_prompt_length = 128
    tokenizer = _fake_tokenizer()

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = os.path.join(tmpdir, "cache.json")
        data = {
            "metadata": make_sent_metadata(
                config,
                status="complete",
                tokenizer=tokenizer,
                max_prompt_length=128,
                dataset_name="open-rs",
                split="train",
            ),
            "indices": [0],
            "entropies": [0.1],
            "clusters": [[]],
        }
        with open(cache_path, "w") as f:
            json.dump(data, f)

        is_valid, _ = gsm8k_loader._validate_cache(
            cache_path,
            config=SENTConfig(),
            tokenizer=tokenizer,
            max_prompt_length=128,
            model_id=config.model.model_id,
            dataset_name="open-rs",
            split="train",
            split_seed=42,
            split_ratios=(0.90, 0.05, 0.05),
        )
        assert is_valid is True

        is_valid, msg = gsm8k_loader._validate_cache(
            cache_path,
            config=SENTConfig(),
            tokenizer=tokenizer,
            max_prompt_length=128,
            model_id=config.model.model_id,
            dataset_name="dapo-math-17k",
            split="train",
            split_seed=42,
            split_ratios=(0.90, 0.05, 0.05),
        )
        assert is_valid is False
        assert "mismatch" in msg


def test_sent_ordering_only_applies_to_train_split(monkeypatch):
    monkeypatch.setattr(
        "src.data.math_dataset.load_dataset",
        lambda *args, **kwargs: {"train": _rows(8)},
    )
    tokenizer = _fake_tokenizer()

    dataloader = gsm8k_loader.create_grpo_dataloader(
        tokenizer=tokenizer,
        dataset_name="open-rs",
        split="validation",
        batch_size=1,
        use_sent=True,
    )

    assert dataloader.uses_sent_curriculum is False
    assert isinstance(dataloader.dataset, GRPOMathDataset)


def test_dataset_name_gsm8k_generic_path_still_works(monkeypatch):
    monkeypatch.setattr(
        "src.data.math_dataset.load_dataset",
        lambda *args, **kwargs: {
            "train": [{"question": "Q?", "answer": "#### 42"}],
            "test": [{"question": "T?", "answer": "#### 24"}],
        },
    )
    dataset = GRPOMathDataset(
        tokenizer=_fake_tokenizer(),
        dataset_name="gsm8k",
        split="train",
        max_prompt_length=8,
    )

    assert len(dataset) == 1
    assert dataset[0]["question"] == "Q?"
    assert dataset[0]["answer"] == "42"


def test_tiny_two_stage_sent_integration(monkeypatch):
    monkeypatch.setattr(
        "src.data.math_dataset.load_dataset",
        lambda *args, **kwargs: {"train": _rows(8)},
    )
    tokenizer = _fake_tokenizer()

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = os.path.join(tmpdir, "cache.json")
        config = get_8gb_vram_config()
        config.training.dataset_name = "open-rs"
        metadata = make_sent_metadata(
            config,
            status="complete",
            tokenizer=tokenizer,
            max_prompt_length=8,
            dataset_name="open-rs",
            split="train",
        )
        cache = {
            "metadata": metadata,
            "indices": [0, 1, 2, 3, 4, 5],
            "entropies": [0.1, 0.2, 0.3, 0.7, 0.8, 0.9],
            "clusters": [[] for _ in range(6)],
        }
        with open(cache_path, "w") as f:
            json.dump(cache, f)

        dataloader = gsm8k_loader.create_grpo_dataloader(
            tokenizer=tokenizer,
            dataset_name="open-rs",
            split="train",
            batch_size=1,
            max_prompt_length=8,
            use_sent=True,
            cache_path=cache_path,
            sent_config=SENTConfig(),
            num_stages=2,
            model_id=config.model.model_id,
        )

        assert dataloader.uses_sent_curriculum is True
        dataloader.dataset.set_stage(1)
        assert len(dataloader.dataset) == 3
        assert dataloader.dataset[0]["sent_rank"] == 0
        dataloader.dataset.set_stage(2)
        assert len(dataloader.dataset) == 3
        assert dataloader.dataset[0]["sent_rank"] == 3
