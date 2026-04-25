from types import SimpleNamespace

import torch

from scripts.filter_dataset_by_response_length_vllm import (
    compute_sent_from_completions,
    default_sent_cache_path,
    format_eta,
    should_keep_prompt,
    write_sent_cache_from_progress,
)


def test_filtering_decision_keeps_when_four_of_four_finish_below_strict_cap():
    kept, finished_count = should_keep_prompt(
        [100, 200, 639, 300],
        ["stop", "stop", "eos_token", "stop"],
        max_response_length=640,
        keep_min_finished=4,
    )

    assert kept is True
    assert finished_count == 4


def test_filtering_decision_rejects_when_fewer_than_keep_min_finished_finish():
    kept, finished_count = should_keep_prompt(
        [100, 200, 639, 640],
        ["stop", "stop", "stop", "stop"],
        max_response_length=640,
        keep_min_finished=4,
    )

    assert kept is False
    assert finished_count == 3


def test_filtering_decision_treats_length_finish_reason_as_not_finished():
    kept, finished_count = should_keep_prompt(
        [100, 120, 140, 200],
        ["stop", "length", "length", "stop"],
        max_response_length=640,
        keep_min_finished=4,
    )

    assert kept is False
    assert finished_count == 2


def test_eta_formatter_returns_remaining_and_finish_strings():
    remaining, finish = format_eta(processed=5, total=10, elapsed_seconds=10.0)

    assert remaining == "00:00:10"
    assert finish != "unknown"


def test_default_sent_cache_path_uses_filtered_640_stem():
    assert (
        default_sent_cache_path("dapo-open-rs-lenfilter-640")
        == "data/cache/dapo_open_rs_lenfilter_640_sent_sorted.pt"
    )


def test_compute_sent_from_completions_reuses_filter_samples():
    class Verifier:
        def extract_final_answer(self, text):
            return text.rsplit("=", 1)[-1].strip()

        def normalize_number(self, text):
            return float(text)

    entropy, clusters = compute_sent_from_completions(
        Verifier(),
        ["x = 1", "answer = 1", "x = 2", "answer = 2"],
    )

    assert entropy > 0
    assert sorted(cluster["count"] for cluster in clusters) == [2, 2]


def test_write_sent_cache_from_kept_train_progress(tmp_path):
    class Tokenizer:
        name_or_path = "tok"
        chat_template = ""

    args = SimpleNamespace(
        dataset_name="dapo-math-17k",
        output_dataset_name="dapo-open-rs-lenfilter-640",
        max_response_length=640,
        max_prompt_length=512,
        model_id="model",
        split_seed=42,
        num_samples=4,
        temperature=1.0,
        seed=1,
    )
    progress_rows = [
        {
            "original_id": "a",
            "length_filter_kept": True,
            "length_filter_sent_entropy": 0.5,
            "length_filter_sent_clusters": [{"answer": "1", "count": 4}],
        },
        {
            "original_id": "b",
            "length_filter_kept": False,
        },
        {
            "original_id": "c",
            "length_filter_kept": True,
            "length_filter_sent_entropy": 0.1,
            "length_filter_sent_clusters": [{"answer": "2", "count": 4}],
        },
    ]
    cache_path = tmp_path / "sent.pt"

    summary = write_sent_cache_from_progress(
        progress_rows=progress_rows,
        cache_path=str(cache_path),
        args=args,
        tokenizer=Tokenizer(),
    )
    cache = torch.load(cache_path, weights_only=False)

    assert summary["sent_cache_entries"] == 2
    assert cache["metadata"]["dataset_name"] == "dapo-open-rs-lenfilter-640"
    assert cache["indices"] == [1, 0]
    assert cache["example_ids"] == ["c", "a"]
