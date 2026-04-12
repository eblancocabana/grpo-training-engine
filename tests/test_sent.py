import pytest
import json
import os
import tempfile
import torch
from unittest.mock import Mock, MagicMock, patch

from src.data.sent_calculator import SemanticEntropyCalculator
from src.data.gsm8k_loader import (
    GRPOGSM8KDataset,
    SENTGSM8KDataset,
    _make_sent_cache_key,
    format_grpo_prompt,
    _validate_cache,
    create_grpo_dataloader,
)
from src.grpo.verifier import RuleBasedVerifier
from src.utils.config import SENTConfig, Config, get_8gb_vram_config


def _configure_mock_tokenizer(mock_tokenizer: Mock, template: str = "<chat>") -> Mock:
    mock_tokenizer.name_or_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    mock_tokenizer.chat_template = template
    return mock_tokenizer


def _sent_metadata_for(
    sent_config: SENTConfig,
    tokenizer: Mock,
    max_prompt_length: int,
    model_id: str | None = None,
) -> dict:
    if model_id is None:
        tokenizer_model_id = getattr(tokenizer, "name_or_path", None)
        if isinstance(tokenizer_model_id, str):
            model_id = tokenizer_model_id
    return {
        "status": "complete",
        "config_hash": "abc",
        "sent_cache_key": _make_sent_cache_key(
            sent_config,
            tokenizer=tokenizer,
            max_prompt_length=max_prompt_length,
            model_id=model_id,
        ),
    }


class TestSemanticEntropyCalculator:
    """Tests for SemanticEntropyCalculator."""

    def test_cluster_by_answer(self):
        """Test clustering by answer equivalence."""
        verifier = RuleBasedVerifier()
        
        mock_model = Mock()
        mock_tokenizer = _configure_mock_tokenizer(Mock())
        config = get_8gb_vram_config()
        
        calc = SemanticEntropyCalculator(mock_model, mock_tokenizer, verifier, config)
        
        responses = [
            "The answer is \\boxed{42}.",  # answer = 42
            "So the final answer is 42.",  # answer = 42
            "The answer is \\boxed{42}.",  # answer = 42
            "The answer is \\boxed{100}.", # answer = 100
        ]
        
        clusters = calc._cluster_by_answer(responses)
        
        assert len(clusters) == 2
        
        cluster_42 = [c for c in clusters if c["answer"] == 42]
        assert len(cluster_42) == 1
        assert len(cluster_42[0]["indices"]) == 3
        
        cluster_100 = [c for c in clusters if c["answer"] == 100]
        assert len(cluster_100) == 1
        assert len(cluster_100[0]["indices"]) == 1

    def test_compute_semantic_entropy_single_cluster(self):
        """Test entropy calculation with single cluster (should be ~0)."""
        verifier = RuleBasedVerifier()
        
        mock_model = Mock()
        mock_tokenizer = _configure_mock_tokenizer(Mock())
        config = get_8gb_vram_config()
        
        calc = SemanticEntropyCalculator(mock_model, mock_tokenizer, verifier, config)
        
        clusters = [
            {"answer": 42, "indices": [0, 1, 2]}
        ]
        
        entropy = calc._compute_semantic_entropy(clusters, 3)
        
        assert entropy < 0.001

    def test_compute_semantic_entropy_two_clusters(self):
        """Test entropy calculation with two equal clusters."""
        verifier = RuleBasedVerifier()
        
        mock_model = Mock()
        mock_tokenizer = Mock()
        config = get_8gb_vram_config()
        
        calc = SemanticEntropyCalculator(mock_model, mock_tokenizer, verifier, config)
        
        clusters = [
            {"answer": 42, "indices": [0]},
            {"answer": 100, "indices": [1]},
        ]
        
        entropy = calc._compute_semantic_entropy(clusters, 2)
        
        # Two clusters with 1 sample each out of 2 total.
        # P(C1) = P(C2) = 0.5
        # H = -2 * (0.5 * log(0.5)) = log(2) ≈ 0.693
        import math
        expected = math.log(2)
        
        assert abs(entropy - expected) < 0.01

    def test_compute_semantic_entropy_empty(self):
        """Test entropy with empty clusters."""
        verifier = RuleBasedVerifier()
        
        mock_model = Mock()
        mock_tokenizer = Mock()
        config = get_8gb_vram_config()
        
        calc = SemanticEntropyCalculator(mock_model, mock_tokenizer, verifier, config)
        
        entropy = calc._compute_semantic_entropy([], 0)
        
        assert entropy == 0.0

    def test_save_load_cache(self):
        """Test cache save and load."""
        verifier = RuleBasedVerifier()
        
        mock_model = Mock()
        mock_tokenizer = Mock()
        config = get_8gb_vram_config()
        
        calc = SemanticEntropyCalculator(mock_model, mock_tokenizer, verifier, config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "cache.json")
            
            data = {
                "metadata": calc._make_metadata(status="complete"),
                "indices": [0, 1, 2],
                "entropies": [0.1, 0.5, 0.9],
                "clusters": [[], [], []]
            }
            
            calc.save_cache(cache_path, data)
            
            loaded = calc.load_cache(cache_path)
            
            assert loaded["metadata"]["status"] == "complete"
            assert loaded["indices"] == [0, 1, 2]
            assert loaded["entropies"] == [0.1, 0.5, 0.9]

    def test_make_metadata(self):
        """Test metadata creation."""
        verifier = RuleBasedVerifier()
        
        mock_model = Mock()
        mock_tokenizer = Mock()
        config = get_8gb_vram_config()
        
        calc = SemanticEntropyCalculator(mock_model, mock_tokenizer, verifier, config)
        
        meta = calc._make_metadata(status="test_status")

        assert meta["version"] == "sent_v1"
        assert meta["status"] == "test_status"
        assert "config_hash" in meta
        assert "sent_cache_key" in meta
        assert "created_at" in meta

    def test_sent_uses_formatted_prompt_and_prompt_cap(self):
        verifier = RuleBasedVerifier()
        config = get_8gb_vram_config()
        config.training.max_prompt_length = 17

        mock_model = Mock()
        mock_model.eval.return_value = None
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 4
        mock_tokenizer.apply_chat_template.return_value = "<chat>question</chat>"
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
        }
        mock_tokenizer.decode.return_value = "answer"

        calc = SemanticEntropyCalculator(mock_model, mock_tokenizer, verifier, config)
        calc.compute_entropy_for_query("question", num_samples=1)

        mock_tokenizer.apply_chat_template.assert_called_once()
        _, kwargs = mock_tokenizer.call_args
        assert kwargs["max_length"] == 17

    def test_format_grpo_prompt_falls_back_when_chat_template_render_fails(self):
        tokenizer = Mock()
        tokenizer.chat_template = "<chat>"
        tokenizer.apply_chat_template.side_effect = ValueError("no chat template")

        prompt = format_grpo_prompt(tokenizer, "question")

        assert prompt == "question"


class TestSENTConfig:
    """Tests for SENTConfig."""

    def test_sent_config_defaults(self):
        """Test SENTConfig default values."""
        sent = SENTConfig()
        
        assert sent.enabled is True
        assert sent.num_samples == 4
        assert sent.temperature == 1.0
        assert sent.cache_path == "data/cache/gsm8k_sent_sorted.pt"
        assert sent.checkpoint_interval == 100
        assert sent.curriculum_stages == 2
        assert sent.resume_from_checkpoint is True
        assert sent.seed is None

    def test_config_includes_sent(self):
        """Test Config includes sent field."""
        config = Config()
        
        assert hasattr(config, "sent")
        assert isinstance(config.sent, SENTConfig)

    def test_get_8gb_vram_config_sent(self):
        """Test get_8gb_vram_config has SENT values."""
        config = get_8gb_vram_config()
        
        assert config.sent.enabled is True
        assert config.sent.num_samples == 4
        assert config.sent.cache_path == "data/cache/gsm8k_sent_sorted.pt"

    def test_config_serialization(self):
        """Test Config to_dict and from_dict with sent."""
        config = get_8gb_vram_config()
        
        config_dict = config.to_dict()
        
        assert "sent" in config_dict
        assert config_dict["sent"]["enabled"] is True
        
        new_config = Config.from_dict(config_dict)
        
        assert new_config.sent.enabled is True
        assert new_config.sent.num_samples == 4


class TestValidateCache:
    """Tests for cache validation."""

    def test_validate_cache_nonexistent(self):
        """Test validation fails for nonexistent cache."""
        is_valid, msg = _validate_cache("/nonexistent/path.pt")
        
        assert is_valid is False
        assert "does not exist" in msg

    def test_validate_cache_invalid_status(self):
        """Test validation fails for incomplete cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "cache.json")
            
            data = {
                "metadata": {"status": "in_progress"},
                "indices": [0, 1, 2],
                "entropies": [0.1, 0.2, 0.3],
                "clusters": [[], [], []]
            }
            
            import json
            with open(cache_path, 'w') as f:
                json.dump(data, f)
            
            is_valid, msg = _validate_cache(cache_path)
            
            assert is_valid is False
            assert "complete" in msg

    def test_validate_cache_valid(self):
        """Test validation passes for complete cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "cache.json")
            config = SENTConfig()
            tokenizer = Mock()
            tokenizer.name_or_path = "tok"
            tokenizer.chat_template = "<chat>"
            
            data = {
                "metadata": {
                    "status": "complete",
                    "config_hash": "abc123",
                    "sent_cache_key": _make_sent_cache_key(
                        config,
                        tokenizer=tokenizer,
                        max_prompt_length=128,
                        model_id="model-a",
                    ),
                },
                "indices": [0, 1, 2],
                "entropies": [0.1, 0.2, 0.3],
                "clusters": [[], [], []]
            }
            
            import json
            with open(cache_path, 'w') as f:
                json.dump(data, f)
            
            is_valid, msg = _validate_cache(
                cache_path,
                config=config,
                tokenizer=tokenizer,
                max_prompt_length=128,
                model_id="model-a",
            )
            
            assert is_valid is True

    def test_validate_cache_rejects_model_id_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "cache.json")
            tokenizer = _configure_mock_tokenizer(Mock())
            data = {
                "metadata": {
                    "status": "complete",
                    "config_hash": "abc123",
                    "sent_cache_key": _make_sent_cache_key(
                        SENTConfig(),
                        tokenizer=tokenizer,
                        max_prompt_length=128,
                        model_id="model-a",
                    ),
                },
                "indices": [0],
                "entropies": [0.1],
                "clusters": [[]],
            }
            with open(cache_path, "w") as f:
                json.dump(data, f)

            is_valid, msg = _validate_cache(
                cache_path,
                config=SENTConfig(),
                tokenizer=tokenizer,
                max_prompt_length=128,
                model_id="model-b",
            )

            assert is_valid is False
            assert "mismatch" in msg

    def test_validate_cache_rejects_legacy_hash_only_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "cache.json")
            data = {
                "metadata": {"status": "complete", "config_hash": "abc123"},
                "indices": [0],
                "entropies": [0.1],
                "clusters": [[]],
            }
            with open(cache_path, "w") as f:
                json.dump(data, f)

            tokenizer = _configure_mock_tokenizer(Mock())
            is_valid, msg = _validate_cache(
                cache_path,
                config=SENTConfig(),
                tokenizer=tokenizer,
                max_prompt_length=128,
            )

            assert is_valid is False
            assert "compatibility key" in msg


class TestSENTGSM8KDataset:
    """Tests for SENTGSM8KDataset."""

    def test_dataset_without_cache_raises(self):
        """Test dataset raises error when cache doesn't exist."""
        with patch("src.data.gsm8k_loader.load_dataset") as mock_load:
            mock_load.return_value = [{"question": "test?", "answer": "42"}]
            
            with pytest.raises(ValueError, match="cache invalid"):
                SENTGSM8KDataset(
                    tokenizer=Mock(),
                    use_sent=True,
                    cache_path="/nonexistent/cache.pt"
                )

    def test_dataset_with_mock_cache(self):
        """Test dataset works with mock cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "cache.json")
            sent_config = SENTConfig()
            mock_tokenizer = _configure_mock_tokenizer(Mock())
            
            cache_data = {
                "metadata": _sent_metadata_for(sent_config, mock_tokenizer, 3),
                "indices": [0, 1, 2, 3, 4],
                "entropies": [0.1, 0.2, 0.3, 0.4, 0.5],
                "clusters": [[], [], [], [], []]
            }
            
            import json
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f)
            
            with patch("src.data.gsm8k_loader.load_dataset") as mock_load:
                mock_load.return_value = [{"question": f"Q{i}?", "answer": str(i)} for i in range(5)]
                
                mock_tokenizer = _configure_mock_tokenizer(Mock())
                mock_tokenizer.apply_chat_template.return_value = "formatted prompt"
                mock_tokenizer.return_value = {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}
                
                dataset = SENTGSM8KDataset(
                    tokenizer=mock_tokenizer,
                    use_sent=True,
                    cache_path=cache_path,
                    max_prompt_length=3,
                    sent_config=sent_config,
                    num_stages=2
                )
                
                info = dataset.get_stage_info()
                assert info["num_stages"] == 2
                assert info["total_samples"] == 5
                
                dataset.set_stage(1)
                assert len(dataset) == 2
                
                dataset.set_stage(2)
                assert len(dataset) == 3


class TestCreateDataloaderFallback:
    def test_missing_sent_cache_falls_back_to_standard_dataset(self):
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 0
        mock_tokenizer.padding_side = "right"
        mock_tokenizer.apply_chat_template.return_value = "formatted prompt"
        mock_tokenizer.return_value = {
            "input_ids": [1, 2, 3],
            "attention_mask": [1, 1, 1],
        }

        with patch("src.data.gsm8k_loader.load_dataset") as mock_load:
            mock_load.return_value = [{"question": "Q?", "answer": "#### 42"}]
            dataloader = create_grpo_dataloader(
                tokenizer=mock_tokenizer,
                split="train",
                batch_size=1,
                use_sent=True,
                cache_path="/nonexistent/cache.pt",
            )

        assert isinstance(dataloader.dataset, GRPOGSM8KDataset)

    def test_invalid_existing_sent_cache_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "cache.json")
            with open(cache_path, "w") as f:
                json.dump({"metadata": {"status": "complete", "config_hash": "legacy"}}, f)

            mock_tokenizer = Mock()
            mock_tokenizer.pad_token_id = 0
            mock_tokenizer.eos_token_id = 0
            mock_tokenizer.padding_side = "right"
            mock_tokenizer.apply_chat_template.return_value = "formatted prompt"
            mock_tokenizer.return_value = {
                "input_ids": [1, 2, 3],
                "attention_mask": [1, 1, 1],
            }

            with pytest.raises(ValueError, match="SENT cache invalid"):
                create_grpo_dataloader(
                    tokenizer=mock_tokenizer,
                    split="train",
                    batch_size=1,
                    use_sent=True,
                    cache_path=cache_path,
                )

    def test_stage_slicing(self):
        """Test stage slicing divides data correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "cache.json")
            sent_config = SENTConfig()
            mock_tokenizer = _configure_mock_tokenizer(Mock())
            
            indices = list(range(100))
            entropies = [i * 0.01 for i in range(100)]
            
            cache_data = {
                "metadata": _sent_metadata_for(sent_config, mock_tokenizer, 1),
                "indices": indices,
                "entropies": entropies,
                "clusters": [[] for _ in range(100)]
            }
            
            import json
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f)
            
            with patch("src.data.gsm8k_loader.load_dataset") as mock_load:
                mock_load.return_value = [{"question": f"Q{i}?", "answer": str(i)} for i in range(100)]
                
                mock_tokenizer = _configure_mock_tokenizer(mock_tokenizer)
                mock_tokenizer.apply_chat_template.return_value = "prompt"
                mock_tokenizer.return_value = {"input_ids": [1], "attention_mask": [1]}
                
                dataset = SENTGSM8KDataset(
                    tokenizer=mock_tokenizer,
                    use_sent=True,
                    cache_path=cache_path,
                    max_prompt_length=1,
                    sent_config=sent_config,
                    num_stages=2
                )
                
                dataset.set_stage(1)
                stage1_size = len(dataset)
                
                dataset.set_stage(2)
                stage2_size = len(dataset)
                
                assert stage1_size == 50
                assert stage2_size == 50


class TestCheckpointResume:
    """Tests for checkpoint and resume functionality."""

    def test_checkpoint_resume(self):
        """Test that process_dataset can resume from a partial checkpoint."""
        verifier = RuleBasedVerifier()
        mock_model = Mock()
        mock_tokenizer = _configure_mock_tokenizer(Mock())
        config = get_8gb_vram_config()

        calc = SemanticEntropyCalculator(mock_model, mock_tokenizer, verifier, config)
        calc.checkpoint_interval = 2

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "cache.json")

            # Simulate a partial cache (already processed 3 items)
            partial_data = {
                "metadata": {"status": "in_progress", "version": "sent_v1"},
                "indices": [0, 1, 2],
                "entropies": [0.1, 0.5, 0.3],
                "clusters": [[], [], []],
            }
            import json
            with open(cache_path, 'w') as f:
                json.dump(partial_data, f)

            # Mock compute_entropy_for_batch to avoid needing a real model
            def mock_batch(questions, num_samples=4):
                return [(0.7, {"clusters": []}) for _ in questions]
            calc.compute_entropy_for_batch = mock_batch

            dataset = [{"id": i, "question": f"Q{i}?"} for i in range(5)]

            calc.process_dataset(dataset, cache_path, resume=True)

            loaded = calc.load_cache(cache_path)
            assert loaded["metadata"]["status"] == "complete"
            # Should have all 5 entries (3 resumed + 2 new)
            assert len(loaded["entropies"]) == 5


class TestDatasetOrdering:
    """Tests for dataset ordering by entropy."""

    def test_entropies_sorted_ascending(self):
        """Test that cache entropies are in non-decreasing order after processing."""
        verifier = RuleBasedVerifier()
        mock_model = Mock()
        mock_tokenizer = _configure_mock_tokenizer(Mock())
        config = get_8gb_vram_config()

        calc = SemanticEntropyCalculator(mock_model, mock_tokenizer, verifier, config)

        # Return entropies in non-sorted order
        entropy_values = [0.9, 0.1, 0.5, 0.3, 0.7]
        call_count = [0]

        def mock_batch(questions, num_samples=4):
            results = []
            for _ in questions:
                val = entropy_values[call_count[0]]
                call_count[0] += 1
                results.append((val, {"clusters": []}))
            return results

        calc.compute_entropy_for_batch = mock_batch

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "cache.json")
            dataset = [{"id": i, "question": f"Q{i}?"} for i in range(5)]

            calc.process_dataset(dataset, cache_path, resume=False)

            loaded = calc.load_cache(cache_path)
            entropies = loaded["entropies"]

            # Verify non-decreasing order
            for i in range(len(entropies) - 1):
                assert entropies[i] <= entropies[i + 1], \
                    f"Entropies not sorted: {entropies[i]} > {entropies[i+1]} at index {i}"


class TestDataloaderCompatibility:
    """Tests for DataLoader compatibility with SENT."""

    def test_create_grpo_dataloader_with_sent(self):
        """Test create_grpo_dataloader works with use_sent=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "cache.json")
            sent_config = SENTConfig()
            mock_tokenizer = _configure_mock_tokenizer(Mock())

            cache_data = {
                "metadata": _sent_metadata_for(sent_config, mock_tokenizer, 2),
                "indices": [0, 1, 2, 3],
                "entropies": [0.1, 0.2, 0.3, 0.4],
                "clusters": [[], [], [], []],
            }

            import json
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f)

            with patch("src.data.gsm8k_loader.load_dataset") as mock_load:
                mock_load.return_value = [
                    {"question": f"Q{i}?", "answer": f"#### {i}"} for i in range(4)
                ]

                mock_tokenizer = _configure_mock_tokenizer(mock_tokenizer)
                mock_tokenizer.apply_chat_template.return_value = "formatted"
                mock_tokenizer.return_value = {"input_ids": [1, 2], "attention_mask": [1, 1]}
                mock_tokenizer.pad.return_value = {
                    "input_ids": torch.tensor([[1, 2], [1, 2]]),
                    "attention_mask": torch.tensor([[1, 1], [1, 1]]),
                }

                dl = create_grpo_dataloader(
                    tokenizer=mock_tokenizer,
                    use_sent=True,
                    batch_size=2,
                    cache_path=cache_path,
                    max_prompt_length=2,
                    sent_config=sent_config,
                    shuffle=False,
                )

                batch = next(iter(dl))
                assert "input_ids" in batch
                assert "questions" in batch
                assert "answers" in batch
                assert batch["input_ids"].shape[0] == 2

    def test_create_grpo_dataloader_without_sent(self):
        """Test create_grpo_dataloader backward compatibility with use_sent=False."""
        with patch("src.data.gsm8k_loader.load_dataset") as mock_load:
            mock_load.return_value = [
                {"question": f"Q{i}?", "answer": f"#### {i}"} for i in range(4)
            ]

            mock_tokenizer = Mock()
            mock_tokenizer.apply_chat_template.return_value = "formatted"
            mock_tokenizer.return_value = {"input_ids": [1, 2], "attention_mask": [1, 1]}
            mock_tokenizer.pad.return_value = {
                "input_ids": torch.tensor([[1, 2], [1, 2]]),
                "attention_mask": torch.tensor([[1, 1], [1, 1]]),
            }

            dl = create_grpo_dataloader(
                tokenizer=mock_tokenizer,
                use_sent=False,
                batch_size=2,
                shuffle=True,
            )

            batch = next(iter(dl))
            assert "input_ids" in batch
            assert "questions" in batch

    def test_create_grpo_dataloader_fallback_restores_standard_shuffle_semantics(self):
        """Missing SENT cache should behave like plain GSM8K training."""
        mock_tokenizer = _configure_mock_tokenizer(Mock())
        mock_tokenizer.apply_chat_template.return_value = "formatted"
        mock_tokenizer.return_value = {"input_ids": [1, 2], "attention_mask": [1, 1]}

        fake_loader = type("FakeLoader", (), {})()

        with patch("src.data.gsm8k_loader.load_dataset") as mock_load, patch(
            "src.data.gsm8k_loader.DataLoader", return_value=fake_loader
        ) as dataloader_cls:
            mock_load.return_value = [
                {"question": f"Q{i}?", "answer": f"#### {i}"} for i in range(4)
            ]

            dl = create_grpo_dataloader(
                tokenizer=mock_tokenizer,
                use_sent=True,
                batch_size=2,
                cache_path="/tmp/does-not-exist.json",
                max_prompt_length=2,
                sent_config=SENTConfig(),
                shuffle=False,
            )

        assert dl is fake_loader
        assert dataloader_cls.call_args.kwargs["shuffle"] is True
        assert getattr(dl, "uses_sent_curriculum") is False


class TestEndToEndPreprocess:
    """End-to-end test with mocked model (no GPU required)."""

    def test_full_pipeline_mocked(self):
        """Test full pipeline: calculator -> cache -> dataset loading."""
        verifier = RuleBasedVerifier()
        mock_model = Mock()
        mock_tokenizer = _configure_mock_tokenizer(Mock())
        config = get_8gb_vram_config()
        config.training.max_prompt_length = 1

        calc = SemanticEntropyCalculator(mock_model, mock_tokenizer, verifier, config)

        # Mock compute to return predictable entropies
        entropies_out = [0.8, 0.2, 0.5, 0.1, 0.9]
        call_idx = [0]

        def mock_batch(questions, num_samples=4):
            results = []
            for _ in questions:
                val = entropies_out[call_idx[0]]
                call_idx[0] += 1
                results.append((val, {"clusters": [{"answer": 42, "count": 1}]}))
            return results

        calc.compute_entropy_for_batch = mock_batch

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "cache.json")
            dataset = [{"id": i, "question": f"What is {i}+1?"} for i in range(5)]

            # Step 1: Generate cache
            calc.process_dataset(dataset, cache_path, resume=False)

            # Step 2: Verify cache
            loaded = calc.load_cache(cache_path)
            assert loaded["metadata"]["status"] == "complete"
            assert len(loaded["entropies"]) == 5

            # Step 3: Verify sorting (easy -> hard)
            ents = loaded["entropies"]
            assert all(ents[i] <= ents[i + 1] for i in range(len(ents) - 1))

            # Step 4: Load into SENTGSM8KDataset
            with patch("src.data.gsm8k_loader.load_dataset") as mock_load:
                mock_load.return_value = [
                    {"question": f"What is {i}+1?", "answer": f"#### {i+1}"} for i in range(5)
                ]

                mock_tok = _configure_mock_tokenizer(Mock())
                mock_tok.apply_chat_template.return_value = "prompt"
                mock_tok.return_value = {"input_ids": [1], "attention_mask": [1]}

                ds = SENTGSM8KDataset(
                    tokenizer=mock_tok,
                    use_sent=True,
                    cache_path=cache_path,
                    max_prompt_length=1,
                    num_stages=1,
                )

                assert len(ds) == 5
                sample = ds[0]
                assert "question" in sample


class TestSentCacheIndexContract:
    def test_process_dataset_stores_positional_indices_and_example_ids(self):
        verifier = RuleBasedVerifier()
        mock_model = Mock()
        mock_tokenizer = _configure_mock_tokenizer(Mock())
        config = get_8gb_vram_config()

        calc = SemanticEntropyCalculator(mock_model, mock_tokenizer, verifier, config)
        calc.compute_entropy_for_batch = lambda questions, num_samples=4: [
            (0.1 + idx, {"clusters": []}) for idx, _ in enumerate(questions)
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "cache.json")
            dataset = [
                {"id": "sample-a", "question": "Q0?"},
                {"id": "sample-b", "question": "Q1?"},
            ]

            calc.process_dataset(dataset, cache_path, resume=False)
            loaded = calc.load_cache(cache_path)

        assert loaded["indices"] == [0, 1]
        assert loaded["example_ids"] == ["sample-a", "sample-b"]

    def test_process_dataset_resume_migrates_legacy_ids_to_example_ids(self):
        verifier = RuleBasedVerifier()
        mock_model = Mock()
        mock_tokenizer = _configure_mock_tokenizer(Mock())
        config = get_8gb_vram_config()

        calc = SemanticEntropyCalculator(mock_model, mock_tokenizer, verifier, config)
        calc.compute_entropy_for_batch = lambda questions, num_samples=4: [
            (0.2, {"clusters": []}) for _ in questions
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "cache.json")
            with open(cache_path, "w") as f:
                json.dump(
                    {
                        "metadata": {"status": "in_progress", "version": "sent_v1"},
                        "indices": ["sample-a"],
                        "entropies": [0.1],
                        "clusters": [[]],
                    },
                    f,
                )

            dataset = [
                {"id": "sample-a", "question": "Q0?"},
                {"id": "sample-b", "question": "Q1?"},
            ]
            calc.process_dataset(dataset, cache_path, resume=True)
            loaded = calc.load_cache(cache_path)

        assert loaded["indices"] == [0, 1]
        assert loaded["example_ids"] == ["sample-a", "sample-b"]

    def test_sent_dataset_rejects_non_positional_indices(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "cache.json")
            sent_config = SENTConfig()
            mock_tokenizer = _configure_mock_tokenizer(Mock())

            cache_data = {
                "metadata": _sent_metadata_for(sent_config, mock_tokenizer, 2),
                "indices": ["sample-a"],
                "example_ids": ["sample-a"],
                "entropies": [0.1],
                "clusters": [[]],
            }

            with open(cache_path, "w") as f:
                json.dump(cache_data, f)

            with patch("src.data.gsm8k_loader.load_dataset") as mock_load:
                mock_load.return_value = [{"question": "Q0?", "answer": "#### 0"}]
                mock_tokenizer.apply_chat_template.return_value = "formatted"
                mock_tokenizer.return_value = {
                    "input_ids": [1, 2],
                    "attention_mask": [1, 1],
                }

                with pytest.raises(ValueError, match="positional dataset indices"):
                    SENTGSM8KDataset(
                        tokenizer=mock_tokenizer,
                        use_sent=True,
                        cache_path=cache_path,
                        max_prompt_length=2,
                        num_stages=1,
                    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
