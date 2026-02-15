import pytest
import json
import os
import tempfile
import torch
from unittest.mock import Mock, MagicMock, patch

from src.data.sent_calculator import SemanticEntropyCalculator
from src.data.gsm8k_loader import SENTGSM8KDataset, _validate_cache, create_grpo_dataloader
from src.grpo.verifier import RuleBasedVerifier
from src.utils.config import SENTConfig, Config, get_8gb_vram_config


class TestSemanticEntropyCalculator:
    """Tests for SemanticEntropyCalculator."""

    def test_cluster_by_answer(self):
        """Test clustering by answer equivalence."""
        verifier = RuleBasedVerifier()
        
        mock_model = Mock()
        mock_tokenizer = Mock()
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
        mock_tokenizer = Mock()
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
        assert "created_at" in meta


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
            
            data = {
                "metadata": {"status": "complete", "config_hash": "abc123"},
                "indices": [0, 1, 2],
                "entropies": [0.1, 0.2, 0.3],
                "clusters": [[], [], []]
            }
            
            import json
            with open(cache_path, 'w') as f:
                json.dump(data, f)
            
            is_valid, msg = _validate_cache(cache_path)
            
            assert is_valid is True


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
            
            cache_data = {
                "metadata": {"status": "complete", "config_hash": "abc"},
                "indices": [0, 1, 2, 3, 4],
                "entropies": [0.1, 0.2, 0.3, 0.4, 0.5],
                "clusters": [[], [], [], [], []]
            }
            
            import json
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f)
            
            with patch("src.data.gsm8k_loader.load_dataset") as mock_load:
                mock_load.return_value = [{"question": f"Q{i}?", "answer": str(i)} for i in range(5)]
                
                mock_tokenizer = Mock()
                mock_tokenizer.apply_chat_template.return_value = "formatted prompt"
                mock_tokenizer.return_value = {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}
                
                dataset = SENTGSM8KDataset(
                    tokenizer=mock_tokenizer,
                    use_sent=True,
                    cache_path=cache_path,
                    num_stages=2
                )
                
                info = dataset.get_stage_info()
                assert info["num_stages"] == 2
                assert info["total_samples"] == 5
                
                dataset.set_stage(1)
                assert len(dataset) == 2
                
                dataset.set_stage(2)
                assert len(dataset) == 3

    def test_stage_slicing(self):
        """Test stage slicing divides data correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = os.path.join(tmpdir, "cache.json")
            
            indices = list(range(100))
            entropies = [i * 0.01 for i in range(100)]
            
            cache_data = {
                "metadata": {"status": "complete", "config_hash": "abc"},
                "indices": indices,
                "entropies": entropies,
                "clusters": [[] for _ in range(100)]
            }
            
            import json
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f)
            
            with patch("src.data.gsm8k_loader.load_dataset") as mock_load:
                mock_load.return_value = [{"question": f"Q{i}?", "answer": str(i)} for i in range(100)]
                
                mock_tokenizer = Mock()
                mock_tokenizer.apply_chat_template.return_value = "prompt"
                mock_tokenizer.return_value = {"input_ids": [1], "attention_mask": [1]}
                
                dataset = SENTGSM8KDataset(
                    tokenizer=mock_tokenizer,
                    use_sent=True,
                    cache_path=cache_path,
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
        mock_tokenizer = Mock()
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
        mock_tokenizer = Mock()
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

            cache_data = {
                "metadata": {"status": "complete", "config_hash": "abc"},
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

                mock_tokenizer = Mock()
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


class TestEndToEndPreprocess:
    """End-to-end test with mocked model (no GPU required)."""

    def test_full_pipeline_mocked(self):
        """Test full pipeline: calculator -> cache -> dataset loading."""
        verifier = RuleBasedVerifier()
        mock_model = Mock()
        mock_tokenizer = Mock()
        config = get_8gb_vram_config()

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

                mock_tok = Mock()
                mock_tok.apply_chat_template.return_value = "prompt"
                mock_tok.return_value = {"input_ids": [1], "attention_mask": [1]}

                ds = SENTGSM8KDataset(
                    tokenizer=mock_tok,
                    use_sent=True,
                    cache_path=cache_path,
                    num_stages=1,
                )

                assert len(ds) == 5
                sample = ds[0]
                assert "question" in sample


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
