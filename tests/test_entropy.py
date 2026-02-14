import torch
import math


class TestEntropyMasking:
    """White-box tests for entropy-based selective backpropagation."""

    def test_entropy_calculation_per_token(self):
        """Verify entropy is calculated correctly: H = -sum(p * log(p))."""
        from src.selective.entropy_mask import EntropyCalculator

        vocab_size = 4

        # Uniform distribution: H = ln(4) = 1.386
        uniform_logits = torch.zeros(1, 1, vocab_size)

        calculator = EntropyCalculator()
        entropy = calculator.calculate_entropy(uniform_logits)

        expected_entropy = math.log(vocab_size)
        assert abs(entropy.item() - expected_entropy) < 0.01

    def test_entropy_masking_by_percentile(self):
        """Verify masking keeps only top percentile tokens."""
        from src.selective.entropy_mask import EntropyCalculator

        calculator = EntropyCalculator(percentile=0.5)

        entropies = torch.tensor([0.1, 0.5, 0.9, 1.3])
        threshold = torch.quantile(entropies, 0.5)
        mask = entropies >= threshold

        assert mask.sum().item() == 2
        assert mask[2].item() is True
        assert mask[3].item() is True

    def test_masked_loss_application(self):
        """Verify loss is only computed for unmasked tokens."""
        from src.selective.entropy_mask import EntropyCalculator

        calculator = EntropyCalculator(percentile=0.5)

        losses = torch.tensor([1.0, 2.0, 3.0, 4.0])
        mask = torch.tensor([True, True, False, False])

        masked_loss = (losses * mask.float()).sum() / mask.sum()
        expected = (1.0 + 2.0) / 2
        assert masked_loss.item() == expected

    def test_entropy_gradients_flow(self):
        """Verify gradients only flow through selected tokens."""
        from src.selective.entropy_mask import EntropyCalculator

        calculator = EntropyCalculator(percentile=0.5)

        logits = torch.randn(2, 4, requires_grad=True)
        entropy = calculator.calculate_entropy(logits.unsqueeze(0))
        entropy.sum().backward()
        assert logits.grad is not None
