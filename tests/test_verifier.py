from src.grpo.verifier import RuleBasedVerifier


class TestVerifier:
    """Comprehensive tests for answer verification."""

    def test_extract_answer_from_boxed(self):
        verifier = RuleBasedVerifier()
        text = "The answer is \\boxed{42}."
        extracted = verifier.extract_final_answer(text)
        assert extracted == "42"

    def test_extract_answer_from_number(self):
        verifier = RuleBasedVerifier()
        text = "First we calculate 10 + 20 = 30, then 30 + 5 = 35."
        extracted = verifier.extract_final_answer(text)
        assert extracted is not None
        assert extracted.replace(",", "").replace(".", "").isdigit()

    def test_verify_exact_match(self):
        verifier = RuleBasedVerifier()
        response = "\\boxed{123}"
        ground_truth = "123"
        result = verifier.verify(response, ground_truth)
        reward = result[0]
        assert reward == 1.0

    def test_verify_wrong_answer(self):
        verifier = RuleBasedVerifier()
        response = "\\boxed{456}"
        ground_truth = "123"
        result = verifier.verify(response, ground_truth)
        reward = result[0]
        assert reward == 0.0

    def test_verify_empty_response(self):
        verifier = RuleBasedVerifier()
        result = verifier.verify("", "123")
        reward = result[0]
        assert reward == 0.0

    def test_verify_batch(self):
        verifier = RuleBasedVerifier()
        responses = ["\\boxed{1}", "\\boxed{2}", "\\boxed{3}"]
        ground_truths = ["1", "2", "4"]
        results = [verifier.verify(r, g) for r, g in zip(responses, ground_truths)]
        assert len(results) == 3
        assert results[0][0] == 1.0
        assert results[1][0] == 1.0
        assert results[2][0] == 0.0
