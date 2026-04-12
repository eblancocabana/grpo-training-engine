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

    def test_extract_final_answer_prefers_post_think_answer(self):
        verifier = RuleBasedVerifier()
        text = (
            "<think>Compute 2+2 = 4. Let me verify 4-1 = 3</think> Final Answer: 4"
        )
        extracted = verifier.extract_final_answer(text)
        assert extracted == "4"

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

    def test_extracts_equals_answer_from_think_tail(self):
        verifier = RuleBasedVerifier()
        text = (
            "She worked 2 hours a day for 4 days, which is 8 hours in total.\n\n"
            "At $22 per hour, she earned $176.\n\n"
            "Finally, profit is earnings minus supplies: $176 minus $54 equals $122.\n"
            "</think>\n\n**"
        )
        extracted = verifier.extract_final_answer(text)
        assert extracted == "122"

    def test_extracts_equals_answer_from_think_tail_with_decimal_money(self):
        verifier = RuleBasedVerifier()
        text = (
            "She sells 4 subscriptions, 1 subscription, 2 subscriptions, and twice that amount.\n"
            "Finally, I'll add up all the earnings: $20.00 + $5.00 + $10.00 + $20.00, "
            "which equals $55.00.\n"
            "</think>\n\n"
            "To determine how much money Maggie earned..."
        )
        extracted = verifier.extract_final_answer(text)
        assert extracted == "55.00"

    def test_extracts_latex_boxed_number_with_spacing_commands(self):
        verifier = RuleBasedVerifier()
        text = (
            "</think>\n\n"
            "The original price of the car was \\(\\boxed{10,\\!000}\\)."
        )
        extracted = verifier.extract_final_answer(text)
        assert extracted == "10,000"

    def test_incomplete_response_does_not_extract_intermediate_think_number(self):
        verifier = RuleBasedVerifier()
        text = (
            "He can buy 12 artichokes. It takes 3 artichokes to make 5 ounces of dip.\n"
            "To find the ounces per artichoke, we compute 5/3.\n"
            "</think>\n\n"
            "To determine the ounces of dip:\n"
            "\\[\n"
            "\\text{Ounces per Artichoke} = \\frac{5"
        )
        extracted = verifier.extract_final_answer(text)
        assert extracted is None

    def test_verify_uses_strict_training_extraction_for_incomplete_reasoning(self):
        verifier = RuleBasedVerifier()
        response = (
            "He can buy 12 artichokes. It takes 3 artichokes to make 5 ounces of dip.\n"
            "To find the ounces per artichoke, we compute 5/3.\n"
            "</think>\n\n"
            "To determine the ounces of dip:\n"
            "\\[\n"
            "\\text{Ounces per Artichoke} = \\frac{5"
        )
        reward, info = verifier.verify(response, "20")
        assert reward == 0.0
        assert info["extracted_answer"] is None

    def test_verify_uses_strict_training_extraction_for_truncated_reasoning_tail(self):
        verifier = RuleBasedVerifier()
        response = (
            "She started at 99 pounds. She dropped 12 pounds, going to 87.\n"
            "Then she added back 24 pounds and reached 111.\n"
            "Then she dropped three times the original 12 pounds, so 12 * 3 = 36 pounds. "
            "So she lost"
        )
        reward, info = verifier.verify(response, "81")
        assert reward == 0.0
        assert info["extracted_answer"] is None

    def test_verify_prefers_high_confidence_think_tail_answer(self):
        verifier = RuleBasedVerifier()
        response = (
            "She worked 2 hours a day for 4 days, which is 8 hours in total.\n"
            "At $22 per hour, she earned $176.\n"
            "Finally, profit is earnings minus supplies: $176 minus $54 equals $122.\n"
            "</think>\n\n**"
        )
        reward, info = verifier.verify(response, "122")
        assert reward == 1.0
        assert info["extracted_answer"] == "122"
