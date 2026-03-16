import torch
import torch.nn.functional as F
import pytest
from transformers import LlamaConfig, LlamaForCausalLM


assert torch.cuda.is_available(), "CUDA is required for this test suite"


def _make_tiny_model() -> LlamaForCausalLM:
    torch.manual_seed(123)
    config = LlamaConfig(
        vocab_size=97,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        use_cache=True,
    )
    model = LlamaForCausalLM(config).to("cuda")
    model.eval()
    return model


def _make_loop(model):
    from src.grpo.trainer import GRPOTrainerLoop
    from src.utils.config import get_8gb_vram_config

    class _NoOpMemoryManager:
        @staticmethod
        def clear_cache() -> None:
            return None

    loop = GRPOTrainerLoop(get_8gb_vram_config())
    loop.model = model
    loop.device = "cuda"
    loop.memory_manager = _NoOpMemoryManager()
    return loop


def _build_dense_old_log_probs(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
    response_ids: torch.Tensor,
    response_mask: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    prompt_ids_expanded = prompt_ids.repeat_interleave(group_size, dim=0)
    prompt_mask_expanded = prompt_mask.repeat_interleave(group_size, dim=0)
    all_input_ids = torch.cat([prompt_ids_expanded, response_ids], dim=1)
    all_attention_mask = torch.cat([prompt_mask_expanded, response_mask], dim=1)

    prompt_width = prompt_ids.shape[1]
    response_only_mask = torch.cat(
        [
            torch.zeros(
                response_ids.shape[0],
                prompt_width,
                dtype=response_mask.dtype,
                device=response_mask.device,
            ),
            response_mask,
        ],
        dim=1,
    )

    with torch.no_grad():
        outputs = model(
            input_ids=all_input_ids,
            attention_mask=all_attention_mask,
            use_cache=False,
        )
        logits = outputs.logits[:, :-1, :]
        targets = all_input_ids[:, 1:]
        token_log_probs = -F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            reduction="none",
        ).view(targets.shape)

    return token_log_probs * response_only_mask[:, 1:], response_only_mask[:, 1:]


def _build_cached_old_log_probs(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
    response_ids: torch.Tensor,
    response_mask: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    loop = _make_loop(model)
    prompt_width = prompt_ids.shape[1]
    response_width = response_ids.shape[1]
    num_samples = response_ids.shape[0]
    full_seq_len = prompt_width + response_width - 1
    all_old_log_probs = torch.zeros(
        num_samples,
        full_seq_len,
        dtype=torch.float32,
        device=response_ids.device,
    )

    with torch.no_grad():
        for prompt_idx in range(prompt_ids.shape[0]):
            single_ids = prompt_ids[prompt_idx : prompt_idx + 1]
            single_mask = prompt_mask[prompt_idx : prompt_idx + 1]

            first_real = single_mask[0].argmax().item()
            real_ids = single_ids[:, first_real:]
            real_mask = single_mask[:, first_real:]
            real_prompt_len = real_ids.shape[1]

            prefix_cache = loop._prefill_prompt_cache(real_ids, real_mask)
            group_start = prompt_idx * group_size
            group_end = group_start + group_size
            group_resp_ids = response_ids[group_start:group_end]
            group_resp_mask = response_mask[group_start:group_end]

            mb_cache = loop._expand_prefix_cache(prefix_cache, group_size)
            anchor = real_ids[:, -1:].expand(group_size, -1)
            scorer_input_ids = torch.cat([anchor, group_resp_ids], dim=1)
            scorer_attention_mask = torch.cat(
                [
                    torch.ones(
                        group_size,
                        real_prompt_len,
                        dtype=real_mask.dtype,
                        device=real_mask.device,
                    ),
                    group_resp_mask,
                ],
                dim=1,
            )

            outputs = model(
                input_ids=scorer_input_ids,
                attention_mask=scorer_attention_mask,
                past_key_values=mb_cache,
                use_cache=True,
            )

            compact_logits = outputs.logits[:, :-1, :]
            compact_targets = group_resp_ids
            compact_log_probs = -F.cross_entropy(
                compact_logits.reshape(-1, compact_logits.size(-1)),
                compact_targets.reshape(-1),
                reduction="none",
            ).view(compact_targets.shape)
            compact_log_probs = compact_log_probs * group_resp_mask

            all_old_log_probs[
                group_start:group_end,
                prompt_width - 1 : prompt_width - 1 + response_width,
            ] = compact_log_probs

    return all_old_log_probs


def _make_left_padded_prompts() -> tuple[torch.Tensor, torch.Tensor]:
    prompt_ids = torch.tensor(
        [
            [0, 0, 11, 12, 13],
            [0, 21, 22, 23, 24],
        ],
        device="cuda",
        dtype=torch.long,
    )
    prompt_mask = torch.tensor(
        [
            [0, 0, 1, 1, 1],
            [0, 1, 1, 1, 1],
        ],
        device="cuda",
        dtype=torch.long,
    )
    return prompt_ids, prompt_mask


def _make_responses() -> tuple[torch.Tensor, torch.Tensor]:
    response_ids = torch.tensor(
        [
            [31, 32, 33, 0],
            [34, 35, 0, 0],
            [36, 37, 38, 39],
            [41, 42, 0, 0],
            [43, 44, 45, 0],
            [46, 47, 48, 49],
        ],
        device="cuda",
        dtype=torch.long,
    )
    response_mask = torch.tensor(
        [
            [1, 1, 1, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 1],
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [1, 1, 1, 1],
        ],
        device="cuda",
        dtype=torch.long,
    )
    return response_ids, response_mask


def _build_response_only_mask(
    prompt_ids: torch.Tensor,
    response_ids: torch.Tensor,
    response_mask: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    prompt_width = prompt_ids.shape[1]
    return torch.cat(
        [
            torch.zeros(
                response_ids.shape[0],
                prompt_width,
                dtype=response_mask.dtype,
                device=response_mask.device,
            ),
            response_mask,
        ],
        dim=1,
    )


class TestOldLogProbsPromptCache:
    def test_cached_response_scoring_matches_dense_left_padded_batch(self):
        model = _make_tiny_model()
        prompt_ids, prompt_mask = _make_left_padded_prompts()
        response_ids, response_mask = _make_responses()

        dense_old_log_probs, response_only_mask = _build_dense_old_log_probs(
            model, prompt_ids, prompt_mask, response_ids, response_mask, group_size=3
        )
        cached_old_log_probs = _build_cached_old_log_probs(
            model, prompt_ids, prompt_mask, response_ids, response_mask, group_size=3
        )

        torch.testing.assert_close(
            cached_old_log_probs, dense_old_log_probs, rtol=1e-5, atol=1e-5
        )
        assert (
            torch.count_nonzero(cached_old_log_probs[:, : prompt_ids.shape[1] - 1]) == 0
        )
        torch.testing.assert_close(
            cached_old_log_probs * response_only_mask,
            dense_old_log_probs * response_only_mask,
            rtol=1e-5,
            atol=1e-5,
        )

    def test_cached_response_scoring_matches_dense_single_token_prompt(self):
        model = _make_tiny_model()
        prompt_ids = torch.tensor([[17], [23]], device="cuda", dtype=torch.long)
        prompt_mask = torch.ones_like(prompt_ids)
        response_ids = torch.tensor(
            [[51, 52, 0], [53, 0, 0], [54, 55, 56], [57, 58, 0]],
            device="cuda",
            dtype=torch.long,
        )
        response_mask = torch.tensor(
            [[1, 1, 0], [1, 0, 0], [1, 1, 1], [1, 1, 0]],
            device="cuda",
            dtype=torch.long,
        )

        dense_old_log_probs, _ = _build_dense_old_log_probs(
            model, prompt_ids, prompt_mask, response_ids, response_mask, group_size=2
        )
        cached_old_log_probs = _build_cached_old_log_probs(
            model, prompt_ids, prompt_mask, response_ids, response_mask, group_size=2
        )

        torch.testing.assert_close(
            cached_old_log_probs, dense_old_log_probs, rtol=1e-5, atol=1e-5
        )

    def test_cached_and_dense_old_log_probs_produce_same_grpo_loss(self):
        from src.grpo.algorithm import GRPOTrainer

        model = _make_tiny_model()
        prompt_ids, prompt_mask = _make_left_padded_prompts()
        response_ids, response_mask = _make_responses()
        dense_old_log_probs, response_only_mask = _build_dense_old_log_probs(
            model, prompt_ids, prompt_mask, response_ids, response_mask, group_size=3
        )
        cached_old_log_probs = _build_cached_old_log_probs(
            model, prompt_ids, prompt_mask, response_ids, response_mask, group_size=3
        )

        prompt_ids_expanded = prompt_ids.repeat_interleave(3, dim=0)
        prompt_mask_expanded = prompt_mask.repeat_interleave(3, dim=0)
        all_input_ids = torch.cat([prompt_ids_expanded, response_ids], dim=1)
        all_attention_mask = torch.cat([prompt_mask_expanded, response_mask], dim=1)
        advantages = torch.tensor([1.0, -1.0, 0.5, 0.25, -0.75, 0.0], device="cuda")

        with torch.no_grad():
            logits = model(
                input_ids=all_input_ids,
                attention_mask=all_attention_mask,
                use_cache=False,
            ).logits[:, :-1, :]

        trainer = GRPOTrainer(group_size=3, use_kl=False, use_triton_kernels=False)
        dense_loss, dense_metrics = trainer.compute_grpo_loss(
            policy_logits=logits,
            advantages=advantages,
            old_log_probs=dense_old_log_probs,
            target_ids=all_input_ids[:, 1:],
            attention_mask=response_only_mask,
        )
        cached_loss, cached_metrics = trainer.compute_grpo_loss(
            policy_logits=logits,
            advantages=advantages,
            old_log_probs=cached_old_log_probs,
            target_ids=all_input_ids[:, 1:],
            attention_mask=response_only_mask,
        )

        torch.testing.assert_close(cached_loss, dense_loss, rtol=1e-5, atol=1e-5)
        assert cached_metrics.keys() == dense_metrics.keys()
        for key, value in dense_metrics.items():
            assert cached_metrics[key] == pytest.approx(value, rel=1e-5, abs=1e-5)

    def test_production_cached_method_matches_dense_reference_with_micro_batches(self):
        from src.grpo.trainer import GRPOTrainerLoop

        if not hasattr(GRPOTrainerLoop, "_compute_old_log_probs_with_prompt_cache"):
            pytest.skip(
                "Direct production cached old_log_probs method is not available on main."
            )

        model = _make_tiny_model()
        loop = _make_loop(model)
        loop.config.grpo.group_size = 3
        loop._gen_micro_batch = 2

        prompt_ids, prompt_mask = _make_left_padded_prompts()
        response_ids, response_mask = _make_responses()
        response_only_mask = _build_response_only_mask(
            prompt_ids, response_ids, response_mask, group_size=3
        )

        dense_old_log_probs, _ = _build_dense_old_log_probs(
            model, prompt_ids, prompt_mask, response_ids, response_mask, group_size=3
        )

        with torch.no_grad():
            production_old_log_probs = loop._compute_old_log_probs_with_prompt_cache(
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                response_ids=response_ids,
                response_mask=response_mask,
                response_only_mask=response_only_mask,
            )

        torch.testing.assert_close(
            production_old_log_probs, dense_old_log_probs, rtol=1e-5, atol=1e-5
        )
