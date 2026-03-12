from optimizer.observation import DeepTraceTargetSelector


def test_selector_prefers_triton_decode_from_kernel_names() -> None:
    selector = DeepTraceTargetSelector()
    observation = selector.select_from_summary(
        frontier_target="main",
        summary={
            "key_averages": [
                {"name": "_paged_attention_decode_kernel"},
                {"name": "_paged_kv_update_kernel"},
            ]
        },
    )

    assert observation.selected_target == "generation_triton_decode"
    assert observation.target_family == "generation"


def test_selector_falls_back_to_generation_general() -> None:
    selector = DeepTraceTargetSelector()
    observation = selector.select_from_summary(frontier_target="main", summary={})

    assert observation.selected_target == "generation_general"
    assert observation.diagnostics["fallback"] is True
