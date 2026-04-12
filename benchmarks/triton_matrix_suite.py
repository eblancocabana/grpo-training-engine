#!/usr/bin/env python3
"""
Triton matrix benchmark suite for per-kernel ablation on the current training path.

This suite keeps the ING INF benchmark harness style, but focuses on isolating
which Triton-enabled subsystems help or hurt the default DeepSeek GRPO workload.
"""

from __future__ import annotations

import argparse
from typing import List, Optional

from ing_inf_suite import BenchmarkConfig, IngInfBenchmarkSuite


class TritonMatrixBenchmarkSuite(IngInfBenchmarkSuite):
    """Per-kernel Triton ablation suite for the default training configuration."""

    def __init__(
        self,
        output_dir: str = "./benchmarks/output/triton_matrix",
        run_prefix: str = "triton_matrix",
    ):
        super().__init__(output_dir=output_dir, run_prefix=run_prefix)
        self.suite_name = "Triton Matrix Benchmark Suite"
        self.report_title = "Triton Matrix Benchmark Report"

    def define_test_matrix(
        self, filter_configs: Optional[List[str]] = None
    ) -> List[BenchmarkConfig]:
        configs = [
            BenchmarkConfig(
                name="torch_baseline",
                description="Current default path with all Triton kernels disabled",
                triton=False,
                gradient_accumulation_steps=16,
                group_size=4,
                steps=20,
                tags=["baseline", "torch", "all_off"],
            ),
            BenchmarkConfig(
                name="triton_all",
                description="Current default path with all Triton kernel families enabled",
                triton=True,
                gradient_accumulation_steps=16,
                group_size=4,
                triton_generation=True,
                triton_grpo_loss=True,
                triton_entropy_mask=True,
                triton_lora=True,
                steps=20,
                tags=["baseline", "triton", "all_on"],
            ),
            BenchmarkConfig(
                name="triton_generation_only",
                description="Only Triton generation enabled; loss, entropy, and LoRA use torch",
                triton=True,
                gradient_accumulation_steps=16,
                group_size=4,
                triton_generation=True,
                triton_grpo_loss=False,
                triton_entropy_mask=False,
                triton_lora=False,
                steps=20,
                tags=["single", "generation"],
            ),
            BenchmarkConfig(
                name="triton_grpo_only",
                description="Only Triton GRPO loss enabled; rollout and other kernels use torch",
                triton=True,
                gradient_accumulation_steps=16,
                group_size=4,
                triton_generation=False,
                triton_grpo_loss=True,
                triton_entropy_mask=False,
                triton_lora=False,
                steps=20,
                tags=["single", "grpo"],
            ),
            BenchmarkConfig(
                name="triton_entropy_only",
                description="Only Triton entropy masking enabled; all other kernel families use torch",
                triton=True,
                gradient_accumulation_steps=16,
                group_size=4,
                triton_generation=False,
                triton_grpo_loss=False,
                triton_entropy_mask=True,
                triton_lora=False,
                steps=20,
                tags=["single", "entropy"],
            ),
            BenchmarkConfig(
                name="triton_lora_only",
                description="Only Triton LoRA forward enabled; generation, loss, and entropy use torch",
                triton=True,
                gradient_accumulation_steps=16,
                group_size=4,
                triton_generation=False,
                triton_grpo_loss=False,
                triton_entropy_mask=False,
                triton_lora=True,
                steps=20,
                tags=["single", "lora"],
            ),
            BenchmarkConfig(
                name="triton_all_except_generation",
                description="All Triton kernel families enabled except paged-KV generation",
                triton=True,
                gradient_accumulation_steps=16,
                group_size=4,
                triton_generation=False,
                triton_grpo_loss=True,
                triton_entropy_mask=True,
                triton_lora=True,
                steps=20,
                tags=["minus_one", "generation"],
            ),
            BenchmarkConfig(
                name="triton_all_except_grpo",
                description="All Triton kernel families enabled except fused GRPO loss",
                triton=True,
                gradient_accumulation_steps=16,
                group_size=4,
                triton_generation=True,
                triton_grpo_loss=False,
                triton_entropy_mask=True,
                triton_lora=True,
                steps=20,
                tags=["minus_one", "grpo"],
            ),
            BenchmarkConfig(
                name="triton_all_except_entropy",
                description="All Triton kernel families enabled except entropy masking",
                triton=True,
                gradient_accumulation_steps=16,
                group_size=4,
                triton_generation=True,
                triton_grpo_loss=True,
                triton_entropy_mask=False,
                triton_lora=True,
                steps=20,
                tags=["minus_one", "entropy"],
            ),
            BenchmarkConfig(
                name="triton_all_except_lora",
                description="All Triton kernel families enabled except LoRA forward",
                triton=True,
                gradient_accumulation_steps=16,
                group_size=4,
                triton_generation=True,
                triton_grpo_loss=True,
                triton_entropy_mask=True,
                triton_lora=False,
                steps=20,
                tags=["minus_one", "lora"],
            ),
        ]

        if filter_configs:
            filter_set = {name.strip() for name in filter_configs if name.strip()}
            if filter_set:
                configs = [c for c in configs if c.name in filter_set]

        return configs


def main() -> None:
    parser = argparse.ArgumentParser(description="Triton Matrix Benchmark Suite")
    parser.add_argument(
        "--filter",
        type=str,
        help="Filter by tag (e.g. 'single', 'minus_one', 'baseline')",
    )
    parser.add_argument(
        "--filter-configs",
        type=str,
        help="Comma-separated list of config names to run",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./benchmarks/output/triton_matrix"
    )
    parser.add_argument(
        "--run-prefix",
        type=str,
        default="triton_matrix",
        help="Prefix for per-run output directories and WandB run names.",
    )
    parser.add_argument("--list", action="store_true", help="List all test configs")

    args = parser.parse_args()

    suite = TritonMatrixBenchmarkSuite(
        output_dir=args.output_dir,
        run_prefix=args.run_prefix,
    )

    filter_configs = None
    if args.filter_configs:
        filter_configs = [
            name.strip() for name in args.filter_configs.split(",") if name.strip()
        ]

    if args.list:
        configs = suite.define_test_matrix(filter_configs=filter_configs)
        print(f"Total configs: {len(configs)}")
        for config in configs:
            tag_list = config.tags or []
            print(
                f"  {config.name}: {config.description} [tags: {', '.join(tag_list)}]"
            )
        return

    suite.run_all(filter_tag=args.filter, filter_configs=filter_configs)


if __name__ == "__main__":
    main()
