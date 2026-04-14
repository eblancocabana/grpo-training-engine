#!/usr/bin/env python3
"""
Focused benchmark suite for the 512-token response-length training workload.

This suite is narrower than the general ING INF benchmark matrix. It is meant
to answer one practical question for the RTX 3060 Ti target:

Which backend mix should be the default when max_response_length=512?

The matrix includes:
1. Core backend comparison at group_size=4 / grad_accum=16.
2. Follow-up tuning runs around the most promising candidate.
"""

from __future__ import annotations

import argparse
import re
import time
from dataclasses import replace
from typing import Dict, List, Optional

from ing_inf_suite import BenchmarkConfig, BenchmarkResult, IngInfBenchmarkSuite


class ResponseLen512BenchmarkSuite(IngInfBenchmarkSuite):
    """Focused benchmark suite for max_response_length=512."""

    CORE_CONFIG_NAMES = [
        "triton_gen_off_g4_ga16",
        "triton_gen_auto_g4_ga16",
        "triton_gen_on_g4_ga16",
        "torch_baseline_g4_ga16",
    ]

    def __init__(
        self,
        output_dir: str = "./benchmarks/output/response_len_512",
        run_prefix: str = "response_len_512",
        *,
        steps: int = 20,
        use_wandb: bool = False,
    ):
        super().__init__(output_dir=output_dir, run_prefix=run_prefix)
        self.suite_name = "Response Length 512 Benchmark Suite"
        self.report_title = "Response Length 512 Benchmark Report"
        self.steps = steps
        self.use_wandb = use_wandb

    def _common_kwargs(self) -> Dict[str, object]:
        return dict(
            batch_size=1,
            lora_rank=16,
            lora_adapter_quant="none",
            max_prompt_length=128,
            max_response_length=512,
            disable_sent=True,
            steps=self.steps,
            use_wandb=self.use_wandb,
        )

    def _core_configs(self) -> List[BenchmarkConfig]:
        common = self._common_kwargs()
        return [
            BenchmarkConfig(
                name="triton_gen_off_g4_ga16",
                description=(
                    "Recommended candidate: Triton kernels on, torch generation, "
                    "group_size=4, grad_accum=16"
                ),
                triton=True,
                gradient_accumulation_steps=16,
                group_size=4,
                triton_generation=False,
                triton_generation_mode="off",
                triton_grpo_loss=True,
                triton_entropy_mask=True,
                triton_lora=True,
                tags=["core", "recommended", "generation_off", "g4", "ga16"],
                **common,
            ),
            BenchmarkConfig(
                name="triton_gen_auto_g4_ga16",
                description=(
                    "Auto generation policy with Triton kernels on, "
                    "group_size=4, grad_accum=16"
                ),
                triton=True,
                gradient_accumulation_steps=16,
                group_size=4,
                triton_generation=True,
                triton_generation_mode="auto",
                triton_grpo_loss=True,
                triton_entropy_mask=True,
                triton_lora=True,
                tags=["core", "generation_auto", "g4", "ga16"],
                **common,
            ),
            BenchmarkConfig(
                name="triton_gen_on_g4_ga16",
                description=(
                    "Forced Triton generation with Triton kernels on, "
                    "group_size=4, grad_accum=16"
                ),
                triton=True,
                gradient_accumulation_steps=16,
                group_size=4,
                triton_generation=True,
                triton_generation_mode="on",
                triton_grpo_loss=True,
                triton_entropy_mask=True,
                triton_lora=True,
                tags=["core", "generation_on", "g4", "ga16"],
                **common,
            ),
            BenchmarkConfig(
                name="torch_baseline_g4_ga16",
                description="Full torch baseline, group_size=4, grad_accum=16",
                triton=False,
                gradient_accumulation_steps=16,
                group_size=4,
                tags=["core", "torch", "baseline", "g4", "ga16"],
                **common,
            ),
        ]

    def _family_prefix(self, config_name: str) -> str:
        return re.sub(r"_g\d+_ga\d+$", "", config_name)

    def _backend_label(self, config: BenchmarkConfig) -> str:
        if not config.triton:
            return "Full torch baseline"
        mode = config.triton_generation_mode
        if mode == "off" or not config.triton_generation:
            return "Triton kernels on, torch generation"
        if mode == "auto":
            return "Triton kernels on, auto generation"
        return "Triton kernels on, forced Triton generation"

    def _followup_config(
        self,
        base_config: BenchmarkConfig,
        *,
        group_size: int,
        grad_accum: int,
    ) -> BenchmarkConfig:
        prefix = self._family_prefix(base_config.name)
        name = f"{prefix}_g{group_size}_ga{grad_accum}"

        if group_size == 8 and grad_accum == 16:
            description = (
                f"Rollout-diversity follow-up: {self._backend_label(base_config)}, "
                f"group_size=8, grad_accum=16"
            )
            tags = ["optional", "winner_followup", f"g{group_size}", f"ga{grad_accum}"]
        else:
            description = (
                f"Winner follow-up: {self._backend_label(base_config)}, "
                f"group_size={group_size}, grad_accum={grad_accum}"
            )
            tags = ["tuning", "winner_followup", f"g{group_size}", f"ga{grad_accum}"]

        return replace(
            base_config,
            name=name,
            description=description,
            gradient_accumulation_steps=grad_accum,
            group_size=group_size,
            tags=tags,
        )

    def _all_configs(self) -> List[BenchmarkConfig]:
        configs: List[BenchmarkConfig] = []
        for core_config in self._core_configs():
            configs.append(core_config)
            configs.append(
                self._followup_config(core_config, group_size=4, grad_accum=8)
            )
            configs.append(
                self._followup_config(core_config, group_size=4, grad_accum=32)
            )
            configs.append(
                self._followup_config(core_config, group_size=8, grad_accum=16)
            )
        return configs

    def define_test_matrix(
        self, filter_configs: Optional[List[str]] = None
    ) -> List[BenchmarkConfig]:
        configs = self._all_configs()

        if filter_configs:
            filter_set = {name.strip() for name in filter_configs if name.strip()}
            if filter_set:
                configs = [c for c in configs if c.name in filter_set]

        return configs

    def _estimate_runtime_minutes(self, configs: List[BenchmarkConfig]) -> float:
        est_time_min = 0.0
        for config in configs:
            base_time = 25 * (config.steps / 30)
            if not config.triton:
                base_time *= 2.5
            base_time *= config.max_response_length / 1024
            est_time_min += base_time
        return est_time_min

    def _select_winner(
        self, results: List[BenchmarkResult]
    ) -> Optional[BenchmarkResult]:
        successful = [result for result in results if result.success]
        if not successful:
            return None

        successful_with_steps = [
            result for result in successful if result.step_time_avg_s is not None
        ]
        if successful_with_steps:
            return min(
                successful_with_steps,
                key=lambda result: (
                    result.step_time_avg_s or float("inf"),
                    -(result.tokens_per_sec or 0.0),
                    result.duration_s,
                ),
            )

        return min(successful, key=lambda result: result.duration_s)

    def _winner_followups(
        self,
        winner_config: BenchmarkConfig,
        *,
        include_rollout_diversity: bool,
    ) -> List[BenchmarkConfig]:
        followups = [
            self._followup_config(winner_config, group_size=4, grad_accum=8),
            self._followup_config(winner_config, group_size=4, grad_accum=32),
        ]
        if include_rollout_diversity:
            followups.append(
                self._followup_config(winner_config, group_size=8, grad_accum=16)
            )
        return followups

    def _run_sequence(
        self,
        configs: List[BenchmarkConfig],
        *,
        total_runs: int,
        start_index: int,
        start_time: float,
    ) -> List[BenchmarkResult]:
        phase_results: List[BenchmarkResult] = []
        for index, config in enumerate(configs, start=start_index):
            print(f"\n[{index}/{total_runs}] ", end="")
            result = self.run_benchmark(config)
            self.results.append(result)
            phase_results.append(result)
            self._save_results()

            elapsed = (time.time() - start_time) / 60
            print(f"\n[PROGRESS] {index}/{total_runs} complete")
            print(f"[PROGRESS] Elapsed: {elapsed:.1f} min")

        return phase_results

    def run_auto_workflow(self, *, include_rollout_diversity: bool = False) -> None:
        self.results = []

        core_configs = self._core_configs()
        total_runs = len(core_configs) + 2 + (1 if include_rollout_diversity else 0)
        planned_est_min = self._estimate_runtime_minutes(core_configs)

        print(f"\n{'='*60}")
        print(f"{self.suite_name}")
        print(f"{'='*60}")
        print("Workflow:")
        print("1. Run core 512-token comparison matrix")
        print("2. Pick the fastest successful core config by step time")
        print("3. Run winner follow-ups at grad_accum=8 and grad_accum=32")
        if include_rollout_diversity:
            print("4. Run rollout-diversity follow-up at group_size=8")
        print(f"Initial core runs: {len(core_configs)}")
        print(f"Planned total runs: {total_runs}")
        print(f"Output directory: {self.output_dir}")
        print(f"Initial core estimate: ~{planned_est_min:.1f} minutes")
        print(f"Train script: {self.train_script}")
        print(f"{'='*60}")
        print("[INFO] Auto-starting in 3 seconds...")
        time.sleep(3)

        start_time = time.time()
        core_results = self._run_sequence(
            core_configs,
            total_runs=total_runs,
            start_index=1,
            start_time=start_time,
        )

        winner = self._select_winner(core_results)
        if winner is None:
            print("\n[WARN] No successful core runs. Skipping winner follow-ups.")
            total_elapsed = (time.time() - start_time) / 60
            print(f"\n{'='*60}")
            print("Benchmark workflow complete")
            print(f"Total time: {total_elapsed:.1f} minutes")
            print(f"{'='*60}")
            self._generate_report()
            return

        config_map = {config.name: config for config in self._all_configs()}
        winner_config = config_map[winner.config_name]
        followups = self._winner_followups(
            winner_config,
            include_rollout_diversity=include_rollout_diversity,
        )

        print(
            f"\n[WINNER] {winner.config_name} selected for follow-ups "
            f"(step_time={winner.step_time_avg_s or -1.0:.2f}s, "
            f"tok/s={winner.tokens_per_sec or -1.0:.2f})"
        )

        self._run_sequence(
            followups,
            total_runs=total_runs,
            start_index=len(core_configs) + 1,
            start_time=start_time,
        )

        total_elapsed = (time.time() - start_time) / 60
        print(f"\n{'='*60}")
        print("Benchmark workflow complete")
        print(f"Total time: {total_elapsed:.1f} minutes ({total_elapsed/60:.1f} hours)")
        print(f"{'='*60}")

        self._generate_report()

    def run_all(
        self,
        filter_tag: Optional[str] = None,
        filter_configs: Optional[List[str]] = None,
        *,
        include_rollout_diversity: bool = False,
    ):
        if filter_tag or filter_configs:
            super().run_all(filter_tag=filter_tag, filter_configs=filter_configs)
            return
        self.run_auto_workflow(
            include_rollout_diversity=include_rollout_diversity,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Response Length 512 Benchmark Suite")
    parser.add_argument(
        "--filter",
        type=str,
        help="Filter by tag (e.g. 'core', 'tuning', 'optional')",
    )
    parser.add_argument(
        "--filter-configs",
        type=str,
        help="Comma-separated list of config names to run",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./benchmarks/output/response_len_512",
    )
    parser.add_argument(
        "--run-prefix",
        type=str,
        default="response_len_512",
        help="Prefix for per-run output directories and WandB run names.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="Training steps per config.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable WandB logging for the suite.",
    )
    parser.add_argument(
        "--include-rollout-diversity",
        action="store_true",
        help="After winner selection, also run the optional group_size=8 follow-up.",
    )
    parser.add_argument("--list", action="store_true", help="List all test configs")

    args = parser.parse_args()

    suite = ResponseLen512BenchmarkSuite(
        output_dir=args.output_dir,
        run_prefix=args.run_prefix,
        steps=args.steps,
        use_wandb=args.wandb,
    )

    filter_configs = None
    if args.filter_configs:
        filter_configs = [
            name.strip() for name in args.filter_configs.split(",") if name.strip()
        ]

    if args.list:
        configs = suite.define_test_matrix(filter_configs=filter_configs)
        print(f"Total configs: {len(configs)}")
        print("Default workflow without filters:")
        print("  1. Run the 4 core configs")
        print("  2. Pick the fastest successful core config by step time")
        print("  3. Run the winner again with grad_accum=8 and grad_accum=32")
        print("  4. Optionally run the winner again with group_size=8")
        for config in configs:
            tag_list = config.tags or []
            print(
                f"  {config.name}: {config.description} [tags: {', '.join(tag_list)}]"
            )
        return

    suite.run_all(
        filter_tag=args.filter,
        filter_configs=filter_configs,
        include_rollout_diversity=args.include_rollout_diversity,
    )


if __name__ == "__main__":
    main()
