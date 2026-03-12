from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from optimizer.adapters.benchmark import CompareBenchAdapter
from optimizer.adapters.profiler import build_profiler_command
from optimizer.adapters.triton import build_triton_command
from optimizer.runtime import OptimizerRuntime, RuntimeConfig


@dataclass(frozen=True)
class OptimizerTemplate:
    baseline: str
    candidates: list[str]
    benchmark_steps: int
    triton_mode: str
    artifacts_dir: Path
    iteration_budget: int | None
    open_ended: bool
    trace_summary_path: Path | None
    allow_recovered_oom_promotion: bool

    @classmethod
    def load(cls, path: str | Path) -> "OptimizerTemplate":
        payload = _load_json_object(path)
        template_path = Path(path).resolve()
        benchmark = _require_mapping(payload.get("benchmark"), "benchmark")
        output = _require_mapping(payload.get("output"), "output")
        policy = _require_mapping(payload.get("policy"), "policy")
        candidates = payload.get("candidates")
        if not isinstance(candidates, list) or not all(
            isinstance(item, str) for item in candidates
        ):
            raise ValueError("Template field 'candidates' must be a list of strings.")

        return cls(
            baseline=_require_str(payload, "baseline"),
            candidates=list(candidates),
            benchmark_steps=_required_int(
                benchmark.get("steps", 5), field_name="benchmark.steps"
            ),
            triton_mode=str(benchmark.get("triton", "auto")),
            artifacts_dir=Path(
                str(output.get("artifacts_dir", "optimizer/artifacts/records"))
            ),
            iteration_budget=_optional_int(payload.get("iteration_budget"), default=1),
            open_ended=_optional_bool(payload.get("open_ended"), default=False),
            trace_summary_path=_optional_path(
                payload.get("trace_summary_path"),
                template_path=template_path,
            ),
            allow_recovered_oom_promotion=_optional_bool(
                policy.get("allow_recovered_oom_promotion"),
                default=True,
            ),
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sequential optimizer entrypoint")
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    template = OptimizerTemplate.load(args.template)
    repo_root = Path(__file__).resolve().parent.parent

    if args.dry_run:
        _print_dry_run(template=template, repo_root=repo_root)
        return 0

    runtime = OptimizerRuntime(
        repo_root=repo_root,
        artifacts_dir=repo_root / template.artifacts_dir,
    )
    session, iterations = runtime.run_control_loop(
        RuntimeConfig(
            baseline=template.baseline,
            iteration_budget=template.iteration_budget,
            open_ended=template.open_ended,
            candidate_targets=template.candidates,
            allow_recovered_oom_promotion=template.allow_recovered_oom_promotion,
            benchmark_steps=template.benchmark_steps,
            benchmark_triton_mode=template.triton_mode,
            trace_summary_path=(
                None
                if template.trace_summary_path is None
                else str(template.trace_summary_path)
            ),
        )
    )
    print(f"status={session.status}")
    print(f"session_id={session.session_id}")
    print(f"iterations_completed={session.iterations_completed}")
    for iteration in iterations:
        print(
            "iteration=%s candidate_id=%s selected_target=%s worktree_path=%s mutation_plan_path=%s"
            % (
                iteration.iteration_index,
                iteration.candidate_id,
                iteration.selected_target,
                iteration.worktree_path,
                iteration.mutation_plan_path,
            )
        )
    return 0


def _print_dry_run(*, template: OptimizerTemplate, repo_root: Path) -> None:
    adapter = CompareBenchAdapter(
        repo_root=repo_root,
        steps=template.benchmark_steps,
        triton_mode=template.triton_mode,
    )
    sample_candidate = template.candidates[0] if template.candidates else "<candidate>"

    print("mode=dry-run")
    print("sequential=true")
    print("decisive_tool=compare_bench.sh")
    print("diagnostic_tools=compare_triton.py,profiler")
    print(f"baseline={template.baseline}")
    print(f"candidate_count={len(template.candidates)}")
    print(f"benchmark_steps={template.benchmark_steps}")
    print(f"triton_mode={template.triton_mode}")
    print(f"artifacts_dir={template.artifacts_dir}")
    print(f"iteration_budget={template.iteration_budget}")
    print(f"open_ended={template.open_ended}")
    print(f"trace_summary_path={template.trace_summary_path}")
    print(f"allow_recovered_oom_promotion={template.allow_recovered_oom_promotion}")
    print(f"iteration_targets={template.candidates}")
    print(
        "sample_benchmark_command=%s"
        % " ".join(
            adapter.build_command(
                frontier_target=template.baseline,
                candidate_target=sample_candidate,
            )
        )
    )
    print(
        "sample_triton_command=%s"
        % " ".join(
            build_triton_command(
                repo_root=repo_root,
                max_steps=template.benchmark_steps,
            )
        )
    )
    print(
        "sample_profiler_command=%s"
        % " ".join(
            build_profiler_command(
                repo_root=repo_root,
                target=sample_candidate,
                max_steps=template.benchmark_steps,
            )
        )
    )
    print("handoff_mode=external_ai_agent")
    print("next_step=generate_observation_worktree_and_mutation_plan")


def _load_json_object(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Optimizer template root must be a JSON object.")
    return {str(key): value for key, value in payload.items()}


def _require_mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"Template field '{name}' must be an object.")
    return {str(key): item for key, item in value.items()}


def _require_str(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise ValueError(f"Template field '{key}' must be a string.")
    return value


def _optional_int(value: object, *, default: int | None) -> int | None:
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value)
    raise ValueError(
        f"Expected optional integer-compatible value, got {type(value)!r}."
    )


def _required_int(value: object, *, field_name: str) -> int:
    parsed = _optional_int(value, default=None)
    if parsed is None:
        raise ValueError(f"Template field '{field_name}' must be an integer.")
    return parsed


def _optional_path(value: object, *, template_path: Path) -> Path | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        candidate = Path(value)
        if candidate.is_absolute():
            return candidate
        return (template_path.parent / candidate).resolve()
    raise ValueError(f"Expected optional path string, got {type(value)!r}.")


def _optional_bool(value: object, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(
        f"Expected optional boolean-compatible value, got {type(value)!r}."
    )


if __name__ == "__main__":
    sys.exit(main())
