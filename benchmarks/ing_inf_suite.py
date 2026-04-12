#!/usr/bin/env python3
"""
ING INF (Ingeniería Informática) Benchmark Suite
Comprehensive performance testing for system optimization analysis.

Tests throughput, VRAM usage, kernel performance, and step time breakdowns
across multiple configurations relevant to the engineering-focused PFG.
"""

import subprocess
import json
import time
import os
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any, List, Dict, Optional
import re

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def note_cleanup_moved() -> None:
    print("[CLEANUP] train.py handles stale-process cleanup and GPU cache.")


def get_conda_python() -> str:
    """Get the Python executable from conda environment."""
    # Try to find conda python
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "grpo-3060ti")
    
    # Common paths
    possible_paths = [
        f"{os.environ.get('HOME', '/home/ndk')}/.conda/envs/{conda_env}/bin/python",
        f"/opt/anaconda/envs/{conda_env}/bin/python",
        f"{os.environ.get('CONDA_PREFIX', '')}/bin/python",
        sys.executable,  # Fallback to current python
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return sys.executable


def run_with_conda(cmd: List[str], **kwargs) -> subprocess.Popen[str]:
    """Run command with conda environment activated."""
    python_exe = get_conda_python()
    
    # Replace 'python' with conda python path
    if cmd[0] in ['python', 'python3', sys.executable]:
        cmd[0] = python_exe
    
    # Set environment variables for clean run
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    env['CUDA_VISIBLE_DEVICES'] = '0'
    # WANDB enabled - remove WANDB_DISABLED
    
    return subprocess.Popen(cmd, env=env, **kwargs)


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark run."""
    name: str
    description: str
    # Core params
    triton: bool = True
    gradient_accumulation_steps: int = 16
    group_size: int = 4
    batch_size: int = 1
    # LoRA params
    lora_rank: int = 16
    lora_adapter_quant: str = "8bit"
    # Generation params
    max_prompt_length: int = 128
    max_response_length: int = 1024
    # Optimization params
    use_entropy_mask: bool = True
    enable_gradient_checkpointing: bool = True
    disable_sent: bool = False
    # Tuning
    triton_lora_prefer_base: bool = False
    triton_generation_mode: Optional[str] = None
    triton_generation: Optional[bool] = None
    triton_grpo_loss: Optional[bool] = None
    triton_entropy_mask: Optional[bool] = None
    triton_lora: Optional[bool] = None
    # Profile
    profile: bool = False
    # Steps
    steps: int = 20
    # Tags
    tags: Optional[List[str]] = None
    # WandB configuration
    use_wandb: bool = True
    wandb_project: str = "grpo-training"
    wandb_run_name: Optional[str] = None
    wandb_tags: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.wandb_tags is None:
            self.wandb_tags = []
    
    @property
    def effective_batch(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps * self.group_size
    
    def to_args(self) -> List[str]:
        """Convert config to command line arguments."""
        args = [
            "--no-checkpoints",
            "--no-initial-benchmark",
            f"--group-size", str(self.group_size),
            f"--lora-rank", str(self.lora_rank),
            f"--lora-adapter-quant", self.lora_adapter_quant,
            f"--max-prompt-length", str(self.max_prompt_length),
            f"--max-response-length", str(self.max_response_length),
            f"--gradient-accumulation-steps", str(self.gradient_accumulation_steps),
            f"--max-steps", str(self.steps),
        ]
        
        # WandB
        if self.use_wandb:
            args.append("--wandb")
            args.extend(["--wandb-project", self.wandb_project])
            if self.wandb_run_name:
                args.extend(["--wandb-run-name", self.wandb_run_name])
            if self.wandb_tags:
                args.extend(["--wandb-tags"] + self.wandb_tags)
        else:
            args.append("--no-wandb")
        
        # Triton
        if self.triton:
            args.append("--use-triton")
        else:
            args.append("--no-triton")

        if self.triton_generation_mode is not None:
            args.extend(["--triton-generation-mode", self.triton_generation_mode])

        if self.triton_generation is True:
            args.append("--triton-generation")
        elif self.triton_generation is False:
            args.append("--no-triton-generation")

        if self.triton_grpo_loss is True:
            args.append("--triton-grpo-loss")
        elif self.triton_grpo_loss is False:
            args.append("--no-triton-grpo-loss")

        if self.triton_entropy_mask is True:
            args.append("--triton-entropy-mask")
        elif self.triton_entropy_mask is False:
            args.append("--no-triton-entropy-mask")

        if self.triton_lora is True:
            args.append("--triton-lora")
        elif self.triton_lora is False:
            args.append("--no-triton-lora")
        
        # Triton LoRA prefer base
        if self.triton and self.triton_lora_prefer_base:
            args.append("--triton-lora-prefer-base")
        
        # Entropy mask
        if self.use_entropy_mask:
            args.append("--use-entropy-mask")
        else:
            args.append("--no-mask-truncated")

        if self.disable_sent:
            args.append("--no-sent")
        
        # Profile
        if self.profile:
            args.append("--profile")
        
        return args


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    config_name: str
    success: bool
    duration_s: float
    steps_completed: int
    
    # Performance metrics
    step_time_avg_s: Optional[float] = None
    step_time_min_s: Optional[float] = None
    step_time_max_s: Optional[float] = None
    tokens_per_sec: Optional[float] = None
    tokens_per_sec_min: Optional[float] = None
    tokens_per_sec_max: Optional[float] = None
    
    # VRAM metrics
    vram_peak_gb: Optional[float] = None
    vram_avg_gb: Optional[float] = None
    
    # Quality metrics
    loss_avg: Optional[float] = None
    loss_final: Optional[float] = None
    reward_avg: Optional[float] = None
    reward_final: Optional[float] = None
    
    # Error info
    error_message: Optional[str] = None
    oom_events: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class IngInfBenchmarkSuite:
    """Benchmark suite for ING INF PFG."""
    
    def __init__(self, output_dir: str = "./benchmarks/output/ing_inf", run_prefix: str = "ing_inf"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[BenchmarkResult] = []
        self.suite_name = "ING INF Benchmark Suite"
        self.report_title = "ING INF Benchmark Report"
        self.run_prefix = self._normalize_prefix(run_prefix)
        
        # Find train.py
        self.train_script = Path(__file__).parent.parent / "train.py"
        if not self.train_script.exists():
            raise FileNotFoundError(f"train.py not found at {self.train_script}")

    def _normalize_prefix(self, prefix: Optional[str]) -> str:
        """Normalize a prefix for output directories and WandB run names."""
        if prefix is None:
            return ""
        normalized = prefix.strip().strip("_-")
        return normalized

    def _prefixed_name(self, name: str) -> str:
        """Apply the suite prefix to a config/run name."""
        if not self.run_prefix:
            return name
        return f"{self.run_prefix}_{name}"
    
    def _parse_log(self, log_path: Path) -> Dict[str, Any]:
        """Parse training log for metrics."""
        metrics: Dict[str, Any] = {
            "steps_completed": 0,
            "step_times": [],
            "tokens_per_sec": [],
            "losses": [],
            "rewards": [],
            "vram_samples": [],
            "oom_events": 0,
            "error": None,
        }
        
        if not log_path.exists():
            return metrics
        
        content = log_path.read_text(encoding="utf-8", errors="ignore")

        # tqdm rewrites the same progress line multiple times using carriage returns.
        # Parse per-step metrics from the latest segment for each reported step.
        step_records: Dict[int, Dict[str, float]] = {}
        for segment in re.split(r'[\r\n]+', content):
            step_match = re.search(r'step=(\d+)', segment)
            if not step_match:
                continue

            step = int(step_match.group(1))
            record = step_records.setdefault(step, {})

            step_time_match = re.search(r'(\d+\.?\d*)s/it', segment)
            if step_time_match:
                record["step_time"] = float(step_time_match.group(1))

            tokens_match = re.search(r'tokens_per_sec[=:]\s*([\d.]+)', segment)
            if tokens_match:
                record["tokens_per_sec"] = float(tokens_match.group(1))

            loss_match = re.search(r'loss[=:]\s*(-?[\d.]+)', segment)
            if loss_match:
                record["loss"] = float(loss_match.group(1))

            reward_match = re.search(r'reward[=:]\s*(-?[\d.]+)', segment)
            if reward_match:
                record["reward"] = float(reward_match.group(1))

        metrics["steps_completed"] = len(step_records)
        metrics["step_times"] = [
            record["step_time"]
            for _, record in sorted(step_records.items())
            if "step_time" in record
        ]
        metrics["tokens_per_sec"] = [
            record["tokens_per_sec"]
            for _, record in sorted(step_records.items())
            if "tokens_per_sec" in record
        ]
        metrics["losses"] = [
            record["loss"]
            for _, record in sorted(step_records.items())
            if "loss" in record
        ]
        metrics["rewards"] = [
            record["reward"]
            for _, record in sorted(step_records.items())
            if "reward" in record
        ]
        
        # Parse VRAM from log
        vram_pattern = r'VRAM:\s*([\d.]+)\s*GB'
        for match in re.finditer(vram_pattern, content):
            try:
                metrics["vram_samples"].append(float(match.group(1)))
            except ValueError:
                pass
        
        # Check for OOM
        if "CUDA out of memory" in content or "OOM" in content:
            metrics["oom_events"] += 1
        
        # Check for errors
        if "Traceback" in content or "Error" in content:
            lines = content.split('\n')
            for line in reversed(lines):
                if 'Error' in line or 'Exception' in line:
                    metrics["error"] = line.strip()
                    break
        
        return metrics
    
    def run_benchmark(self, config: BenchmarkConfig) -> BenchmarkResult:
        """Run a single benchmark configuration."""
        print(f"\n{'='*60}")
        print(f"Running: {config.name}")
        print(f"Description: {config.description}")
        print(f"Effective batch: {config.effective_batch}")
        print(f"{'='*60}")
        
        note_cleanup_moved()
        
        run_name = self._prefixed_name(config.name)
        run_dir = self.output_dir / run_name.replace(" ", "_").replace("/", "_")
        run_dir.mkdir(parents=True, exist_ok=True)
        
        log_path = run_dir / "train.log"
        
        # Set WandB run name if not set
        if config.use_wandb and not config.wandb_run_name:
            # Clear, concise name with key params
            config.wandb_run_name = self._prefixed_name(f"INF-{config.name}")
            base_tags = config.tags or []
            config.wandb_tags = base_tags + ["ing_inf", "benchmark"]
        
        # Use conda python
        python_exe = get_conda_python()
        cmd = [
            python_exe,
            str(self.train_script),
            *config.to_args(),
            "--output-dir", str(run_dir),
        ]
        
        print(f"Python: {python_exe}")
        print(f"Command: {' '.join(cmd[:5])} ...")  # Truncate for readability
        print(f"Log: {log_path}")
        
        start_time = time.time()
        
        process = None
        try:
            with open(log_path, 'w') as log_file:
                process = run_with_conda(
                    cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    cwd=str(self.train_script.parent),
                )
                process.wait(timeout=3600)
                success = process.returncode == 0
        except subprocess.TimeoutExpired:
            if process is not None:
                process.kill()
            success = False
            print("[TIMEOUT] Benchmark exceeded 1 hour")
        except Exception as e:
            success = False
            print(f"[ERROR] {e}")
        
        duration = time.time() - start_time
        
        metrics = self._parse_log(log_path)
        
        step_times = metrics.get("step_times", [])
        tokens_per_sec = metrics.get("tokens_per_sec", [])
        losses = metrics.get("losses", [])
        rewards = metrics.get("rewards", [])
        vram_samples = metrics.get("vram_samples", [])
        
        result = BenchmarkResult(
            config_name=config.name,
            success=success,
            duration_s=duration,
            steps_completed=metrics.get("steps_completed", 0),
            step_time_avg_s=sum(step_times) / len(step_times) if step_times else None,
            step_time_min_s=min(step_times) if step_times else None,
            step_time_max_s=max(step_times) if step_times else None,
            tokens_per_sec=sum(tokens_per_sec) / len(tokens_per_sec) if tokens_per_sec else None,
            tokens_per_sec_min=min(tokens_per_sec) if tokens_per_sec else None,
            tokens_per_sec_max=max(tokens_per_sec) if tokens_per_sec else None,
            vram_peak_gb=max(vram_samples) if vram_samples else None,
            vram_avg_gb=sum(vram_samples) / len(vram_samples) if vram_samples else None,
            loss_avg=sum(losses) / len(losses) if losses else None,
            loss_final=losses[-1] if losses else None,
            reward_avg=sum(rewards) / len(rewards) if rewards else None,
            reward_final=rewards[-1] if rewards else None,
            error_message=metrics.get("error"),
            oom_events=metrics.get("oom_events", 0),
        )
        
        print(f"\nResults for {config.name}:")
        print(f"  Success: {success}")
        print(f"  Duration: {duration:.1f}s")
        print(f"  Steps: {result.steps_completed}")
        if result.step_time_avg_s:
            print(f"  Step time: {result.step_time_avg_s:.2f}s")
        if result.tokens_per_sec:
            print(f"  Throughput: {result.tokens_per_sec:.2f} tok/s")
        if result.vram_peak_gb:
            print(f"  VRAM peak: {result.vram_peak_gb:.2f} GB")
        
        return result
    
    def define_test_matrix(self, filter_configs: Optional[List[str]] = None) -> List[BenchmarkConfig]:
        """Define the complete test matrix for ING INF."""
        configs = []
        
        # === BASELINE TESTS ===
        configs.append(BenchmarkConfig(
            name="triton_on_baseline",
            description="Triton-on with grad_accum=4 (matches frontier benchmark)",
            triton=True,
            gradient_accumulation_steps=4,
            group_size=4,
            disable_sent=True,
            steps=20,
            tags=["baseline", "triton_on", "frontier"],
        ))
        
        configs.append(BenchmarkConfig(
            name="triton_on_current",
            description="Triton-on with grad_accum=16 (current config)",
            triton=True,
            gradient_accumulation_steps=16,
            group_size=4,
            disable_sent=True,
            steps=20,
            tags=["baseline", "triton_on", "current"],
        ))
        
        configs.append(BenchmarkConfig(
            name="triton_off_baseline",
            description="Triton-off with grad_accum=4",
            triton=False,
            gradient_accumulation_steps=4,
            group_size=4,
            disable_sent=True,
            steps=20,
            tags=["baseline", "triton_off"],
        ))
        
        configs.append(BenchmarkConfig(
            name="triton_off_current",
            description="Triton-off with grad_accum=16",
            triton=False,
            gradient_accumulation_steps=16,
            group_size=4,
            disable_sent=True,
            steps=20,
            tags=["baseline", "triton_off"],
        ))
        
        # === GRADIENT ACCUMULATION SWEEP ===
        for ga in [1, 2, 4, 8, 16, 32]:
            configs.append(BenchmarkConfig(
                name=f"triton_on_grad_accum_{ga}",
                description=f"Triton-on with grad_accum={ga} (effective batch={4*ga})",
                triton=True,
                gradient_accumulation_steps=ga,
                group_size=4,
                disable_sent=True,
                steps=20,
                tags=["sweep", "grad_accum", "triton_on"],
            ))
            
            configs.append(BenchmarkConfig(
                name=f"triton_off_grad_accum_{ga}",
                description=f"Triton-off with grad_accum={ga} (effective batch={4*ga})",
                triton=False,
                gradient_accumulation_steps=ga,
                group_size=4,
                disable_sent=True,
                steps=20,
                tags=["sweep", "grad_accum", "triton_off"],
            ))
        
        # === GROUP SIZE SWEEP ===
        for gs in [2, 4, 8, 16]:
            configs.append(BenchmarkConfig(
                name=f"triton_on_group_size_{gs}",
                description=f"Triton-on with group_size={gs}",
                triton=True,
                gradient_accumulation_steps=4,
                group_size=gs,
                disable_sent=True,
                steps=20,
                tags=["sweep", "group_size", "triton_on"],
            ))
        
        # === LORA RANK SWEEP ===
        for rank in [4, 8, 16, 32, 64]:
            configs.append(BenchmarkConfig(
                name=f"triton_on_lora_rank_{rank}",
                description=f"Triton-on with LoRA rank={rank}",
                triton=True,
                gradient_accumulation_steps=4,
                group_size=4,
                lora_rank=rank,
                disable_sent=True,
                steps=20,
                tags=["sweep", "lora_rank", "triton_on"],
            ))
        
        # === SEQUENCE LENGTH SWEEP ===
        for max_len in [256, 512, 768, 1024]:
            configs.append(BenchmarkConfig(
                name=f"triton_on_response_len_{max_len}",
                description=f"Triton-on with max_response_length={max_len}",
                triton=True,
                gradient_accumulation_steps=4,
                group_size=4,
                max_response_length=max_len,
                disable_sent=True,
                steps=20,
                tags=["sweep", "response_length", "triton_on"],
            ))
        
        # === ENTROPY MASK TESTS ===
        configs.append(BenchmarkConfig(
            name="triton_on_entropy_mask_on",
            description="Triton-on with entropy mask enabled",
            triton=True,
            gradient_accumulation_steps=4,
            group_size=4,
            use_entropy_mask=True,
            disable_sent=True,
            steps=20,
            tags=["feature", "entropy_mask", "triton_on"],
        ))
        
        configs.append(BenchmarkConfig(
            name="triton_on_entropy_mask_off",
            description="Triton-on with entropy mask disabled",
            triton=True,
            gradient_accumulation_steps=4,
            group_size=4,
            use_entropy_mask=False,
            disable_sent=True,
            steps=20,
            tags=["feature", "entropy_mask", "triton_on"],
        ))
        
        # === GRADIENT CHECKPOINTING TESTS ===
        configs.append(BenchmarkConfig(
            name="triton_on_checkpointing_on",
            description="Triton-on with gradient checkpointing enabled",
            triton=True,
            gradient_accumulation_steps=4,
            group_size=4,
            enable_gradient_checkpointing=True,
            disable_sent=True,
            steps=20,
            tags=["feature", "checkpointing", "triton_on"],
        ))
        
        configs.append(BenchmarkConfig(
            name="triton_on_checkpointing_off",
            description="Triton-on with gradient checkpointing disabled",
            triton=True,
            gradient_accumulation_steps=4,
            group_size=4,
            enable_gradient_checkpointing=False,
            disable_sent=True,
            steps=20,
            tags=["feature", "checkpointing", "triton_on"],
        ))
        
        # === LORA QUANTIZATION TESTS ===
        for quant in ["4bit", "8bit", "none"]:
            configs.append(BenchmarkConfig(
                name=f"triton_on_lora_quant_{quant}",
                description=f"Triton-on with LoRA {quant} quantization",
                triton=True,
                gradient_accumulation_steps=4,
                group_size=4,
                lora_adapter_quant=quant,
                disable_sent=True,
                steps=20,
                tags=["feature", "lora_quant", "triton_on"],
            ))
        
        # === PROFILE RUN (shorter) ===
        configs.append(BenchmarkConfig(
            name="triton_on_profile",
            description="Triton-on with profiler enabled (20 steps)",
            triton=True,
            gradient_accumulation_steps=4,
            group_size=4,
            profile=True,
            disable_sent=True,
            steps=20,
            tags=["profile", "triton_on"],
        ))
        
        if filter_configs:
            filter_set = {name.strip() for name in filter_configs if name.strip()}
            if filter_set:
                configs = [c for c in configs if c.name in filter_set]

        return configs
    
    def run_all(self, filter_tag: Optional[str] = None, filter_configs: Optional[List[str]] = None):
        """Run all benchmarks in the test matrix."""
        configs = self.define_test_matrix(filter_configs=filter_configs)
        
        if filter_tag:
            configs = [c for c in configs if filter_tag in (c.tags or [])]
            print(f"Filtered to {len(configs)} configs with tag '{filter_tag}'")
        
        # Calculate estimated time based on config
        est_time_min = 0
        for c in configs:
            base_time = 25  # base time for 30 steps
            # Adjust for steps
            base_time = base_time * (c.steps / 30)
            # Triton-off is slower
            if not c.triton:
                base_time *= 2.5
            # Longer sequences take more time
            base_time *= (c.max_response_length / 1024)
            est_time_min += base_time
        
        est_hours = est_time_min / 60
        
        print(f"\n{'='*60}")
        print(f"{self.suite_name}")
        print(f"{'='*60}")
        print(f"Total configs: {len(configs)}")
        print(f"Output directory: {self.output_dir}")
        print(f"Estimated time: ~{est_hours:.1f} hours ({est_time_min:.0f} minutes)")
        print(f"Python executable: {get_conda_python()}")
        print(f"{'='*60}")
        
        # Auto-start (no confirmation needed for screen sessions)
        print("[INFO] Auto-starting in 3 seconds...")
        time.sleep(3)
        
        start_time = time.time()
        
        for i, config in enumerate(configs, 1):
            print(f"\n[{i}/{len(configs)}] ", end="")
            result = self.run_benchmark(config)
            self.results.append(result)
            
            # Save intermediate results
            self._save_results()
            
            # Progress update
            elapsed = (time.time() - start_time) / 60
            remaining = (est_time_min - elapsed) if i < len(configs) else 0
            print(f"\n[PROGRESS] {i}/{len(configs)} complete")
            print(f"[PROGRESS] Elapsed: {elapsed:.1f} min, Est. remaining: {remaining:.1f} min")
        
        total_elapsed = (time.time() - start_time) / 60
        print(f"\n{'='*60}")
        print(f"All benchmarks complete!")
        print(f"Total time: {total_elapsed:.1f} minutes ({total_elapsed/60:.1f} hours)")
        print(f"{'='*60}")
        
        self._generate_report()
    
    def _save_results(self):
        """Save results to JSON."""
        results_path = self.output_dir / "results.json"
        data = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "total_runs": len(self.results),
            "successful_runs": sum(1 for r in self.results if r.success),
            "results": [r.to_dict() for r in self.results],
        }
        results_path.write_text(json.dumps(data, indent=2))
    
    def _generate_report(self):
        """Generate markdown report."""
        report_path = self.output_dir / "report.md"
        
        lines = [
            f"# {self.report_title}",
            "",
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total runs: {len(self.results)}",
            f"Successful: {sum(1 for r in self.results if r.success)}",
            "",
            "## Summary",
            "",
            "| Config | Status | Step Time (s) | Tok/s | VRAM (GB) |",
            "|--------|--------|---------------|-------|-----------|",
        ]
        
        for r in self.results:
            status = "✅" if r.success else "❌"
            step_time = f"{r.step_time_avg_s:.2f}" if r.step_time_avg_s else "N/A"
            tok_s = f"{r.tokens_per_sec:.2f}" if r.tokens_per_sec else "N/A"
            vram = f"{r.vram_peak_gb:.2f}" if r.vram_peak_gb else "N/A"
            lines.append(f"| {r.config_name} | {status} | {step_time} | {tok_s} | {vram} |")
        
        lines.extend([
            "",
            "## Detailed Results",
            "",
        ])
        
        for r in self.results:
            lines.extend([
                f"### {r.config_name}",
                "",
                f"- Success: {r.success}",
                f"- Duration: {r.duration_s:.1f}s",
                f"- Steps: {r.steps_completed}",
            ])
            if r.step_time_avg_s:
                lines.append(f"- Step time: {r.step_time_avg_s:.2f}s (min: {r.step_time_min_s:.2f}, max: {r.step_time_max_s:.2f})")
            if r.tokens_per_sec:
                lines.append(f"- Throughput: {r.tokens_per_sec:.2f} tok/s")
            if r.vram_peak_gb:
                lines.append(f"- VRAM peak: {r.vram_peak_gb:.2f} GB")
            if r.error_message:
                lines.append(f"- Error: {r.error_message}")
            lines.append("")
        
        report_path.write_text("\n".join(lines))
        print(f"\nReport saved to: {report_path}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="ING INF Benchmark Suite")
    parser.add_argument("--filter", type=str, help="Filter by tag (e.g., 'baseline', 'sweep')")
    parser.add_argument(
        "--filter-configs",
        type=str,
        help="Comma-separated list of config names to run",
    )
    parser.add_argument("--output-dir", type=str, default="./benchmarks/output/ing_inf")
    parser.add_argument(
        "--run-prefix",
        type=str,
        default="ing_inf",
        help="Prefix for per-run output directories and WandB run names.",
    )
    parser.add_argument("--list", action="store_true", help="List all test configs")
    
    args = parser.parse_args()
    
    suite = IngInfBenchmarkSuite(output_dir=args.output_dir, run_prefix=args.run_prefix)
    
    filter_configs = None
    if args.filter_configs:
        filter_configs = [name.strip() for name in args.filter_configs.split(",") if name.strip()]

    if args.list:
        configs = suite.define_test_matrix(filter_configs=filter_configs)
        print(f"Total configs: {len(configs)}")
        for c in configs:
            tag_list = c.tags or []
            print(f"  {c.name}: {c.description} [tags: {', '.join(tag_list)}]")
        return
    
    suite.run_all(filter_tag=args.filter, filter_configs=filter_configs)


if __name__ == "__main__":
    main()
