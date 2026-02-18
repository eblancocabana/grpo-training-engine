# VRAM Profiler

A low-level VRAM profiling suite for PyTorch, designed to identify memory bottlenecks and leaks.

## Components

1.  **`vram_profiler.py`**: A modified training runner that hooks into the PyTorch C++ Allocator (`torch.cuda.memory._record_memory_history`) to capture every memory event.
2.  **Web Visualizer** (`app/index.html`): A local D3.js web application to visualize the profile trace.

## Quick Start

### 1. Capture Profile
Run the profiler for a few steps (3 is usually enough to see the steady state).

```bash
# Using your conda environment
python tools/vram_profiler/vram_profiler.py --context all --steps 3 --output vram_snapshot.json
```

**Available Context Levels:**
- `--context all`: Captures full Python/C++ stack traces (High Overhead).
- `--context state`: Captures state allocations only (Moderate Detail).
- `--max-entries N`: Increase buffer size to capture longer training runs (Default: 2,000,000 events).
  > **Note**: If the output exceeds 150MB, it will automatically be compressed to a smaller size.

### 2. Visualize
1.  Open `tools/vram_profiler/app/index.html` in your web browser.
2.  Drag and drop the generated `vram_snapshot.json` file onto the drop zone.
3.  **Timeline**: Hover to see memory usage over time. Click to freeze the "Active Allocations" view at that specific point.
4.  **Bottleneck Detector**: Check the sidebar for suspicious large accumulations.

## Troubleshooting
- **AttributeError**: If you see errors related to `cProfile` or `module profile`, Ensure the script is named `vram_profiler.py` and NOT `profile.py` (which causes a name collision).
