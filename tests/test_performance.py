import torch
import torch.nn as nn
import torch.nn.functional as F
import time


class TestPerformance:
    """Performance-related tests."""

    def test_memory_efficiency(self):
        device = torch.device("cuda")
        initial_memory = torch.cuda.memory_allocated(device)
        model = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 128)).to(
            device
        )
        x = torch.randn(32, 128, device=device)
        y = model(x)
        del y
        torch.cuda.empty_cache()
        final_memory = torch.cuda.memory_allocated(device)
        memory_increase = (final_memory - initial_memory) / (1024**2)
        assert memory_increase < 1024

    def test_computation_time_reasonable(self):
        device = torch.device("cuda")
        model = nn.Linear(256, 256).to(device)
        x = torch.randn(64, 256, device=device)
        for _ in range(10):
            _ = model(x)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            y = model(x)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        assert elapsed < 10.0
