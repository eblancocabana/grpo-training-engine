"""
Pytest configuration and shared fixtures for the test suite.

This module centralizes common test setup, including:
- Fake bitsandbytes module injection for testing without GPU/4-bit quantization
- Path setup for importing src modules
"""

import sys
import os
import types

# Ensure project root is on sys.path so `import src` works when running tests directly
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
del _repo_root

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


class FakeLinear4bit(nn.Module):
    """Fake 4-bit linear layer that mimics bitsandbytes behavior for testing."""
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


@pytest.fixture
def fake_bitsandbytes(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Fixture that injects a fake bitsandbytes module into sys.modules.
    
    Automatically cleaned up by pytest's monkeypatch after each test.
    Use this fixture in any test that imports from src.core.lora or src.core.model_loader.
    """
    fake_bnb = types.ModuleType("bitsandbytes")
    fake_bnb.nn = types.SimpleNamespace()
    fake_bnb.nn.Linear4bit = FakeLinear4bit
    monkeypatch.setitem(sys.modules, "bitsandbytes", fake_bnb)


@pytest.fixture
def fake_linear_4bit() -> type[FakeLinear4bit]:
    """Returns the FakeLinear4bit class for use in tests."""
    return FakeLinear4bit
