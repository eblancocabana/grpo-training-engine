import sys
import os
import types
import importlib
import pytest
import torch

# Ensure project root is on sys.path so `import src` works when running tests directly
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
del _repo_root

import torch.nn as nn
import torch.nn.functional as F


class FakeLinear4bit(nn.Module):
    """Fake 4-bit linear layer that mimics bitsandbytes behavior."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        output = F.linear(x, self.weight, self.bias)
        return output


class _ManualLoRALayerLike(nn.Module):
    def __init__(self, base_layer, rank: int = 16, alpha: int = 32, dropout: float = 0.0):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.lora_A = nn.Linear(base_layer.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base_layer.out_features, bias=False)
        self.lora_dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        base_output = self.base_layer(x)
        if base_output.requires_grad:
            base_output = base_output.clone()
        x_adapt = x.to(self.lora_A.weight.dtype)
        lora_output = self.lora_B(self.lora_A(self.lora_dropout(x_adapt)))
        lora_output = lora_output * self.scaling
        if lora_output.dtype != base_output.dtype:
            lora_output = lora_output.to(base_output.dtype)
        return base_output + lora_output


class TestLoRA:
    """Comprehensive white-box tests for manual LoRA implementation."""

    def setup_fake_bnb(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Inject fake bitsandbytes module using pytest's monkeypatch for automatic cleanup."""
        fake_bnb = types.ModuleType("bitsandbytes")
        fake_bnb.nn = types.SimpleNamespace()  # pyright: ignore[reportAttributeAccessIssue]
        fake_bnb.nn.Linear4bit = FakeLinear4bit
        monkeypatch.setitem(sys.modules, "bitsandbytes", fake_bnb)

    def test_lora_forward_equation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.setup_fake_bnb(monkeypatch)
        from src.core.lora import ManualLoRALayer

        in_features = 8
        out_features = 8
        rank = 4
        alpha = 8

        base = FakeLinear4bit(in_features, out_features)
        lora = ManualLoRALayer(base, rank=rank, alpha=alpha, use_triton=False)  # pyright: ignore[reportCallIssue]

        with torch.no_grad():
            lora.lora_A.weight.fill_(0.1)
            lora.lora_B.weight.fill_(0.2)

        x = torch.ones(1, in_features)
        base_out = base(x)
        a_out = lora.lora_A(x)
        b_out = lora.lora_B(a_out)
        expected = base_out + b_out * (alpha / rank)
        actual = lora(x)
        assert torch.allclose(actual, expected, atol=1e-6)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_lora_triton_matches_torch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.setup_fake_bnb(monkeypatch)
        from src.core.lora import ManualLoRALayer
        from src.triton_kernels import lora_fused_forward, TRITON_AVAILABLE

        if not TRITON_AVAILABLE:
            pytest.skip("Triton not available")

        torch.manual_seed(123)
        torch.cuda.manual_seed_all(123)
        torch.backends.cudnn.deterministic = True

        device = torch.device("cuda")
        in_features = 16
        out_features = 16
        rank = 4
        alpha = 8

        base = FakeLinear4bit(in_features, out_features).to(device, dtype=torch.bfloat16)
        lora = ManualLoRALayer(base, rank=rank, alpha=alpha, use_triton=False).to(device, dtype=torch.bfloat16)  # pyright: ignore[reportCallIssue]

        x = torch.randn(2, 3, in_features, device=device, dtype=torch.bfloat16)
        expected = lora(x)
        lora_fused = _ManualLoRALayerLike(base, rank=rank, alpha=alpha).to(device, dtype=torch.bfloat16)
        lora_fused.lora_A.weight.data.copy_(lora.lora_A.weight.data)
        lora_fused.lora_B.weight.data.copy_(lora.lora_B.weight.data)
        actual = lora_fused_forward(x, lora_layer=lora_fused)

        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_lora_generate_parity(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.setup_fake_bnb(monkeypatch)
        from src.core.lora import ManualLoRALayer
        from src.triton_kernels import TRITON_AVAILABLE

        if not TRITON_AVAILABLE:
            pytest.skip("Triton not available")

        torch.manual_seed(321)
        torch.cuda.manual_seed_all(321)
        torch.backends.cudnn.deterministic = True

        device = torch.device("cuda")
        vocab_size = 32
        hidden = 16
        steps = 4

        embed = nn.Embedding(vocab_size, hidden).to(device)
        base_linear = FakeLinear4bit(hidden, vocab_size).to(device)
        lora_triton = ManualLoRALayer(base_linear, rank=4, alpha=8, use_triton=True).to(
            device
        )
        lora_torch = ManualLoRALayer(base_linear, rank=4, alpha=8, use_triton=False).to(
            device
        )
        lora_torch.lora_A.weight.data.copy_(lora_triton.lora_A.weight.data)
        lora_torch.lora_B.weight.data.copy_(lora_triton.lora_B.weight.data)

        input_ids = torch.randint(0, vocab_size, (1, 3), device=device)

        def generate(model_layer: nn.Module) -> torch.Tensor:
            tokens = input_ids.clone()
            for _ in range(steps):
                x = embed(tokens)
                logits = model_layer(x)[:, -1, :]
                next_token = torch.argmax(logits, dim=-1)
                tokens = torch.cat([tokens, next_token[:, None]], dim=1)
            return tokens[:, -steps:]

        out_triton = generate(lora_triton)
        out_torch = generate(lora_torch)
        assert torch.equal(out_triton, out_torch)

    def test_lora_initialization_distribution(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self.setup_fake_bnb(monkeypatch)
        from src.core.lora import ManualLoRALayer

        base = FakeLinear4bit(16, 16)
        lora = ManualLoRALayer(base, rank=4, alpha=8, use_triton=False)  # pyright: ignore[reportCallIssue]
        assert torch.allclose(lora.lora_B.weight, torch.zeros_like(lora.lora_B.weight))
        assert not torch.allclose(
            lora.lora_A.weight, torch.zeros_like(lora.lora_A.weight)
        )

    def test_lora_gradient_isolation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.setup_fake_bnb(monkeypatch)
        from src.core.lora import ManualLoRALayer

        device = torch.device("cuda")
        base = FakeLinear4bit(8, 8).to(device)
        lora = ManualLoRALayer(base, rank=4, alpha=8, use_triton=False).to(device)  # pyright: ignore[reportCallIssue]
        initial_base_weight = base.weight.clone()
        x = torch.randn(2, 8, device=device)
        out = lora(x)
        loss = out.sum()
        loss.backward()
        assert lora.lora_A.weight.grad is not None
        assert lora.lora_B.weight.grad is not None
        assert base.weight.grad is None
        assert torch.equal(base.weight, initial_base_weight)

    def test_lora_rank_efficiency(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.setup_fake_bnb(monkeypatch)
        from src.core.lora import ManualLoRALayer

        in_features = 1024
        out_features = 1024
        base = FakeLinear4bit(in_features, out_features)
        lora_r8 = ManualLoRALayer(base, rank=8, alpha=16, use_triton=False)  # pyright: ignore[reportCallIssue]
        params_r8 = sum(p.numel() for p in lora_r8.parameters() if p.requires_grad)
        base2 = FakeLinear4bit(in_features, out_features)
        lora_r16 = ManualLoRALayer(base2, rank=16, alpha=32, use_triton=False)  # pyright: ignore[reportCallIssue]
        params_r16 = sum(p.numel() for p in lora_r16.parameters() if p.requires_grad)
        assert params_r16 > params_r8

    def test_inject_lora_layers_selective(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self.setup_fake_bnb(monkeypatch)
        from src.core import lora as lora_module

        importlib.reload(lora_module)
        inject_lora_layers = lora_module.inject_lora_layers
        ManualLoRALayer = lora_module.ManualLoRALayer

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = FakeLinear4bit(16, 16)
                self.k_proj = FakeLinear4bit(16, 16)
                self.v_proj = FakeLinear4bit(16, 16)
                self.other = FakeLinear4bit(16, 16)

        model = TinyModel()
        count = inject_lora_layers(
            model,
            target_modules=["q_proj", "v_proj"],
            rank=4,
            alpha=8,
            use_triton=False,
            verbose=False,
        )  # pyright: ignore[reportCallIssue]
        assert count == 2
        assert isinstance(model.q_proj, ManualLoRALayer)
        assert isinstance(model.v_proj, ManualLoRALayer)
        assert isinstance(model.k_proj, FakeLinear4bit)
        assert isinstance(model.other, FakeLinear4bit)
