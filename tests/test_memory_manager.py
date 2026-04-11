import torch


class TestMemoryManager:
    """Comprehensive tests for memory management."""

    def test_clear_cache_calls_empty_cache(self, monkeypatch):
        from src.core.memory_manager import MemoryManager

        called = {"empty_cache": False}

        def fake_empty_cache():
            called["empty_cache"] = True

        monkeypatch.setattr(torch.cuda, "empty_cache", fake_empty_cache)
        mm = MemoryManager()
        mm.clear_cache()
        assert called["empty_cache"] is True

    def test_clear_cache_aggressive_calls_gc(self, monkeypatch):
        from src.core.memory_manager import MemoryManager
        import gc

        called = {"gc_collect": False}

        def fake_gc_collect():
            called["gc_collect"] = True

        monkeypatch.setattr(gc, "collect", fake_gc_collect)
        mm = MemoryManager()
        mm.clear_cache(aggressive=True)
        assert called["gc_collect"] is True

    def test_memory_stats_format(self):
        from src.core.memory_manager import MemoryManager

        mm = MemoryManager()
        stats = mm.get_memory_stats()
        required_keys = [
            "allocated_gb",
            "reserved_gb",
            "max_allocated_gb",
            "total_gb",
            "free_gb",
            "usage_fraction",
        ]
        for key in required_keys:
            assert key in stats
            assert isinstance(stats[key], float)

    def test_memory_warning_threshold(self):
        from src.core.memory_manager import MemoryManager

        mm = MemoryManager(memory_fraction_warning=0.5)
        mm.get_memory_stats = lambda: {
            "usage_fraction": 0.9,
            "allocated_gb": 6.0,
            "reserved_gb": 6.5,
            "max_allocated_gb": 7.0,
            "total_gb": 8.0,
            "free_gb": 1.5,
        }
        assert mm.check_memory_warning() is True
        mm.get_memory_stats = lambda: {
            "usage_fraction": 0.3,
            "allocated_gb": 2.0,
            "reserved_gb": 2.5,
            "max_allocated_gb": 3.0,
            "total_gb": 8.0,
            "free_gb": 5.5,
        }
        assert mm.check_memory_warning() is False

    def test_selective_checkpointing_by_name(self, monkeypatch):
        from src.core.memory_manager import MemoryManager

        class DummyModule:
            def __init__(self):
                self.gradient_checkpointing = False

        class DummyModel:
            def __init__(self):
                self.layers = {
                    "block.0": DummyModule(),
                    "block.1": DummyModule(),
                    "other": DummyModule(),
                }

            def named_modules(self):
                for name, module in self.layers.items():
                    yield name, module

            def gradient_checkpointing_enable(self):
                pass

            def gradient_checkpointing_disable(self):
                pass

            def get_input_embeddings(self):
                class DummyEmb:
                    def register_forward_hook(self, hook):
                        return None

                return DummyEmb()

        mm = MemoryManager(
            checkpointing_strategy="subset",
            checkpointing_layer_name_patterns=["block"],
        )
        model = DummyModel()

        mm.enable_checkpointing(model)

        assert model.layers["block.0"].gradient_checkpointing is True
        assert model.layers["block.1"].gradient_checkpointing is True
        assert model.layers["other"].gradient_checkpointing is False

    def test_vram_auto_checkpointing_toggle(self):
        from src.core.memory_manager import MemoryManager

        class DummyModule:
            def __init__(self):
                self.gradient_checkpointing = False

        class DummyModel:
            def __init__(self):
                self.layers = {
                    "block.0": DummyModule(),
                    "block.1": DummyModule(),
                }

            def named_modules(self):
                for name, module in self.layers.items():
                    yield name, module

            def gradient_checkpointing_enable(self):
                pass

            def gradient_checkpointing_disable(self):
                pass

            def get_input_embeddings(self):
                class DummyEmb:
                    def register_forward_hook(self, hook):
                        return None

                return DummyEmb()

        mm = MemoryManager(
            checkpointing_strategy="vram_auto",
            checkpointing_vram_enable_threshold=0.8,
            checkpointing_vram_disable_threshold=0.6,
            checkpointing_update_interval_steps=1,
        )
        model = DummyModel()

        mm.get_memory_stats = lambda: {"usage_fraction": 0.85}
        mm.enable_checkpointing(model)
        assert model.layers["block.0"].gradient_checkpointing is True
        assert model.layers["block.1"].gradient_checkpointing is True

        mm.get_peak_memory_stats = lambda: {
            "usage_fraction": 0.55,
            "peak_usage_fraction": 0.55,
        }
        mm.maybe_update_checkpointing(model, step=2)
        assert model.layers["block.0"].gradient_checkpointing is False
        assert model.layers["block.1"].gradient_checkpointing is False

    def test_vram_auto_starts_enabled_until_peak_data_is_observed(self):
        from src.core.memory_manager import MemoryManager

        class DummyModule:
            def __init__(self):
                self.gradient_checkpointing = False

        class DummyModel:
            def __init__(self):
                self.layers = {"block.0": DummyModule()}

            def named_modules(self):
                for name, module in self.layers.items():
                    yield name, module

            def gradient_checkpointing_enable(self):
                pass

            def gradient_checkpointing_disable(self):
                pass

            def get_input_embeddings(self):
                class DummyEmb:
                    def register_forward_hook(self, hook):
                        return None

                return DummyEmb()

        model = DummyModel()
        mm = MemoryManager(checkpointing_strategy="vram_auto")
        mm.enable_checkpointing(model)

        assert model.layers["block.0"].gradient_checkpointing is True

    def test_input_grad_hook_is_cleared_and_reregistered_on_reenable(self):
        from src.core.memory_manager import MemoryManager

        class DummyHandle:
            def __init__(self):
                self.removed = False

            def remove(self):
                self.removed = True

        class DummyEmb:
            def __init__(self):
                self.hook_calls = 0
                self.handles = []

            def register_forward_hook(self, hook):
                self.hook_calls += 1
                handle = DummyHandle()
                self.handles.append(handle)
                return handle

        class DummyModel:
            def __init__(self):
                self.embedding = DummyEmb()
                self.gradient_checkpointing = False

            def named_modules(self):
                yield "layer", self

            def gradient_checkpointing_enable(self):
                pass

            def gradient_checkpointing_disable(self):
                pass

            def get_input_embeddings(self):
                return self.embedding

        model = DummyModel()
        mm = MemoryManager(
            checkpointing_strategy="vram_auto",
            checkpointing_vram_enable_threshold=0.8,
            checkpointing_vram_disable_threshold=0.6,
            checkpointing_update_interval_steps=1,
        )
        mm.get_memory_stats = lambda: {"usage_fraction": 0.85}
        mm.enable_checkpointing(model)
        mm.get_memory_stats = lambda: {"usage_fraction": 0.55}
        mm.maybe_update_checkpointing(model, step=2)
        first_handle = model.embedding.handles[0]
        assert first_handle.removed is True
        assert mm._input_grad_hook_handle is None
        assert mm._input_grad_helper_enabled is False
        mm.get_memory_stats = lambda: {"usage_fraction": 0.85}
        mm.maybe_update_checkpointing(model, step=3)

        assert model.embedding.hook_calls == 2
        assert mm._input_grad_hook_handle is model.embedding.handles[1]
