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
