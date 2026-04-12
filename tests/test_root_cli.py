import sys
from unittest.mock import patch


def test_root_cli_does_not_kill_sibling_processes_by_default(tmp_path):
    import train

    with patch.object(train, "setup_logging"), patch.object(
        train, "clear_gpu_memory"
    ), patch.object(train, "check_system", return_value=True), patch.object(
        train, "kill_stale_train_processes"
    ) as cleanup:
        with patch.object(
            sys,
            "argv",
            [
                "train.py",
                "--dry-run",
                "--output-dir",
                str(tmp_path / "out"),
            ],
        ):
            train.main()

    cleanup.assert_not_called()


def test_root_cli_can_explicitly_cleanup_sibling_processes(tmp_path):
    import train

    with patch.object(train, "setup_logging"), patch.object(
        train, "clear_gpu_memory"
    ), patch.object(train, "check_system", return_value=True), patch.object(
        train, "kill_stale_train_processes"
    ) as cleanup:
        with patch.object(
            sys,
            "argv",
            [
                "train.py",
                "--dry-run",
                "--cleanup-train-processes",
                "--output-dir",
                str(tmp_path / "out"),
            ],
        ):
            train.main()

    cleanup.assert_called_once()


def test_root_cli_entropy_mask_defaults_to_config(tmp_path):
    import train

    config = train.get_8gb_vram_config()
    config.entropy.use_entropy_mask = True

    with patch.object(train, "get_8gb_vram_config", return_value=config), patch.object(
        train, "setup_logging"
    ), patch.object(train, "clear_gpu_memory"), patch.object(
        train, "check_system", return_value=True
    ), patch.object(train, "kill_stale_train_processes"):
        with patch.object(
            sys,
            "argv",
            [
                "train.py",
                "--dry-run",
                "--output-dir",
                str(tmp_path / "out"),
            ],
        ):
            train.main()

    assert config.entropy.use_entropy_mask is True


def test_root_cli_can_disable_entropy_mask(tmp_path):
    import train

    config = train.get_8gb_vram_config()
    config.entropy.use_entropy_mask = True

    with patch.object(train, "get_8gb_vram_config", return_value=config), patch.object(
        train, "setup_logging"
    ), patch.object(train, "clear_gpu_memory"), patch.object(
        train, "check_system", return_value=True
    ), patch.object(train, "kill_stale_train_processes"):
        with patch.object(
            sys,
            "argv",
            [
                "train.py",
                "--dry-run",
                "--no-entropy-mask",
                "--output-dir",
                str(tmp_path / "out"),
            ],
        ):
            train.main()

    assert config.entropy.use_entropy_mask is False


def test_root_cli_can_explicitly_enable_entropy_mask(tmp_path):
    import train

    config = train.get_8gb_vram_config()
    config.entropy.use_entropy_mask = False

    with patch.object(train, "get_8gb_vram_config", return_value=config), patch.object(
        train, "setup_logging"
    ), patch.object(train, "clear_gpu_memory"), patch.object(
        train, "check_system", return_value=True
    ), patch.object(train, "kill_stale_train_processes"):
        with patch.object(
            sys,
            "argv",
            [
                "train.py",
                "--dry-run",
                "--use-entropy-mask",
                "--output-dir",
                str(tmp_path / "out"),
            ],
        ):
            train.main()

    assert config.entropy.use_entropy_mask is True
