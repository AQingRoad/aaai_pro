from pathlib import Path

import pytest

from manu_src.scripts.train.keep_latest_optimizer_state_callback import (
    prune_older_optimizer_states,
)


def _make_checkpoint(root: Path, step: int, *, with_optimizer: bool = True) -> Path:
    checkpoint = root / f"checkpoint-{step}"
    checkpoint.mkdir()
    (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")
    if with_optimizer:
        (checkpoint / "optimizer.pt").write_bytes(b"optimizer")
    return checkpoint


def test_prune_keeps_only_current_optimizer_state(tmp_path: Path) -> None:
    old_checkpoint = _make_checkpoint(tmp_path, 300)
    current_checkpoint = _make_checkpoint(tmp_path, 600)
    future_checkpoint = _make_checkpoint(tmp_path, 900)

    removed = prune_older_optimizer_states(tmp_path, 600)

    assert removed == [old_checkpoint / "optimizer.pt"]
    assert not (old_checkpoint / "optimizer.pt").exists()
    assert (current_checkpoint / "optimizer.pt").exists()
    assert (future_checkpoint / "optimizer.pt").exists()
    assert (old_checkpoint / "adapter_model.safetensors").exists()
    assert (current_checkpoint / "adapter_model.safetensors").exists()


def test_prune_aborts_when_current_optimizer_state_is_missing(tmp_path: Path) -> None:
    old_checkpoint = _make_checkpoint(tmp_path, 300)
    _make_checkpoint(tmp_path, 600, with_optimizer=False)

    with pytest.raises(RuntimeError, match="current checkpoint has no optimizer state"):
        prune_older_optimizer_states(tmp_path, 600)

    assert (old_checkpoint / "optimizer.pt").exists()
