#!/usr/bin/env python3
"""Keep optimizer state only in the latest successfully saved checkpoint.

The callback is loaded as an ms-swift external plugin.  On every successful
checkpoint save, it first verifies that the current checkpoint contains an
optimizer state, then removes optimizer-state files from lower-step
checkpoints.  Model and adapter files are never removed.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

try:
    from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
except ImportError:  # Allows pure-helper tests outside the training environment.
    class TrainerCallback:  # type: ignore[no-redef]
        pass

    TrainerControl = TrainerState = TrainingArguments = Any  # type: ignore[misc,assignment]


LOGGER = logging.getLogger(__name__)
CHECKPOINT_PATTERN = re.compile(r"^checkpoint-(\d+)$")
OPTIMIZER_STATE_PATTERNS = (
    "optimizer.pt",
    "optimizer.bin",
    "optimizer_*.pt",
    "optimizer_*.bin",
)


def _checkpoint_step(path: Path) -> int | None:
    match = CHECKPOINT_PATTERN.fullmatch(path.name)
    return int(match.group(1)) if match else None


def _optimizer_state_files(checkpoint_dir: Path) -> list[Path]:
    files: set[Path] = set()
    for pattern in OPTIMIZER_STATE_PATTERNS:
        files.update(path for path in checkpoint_dir.glob(pattern) if path.is_file())
    return sorted(files)


def prune_older_optimizer_states(
    output_dir: str | Path,
    current_step: int,
) -> list[Path]:
    """Delete optimizer states below ``current_step`` after validating the current one.

    If the current checkpoint or its optimizer state is missing, no older file
    is removed.  This preserves the last resumable checkpoint when a save is
    incomplete.
    """

    output_path = Path(output_dir)
    current_checkpoint = output_path / f"checkpoint-{int(current_step)}"
    current_optimizer_files = _optimizer_state_files(current_checkpoint)
    if not current_checkpoint.is_dir() or not current_optimizer_files:
        raise RuntimeError(
            "Refusing to prune older optimizer states because the current "
            f"checkpoint has no optimizer state: {current_checkpoint}"
        )

    removed: list[Path] = []
    for checkpoint_dir in sorted(output_path.glob("checkpoint-*")):
        step = _checkpoint_step(checkpoint_dir)
        if step is None or step >= int(current_step):
            continue
        for optimizer_file in _optimizer_state_files(checkpoint_dir):
            optimizer_file.unlink()
            removed.append(optimizer_file)
    return removed


class KeepLatestOptimizerStateCallback(TrainerCallback):
    """Retain optimizer state in the newest checkpoint and prune older copies."""

    def on_init_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if bool(getattr(args, "save_only_model", False)):
            raise RuntimeError(
                "KeepLatestOptimizerStateCallback requires --save_only_model false"
            )

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if not bool(getattr(state, "is_world_process_zero", True)):
            return
        removed = prune_older_optimizer_states(
            getattr(args, "output_dir"),
            int(getattr(state, "global_step")),
        )
        LOGGER.info(
            "Kept optimizer state in checkpoint-%s and removed %s older optimizer file(s)",
            getattr(state, "global_step"),
            len(removed),
        )


def _register_callback() -> None:
    try:
        from swift.plugin import extra_callbacks
    except ImportError:
        return
    if not any(isinstance(callback, KeepLatestOptimizerStateCallback) for callback in extra_callbacks):
        extra_callbacks.append(KeepLatestOptimizerStateCallback())


_register_callback()
