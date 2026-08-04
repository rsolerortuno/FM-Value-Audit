"""Effort and compute accounting."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeVar

from fmva.schemas import EffortRecord

T = TypeVar("T")


@dataclass
class EffortTracker:
    """Measure wall and process CPU time for tuning and final fitting."""

    method: str
    seed: int
    gpu_hours: float = 0.0
    configurations: list[str] = field(default_factory=list)
    tuning_wall_seconds: float = 0.0
    final_fit_wall_seconds: float = 0.0
    cpu_seconds: float = 0.0
    selected_configuration: str = "NOT_SELECTED"

    def run_tuning(self, function: Callable[[], T]) -> T:
        """Execute a zero-argument callable and account for tuning effort."""
        wall_start = time.perf_counter()
        cpu_start = time.process_time()
        result = function()
        self.tuning_wall_seconds += time.perf_counter() - wall_start
        self.cpu_seconds += time.process_time() - cpu_start
        return result

    def run_final(self, function: Callable[[], T]) -> T:
        """Execute a zero-argument callable and account for final-fit effort."""
        wall_start = time.perf_counter()
        cpu_start = time.process_time()
        result = function()
        self.final_fit_wall_seconds += time.perf_counter() - wall_start
        self.cpu_seconds += time.process_time() - cpu_start
        return result

    def to_record(self) -> EffortRecord:
        """Freeze the current measurements into a validated record."""
        return EffortRecord(
            method=self.method,
            configurations_tried=self.configurations,
            tuning_wall_seconds=self.tuning_wall_seconds,
            final_fit_wall_seconds=self.final_fit_wall_seconds,
            cpu_seconds=self.cpu_seconds,
            gpu_hours=self.gpu_hours,
            seed=self.seed,
            selected_configuration=self.selected_configuration,
        )
