"""Base clusterer adapter interface."""

from __future__ import annotations

import logging
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from ..schema.records import ClusterAssignment, MethodStatus

logger = logging.getLogger(__name__)


@dataclass
class ClustererResult:
    """Result from a single clusterer run."""

    method: str
    assignments: list[ClusterAssignment] = field(default_factory=list)
    raw_output: Any = None
    runtime_seconds: float = 0.0
    memory_peak_mb: float = 0.0
    status: MethodStatus = MethodStatus.SUCCESS
    error_message: str = ""


class BaseClusterer(ABC):
    """Abstract base class for all clustering method adapters."""

    name: str = "base"

    @abstractmethod
    def prepare_input(
        self, tcr_table: pd.DataFrame, config: dict
    ) -> Any:
        """Transform canonical TCR table into method-specific input."""
        ...

    @abstractmethod
    def run(self, prepared_input: Any, workdir: Path) -> Any:
        """Execute clustering method. Returns raw method-specific output."""
        ...

    @abstractmethod
    def parse_output(self, workdir: Path) -> Any:
        """Parse raw output from workdir."""
        ...

    @abstractmethod
    def normalize(self, raw_output: Any) -> list[ClusterAssignment]:
        """Convert raw output to standardized ClusterAssignment list."""
        ...

    def safe_execute(
        self,
        tcr_table: pd.DataFrame,
        workdir: Path,
        config: Optional[dict] = None,
    ) -> ClustererResult:
        """Execute full pipeline with error handling and timing."""
        config = config or {}
        start = time.time()
        try:
            prepared = self.prepare_input(tcr_table, config)
            raw = self.run(prepared, workdir)
            assignments = self.normalize(raw)
            elapsed = time.time() - start
            return ClustererResult(
                method=self.name,
                assignments=assignments,
                raw_output=raw,
                runtime_seconds=elapsed,
                status=MethodStatus.SUCCESS,
            )
        except Exception as e:
            elapsed = time.time() - start
            logger.error(f"Clusterer {self.name} failed: {e}\n{traceback.format_exc()}")
            return ClustererResult(
                method=self.name,
                runtime_seconds=elapsed,
                status=MethodStatus.FAILED,
                error_message=str(e),
            )
