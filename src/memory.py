from __future__ import annotations

import os

import psutil


class MemoryMonitor:
    """Checks process RSS against the memory budget."""

    def __init__(self, memory_limit_mb: int, flush_ratio: float = 0.70) -> None:
        if memory_limit_mb <= 0:
            raise ValueError("memory_limit_mb must be positive")
        if not 0 < flush_ratio < 1:
            raise ValueError("flush_ratio must be between 0 and 1")
        self.memory_limit_bytes = memory_limit_mb * 1024 * 1024
        self.flush_threshold_bytes = int(self.memory_limit_bytes * flush_ratio)
        self.process = psutil.Process(os.getpid())

    def rss_bytes(self) -> int:
        return int(self.process.memory_info().rss)

    def rss_megabytes(self) -> float:
        return self.rss_bytes() / (1024 * 1024)

    def should_flush(self) -> bool:
        return self.rss_bytes() >= self.flush_threshold_bytes
