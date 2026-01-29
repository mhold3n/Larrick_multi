"""Structured logging utilities."""

from __future__ import annotations

import json
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, TextIO


@dataclass
class LogRecord:
    """Structured log record."""

    level: str
    message: str
    timestamp: float = field(default_factory=time.time)
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "level": self.level,
            "message": self.message,
            "timestamp": self.timestamp,
            **self.data,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())


class StructuredLogger:
    """Simple structured logger with JSON output."""

    def __init__(
        self,
        name: str,
        output: TextIO | None = None,
        min_level: str = "INFO",
    ) -> None:
        self.name = name
        self.output = output or sys.stdout
        self._levels = {"DEBUG": 0, "INFO": 1, "WARN": 2, "ERROR": 3}
        self._min_level = self._levels.get(min_level.upper(), 1)

    def _log(self, level: str, message: str, **data: Any) -> None:
        if self._levels.get(level, 0) < self._min_level:
            return

        record = LogRecord(level=level, message=message, data={"logger": self.name, **data})
        print(record.to_json(), file=self.output)

    def debug(self, message: str, **data: Any) -> None:
        """Log at DEBUG level."""
        self._log("DEBUG", message, **data)

    def info(self, message: str, **data: Any) -> None:
        """Log at INFO level."""
        self._log("INFO", message, **data)

    def warn(self, message: str, **data: Any) -> None:
        """Log at WARN level."""
        self._log("WARN", message, **data)

    def error(self, message: str, **data: Any) -> None:
        """Log at ERROR level."""
        self._log("ERROR", message, **data)

    @contextmanager
    def timer(self, operation: str):
        """Context manager for timing operations.

        Usage:
            with logger.timer("eval_thermo"):
                result = eval_thermo(...)
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.debug(f"{operation} completed", elapsed_ms=elapsed * 1000)


# Global logger cache
_loggers: dict[str, StructuredLogger] = {}


def get_logger(name: str) -> StructuredLogger:
    """Get or create a structured logger.

    Args:
        name: Logger name (typically __name__).

    Returns:
        StructuredLogger instance.
    """
    if name not in _loggers:
        _loggers[name] = StructuredLogger(name)
    return _loggers[name]


def set_log_level(level: str) -> None:
    """Set minimum log level for all loggers.

    Args:
        level: One of DEBUG, INFO, WARN, ERROR.
    """
    for logger in _loggers.values():
        logger._min_level = logger._levels.get(level.upper(), 1)
