"""Small reporting shim for IPOPT runs."""

from __future__ import annotations

import logging


class IPOPTReporter:
    """Structured logger wrapper for solve traces."""

    def __init__(self, logger: logging.Logger | None = None):
        self._log = logger or logging.getLogger(__name__)

    def info(self, msg: str) -> None:
        self._log.info(msg)

    def warning(self, msg: str) -> None:
        self._log.warning(msg)

    def debug(self, msg: str) -> None:
        self._log.debug(msg)
