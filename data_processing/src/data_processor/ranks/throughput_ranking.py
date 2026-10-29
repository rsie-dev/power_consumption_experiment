import logging
from pathlib import Path

from .base_ranking import BaseRanking


class ThroughputRanking(BaseRanking):
    def __init__(self, resources: Path, threading: str, mode: str):
        super().__init__(resources, threading, mode)
        self._logger = logging.getLogger(self.__class__.__name__)

    @property
    def base_column(self) -> str:
        return "throughput"
