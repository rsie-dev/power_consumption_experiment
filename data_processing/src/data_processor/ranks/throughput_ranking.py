import logging
from pathlib import Path

from .base_ranking import BaseRanking


class ThroughputRanking(BaseRanking):
    def __init__(self, resources: Path, logfile: bool, n_boot: int, seed: int | None, threading: str, mode: str):
        super().__init__(resources, logfile, n_boot, seed, threading, mode)
        self._logger = logging.getLogger(self.__class__.__name__)

    def extra_log_name(self, input_file: Path) -> str:
        base = self._get_base_from_input_file(input_file)
        return f"tp_ranking_{base}_{self._threading}_{self._mode}"

    def _get_base_from_input_file(self, input_file: Path) -> str:
        base = input_file.stem.removeprefix("tp_")
        return base

    @property
    def base_column(self) -> str:
        return "throughput"
