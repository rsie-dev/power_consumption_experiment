import logging
from pathlib import Path


from .base_statistics import BaseStatistics


class EnergyEfficiencyStatistics(BaseStatistics):
    def __init__(self, resources: Path, n_boot: int, seed: int | None, threading: str, mode: str, create_tex: bool):
        super().__init__(resources, n_boot, seed, threading, mode, create_tex)
        self._logger = logging.getLogger(self.__class__.__name__)

    def process(self, tp_file: Path):
        pass
