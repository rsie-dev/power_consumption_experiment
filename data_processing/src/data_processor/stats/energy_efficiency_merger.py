import logging
from pathlib import Path

from data_processor.processor import Processor


class EnergyEfficiencyMerger(Processor):
    def __init__(self, resources: Path):
        super().__init__(resources)
        self._logger = logging.getLogger(self.__class__.__name__)

    def merge(self, rank_files: list[Path]):
        pass