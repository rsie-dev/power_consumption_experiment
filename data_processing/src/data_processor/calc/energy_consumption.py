import logging
from pathlib import Path

import tabulate
import pandas as pd

from data_processor.util import FrameIO
from data_processor.constants import GROUP_COLS, ORDER_TOOL, ORDER_STRENGTH


class EnergyConsumption:
    def __init__(self, resources: Path):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._frameio = FrameIO()
        self._resources = resources

    def process(self, used_energy_file: Path, create_tex: bool, no_tool: list, no_dataset: list):
        df = self._frameio.load(used_energy_file)
        df = df[~df["tool"].isin(no_tool)]
        df = df[~df["dataset"].isin(no_dataset)]
