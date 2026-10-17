import logging
from pathlib import Path
from abc import abstractmethod

import pandas as pd

from .base_statistics import BaseStatistics
from .bootstrap_ci_calculator import BootstrapCiCalculator


class EnergyEfficiencyStatistics(BaseStatistics):
    def __init__(self, resources: Path, n_boot: int, seed: int | None, threading: str, mode: str, create_tex: bool):
        super().__init__(resources, threading, mode, create_tex)
        self._logger = logging.getLogger(self.__class__.__name__)
        self._display_unit = "MiB/joule"
        self._bootstrap_ci_calculator = BootstrapCiCalculator(self.factors, n_boot, seed)

    def process(self, ee_file: Path):
        self._log_section("Process %s %s" % (self._threading, self._mode))

        df = self._frameio.load(ee_file)
        comp = self._extract_data(df)
        comp = self._calculate_base(comp)

        self._validate_data(comp)
        self._calculate_statistics(comp)

    @abstractmethod
    def _calculate_base(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def _validate_data(self, comp: pd.DataFrame):
        pass

    def _validate_data_quantity(self, comp: pd.DataFrame, base_column: str):
        super()._validate_data_quantity(comp, base_column)
        n_negative = (comp[base_column] < 0).sum()
        print("Negative energy count         %d" % n_negative)

    @abstractmethod
    def _calculate_statistics(self, comp: pd.DataFrame):
        pass
