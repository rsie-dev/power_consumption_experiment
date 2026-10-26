import logging
from pathlib import Path

import pandas as pd
import numpy as np
from statsmodels.regression.linear_model import RegressionResultsWrapper

from data_processor import ureg
from .energy_efficiency_statistics import EnergyEfficiencyStatistics


class TotalEnergyEfficiencyStatistics(EnergyEfficiencyStatistics):
    def __init__(self, resources: Path, logfile: bool, n_boot: int, seed: int | None,
                 threading: str, mode: str, create_tex: bool):
        super().__init__(resources, logfile, n_boot, seed, threading, mode, create_tex)
        self._logger = logging.getLogger(self.__class__.__name__)

    @property
    def base_column(self) -> str:
        return "energy_efficiency_total"

    @property
    def effect_name(self) -> str:
        return "total"

    def _calculate_base(self, df: pd.DataFrame) -> pd.DataFrame:
        self._logger.info("Log-transform energy efficiency total")
        reference = 1 * ureg.byte / ureg.joule
        normalized_total = df["energy_efficiency_total"] / reference
        df["log_energy_efficiency_total"] = np.log(normalized_total).pint.magnitude
        return df

    def create_full_model(self, df: pd.DataFrame) -> RegressionResultsWrapper:
        full_model = self._stat_model.fit_full_model(df, "log_energy_efficiency_total", self.factors)
        return full_model
