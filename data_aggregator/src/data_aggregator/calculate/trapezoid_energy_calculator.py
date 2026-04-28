import logging

import pandas as pd

from .energy_calculator import EnergyCalculator


class TrapezoidEnergyCalculator(EnergyCalculator):
    def __init__(self):
        super().__init__()
        self._logger = logging.getLogger(self.__class__.__name__)

    def calculate_energy(self, df: pd.DataFrame) -> pd.DataFrame:
        df["power_duration"] = df.groupby("run")["timestamp"].diff().dt.total_seconds().astype("pint[second]")

        # average power between previous and current row
        df["power_avg"] = (df["power"].shift() + df["power"]) / 2
        # trapezoid energy per row
        df["energy_used"] = df["power_avg"] * df["power_duration"]

        df["energy_used"] = df["energy_used"].pint.to("joule")
        del df['power_avg']

        return df
