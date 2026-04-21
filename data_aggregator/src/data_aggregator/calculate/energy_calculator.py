import logging

import pandas as pd


class EnergyCalculator:
    def __init__(self):
        self._logger = logging.getLogger(self.__class__.__name__)

    def calculate_energy(self, df: pd.DataFrame) -> pd.DataFrame:
        df["power_duration"] = df.groupby("run")["timestamp"].diff().dt.total_seconds().astype("pint[second]")
        df["energy_used"] = df["power"] * df["power_duration"]
        df["energy_used"] = df["energy_used"].pint.to("joule")
        return df
