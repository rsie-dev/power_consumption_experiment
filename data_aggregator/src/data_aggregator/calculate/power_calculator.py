import logging

import pandas as pd


class PowerCalculator:
    def __init__(self):
        self._logger = logging.getLogger(self.__class__.__name__)

    def calculate_power(self, df: pd.DataFrame) -> pd.DataFrame:
        df["duration"] = df.groupby("run")["timestamp"].diff().dt.total_seconds().astype("pint[second]")
        df["power"] = df["energy"] * df["duration"]
        return df
