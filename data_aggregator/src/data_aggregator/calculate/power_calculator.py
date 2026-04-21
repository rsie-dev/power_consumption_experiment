import logging

import pandas as pd


class PowerCalculator:
    def __init__(self):
        self._logger = logging.getLogger(self.__class__.__name__)

    def calculate_power(self, df: pd.DataFrame) -> pd.DataFrame:
        df["power"] = df.voltage * df.current
        df["power"] = df["power"].pint.to("watt")
        return df
