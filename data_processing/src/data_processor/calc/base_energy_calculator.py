import pandas as pd

from .calculator import Calculator


class BaseEnergyCalculator(Calculator):
    def _calculate_average_power(self, df: pd.DataFrame) -> pd.DataFrame:
        df["average_power"] = df["energy"] / df["duration"]
        df["average_power"] = df["average_power"].pint.to("watt")
        return df
