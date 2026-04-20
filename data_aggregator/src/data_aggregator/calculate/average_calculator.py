import logging

import pandas as pd


class AverageCalculator:
    def __init__(self):
        self._logger = logging.getLogger(self.__class__.__name__)

    def calculate_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        power_mean = df['power'].mean()
        power_std = df['power'].std()
        power_var = df['power'].var()

        df = pd.DataFrame(
            {
                "runs": pd.Series([len(df.index)], dtype="int"),
                "power_average": pd.Series([power_mean]).astype("pint[ampere·second·volt]"),
                "power_std": pd.Series([power_std]).astype("pint[ampere·second·volt]"),
                "power_var": pd.Series([power_var]).astype("pint[(ampere·second·volt)^2]"),
            }
        )
        return df
