import logging
from pathlib import Path

import pandas as pd

from data_aggregator.util import FrameIO


class AverageCalculator:
    def __init__(self, resources_folder: Path):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._resources_folder = resources_folder

    def calculate(self, power_file: Path):
        self._logger.info("Calculate averages for: %s", power_file)

        frame_io = FrameIO()
        df = frame_io.load(power_file)
        df = self._calculate_averages(df)

        calculated_name = f"average_power_{power_file.stem[13:]}.csv"
        csv_file = self._resources_folder / calculated_name
        frame_io.persist(df, csv_file)

    def _calculate_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        power_mean = df['power'].mean()

        df = pd.DataFrame(
            {
                "runs": pd.Series([len(df.index)], dtype="int"),
                "power_average": pd.Series([power_mean]).astype("pint[ampere·second·volt]"),
            }
        )
        return df
