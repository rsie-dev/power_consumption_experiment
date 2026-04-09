import logging
from pathlib import Path

import pandas as pd
import pint_pandas  # needed to convert to pint columns

from data_aggregator.util import FramePersist


class AverageCalculator:
    def __init__(self, resources_folder: Path):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._resources_folder = resources_folder

    def calculate(self, power_file: Path):
        self._logger.info("Calculate averages for: %s", power_file)

        df = self._load(power_file)
        df = self._calculate_averages(df)

        calculated_name = f"average_power_{power_file.stem[13:]}.csv"
        csv_file = self._resources_folder / calculated_name
        frame_persist = FramePersist()
        frame_persist.persist(df, csv_file)

    def _calculate_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        print(df.dtypes)
        print(df.head())
        mean_power = df['power'].mean()
        print("mean: %s" % mean_power)

        df = pd.DataFrame(
            {
                "runs": pd.Series([len(df.index)], dtype="int"),
                "average_power": pd.Series([mean_power]).astype("pint[ampere·second·volt]"),
            }
        )
        return df

    def _load(self, in_file: Path) -> pd.DataFrame:
        df = pd.read_csv(in_file, header=[0, 1])

        names = df.columns.get_level_values(0)
        units = df.columns.get_level_values(1)

        df.columns = names  # flatten
        for col, unit in zip(names, units):  # apply units
            if unit != "No Unit":
                df[col] = df[col].astype(f"pint[{unit}]")

        self._logger.warning(df.dtypes)
        self._logger.warning(df.head())

        return df
