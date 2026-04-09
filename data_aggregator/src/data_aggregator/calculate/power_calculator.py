import logging
from pathlib import Path

import pandas as pd
import pint_pandas  # needed to convert to pint columns

from data_aggregator.util import FrameIO


class PowerCalculator:
    def __init__(self, resources_folder: Path):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._resources_folder = resources_folder

    def calculate(self, preprocessed: Path):
        self._logger.info("Calculate power for: %s", preprocessed)
        df = self._load(preprocessed)

        df = self._calculate_power(df)

        calculated_name = f"calculated_{preprocessed.stem[13:]}"
        csv_file = self._resources_folder / f"{calculated_name}.csv"
        frame_io = FrameIO()
        frame_io.persist(df, csv_file)

    def _calculate_power(self, df: pd.DataFrame) -> pd.DataFrame:
        df["duration"] = df.groupby("run")["timestamp"].diff().dt.total_seconds().astype("pint[second]")
        df["power"] = df["energy"] * df["duration"]
        return df

    def _load(self, in_file: Path) -> pd.DataFrame:
        df = pd.read_csv(in_file, header=[0, 1])

        names = df.columns.get_level_values(0)
        units = df.columns.get_level_values(1)

        df.columns = names  # flatten
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        for col, unit in zip(names, units):  # apply units
            if unit != "No Unit":
                df[col] = df[col].astype(f"pint[{unit}]")

        self._logger.warning(df.dtypes)
        self._logger.warning(df.head())

        return df
