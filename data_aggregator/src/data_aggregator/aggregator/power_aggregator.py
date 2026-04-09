import logging
from pathlib import Path

import pandas as pd

from data_aggregator.util import FrameIO


class PowerAggregator:
    def __init__(self, resources_folder: Path):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._resources_folder = resources_folder

    def aggregate(self, power_data: Path):
        df = self._load(power_data)
        df = self._aggregate_power(df)

        calculated_name = f"used_power_{power_data.stem[13:]}.csv"
        csv_file = self._resources_folder / calculated_name
        frame_io = FrameIO()
        frame_io.persist(df, csv_file)

    def _aggregate_power(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.groupby("run")["power"].sum().reset_index()
        return result

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
