import logging
from pathlib import Path

import pandas as pd

from data_aggregator.util import FrameIO


class PowerAggregator:
    def __init__(self, resources_folder: Path):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._resources_folder = resources_folder

    def aggregate(self, power_datas: list[Path]):
        frame_io = FrameIO()
        for power_data in power_datas:
            df = frame_io.load(power_data)
            df = self._aggregate_power(df)

            calculated_name = f"used_power_{power_data.stem[13:]}.csv"
            csv_file = self._resources_folder / calculated_name
            frame_io.persist(df, csv_file)

    def _aggregate_power(self, df: pd.DataFrame) -> pd.DataFrame:
        result = (
            df.groupby("run")
            .agg(
                power=("power", "sum"),
                real=("real", "first"),
                size=("size", "first"),
            )
            .reset_index()
        )
        for i, column in enumerate(["host", "tool", "dataset", "mode", "strength", "threading"]):
            result.insert(loc=i, column=column, value=df.iloc[0][column])
        return result
