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
        all_df = []
        for power_data in power_datas:
            df = frame_io.load(power_data)
            df = self.aggregate_power(df)
            all_df.append(df)

            calculated_name = f"used_power_{power_data.stem[13:]}.csv"
            csv_file = self._resources_folder / calculated_name
            frame_io.persist(df, csv_file)
        df_all = pd.concat(all_df)
        csv_file = self._resources_folder / "used_power.csv"
        frame_io.persist(df_all, csv_file)

    def aggregate_power(self, df: pd.DataFrame) -> pd.DataFrame:
        result = (
            df.groupby("run")
            .agg(
                power=("power", "sum"),
                power_duration=("power_duration", "sum"),
                energy=("energy_used", "sum"),
                real=("real", "first"),
                size=("size", "first"),
            )
            .reset_index()
        )
        for i, column in enumerate(["host", "tool", "dataset", "mode", "strength", "threading"]):
            result.insert(loc=i, column=column, value=df.iloc[0][column])
        return result
