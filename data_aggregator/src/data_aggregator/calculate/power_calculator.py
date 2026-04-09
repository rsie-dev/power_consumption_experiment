import logging
from pathlib import Path

import pandas as pd

from data_aggregator.util import FrameIO


class PowerCalculator:
    def __init__(self, resources_folder: Path):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._resources_folder = resources_folder

    def calculate(self, preprocessed: Path):
        self._logger.info("Calculate power for: %s", preprocessed)

        frame_io = FrameIO()
        df = frame_io.load(preprocessed)
        df = self._calculate_power(df)

        calculated_name = f"calculated_{preprocessed.stem[13:]}"
        csv_file = self._resources_folder / f"{calculated_name}.csv"
        frame_io.persist(df, csv_file)

    def _calculate_power(self, df: pd.DataFrame) -> pd.DataFrame:
        df["duration"] = df.groupby("run")["timestamp"].diff().dt.total_seconds().astype("pint[second]")
        df["power"] = df["energy"] * df["duration"]
        return df
