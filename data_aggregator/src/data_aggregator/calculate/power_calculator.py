import logging
from pathlib import Path

import pandas as pd
import pint_pandas  # needed to convert to pint columns

from data_aggregator.util import FramePersist


class PowerCalculator:
    def __init__(self, resources_folder: Path):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._resources_folder = resources_folder

    def calculate(self, preprocessed: Path):
        self._logger.info("Calculate power for: %s", preprocessed)
        df = pd.read_csv(preprocessed, header=[0, 1])

        names = df.columns.get_level_values(0)
        units = df.columns.get_level_values(1)

        df.columns = names  # flatten
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        for col, unit in zip(names, units):  # apply units
            if unit != "No Unit":
                df[col] = df[col].astype(f"pint[{unit}]")

        df["duration"] = df["timestamp"].diff().dt.total_seconds().astype("pint[second]")
        df["power"] = df["energy"] * df["duration"]

        calculated_name = f"calculated_{preprocessed.stem[13:]}"
        csv_file = self._resources_folder / f"{calculated_name}.csv"
        frame_persist = FramePersist()
        frame_persist.persist(df, csv_file)
