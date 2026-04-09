import logging
from pathlib import Path

import pandas as pd


class FrameIO:
    def __init__(self):
        self._logger = logging.getLogger(self.__class__.__name__)

    def load(self, in_file: Path) -> pd.DataFrame:
        df = pd.read_csv(in_file, header=[0, 1])

        names = df.columns.get_level_values(0)
        units = df.columns.get_level_values(1)

        df.columns = names  # flatten
        if 'timestamp' in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        for col, unit in zip(names, units):  # apply units
            if unit != "No Unit":
                df[col] = df[col].astype(f"pint[{unit}]")

        return df

    def persist(self, df: pd.DataFrame, target_path: Path) -> None:
        self._logger.info("Generate: %s", target_path)
        df = df.pint.dequantify()
        df.to_csv(target_path, encoding='UTF_8', index=False, header=True)
