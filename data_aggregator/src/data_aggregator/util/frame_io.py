import logging
from pathlib import Path
from io import StringIO

import pandas as pd



class FrameIO:
    def __init__(self):
        self._logger = logging.getLogger(self.__class__.__name__)

    def load(self, in_file: Path, time_columns: list[str] | None=None) -> pd.DataFrame:
        df = pd.read_csv(in_file, header=[0, 1])
        return self.adjust_pint_columns(df, time_columns)

    def load_str(self, data: str) -> pd.DataFrame:
        df = pd.read_csv(StringIO(data), header=[0, 1])
        return self.adjust_pint_columns(df)

    def adjust_pint_columns(self, df: pd.DataFrame, time_columns: list[str] | None=None) -> pd.DataFrame:
        names = df.columns.get_level_values(0)
        units = df.columns.get_level_values(1)

        df.columns = names  # flatten
        if time_columns is None:
            time_columns = ['timestamp']
        for time_column in time_columns:
            if time_column in df.columns:
                df[time_column ] = pd.to_datetime(df[time_column], format="ISO8601")
        for col, unit in zip(names, units):  # apply units
            if unit != "No Unit":
                df[col] = df[col].astype(f"pint[{unit}]")

        return df

    def persist(self, df: pd.DataFrame, target_path: Path) -> None:
        self._logger.info("Generate: %s", target_path)

        if 'timestamp' in df.columns:
            df["timestamp"] = df["timestamp"].apply(lambda x: x.isoformat(timespec="milliseconds"))

        df = df.pint.dequantify()

        df.to_csv(target_path, encoding='UTF_8', index=False, header=True)
