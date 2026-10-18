import logging
from pathlib import Path

import pandas as pd

from data_processor.processor import Processor


class EnergyEfficiencyMerger(Processor):
    def __init__(self, resources: Path):
        super().__init__(resources)
        self._logger = logging.getLogger(self.__class__.__name__)

    def merge(self, rank_files: list[Path]):
        rank1_df = self._load_rank_file(rank_files[0])
        rank2_df = self._load_rank_file(rank_files[1])

        print("1:")
        print(rank1_df)
        print("2:")
        print(rank2_df)


    def _load_rank_file(self, file_name: Path) -> pd.DataFrame:
        df = self._frameio.load(file_name)
        df['df'] = df['df'].astype(int)
        for column in ["omega_squared", "ci_low", "ci_high"]:
            df["%s_percent" % column] = df[column] * 100
        df["cumulative_omega_squared_percent"] = df["omega_squared"].cumsum() * 100
        df = df[["Effect", "df",
                 "Rank",
                 "omega_squared_percent", "ci_low_percent", "ci_high_percent",
                 "cumulative_omega_squared_percent"]]
        return df
