import logging
from pathlib import Path

import tabulate
import pandas as pd

from data_processor.data_set import get_dataset_size
from data_processor.constants import GROUP_COLS
from .calc_params import EnergyParams
from .calculator import Calculator


class Throughput(Calculator):
    def __init__(self, resources: Path):
        super().__init__(resources)
        self._logger = logging.getLogger(self.__class__.__name__)

    def process(self, params: EnergyParams):
        df = self._load(params)

        result_df = self._calculate_throughput(df)

        self._print_table(result_df)
        tp_file = "tp_%s" % params.used_energy_file.stem.removeprefix("used_energy_") + ".csv"
        csv_df = result_df.drop(columns=['energy', 'size'])
        csv_df = self._order_data(csv_df)
        self._create_csv(tp_file, csv_df)

    def _print_table(self, df: pd.DataFrame):
        table_df = self._aggregate_throughput(df)
        table_df= self._order_data(table_df)

        table_df["average_duration"] = table_df["average_duration"].astype(float)
        table_df["average_throughput"] = table_df["average_throughput"].pint.to("MiB/s")
        table_df["average_throughput"] = table_df["average_throughput"].astype(float)

        cols = table_df.columns.tolist()

        table_entries = []
        for _, row in table_df.iterrows():
            table_entries.append(row.values[:])

        headers = cols[:-2]
        headers.append("average duration (s)")
        headers.append("average throughput (MiB/s)")
        table_str = tabulate.tabulate(table_entries,
                                      headers=headers,
                                      tablefmt="simple"
                                      )
        print(table_str)

    def _calculate_throughput(self, df: pd.DataFrame) -> pd.DataFrame:
        df["throughput"] = df["dataset"].map(get_dataset_size) / df["duration"]

        return df

    def _aggregate_throughput(self, df: pd.DataFrame) -> pd.DataFrame:
        result_df = (
            df.groupby(GROUP_COLS, as_index=False)
            .agg(
                num_runs=("run", "count"),
                average_duration=("duration", "mean"),
                average_throughput=("throughput", "mean"),
            )
        )

        return result_df
