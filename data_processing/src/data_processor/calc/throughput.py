import logging
from pathlib import Path

import tabulate
import pandas as pd

from data_processor.util import FrameIO
from data_processor.data_set import dataset_from_str


class Throughput:
    def __init__(self, resources: Path):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._resources = resources

    def process(self, used_energy_file: Path, create_tex: bool, no_tool: list, no_dataset: list):
        frameio = FrameIO()
        self._logger.debug("loading %s", used_energy_file)
        df = frameio.load(used_energy_file)
        df = df[~df["tool"].isin(no_tool)]
        df = df[~df["dataset"].isin(no_dataset)]

        result_df = self._calculate_throughput(df)
        self._print_table(result_df)
        self._create_csv(used_energy_file, result_df)

    def _print_table(self, df: pd.DataFrame):
        table_df = df.drop(columns=["num_runs"])
        table_df["average_real"] = table_df["average_real"].astype(float)
        table_df["throughput"] = table_df["throughput"].pint.to("MiB/s")
        table_df["throughput"] = table_df["throughput"].astype(float)

        table_entries = []
        for _, row in table_df.iterrows():
            table_entries.append(row.values[:])

        headers = list(table_df.columns)[:-2]
        headers.append("average_real (s)")
        headers.append("throughput (MiB/s")
        table_str = tabulate.tabulate(table_entries,
                                      headers=headers,
                                      tablefmt="simple"
                                      )
        print(table_str)

    def _create_csv(self, used_energy_file: Path, df: pd.DataFrame):
        tp_file = self._resources / ("tp_%s" % used_energy_file.stem.removeprefix("used_energy_") + ".csv")
        frameio = FrameIO()
        frameio.persist(df, tp_file)

    def _calculate_throughput(self, df: pd.DataFrame) -> pd.DataFrame:
        group_cols = ["host", "tool", "dataset", "mode", "strength", "threading"]
        result_df = (
            df.groupby(group_cols, as_index=False)
            .agg(
                average_real=("real", "mean"),
                num_runs=("run", "count"),
            )
        )

        def dataset_map(str_ds):
            return dataset_from_str(str_ds).value

        result_df["throughput"] = (
                result_df["dataset"].map(dataset_map) / result_df["average_real"]
        )
        return result_df
