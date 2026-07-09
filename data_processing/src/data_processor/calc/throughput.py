import logging
from pathlib import Path

import tabulate
import pandas as pd

from data_processor.util import FrameIO
from data_processor.data_set import dataset_from_str
from data_processor.constants import GROUP_COLS, ORDER_TOOL, ORDER_STRENGTH


class Throughput:
    def __init__(self, resources: Path):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._frameio = FrameIO()
        self._resources = resources

    def process(self, used_energy_file: Path, create_tex: bool, no_tool: list, no_dataset: list):
        self._logger.debug("loading %s", used_energy_file)
        df = self._frameio.load(used_energy_file)
        df = df[~df["tool"].isin(no_tool)]
        df = df[~df["dataset"].isin(no_dataset)]

        result_df = self._calculate_throughput(df)
        result_df["_tool_key"] = result_df["tool"].apply(ORDER_TOOL.index)
        result_df["_strength_key"] = result_df["strength"].apply(ORDER_STRENGTH.index)
        result_df = result_df.sort_values(
            by=["host", "_tool_key", "dataset", "mode", "_strength_key"],
            #ascending=[True]
        ).drop(columns=["_tool_key", "_strength_key"])

        self._print_table(result_df)
        self._create_csv(used_energy_file, result_df)

    def _print_table(self, df: pd.DataFrame):
        table_df = df.drop(columns=[])
        table_df["average_real"] = table_df["average_real"].astype(float)
        table_df["average_throughput"] = table_df["average_throughput"].pint.to("MiB/s")
        table_df["average_throughput"] = table_df["average_throughput"].astype(float)

        cols = table_df.columns.tolist()
        i = cols.index("num_runs")
        cols.insert(i - 1, cols.pop(i))
        table_df = table_df[cols]

        table_entries = []
        for _, row in table_df.iterrows():
            table_entries.append(row.values[:])

        headers = cols[:-2]
        headers.append("average real (s)")
        headers.append("average throughput (MiB/s)")
        table_str = tabulate.tabulate(table_entries,
                                      headers=headers,
                                      tablefmt="simple"
                                      )
        print(table_str)

    def _create_csv(self, used_energy_file: Path, df: pd.DataFrame):
        tp_file = self._resources / ("tp_%s" % used_energy_file.stem.removeprefix("used_energy_") + ".csv")
        self._frameio.persist(df, tp_file)

    def _calculate_throughput(self, df: pd.DataFrame) -> pd.DataFrame:
        def dataset_map(str_ds):
            return dataset_from_str(str_ds).value

        df["throughput"] = df["dataset"].map(dataset_map) / df["real"]

        result_df = (
            df.groupby(GROUP_COLS, as_index=False)
            .agg(
                num_runs=("run", "count"),
                average_real=("real", "mean"),
                average_throughput=("throughput", "mean"),
            )
        )

        return result_df
