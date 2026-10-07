import logging
from pathlib import Path

import tabulate
from tabulate import SEPARATING_LINE
import pandas as pd

from data_processor.constants import GROUP_COLS
from data_processor.processor import Processor


class ThroughputStatistics(Processor):
    VALUE_COLS = ["throughput"]

    def __init__(self, resources: Path):
        super().__init__(resources)
        self._logger = logging.getLogger(self.__class__.__name__)

    def process(self, tp_file: Path):
        df = self._frameio.load(tp_file)
        stats_df = self._calculate_statistics(df)

        self._print_table(stats_df)

        #stat_file = "stats_tp_" + tp_file.stem.removeprefix("tp_") + ".csv"
        #self._create_csv(stat_file, stats_df)

    def _calculate_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        stats_df = pd.concat(
            [
                df.groupby(GROUP_COLS, as_index=False)
                .agg(
                    num_runs=("run", "size"),
                    throughput=("throughput", stat),
                )
                .assign(stat=name)
                for name, stat in [
                ("min", "min"),
                ("max", "max"),
                ("mean", "mean"),
                ("stdev", "std"),
            ]
            ],
            ignore_index=True,
                )

        stats_df = stats_df[
            ["stat"] + GROUP_COLS + ["num_runs"] + self.VALUE_COLS
            ]

        stats_df["stat"] = pd.Categorical(
            stats_df["stat"],
            categories=["min", "max", "mean", "stdev"],
            ordered=True,
        )
        stats_df = stats_df.sort_values(
            GROUP_COLS + ["stat"],
            ignore_index=True,
            )

        return stats_df

    def _print_table(self, df):
        table_df = df.copy()
        table_df["throughput"] = table_df["throughput"].pint.to("MiB/s")
        table_entries = self._create_table_entries(table_df)
        unit_tp = str(table_df["throughput"].dtype.units)
        headers = ["stat", "host", "tool", "dataset", "mode", "strength", "threading", "num runs",
                   "throughput (%s)" % unit_tp]
        table_str = tabulate.tabulate(table_entries,
                                      headers=headers,
                                      tablefmt="simple"
                                      )
        print(table_str)

    def _create_table_entries(self, df):
        table_df = df.copy()
        for col in self.VALUE_COLS:
            table_df[col] = table_df[col].astype(float)
        table_entries = []
        groups = list(table_df.groupby(GROUP_COLS, sort=False))
        for i, (_, block) in enumerate(groups):
            for _, row in block.iterrows():
                entry = list(row.values[:1 + len(GROUP_COLS) + 1 + len(self.VALUE_COLS)])
                table_entries.append(entry)

            if i < len(groups) -1:
                table_entries.append(SEPARATING_LINE)
        return table_entries
