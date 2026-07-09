import logging
from pathlib import Path

import tabulate
from tabulate import SEPARATING_LINE
import humanize
import pandas as pd

from data_processor.util import FrameIO
from data_processor.constants import GROUP_COLS


class Statistics:
    VALUE_COLS = ["energy", "real", "size"]

    def __init__(self, resources: Path):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._resources = resources

    def process(self, used_power_file: Path):
        frameio = FrameIO()
        self._logger.debug("loading %s", used_power_file)
        df = frameio.load(used_power_file)
        stats_df = self._calculate_statistics(df)

        self._print_table(stats_df)

        stat_file = self._resources / ("stats_" + used_power_file.stem.removeprefix("used_energy_") + ".csv")
        frameio.persist(stats_df, stat_file)

    def _calculate_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        stats_df = pd.concat(
            [
                df.groupby(GROUP_COLS, as_index=False)
                .agg(
                    num_runs=("run", "size"),
                    energy=("energy", stat),
                    real=("real", stat),
                    size=("size", stat),
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
        table_entries = self._create_table_entries(df)
        unit_energy = str(df["energy"].dtype.units)
        unit_times = str(df["real"].dtype.units)
        unit_size = str(df["size"].dtype.units)
        headers = ["stat", "host", "tool", "dataset", "mode", "strength", "threading", "num runs",
                   "energy (%s)" % unit_energy, "real (%s)" % unit_times, "size (%s)" % unit_size]
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
                entry = list(row.values[:1 + len(GROUP_COLS) + 1 + len(self.VALUE_COLS) - 1])
                if pd.isna(row["size"]):
                    entry.append(None)
                else:
                    entry.append(humanize.naturalsize(row["size"], binary=True))
                table_entries.append(entry)

            if i < len(groups) -1:
                table_entries.append(SEPARATING_LINE)
        return table_entries
