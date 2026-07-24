import logging
from pathlib import Path

import tabulate
import pandas as pd

from data_processor.util import FrameIO
from data_processor.constants import GROUP_COLS, ORDER_TOOL, ORDER_STRENGTH


class EnergyConsumption:
    def __init__(self, resources: Path):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._frameio = FrameIO()
        self._resources = resources

    def process(self, used_energy_file: Path, create_tex: bool, no_tool: list, no_dataset: list, idle_power: Path):
        idle_power_df = self._frameio.load(idle_power)
        df = self._frameio.load(used_energy_file)
        df = df[~df["tool"].isin(no_tool)]
        df = df[~df["dataset"].isin(no_dataset)]

        energy_df = self._calculate_energy_consumption(df, idle_power_df)

        energy_df["_tool_key"] = energy_df["tool"].apply(ORDER_TOOL.index)
        energy_df["_strength_key"] = energy_df["strength"].apply(ORDER_STRENGTH.index)
        energy_df = energy_df.sort_values(
            by=["host", "_tool_key", "dataset", "mode", "_strength_key"],
        ).drop(columns=["_tool_key", "_strength_key"])

        self._print_table(energy_df)
        #self._create_csv(used_energy_file, energy_df)

    def _print_table(self, df: pd.DataFrame) -> None:
        table_df = df.copy()
        unit_energy = str(table_df["average_energy_total"].dtype.units)

        table_entries = []
        for col in ["average_energy_total"]:
            table_df[col] = table_df[col].astype(float)
        for _, row in table_df.iterrows():
            table_entries.append(row.values[:])

        cols = table_df.columns.tolist()
        headers = cols[:-1]
        headers.append("average energy total (%s)" % unit_energy)
        table_str = tabulate.tabulate(table_entries,
                                      headers=headers,
                                      tablefmt="simple"
                                      )
        print(table_str)

    def _create_csv(self, used_energy_file: Path, df: pd.DataFrame) -> None:
        tp_file = self._resources / ("energy_consumption_%s" % used_energy_file.stem.removeprefix("used_energy_") + ".csv")
        self._frameio.persist(df, tp_file)

    def _calculate_energy_consumption(self, df: pd.DataFrame, idle_power_df: pd.DataFrame) -> pd.DataFrame:
        result_df = (
            df.groupby(GROUP_COLS, as_index=False)
            .agg(
                num_runs=("run", "size"),
                average_energy_total=("energy", "mean"),
            )
        )
        return result_df
