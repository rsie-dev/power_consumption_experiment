import logging
from pathlib import Path

import tabulate
import pandas as pd

from data_processor.util import FrameIO
from data_processor.constants import GROUP_COLS, ORDER_TOOL, ORDER_STRENGTH


class Power:
    def __init__(self, resources: Path):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._frameio = FrameIO()
        self._resources = resources

    def process(self, used_energy_file: Path, create_tex: bool, no_tool: list, no_dataset: list):
        self._logger.debug("loading %s", used_energy_file)
        df = self._frameio.load(used_energy_file)
        df = df[~df["tool"].isin(no_tool)]
        df = df[~df["dataset"].isin(no_dataset)]

        power_df = self._calculate_power(df)
        power_df["average_power"] = power_df["average_power"].pint.to("watt")

        power_df["_tool_key"] = power_df["tool"].apply(ORDER_TOOL.index)
        power_df["_strength_key"] = power_df["strength"].apply(ORDER_STRENGTH.index)
        power_df = power_df.sort_values(
            by=["host", "_tool_key", "dataset", "mode", "_strength_key"],
        ).drop(columns=["_tool_key", "_strength_key"])

        self._print_table(power_df)
        self._create_csv(used_energy_file, power_df)

    def _print_table(self, df: pd.DataFrame):
        table_df = df.copy()
        unit_energy = str(table_df["sum_energy"].dtype.units)
        unit_duration = str(table_df["sum_duration"].dtype.units)
        unit_power = str(table_df["average_power"].dtype.units)

        table_entries = []
        for col in ["sum_energy", "sum_duration", "average_power"]:
            table_df[col] = table_df[col].astype(float)
        for _, row in table_df.iterrows():
            table_entries.append(row.values[:])

        cols = table_df.columns.tolist()
        headers = cols[:-3]
        headers.append("total energy (%s)" % unit_energy)
        headers.append("total duration (%s)" % unit_duration)
        headers.append("average power (%s)" % unit_power)
        table_str = tabulate.tabulate(table_entries,
                                      headers=headers,
                                      tablefmt="simple"
                                      )
        print(table_str)

    def _create_csv(self, used_energy_file: Path, df: pd.DataFrame):
        tp_file = self._resources / ("power_%s" % used_energy_file.stem.removeprefix("used_energy_") + ".csv")
        self._frameio.persist(df, tp_file)

    def _calculate_power(self, df: pd.DataFrame) -> pd.DataFrame:
        result_df = (
            df.groupby(GROUP_COLS, as_index=False)
            .agg(
                num_runs=("run", "size"),
                sum_energy=("energy", "sum"),
                sum_duration=("duration", "sum"),
            )
        )
        result_df["average_power"] = result_df["sum_energy"] / result_df["sum_duration"]

        return result_df
