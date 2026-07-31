import logging
from pathlib import Path
from dataclasses import dataclass

import tabulate
import pandas as pd

from data_processor.constants import GROUP_COLS, ORDER_TOOL, ORDER_STRENGTH
from data_processor.data_set import dataset_from_str
from .calc_params import EnergyParams
from .calculator import Calculator


class EnergyEfficiency(Calculator):
    @dataclass(frozen=True)
    class Params(EnergyParams):
        idle_power: Path

    def __init__(self, resources: Path):
        super().__init__(resources)
        self._logger = logging.getLogger(self.__class__.__name__)

    def process(self, params: Params):
        df = self._load(params)
        idle_power_df = self._frameio.load(params.idle_power)

        energy_df = self._calculate_energy_efficiency(df, idle_power_df)

        energy_df["_tool_key"] = energy_df["tool"].apply(ORDER_TOOL.index)
        energy_df["_strength_key"] = energy_df["strength"].apply(ORDER_STRENGTH.index)
        energy_df = energy_df.sort_values(
            by=["host", "_tool_key", "dataset", "mode", "_strength_key"],
        ).drop(columns=["_tool_key", "_strength_key"])

        self._print_table(energy_df)
        energy_file = "energy_efficiency_%s" % params.used_energy_file.stem.removeprefix("used_energy_") + ".csv"
        self._create_csv(energy_file, energy_df)

    def _print_table(self, df: pd.DataFrame) -> None:
        table_df = df.copy()
        for c in ["average_energy_efficiency_total", "average_energy_efficiency_net"]:
            table_df[c] = table_df[c].pint.to("MiB/joule")
        unit_energy = str(table_df["average_energy_efficiency_total"].dtype.units)

        table_entries = []
        cols = table_df.columns.tolist()
        energy_cols = cols[len(GROUP_COLS) + 1:]
        for col in energy_cols:
            table_df[col] = table_df[col].astype(float)
        for _, row in table_df.iterrows():
            table_entries.append(row.values[:])

        headers = cols[:len(GROUP_COLS) + 1]
        for column in energy_cols:
            headers.append("%s (%s)" % (column.replace("_", " "), unit_energy))

        table_str = tabulate.tabulate(table_entries,
                                      headers=headers,
                                      tablefmt="simple"
                                      )
        print(table_str)

    def _calculate_energy_efficiency(self, df: pd.DataFrame, idle_power_df: pd.DataFrame) -> pd.DataFrame:
        def get_idle(host: str):
            result = idle_power_df.loc[idle_power_df["host"] == host, "average_power"]
            average_power = result.iloc[0]
            return average_power

        def get_data_size(dataset: str):
            return dataset_from_str(dataset).value

        df["energy_efficiency_total"] = df["dataset"].map(get_data_size) / df["energy"]
        df["energy_consumption_net"] = df["energy"] - df["host"].map(get_idle) * df["real"]
        df["energy_efficiency_net"] = df["dataset"].map(get_data_size) / df["energy_consumption_net"]

        result_df = (
            df.groupby(GROUP_COLS, as_index=False)
            .agg(
                num_runs=("run", "size"),
                average_energy_efficiency_total=("energy_efficiency_total", "mean"),
                average_energy_efficiency_net=("energy_efficiency_net", "mean"),
            )
            .reset_index(drop=True)
        )
        return result_df
