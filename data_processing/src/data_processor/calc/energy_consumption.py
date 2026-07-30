import logging
from pathlib import Path
from dataclasses import dataclass

import tabulate
import pandas as pd

from data_processor import ureg
from data_processor.constants import GROUP_COLS, ORDER_TOOL, ORDER_STRENGTH
from .calc_params import EnergyParams
from .calculator import Calculator


class EnergyConsumption(Calculator):
    @dataclass(frozen=True)
    class Params(EnergyParams):
        idle_power: Path

    def __init__(self, resources: Path):
        super().__init__(resources)
        self._logger = logging.getLogger(self.__class__.__name__)
        self._virtual_powers = [0, 1]

    @property
    def virtual_powers(self) -> list:
        return self._virtual_powers

    def process(self, params: Params):
        df = self._load(params)
        idle_power_df = self._frameio.load(params.idle_power)

        energy_df = self._calculate_energy_consumption(df, idle_power_df)

        energy_df["_tool_key"] = energy_df["tool"].apply(ORDER_TOOL.index)
        energy_df["_strength_key"] = energy_df["strength"].apply(ORDER_STRENGTH.index)
        energy_df = energy_df.sort_values(
            by=["host", "_tool_key", "dataset", "mode", "_strength_key"],
        ).drop(columns=["_tool_key", "_strength_key"])

        self._print_table(energy_df)
        energy_file = "energy_consumption_%s" % params.used_energy_file.stem.removeprefix("used_energy_") + ".csv"
        self._create_csv(energy_file, energy_df)

    def _print_table(self, df: pd.DataFrame) -> None:
        table_df = df.copy()
        unit_energy = str(table_df["average_energy_consumption_total"].dtype.units)

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

    def _calculate_energy_consumption(self, df: pd.DataFrame, idle_power_df: pd.DataFrame) -> pd.DataFrame:
        def get_idle(host: str):
            result = idle_power_df.loc[idle_power_df["host"] == host, "average_power"]
            average_power = result.iloc[0]
            return average_power

        df = df.copy()
        df["energy_net"] = df["energy"] - df["host"].map(get_idle) * df["real"]
        virtual_powers = [p * ureg.watt for p in self._virtual_powers]
        for p_virtual  in virtual_powers:
            df["energy_norm_%s" % p_virtual.magnitude] = df["energy_net"] + p_virtual * df["real"]

        result_df = (
            df.groupby(GROUP_COLS, as_index=False)
            .agg(
                num_runs=("run", "size"),
                average_energy_consumption_total=("energy", "mean"),
                average_energy_consumption_net=("energy_net", "mean"),
                **{
                    "average_energy_consumption_norm_%s_watt" % v.magnitude: ("energy_norm_%s" % v.magnitude, "mean")
                    for v in virtual_powers
                },
            )
            .reset_index(drop=True)
        )
        return result_df
