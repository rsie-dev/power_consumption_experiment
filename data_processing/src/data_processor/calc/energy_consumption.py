import logging
from pathlib import Path
from dataclasses import dataclass

import pandas as pd

from data_processor import ureg
from data_processor.constants import GROUP_COLS
from .calc_params import EnergyParams
from .base_energy_calculator import BaseEnergyCalculator


class EnergyConsumption(BaseEnergyCalculator):
    @dataclass(frozen=True)
    class ConsumptionParams(EnergyParams):
        idle_power: Path

    def __init__(self, resources: Path):
        super().__init__(resources)
        self._logger = logging.getLogger(self.__class__.__name__)
        self._virtual_powers = [0, 1]

    @property
    def virtual_powers(self) -> list:
        return self._virtual_powers

    def process(self, params: ConsumptionParams):
        df = self._load(params)
        idle_power_df = self._frameio.load(params.idle_power)

        energy_df = self._calculate_energy_consumption(df, idle_power_df)
        energy_df = self._order_data(energy_df)

        self._print_table(energy_df)
        energy_file = "energy_consumption_%s" % params.used_energy_file.stem.removeprefix("used_energy_") + ".csv"
        self._create_csv(energy_file, energy_df)

    def _print_table(self, df: pd.DataFrame) -> None:
        table_df = df.copy()
        unit_energy = str(table_df["average_energy_consumption_total"].dtype.units)
        table_str = self._pre_print_table(table_df, unit_energy)
        print(table_str)

    def _calculate_energy_consumption(self, df: pd.DataFrame, idle_power_df: pd.DataFrame) -> pd.DataFrame:
        idle_power_df = self._calculate_average_power(idle_power_df)

        def get_idle(host: str):
            return self._lookup_idle_power(host, idle_power_df)

        df["energy_net"] = df["energy"] - df["host"].map(get_idle) * df["duration"]
        virtual_powers = [p * ureg.watt for p in self._virtual_powers]
        for p_virtual  in virtual_powers:
            df["energy_norm_%s" % p_virtual.magnitude] = df["energy_net"] + p_virtual * df["duration"]

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
