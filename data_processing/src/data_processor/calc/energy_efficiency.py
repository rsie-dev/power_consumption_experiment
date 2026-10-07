import logging
from pathlib import Path
from dataclasses import dataclass

import pandas as pd

from data_processor.constants import GROUP_COLS
from data_processor.data_set import get_dataset_size
from .calc_params import EnergyParams
from .base_energy_calculator import BaseEnergyCalculator


class EnergyEfficiency(BaseEnergyCalculator):
    @dataclass(frozen=True)
    class EfficiencyParams(EnergyParams):
        idle_power: Path

    def __init__(self, resources: Path):
        super().__init__(resources)
        self._logger = logging.getLogger(self.__class__.__name__)

    def process(self, params: EfficiencyParams):
        df = self._load(params)
        idle_power_df = self._frameio.load(params.idle_power)

        energy_df = self._calculate_energy_efficiency(df, idle_power_df)
        energy_df = self._order_data(energy_df)

        self._print_table(energy_df)
        energy_file = "energy_efficiency_%s" % params.used_energy_file.stem.removeprefix("used_energy_") + ".csv"
        self._create_csv(energy_file, energy_df)

    def _print_table(self, df: pd.DataFrame) -> None:
        table_df = df.copy()
        for c in ["average_energy_efficiency_total", "average_energy_efficiency_net"]:
            table_df[c] = table_df[c].pint.to("MiB/joule")
        unit_energy = str(table_df["average_energy_efficiency_total"].dtype.units)
        table_str = self._pre_print_table(table_df, unit_energy)
        print(table_str)

    def _calculate_energy_efficiency(self, df: pd.DataFrame, idle_power_df: pd.DataFrame) -> pd.DataFrame:
        idle_power_df = self._calculate_average_power(idle_power_df)

        def get_idle(host: str):
            return self._lookup_idle_power(host, idle_power_df)

        df["energy_efficiency_total"] = df["dataset"].map(get_dataset_size) / df["energy"]
        df["energy_consumption_net"] = df["energy"] - df["host"].map(get_idle) * df["duration"]
        df["energy_efficiency_net"] = df["dataset"].map(get_dataset_size) / df["energy_consumption_net"]

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
