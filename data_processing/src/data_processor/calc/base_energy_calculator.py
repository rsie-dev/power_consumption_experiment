import tabulate
import pandas as pd

from data_processor.constants import GROUP_COLS
from .calculator import Calculator


class BaseEnergyCalculator(Calculator):
    def _calculate_average_power(self, df: pd.DataFrame) -> pd.DataFrame:
        df["average_power"] = df["energy"] / df["duration"]
        df["average_power"] = df["average_power"].pint.to("watt")
        return df

    def _lookup_idle_power(self, host: str, idle_power_df: pd.DataFrame):
        result = idle_power_df.loc[idle_power_df["host"] == host, "average_power"]
        average_power = result.iloc[0]
        return average_power

    def _pre_print_table(self, df: pd.DataFrame, unit_energy) -> str:
        table_entries = []
        cols = df.columns.tolist()
        energy_cols = cols[len(GROUP_COLS) + 1:]
        for col in energy_cols:
            df[col] = df[col].astype(float)
        for _, row in df.iterrows():
            table_entries.append(row.values[:])

        headers = cols[:len(GROUP_COLS) + 1]
        for column in energy_cols:
            headers.append("%s (%s)" % (column.replace("_", " "), unit_energy))

        table_str = tabulate.tabulate(table_entries,
                                      headers=headers,
                                      tablefmt="simple"
                                      )
        return table_str
