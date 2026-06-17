import logging
from pathlib import Path
from statistics import mean, stdev

import tabulate
import humanize

from data_processor.util import FrameIO


class Statistics:
    def __init__(self):
        self._logger = logging.getLogger(self.__class__.__name__)

    def process(self, used_power_file: Path):
        frameio = FrameIO()
        self._logger.debug("loading %s", used_power_file)
        df = frameio.load(used_power_file)
        self._list_statistics(df)

    def _list_statistics(self, df):
        group_cols = ["host", "tool", "dataset", "mode", "strength", "threading"]
        for group_name, group_df in df.groupby(group_cols):
            unit_power = str(group_df["power"].dtype.units)
            values_power = group_df["power"].pint.magnitude
            unit_times = str(group_df["real"].dtype.units)
            values_times = group_df["real"].pint.magnitude
            unit_size = str(group_df["size"].dtype.units)
            values_size = group_df["size"].pint.magnitude

            table_entries = []
            table_entries.append(self._get_stats("power", unit_power, values_power))
            table_entries.append(self._get_stats("times", unit_times, values_times))
            size_entries = self._get_stats("size", unit_size, values_size)
            entries = list(size_entries[:2]) + [humanize.naturalsize(s) for s in size_entries[2:]]
            table_entries.append(entries)

            headers = ["item", "unit", "min", "max", "mean", "std"]
            table_str = tabulate.tabulate(table_entries,
                                          headers=headers,
                                          tablefmt="simple"
                                          )
            print(f"Group: {"_".join(group_name)}")
            print(table_str)

    def _get_stats(self, name: str, unit: str, values):
        return name, unit, min(values), max(values), mean(values), stdev(values)
