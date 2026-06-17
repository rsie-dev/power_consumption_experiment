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
            stats_power = self._get_stats(values_power)
            stats_times = self._get_stats(values_times)
            stats_sizes = [humanize.naturalsize(s) for s in self._get_stats(values_size)]

            table_entries.append(("min", stats_power[0], stats_times[0], stats_sizes[0]))
            table_entries.append(("max", stats_power[1], stats_times[1], stats_sizes[1]))
            table_entries.append(("mean", stats_power[2], stats_times[2], stats_sizes[2]))
            table_entries.append(("std", stats_power[3], stats_times[3], stats_sizes[3]))

            headers = ["stat", "power (%s)" % unit_power, "times (%s)" % unit_times, "size (%s)" % unit_size]
            table_str = tabulate.tabulate(table_entries,
                                          headers=headers,
                                          tablefmt="simple"
                                          )
            print(f"Group: {"_".join(group_name)}")
            print(table_str)
    def _get_stats(self, values):
        return min(values), max(values), mean(values), stdev(values)
