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
            group = "_".join(group_name)
            self._process_group(group, group_df)

    def _process_group(self, group: str, df):
        unit_power = str(df["power"].dtype.units)
        values_power = df["power"].pint.magnitude
        unit_times = str(df["real"].dtype.units)
        values_times = df["real"].pint.magnitude
        unit_size = str(df["size"].dtype.units)
        values_size = df["size"].pint.magnitude

        stats_power = self._get_stats(values_power)
        stats_times = self._get_stats(values_times)
        stats_sizes = {}
        for k,v in self._get_stats(values_size).items():
            if k == "stdev":
                if v == "n/a":
                    stats_sizes[k] = v
                else:
                    stats_sizes[k] = humanize.naturalsize(v, binary=True)
            else:
                stats_sizes[k] = humanize.naturalsize(v, binary=True)

        table_entries = []
        table_entries.append(("min", stats_power["min"], stats_times["min"], stats_sizes["min"]))
        table_entries.append(("max", stats_power["max"], stats_times["max"], stats_sizes["max"]))
        table_entries.append(("mean", stats_power["mean"], stats_times["mean"], stats_sizes["mean"]))
        table_entries.append(("std", stats_power["stdev"], stats_times["stdev"], stats_sizes["stdev"]))

        headers = ["stat", "power (%s)" % unit_power, "times (%s)" % unit_times, "size (%s)" % unit_size]
        table_str = tabulate.tabulate(table_entries,
                                      headers=headers,
                                      tablefmt="simple"
                                      )
        print(f"Group: {group}")
        print(table_str)

    def _get_stats(self, values):
        result = {
            "min": min(values),
            "max": max(values),
            "mean": mean(values),
        }
        if len(values) >= 2:
            result["stdev"] = stdev(values)
        else:
            result["stdev"] = "n/a"
        return result
