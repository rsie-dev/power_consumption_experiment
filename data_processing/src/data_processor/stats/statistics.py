import logging
from pathlib import Path
from statistics import mean, stdev

import tabulate
from tabulate import SEPARATING_LINE
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
        unit_energy = str(df["energy"].dtype.units)
        unit_times = str(df["real"].dtype.units)
        unit_size = str(df["size"].dtype.units)
        table_entries = []
        for _, group_df in df.groupby(group_cols):
            entries = self._process_group(group_df)
            table_entries.extend(entries)
            table_entries.append(SEPARATING_LINE)

        headers = ["stat", "host", "tool", "dataset", "mode", "strength", "threading",
                   "energy (%s)" % unit_energy, "times (%s)" % unit_times, "size (%s)" % unit_size]
        table_str = tabulate.tabulate(table_entries,
                                      headers=headers,
                                      tablefmt="simple"
                                      )
        print(table_str)

    def _process_group(self, df) -> list:
        values_energy = df["energy"].pint.magnitude
        values_times = df["real"].pint.magnitude
        values_size = df["size"].pint.magnitude

        stats_energy = self._post_process_stats(self._get_stats(values_energy))
        stats_times = self._post_process_stats(self._get_stats(values_times))
        stats_sizes = {}
        for k, v in self._get_stats(values_size).items():
            if k == "stdev":
                if v == "n/a":
                    stats_sizes[k] = None
                else:
                    stats_sizes[k] = humanize.naturalsize(v, binary=True)
            else:
                stats_sizes[k] = humanize.naturalsize(v, binary=True)

        table_entries = []
        host = df["host"].iloc[0]
        tool = df["tool"].iloc[0]
        dataset = df["dataset"].iloc[0]
        mode = df["mode"].iloc[0]
        strength = df["strength"].iloc[0]
        threading = df["threading"].iloc[0]
        table_entries.append(("min", host, tool, dataset, mode, strength, threading,
                              stats_energy["min"], stats_times["min"], stats_sizes["min"]))
        table_entries.append(("max", host, tool, dataset, mode, strength, threading,
                              stats_energy["max"], stats_times["max"], stats_sizes["max"]))
        table_entries.append(("mean", host, tool, dataset, mode, strength, threading,
                              stats_energy["mean"], stats_times["mean"], stats_sizes["mean"]))
        table_entries.append(("std", host, tool, dataset, mode, strength, threading,
                              stats_energy["stdev"], stats_times["stdev"], stats_sizes["stdev"]))
        return table_entries

    def _post_process_stats(self, stats: dict) -> dict:
        processed_stats = {}
        for k, v in stats.items():
            if v == "n/a":
                processed_stats[k] = None
            else:
                processed_stats[k] = v
        return processed_stats

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
