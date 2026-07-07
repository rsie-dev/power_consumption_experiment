import logging
from pathlib import Path
from statistics import mean, stdev

import tabulate
from tabulate import SEPARATING_LINE
import humanize
import pandas as pd

from data_processor.util import FrameIO


class Statistics:
    def __init__(self, resources: Path):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._resources = resources

    def process(self, used_power_file: Path):
        frameio = FrameIO()
        self._logger.debug("loading %s", used_power_file)
        df = frameio.load(used_power_file)
        groups = self._collect_statistics(df)

        table_entries = self._create_table_entries(groups)
        unit_energy = str(df["energy"].dtype.units)
        unit_times = str(df["real"].dtype.units)
        unit_size = str(df["size"].dtype.units)
        headers = ["stat", "host", "tool", "dataset", "mode", "strength", "threading",
                   "energy (%s)" % unit_energy, "time (%s)" % unit_times, "size (%s)" % unit_size]
        table_str = tabulate.tabulate(table_entries,
                                      headers=headers,
                                      tablefmt="simple"
                                      )
        print(table_str)
        stat_file = self._resources / ("stats_" + used_power_file.stem.removeprefix("used_energy_") + ".csv")
        stats_df = self._create_stats_df(groups)
        frameio.persist(stats_df, stat_file)

    def _create_stats_df(self, groups) -> pd.DataFrame:
        all_df = []
        for group in groups:
            all_df.append(self._build_df(group, "min"))
            all_df.append(self._build_df(group, "max"))
            all_df.append(self._build_df(group, "mean"))
            all_df.append(self._build_df(group, "stdev"))

        df = pd.concat(all_df)
        df["energy"] = df["energy"].astype("pint[joule]")
        df["time"] = df["time"].astype("pint[second]")
        df["size"] = df["size"].astype("pint[byte]")

        return df

    def _build_df(self, group, key):
        stats_energy = group["energy"]
        stats_times = group["times"]
        stats_sizes = group["sizes"]
        data = {
            "stat": key,
            "host": group["host"],
            "tool": group["tool"],
            "dataset": group["dataset"],
            "mode": group["mode"],
            "strength": group["strength"],
            "threading": group["threading"],
            "energy": stats_energy[key],
            "time": stats_times[key],
            "size": stats_sizes[key],
        }
        df = pd.DataFrame([data])
        return df

    def _create_table_entries(self, groups):
        table_entries = []
        for group in groups:
            entry = self._create_table_entry(group)
            table_entries.extend(entry)
            table_entries.append(SEPARATING_LINE)
        return table_entries

    def _create_table_entry(self, group):
        entry = []

        stats_energy = group["energy"]
        stats_times = group["times"]
        stats_sizes = {}
        for k, v in group["sizes"].items():
            if k == "stdev":
                if v is None:
                    stats_sizes[k] = None
                else:
                    stats_sizes[k] = humanize.naturalsize(v, binary=True)
            else:
                stats_sizes[k] = humanize.naturalsize(v, binary=True)

        entry.append(("min",
                      group["host"], group["tool"], group["dataset"],
                      group["mode"], group["strength"], group["threading"],
                      stats_energy["min"], stats_times["min"], stats_sizes["min"]))
        entry.append(("max",
                      group["host"], group["tool"], group["dataset"],
                      group["mode"], group["strength"], group["threading"],
                      stats_energy["max"], stats_times["max"], stats_sizes["max"]))
        entry.append(("mean",
                      group["host"], group["tool"], group["dataset"],
                      group["mode"], group["strength"], group["threading"],
                      stats_energy["mean"], stats_times["mean"], stats_sizes["mean"]))
        entry.append(("stdev",
                      group["host"], group["tool"], group["dataset"],
                      group["mode"], group["strength"], group["threading"],
                      stats_energy["stdev"], stats_times["stdev"], stats_sizes["stdev"]))

        return entry

    def _collect_statistics(self, df) -> list:
        group_cols = ["host", "tool", "dataset", "mode", "strength", "threading"]
        groups = []
        for _, group_df in df.groupby(group_cols):
            group_stats = self._collects_group_stats(group_df)
            groups.append(group_stats)
        return groups

    def _collects_group_stats(self, df) -> dict:
        values_energy = df["energy"].pint.magnitude
        values_times = df["real"].pint.magnitude
        values_size = df["size"].pint.magnitude
        stats_energy = self._post_process_stats(self._get_stats(values_energy))
        stats_times = self._post_process_stats(self._get_stats(values_times))
        stats_sizes = self._post_process_stats(self._get_stats(values_size))

        result = {
            "host": df["host"].iloc[0],
            "tool": df["tool"].iloc[0],
            "dataset": df["dataset"].iloc[0],
            "mode": df["mode"].iloc[0],
            "strength": df["strength"].iloc[0],
            "threading": df["threading"].iloc[0],
            "energy": stats_energy,
            "times": stats_times,
            "sizes": stats_sizes,
        }
        return result

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
