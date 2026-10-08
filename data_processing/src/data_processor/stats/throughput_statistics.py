import logging
from pathlib import Path

import tabulate
from tabulate import SEPARATING_LINE
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

from data_processor.constants import GROUP_COLS
from data_processor.processor import Processor


class ThroughputStatistics(Processor):
    VALUE_COLS = ["throughput"]
    DRIFT_THRESHOLD = 1.0  # percentage points across runs 1–30

    def __init__(self, resources: Path):
        super().__init__(resources)
        self._logger = logging.getLogger(self.__class__.__name__)

    def process(self, tp_file: Path):
        df = self._frameio.load(tp_file)

        self._validate_temporal_drift(df)
        return

        stats_df = self._calculate_statistics(df)

        self._print_table(stats_df)

        #stat_file = "stats_tp_" + tp_file.stem.removeprefix("tp_") + ".csv"
        #self._create_csv(stat_file, stats_df)

    def _validate_temporal_drift(self, df: pd.DataFrame):
        self._check_temporal_drift(df, "single", "compress", self.DRIFT_THRESHOLD, [])
        self._check_temporal_drift(df, "single", "decompress", self.DRIFT_THRESHOLD, [])
        single_thread_only = ["gzip", "bzip2", "lzop"]
        self._check_temporal_drift(df, "multi", "compress", self.DRIFT_THRESHOLD, single_thread_only)
        self._check_temporal_drift(df, "multi", "decompress", self.DRIFT_THRESHOLD, single_thread_only)

    def _check_temporal_drift(self, df: pd.DataFrame, threading: str, mode: str, threshold: float, excluded: list[str]):
        df = df[~df["tool"].isin(excluded)]
        drift_results = self._calculate_temporal_drift(df, mode, threshold)
        self._show_drift_analysis(threading, mode, threshold, drift_results)
        self._verify_temporal_drift(drift_results)

    def _verify_temporal_drift(self, df: pd.DataFrame):
        matches = df["drift_conclusion"].str.startswith("material", na=False)
        matching_hosts = df.loc[matches, "host"].unique().tolist()
        if matching_hosts:
            raise ValueError("material temporal drift detected on: %s" % matching_hosts)

    def _show_drift_analysis(self, threading: str, mode: str, threshold: float, drift_results: pd.DataFrame) -> None:
        columns = [
            "host",
            "change_run1_to_run30_percent",
            "ci_low_run1_to_run30_percent",
            "ci_high_run1_to_run30_percent"
        ]

        headers = ["host", "change %", "CI low %", "CI high %", "drift conclusion"]
        table_entries = []
        for _, row in drift_results[columns + ["drift_conclusion"]].iterrows():
            table_entries.append(row.values[:])
        table_str = tabulate.tabulate(table_entries,
                                      headers=headers,
                                      tablefmt="simple",
                                      floatfmt=".3f",
                                      )
        print("Drift analysis for %s %s (threshold: %s%%)" % (threading, mode, threshold))
        print(table_str)

    def _calculate_temporal_drift(self, df: pd.DataFrame, mode: str, threshold: float):
        comp = df.copy()
        # comp is the prepared compression dataset
        comp = comp[comp["mode"] == mode]
        comp = comp[comp["threading"] == "single"]

        comp = comp.pint.dequantify()
        comp["log_throughput"] = np.log(comp["throughput"])

        # A cell identifies one experimental configuration
        cell_columns = ["host", "tool", "dataset", "strength"]
        comp["cell"] = comp[cell_columns].astype(str).agg("|".join, axis=1)

        # Remove each configuration's average throughput
        comp["centered_log_throughput"] = (
                comp["log_throughput"]
                - comp.groupby("cell")["log_throughput"].transform("mean")
        )

        # Retain only the variable-name level
        comp.columns = comp.columns.get_level_values(0)
        comp.columns.name = None

        drift_results = []
        for host, host_data in comp.groupby("host", observed=True):
            host_data = host_data.copy()
            host_data["run_centered"] = (
                    host_data["run"] - host_data["run"].mean()
            )

            model = smf.ols(
                "log_throughput ~ C(cell) + run_centered",
                data=host_data
            ).fit(
                cov_type="cluster",
                cov_kwds={"groups": host_data["cell"]}
            )

            beta = model.params["run_centered"]
            lower, upper = model.conf_int().loc["run_centered"]

            drift_results.append({
                "host": host,
                "change_per_run_percent": 100 * (np.exp(beta) - 1),
                "ci_low_per_run_percent": 100 * (np.exp(lower) - 1),
                "ci_high_per_run_percent": 100 * (np.exp(upper) - 1),
                "change_run1_to_run30_percent": 100 * (np.exp(29 * beta) - 1),
                "ci_low_run1_to_run30_percent": 100 * (np.exp(29 * lower) - 1),
                "ci_high_run1_to_run30_percent": 100 * (np.exp(29 * upper) - 1),
                "p_value": model.pvalues["run_centered"]
            })

        drift_results = pd.DataFrame(drift_results)
        drift_results["drift_conclusion"] = drift_results.apply(self._classify_drift, axis=1, threshold=threshold)
        return drift_results

    def _classify_drift(self, row, threshold: float):
        low = row["ci_low_run1_to_run30_percent"]
        high = row["ci_high_run1_to_run30_percent"]

        if low >= -threshold and high <= threshold:
            return "no material drift"
        elif low > threshold:
            return "material increase"
        elif high < -threshold:
            return "material decrease"
        return "inconclusive"

    def _calculate_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        stats_df = pd.concat(
            [
                df.groupby(GROUP_COLS, as_index=False)
                .agg(
                    num_runs=("run", "size"),
                    throughput=("throughput", stat),
                )
                .assign(stat=name)
                for name, stat in [
                ("min", "min"),
                ("max", "max"),
                ("mean", "mean"),
                ("stdev", "std"),
            ]
            ],
            ignore_index=True,
                )

        stats_df = stats_df[
            ["stat"] + GROUP_COLS + ["num_runs"] + self.VALUE_COLS
            ]

        stats_df["stat"] = pd.Categorical(
            stats_df["stat"],
            categories=["min", "max", "mean", "stdev"],
            ordered=True,
        )
        stats_df = stats_df.sort_values(
            GROUP_COLS + ["stat"],
            ignore_index=True,
            )

        return stats_df

    def _print_table(self, df):
        table_df = df.copy()
        table_df["throughput"] = table_df["throughput"].pint.to("MiB/s")
        table_entries = self._create_table_entries(table_df)
        unit_tp = str(table_df["throughput"].dtype.units)
        headers = ["stat", "host", "tool", "dataset", "mode", "strength", "threading", "num runs",
                   "throughput (%s)" % unit_tp]
        table_str = tabulate.tabulate(table_entries,
                                      headers=headers,
                                      tablefmt="simple"
                                      )
        print(table_str)

    def _create_table_entries(self, df):
        table_df = df.copy()
        for col in self.VALUE_COLS:
            table_df[col] = table_df[col].astype(float)
        table_entries = []
        groups = list(table_df.groupby(GROUP_COLS, sort=False))
        for i, (_, block) in enumerate(groups):
            for _, row in block.iterrows():
                entry = list(row.values[:1 + len(GROUP_COLS) + 1 + len(self.VALUE_COLS)])
                table_entries.append(entry)

            if i < len(groups) -1:
                table_entries.append(SEPARATING_LINE)
        return table_entries
