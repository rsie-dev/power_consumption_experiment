import logging

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import tabulate


class DriftAnalyzer:
    def __init__(self, factors: list[str], drift_threshold: float, threading: str, mode: str):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._factors = factors
        self._drift_threshold = drift_threshold
        self._threading = threading
        self._mode = mode

    @property
    def factors(self):
        return self._factors

    def validate_temporal_drift(self, comp: pd.DataFrame):
        self._check_temporal_drift(comp, self._threading, self._mode, self._drift_threshold)

    def _check_temporal_drift(self, comp: pd.DataFrame, threading: str, mode: str, threshold: float):
        drift_results = self._calculate_temporal_drift(comp, threshold)
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
            "ci_high_run1_to_run30_percent",
            "drift_conclusion",
        ]

        headers = ["host", "change %", "CI low %", "CI high %", "drift conclusion"]
        table_df = drift_results[columns]
        table_str = tabulate.tabulate(table_df,
                                      showindex=False,
                                      headers=headers,
                                      floatfmt=".3f",
                                      )
        self._logger.info("Drift analysis for %s %s (threshold: %s%%)", threading, mode, threshold)
        self._logger.info(table_str)

    def _calculate_temporal_drift(self, comp: pd.DataFrame, threshold: float) -> pd.DataFrame:
        comp = comp.copy()

        # A cell identifies one experimental configuration
        comp["cell"] = comp[self.factors].astype(str).agg("|".join, axis=1)

        # Remove each configuration's average throughput
        comp["centered_log_throughput"] = (
                comp["log_throughput"]
                - comp.groupby("cell")["log_throughput"].transform("mean")
        )

        # Flatten the DataFrame
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
        if low > threshold:
            return "material increase"
        if high < -threshold:
            return "material decrease"
        return "inconclusive"
