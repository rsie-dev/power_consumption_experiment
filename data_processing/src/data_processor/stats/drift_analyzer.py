import logging

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from patsy.build import build_design_matrices
from scipy.stats import t
import tabulate


class DriftAnalyzer:
    def __init__(self, factors: list[str], drift_threshold: float, threading: str, mode: str, base_column: str):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._factors = factors
        self._drift_threshold = drift_threshold
        self._threading = threading
        self._mode = mode
        self._base_column = base_column

    @property
    def factors(self):
        return self._factors

    def validate_temporal_drift(self, comp: pd.DataFrame) -> list[str]:
        drift_results = self._calculate_temporal_drift(comp, self._drift_threshold)
        self._show_drift_analysis(self._threading, self._mode, self._drift_threshold, drift_results)
        self._verify_temporal_drift(drift_results)

        hosts = (
            drift_results.loc[drift_results["drift_conclusion"] == "inconclusive", "host"]
            .drop_duplicates()
            .tolist()
        )
        return hosts

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

        table_df = drift_results[columns]
        headers = ["Host", "Change (%)", "CI low %", "CI high (%)", "Drift conclusion"]
        table_str = tabulate.tabulate(
            table_df,
            showindex=False,
            headers=headers,
            floatfmt=".3f",
        )
        ci_method = "cell fixed effects; 95% cell-clustered CRV1 Wald confidence intervals"
        self._logger.info("Run-order screening model (%s):", ci_method)
        self._logger.info("Drift analysis for %s %s (materiality threshold: %s%%)", threading, mode, threshold)
        self._logger.info(table_str)

    def _calculate_temporal_drift(self, comp: pd.DataFrame, threshold: float) -> pd.DataFrame:
        comp = comp.copy()

        # A cell identifies one experimental configuration
        comp["cell"] = comp[self.factors].astype(str).agg("|".join, axis=1)

        # Remove each configuration's average throughput
        log_column = "log_%s" % self._base_column
        comp["centered_%s" % log_column] = (
                comp[log_column]
                - comp.groupby("cell")[log_column].transform("mean")
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
                "%s ~ C(cell) + run_centered" % log_column,
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
        return self._classify_drift_values(low, high, threshold)

    def _classify_drift_values(self, low: float, high: float, threshold: float):
        if low >= -threshold and high <= threshold:
            return "no material drift"
        if low > threshold:
            return "material increase"
        if high < -threshold:
            return "material decrease"
        return "inconclusive"

    def calculate_run_order_effect(self, host: str, df: pd.DataFrame):
        data = df.copy()

        # Ensure numeric run order.
        data["run"] = data["run"].astype(int)

        run_first = data["run"].min()
        run_last = data["run"].max()
        run_midpoint = (run_first + run_last) / 2

        data["run_centered"] = data["run"] - run_midpoint

        sensitivity_formula = f"""
        log_{self._base_column} ~
            C(host, Sum) *
            C(tool, Sum) *
            C(dataset, Sum) *
            C(strength, Sum)
            + run_centered
            + C(host, Sum):run_centered
        """
        sensitivity_model = smf.ols(
            sensitivity_formula,
            data=data,
        ).fit()

        sensitivity_hc3 = sensitivity_model.get_robustcov_results(
            cov_type="HC3",
            use_t=True,
        )

        # All utility × dataset × strength configurations.
        configurations = (
            data[["tool", "dataset", "strength"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )

        first_run_data = configurations.assign(
            host=host,
            run_centered=run_first - run_midpoint,
        )

        last_run_data = configurations.assign(
            host=host,
            run_centered=run_last - run_midpoint,
        )

        design_info = sensitivity_model.model.data.design_info

        x_first = np.asarray(
            build_design_matrices([design_info], first_run_data)[0]
        )

        x_last = np.asarray(
            build_design_matrices([design_info], last_run_data)[0]
        )

        # Adjusted run-first to run-last contrast.
        contrast = x_last.mean(axis=0) - x_first.mean(axis=0)

        log_change = float(contrast @ sensitivity_hc3.params)

        log_change_se = np.sqrt(
            float(
                contrast
                @ sensitivity_hc3.cov_params()
                @ contrast
            )
        )

        critical_value = t.ppf(
            0.975,
            sensitivity_hc3.df_resid,
        )

        log_ci_low = log_change - critical_value * log_change_se
        log_ci_high = log_change + critical_value * log_change_se

        change_pct = 100 * np.expm1(log_change)
        ci_low_pct = 100 * np.expm1(log_ci_low)
        ci_high_pct = 100 * np.expm1(log_ci_high)

        conclusion = self._classify_drift_values(ci_low_pct, ci_high_pct, self._drift_threshold)

        self._logger.info("")
        self._logger.info("Full-factorial run-order sensitivity model (HC3 confidence interval):")
        self._logger.info("%s run %d-%d change: %.3f%%", host, run_first, run_last, change_pct)
        self._logger.info("(95%% HC3 CI: %.3f%% to %.3f%%): %s", ci_low_pct, ci_high_pct, conclusion)

        return sensitivity_model
