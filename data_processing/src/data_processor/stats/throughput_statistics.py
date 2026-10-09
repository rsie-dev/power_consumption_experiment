import logging
from pathlib import Path
from itertools import product

import tabulate
from tabulate import SEPARATING_LINE
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.regression.linear_model import RegressionResultsWrapper
from statsmodels.regression.linear_model import OLSResults
#from patsy import build_design_matrices
from patsy.build import build_design_matrices

from data_processor.constants import GROUP_COLS
from data_processor.processor import Processor
from data_processor.constants import ORDER_STRENGTH


class ThroughputStatistics(Processor):
    VALUE_COLS = ["throughput"]
    DRIFT_THRESHOLD = 1.0  # percentage points across runs 1–30

    def __init__(self, resources: Path):
        super().__init__(resources)
        self._logger = logging.getLogger(self.__class__.__name__)

    def process(self, tp_file: Path):
        df = self._frameio.load(tp_file)

        #self._validate_temporal_drift(df)

        stats_df = self._calculate_statistics(df)
        return

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
        # comp is the prepared compression dataset
        comp = self._extract_comp_dataframe(df, mode)

        # A cell identifies one experimental configuration
        cell_columns = ["host", "tool", "dataset", "strength"]
        comp["cell"] = comp[cell_columns].astype(str).agg("|".join, axis=1)

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

    def _extract_comp_dataframe(self, df: pd.DataFrame, mode: str) -> pd.DataFrame:
        comp = df.copy()
        # comp is the prepared compression dataset
        comp = comp[comp["mode"] == mode]
        comp = comp[comp["threading"] == "single"]

        comp = comp.pint.dequantify()
        comp["log_throughput"] = np.log(comp["throughput"])
        return comp

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
        comp = self._extract_comp_dataframe(df, "compress")

        # Flatten the DataFrame
        comp.columns = comp.columns.get_level_values(0)
        comp.columns.name = None

        formula = """
        log_throughput ~ (
            C(host, Sum)
            + C(tool, Sum)
            + C(dataset, Sum)
            + C(strength, Sum)
        )**2
        """

        compression_model = smf.ols(
            formula=formula,
            data=comp
        ).fit()
        # The fit has the expected dimensions.
        print("Observations:", int(compression_model.nobs))
        print("Model degrees of freedom:", int(compression_model.df_model))
        print("Residual degrees of freedom:", int(compression_model.df_resid))
        # This means the model explains about R^2 * 100% of observed log-throughput variation.
        print("R-squared:", compression_model.rsquared)

        influence = compression_model.get_influence()
        cooks_d = influence.cooks_distance[0]
        threshold = 4 / compression_model.nobs

        print("Maximum Cook's distance:", cooks_d.max())
        print("Screening threshold:", threshold)
        print("Observations above threshold:", (cooks_d > threshold).sum())

        diagnostics = compression_model.model.data.frame[
            ["host", "tool", "dataset", "strength", "run"]
        ].copy()
        diagnostics["cooks_d"] = cooks_d
        print(
            diagnostics.nlargest(10, "cooks_d")
            .to_string(index=False)
        )

        full_formula = """
        log_throughput ~
        C(host, Sum) *
        C(tool, Sum) *
        C(dataset, Sum) *
        C(strength, Sum)
        """
        full_model = smf.ols(full_formula, data=comp).fit()
        print(anova_lm(compression_model, full_model))
        print("R² increase:", full_model.rsquared - compression_model.rsquared)

        three_way_formula = """
        log_throughput ~ (
            C(host, Sum)
            + C(tool, Sum)
            + C(dataset, Sum)
            + C(strength, Sum)
        )**3
        """

        # Determine whether three-way interactions are sufficient
        # or whether the four-way interaction is also needed.
        three_way_model = smf.ols(
            three_way_formula,
            data=comp
        ).fit()
        print(anova_lm(compression_model, three_way_model, full_model))
        print("Three-way R² gain:",
              three_way_model.rsquared - compression_model.rsquared)
        print("Four-way R² gain:",
              full_model.rsquared - three_way_model.rsquared)

        # Make later statistical inference robust to unequal residual variance.
        # -> use HC3 version for confidence intervals and significance tests.
        full_model_hc3 = full_model.get_robustcov_results(
            cov_type="HC3"
        )

        # Determine whether extreme residuals are isolated or concentrated in particular configurations.
        diagnostics = full_model.model.data.frame[
            ["host", "tool", "dataset", "strength", "run"]
        ].copy()
        diagnostics["studentized_residual"] = (
            full_model.get_influence().resid_studentized_internal
        )
        extreme = diagnostics[
            diagnostics["studentized_residual"].abs() > 3
            ]
        print("Extreme observations:", len(extreme))
        extremes = extreme.groupby(
            ["host", "tool", "dataset", "strength"]
        ).size().sort_values(ascending=False).head(10)
        print(extremes.to_string(dtype=False))

        # Test which factors and interactions affect mean compression throughput
        anova_hc3 = anova_lm(
            full_model,
            typ=3,
            robust="hc3"
        )
        print("ANOVA HC3:")
        print(anova_hc3)

        # Rank factors and interactions by explained variation.
        anova_ss = anova_lm(full_model, typ=3)
        mse = (
                anova_ss.loc["Residual", "sum_sq"]
                / anova_ss.loc["Residual", "df"]
        )
        y = full_model.model.endog
        ss_total = np.sum((y - y.mean()) ** 2)
        effects = anova_ss.drop(
            index=["Intercept", "Residual"]
        ).copy()
        effects["omega_squared"] = (
                                   effects["sum_sq"] - effects["df"] * mse
                                   ) / (ss_total + mse)
        effects["omega_squared"] = effects["omega_squared"].clip(lower=0)
        ranked = effects[["df", "omega_squared"]].sort_values("omega_squared", ascending=False)
        ranked["omega_squared_percent"] = ranked["omega_squared"] * 100
        headers = ["Factor", "DF", "omega squared", "omega squared %"]
        table_entries = []
        for idx, row in ranked.iterrows():
            values = list(row.values[:])
            values.insert(0, str(idx))
            table_entries.append(values)
        table_str = tabulate.tabulate(table_entries,
                                      headers=headers,
                                      tablefmt="simple",
                                      floatfmt=".2f",
                                      )
        print("ANOVA rank table:")
        print(table_str)

        # Check that the confidence intervals do not overlap
        #ci_values = self._calculate_ci(comp, full_formula, effects)
        #self._print_ci(ci_values)

        self._show_factor_impacts(comp)
        factors = ["host", "tool", "dataset", "strength"]
        mean_cis_dataset = self._calculate_factor_ci(comp, full_model, full_model_hc3, "dataset", factors)
        self._print_mean_cis("dataset", mean_cis_dataset)

    def _show_factor_impacts(self, comp: pd.DataFrame):
        print("-" * 20 + " Factor means " + "-" * 20)
        means_dataset = self._calculate_mean(comp, "dataset")
        self._print_means("dataset", means_dataset)
        means_tool = self._calculate_mean(comp, "tool")
        self._print_means("tool", means_tool)
        means_host = self._calculate_mean(comp, "host")
        self._print_means("host", means_host)
        means_tool_strength = self._calculate_mean_combined(comp, ["tool", "strength"], ORDER_STRENGTH)
        self._print_means("tool * strength", means_tool_strength, ORDER_STRENGTH)
        means_strength = self._calculate_mean(comp, "strength", ORDER_STRENGTH)
        self._print_means("strength", means_strength)

    def _print_ci(self, sorted_ci_values: pd.DataFrame):
        headers = ["Factor", "omega squared", "CI low", "CI high"]
        table_entries = []
        for idx, row in sorted_ci_values.iterrows():
            values = list(row.values[:])
            values.insert(0, str(idx))
            table_entries.append(values)
        table_str = tabulate.tabulate(table_entries,
                                      headers=headers,
                                      tablefmt="simple",
                                      #floatfmt=".2f",
                                      )
        print("95%% confidence interval table")
        print(table_str)

    def _calculate_ci(self, comp: pd.DataFrame, full_formula: str, effects: pd.DataFrame) -> pd.DataFrame:
        rng = np.random.default_rng(12345)
        n_boot = 1000

        print("Calculating confidence intervals...")

        boot_data = comp.reset_index(drop=True)
        factors = ["host", "tool", "dataset", "strength"]

        groups = list(
            boot_data.groupby(factors, observed=True).indices.values()
        )

        boot_results = []

        for _ in range(n_boot):
            sampled_positions = np.concatenate([
                rng.choice(group, size=len(group), replace=True)
                for group in groups
            ])

            sample = boot_data.iloc[sampled_positions]

            model = smf.ols(full_formula, data=sample).fit()
            table = anova_lm(model, typ=3)

            mse_boot = (
                    table.loc["Residual", "sum_sq"]
                    / table.loc["Residual", "df"]
            )

            y_boot = model.model.endog
            ss_total_boot = np.sum((y_boot - y_boot.mean()) ** 2)

            omega = (
                            table["sum_sq"] - table["df"] * mse_boot
                    ) / (ss_total_boot + mse_boot)

            boot_results.append(
                omega.drop(["Intercept", "Residual"]).clip(lower=0)
            )

        bootstrap_omega = pd.DataFrame(boot_results)

        confidence_intervals = bootstrap_omega.quantile(
            [0.025, 0.975]
        ).T

        confidence_intervals.columns = ["ci_low", "ci_high"]

        results_with_ci = effects[["omega_squared"]].join(
            confidence_intervals
        )

        sorted_values = results_with_ci.sort_values("omega_squared", ascending=False)
        return sorted_values

    def _print_means(self, factor: str, means: pd.DataFrame, extra_columns: list | None = None):
        headers = [factor]
        if extra_columns:
            headers += extra_columns
        else:
            headers += ["Mean throughput MiB/s"]
        table_entries = []
        for idx, row in means.iterrows():
            values = list(row.values[:])
            values.insert(0, str(idx))
            table_entries.append(values)
        table_str = tabulate.tabulate(table_entries,
                                      headers=headers,
                                      tablefmt="simple",
                                      #floatfmt=".2f",
                                      )
        print("Throughput means for: %s" % factor)
        print(table_str)

    def _calculate_mean(self, comp: pd.DataFrame, factor: str, order: list | None = None):
        means = (
            comp.groupby(factor, observed=True)["log_throughput"]
            .mean()
            .pipe(np.exp)
            .div(2 ** 20)
            .sort_values(ascending=False)
        )
        if order:
            means = means.reindex(order)
        means = means.rename("geometric_mean_MiB_s").to_frame()
        return means

    def _calculate_mean_combined(self, comp: pd.DataFrame, factor: str | list[str], order: list | None = None):
        means = (
            comp.groupby(
                factor,
                observed=True
            )["log_throughput"]
            .mean()
            .pipe(np.exp)
            .div(2 ** 20)
        )
        means = means.unstack(factor[-1])

        if order:
            means = means[order]
        return means

    def _calculate_factor_ci(self, comp: pd.DataFrame, full_model: RegressionResultsWrapper, full_model_hc3: OLSResults,
                             factor: str, factors: list[str]):
        levels = {
            factor: comp[factor].unique().tolist()
            for factor in factors
        }

        grid = pd.DataFrame(
            product(*(levels[f] for f in factors)),
            columns=factors
        )

        X_grid = np.asarray(
            build_design_matrices(
                [full_model.model.data.design_info],
                grid
            )[0]
        )

        beta = np.asarray(full_model.params)
        cov = np.asarray(full_model_hc3.cov_params())

        def marginal_mean_ci(factor):
            rows = []
            for level in levels[factor]:
                L = X_grid[grid[factor] == level].mean(axis=0)

                estimate = L @ beta
                se = np.sqrt(L @ cov @ L)

                rows.append({
                    factor: level,
                    "mean_MiB_s": np.exp(estimate) / 2 ** 20,
                    "ci_low": np.exp(estimate - 1.96 * se) / 2 ** 20,
                    "ci_high": np.exp(estimate + 1.96 * se) / 2 ** 20
                })

            return pd.DataFrame(rows)

        mean_cis = marginal_mean_ci(factor)

        mean_cis = mean_cis.sort_values(
            by=["mean_MiB_s"],
            ascending=False,
        )
        return mean_cis

    def _print_mean_cis(self, factor: str, mean_cis: pd.DataFrame):
        headers = [factor, "Mean MiB/s", "CI low", "CI high"]
        table_entries = []
        for _, row in mean_cis.iterrows():
            values = row.values[:]
            table_entries.append(values)
        table_str = tabulate.tabulate(table_entries,
                                      headers=headers,
                                      tablefmt="simple",
                                      #floatfmt=".2f",
                                      )
        print("95%% confidence interval of means for: %s" % factor)
        print(table_str)

    def _calculate_statistics_(self, df: pd.DataFrame) -> pd.DataFrame:
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
