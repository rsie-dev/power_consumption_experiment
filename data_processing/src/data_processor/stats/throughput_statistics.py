import logging
from pathlib import Path
from itertools import product, combinations
import re

import tabulate
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.regression.linear_model import RegressionResultsWrapper
from statsmodels.regression.linear_model import OLSResults
from patsy.build import build_design_matrices
from tqdm import tqdm

from data_processor.processor import Processor
from data_processor.constants import get_factor_levels


class ThroughputStatistics(Processor):
    DRIFT_THRESHOLD = 1.0  # percentage points across runs 1–30

    def __init__(self, resources: Path, n_boot: int, seed: int | None):
        super().__init__(resources)
        self._logger = logging.getLogger(self.__class__.__name__)
        self._n_boot = n_boot
        if seed is None:
            seed = 12345
        self._rng = np.random.default_rng(seed)

    def process(self, tp_file: Path):
        df = self._frameio.load(tp_file)
        factors_single = ["host", "tool", "dataset", "strength"]
        #self._process_threading(df, "single", factors_single)
        self._process_threading(df, "multi", factors_single)

    def _process_threading(self, df: pd.DataFrame, threading: str, factors: list[str]):
        #self._process_threading_mode(df, threading, "compress", factors)
        self._process_threading_mode(df, threading, "decompress", factors)

    def _process_threading_mode(self, df: pd.DataFrame, threading: str, mode: str, factors: list[str]):
        print("")
        print("-" * 20 + " Process %s" % mode + "-" * 20)
        print("")
        self._validate_data_quantity(df, threading, mode, factors)
        self._validate_temporal_drift(df, threading, mode, factors)
        self._calculate_statistics(df, threading, mode, factors)

    def _validate_temporal_drift(self, df: pd.DataFrame, threading: str, mode: str, factors: list[str]):
        print("")
        print("-" * 20 + " Validate temporal drift" + "-" * 20)
        print("")
        self._check_temporal_drift(df, threading, mode, self.DRIFT_THRESHOLD, factors)

    def _check_temporal_drift(self, df: pd.DataFrame, threading: str, mode: str, threshold: float, factors: list[str]):
        drift_results = self._calculate_temporal_drift(df, threading, mode, threshold, factors)
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

    def _calculate_temporal_drift(self, df: pd.DataFrame, threading: str, mode: str, threshold: float, factors: list[str]):
        # comp is the prepared compression dataset
        comp = self._extract_comp_dataframe(df, mode, threading)

        # A cell identifies one experimental configuration
        comp["cell"] = comp[factors].astype(str).agg("|".join, axis=1)

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

    def _extract_comp_dataframe(self, df: pd.DataFrame, mode: str, threading: str) -> pd.DataFrame:
        extracted = df.copy()
        extracted = extracted[extracted["mode"] == mode]
        extracted = extracted[extracted["threading"] == threading]

        extracted = extracted.pint.dequantify()
        extracted["log_throughput"] = np.log(extracted["throughput"])
        return extracted

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

    def _calculate_statistics(self, df: pd.DataFrame, threading: str, mode: str, factors: list[str]) -> pd.DataFrame:
        print("")
        print("-" * 20 + " Calculate statistics " + "-" * 20)
        print("")

        comp = self._extract_comp_dataframe(df, mode, threading)

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
        print("Two-way model:")
        print("Observations:                 %d" % int(compression_model.nobs))
        print("Model degrees of freedom:     %d" % int(compression_model.df_model))
        print("Residual degrees of freedom:  %d" % int(compression_model.df_resid))
        # This means the model explains about R^2 * 100% of observed log-throughput variation.
        print("R-squared:                    %f" % compression_model.rsquared)

        full_formula = """
            log_throughput ~
            C(host, Sum) *
            C(tool, Sum) *
            C(dataset, Sum) *
            C(strength, Sum)
        """
        full_model = smf.ols(full_formula, data=comp).fit()
        print(anova_lm(compression_model, full_model))
        print("Full model:")
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
        print("Three-way R² gain:            %f" % (three_way_model.rsquared - compression_model.rsquared))
        print("Four-way R² gain:             %f" % (full_model.rsquared - three_way_model.rsquared))

        # Now check:
        # 1. residual variance across fitted values -> UI
        # 2. Check residual normality and extreme tails -> UI

        self._print_residual_variance(comp, full_model, factors)

        diagnostics = self._find_observations_that_strongly_influence_model(full_model, factors)
        self._print_observations_that_strongly_influence_model(diagnostics, factors)

        # Determine whether extreme residuals are isolated or concentrated in particular configurations.
        extremes = self._get_diagnostics(full_model, factors)
        self._print_diagnostics(extremes, factors)

        # Make later statistical inference robust to unequal residual variance.
        # Reason: HC3 protects significance tests and confidence intervals against unequal configuration
        # variability and keeps the method consistent with compression.
        # -> use HC3 version for confidence intervals and significance tests.
        full_model_hc3 = full_model.get_robustcov_results(
            cov_type="HC3"
        )

        anova_hc3 = self._calculate_anova_hc3(full_model)
        self._print_anova_hc3(anova_hc3)

        effects, ranked = self._calculate_anova_table(full_model)
        self._print_anova_table(ranked)
        relevant_factors = self._extract_relevant_factors(ranked, 95)
        print("")
        print("Relevant factors: %s" % " > ".join([str(f) for f in relevant_factors]))

        #self._show_factor_means(comp, relevant_factors)
        self._show_factor_means_ci(comp, full_model, full_model_hc3, relevant_factors, factors)

        # Check that the top confidence intervals do not overlap
        ci_values, confidence_intervals = self._calculate_ci(comp, full_formula, effects, factors)
        self._print_ci(ci_values)

        print("")
        effect_ranking = self._create_effect_ranking_table(ranked, confidence_intervals, anova_hc3, factors)
        self._print_effect_ranking_table(threading, mode, effect_ranking)

        throughput_values_df = self._create_throughput_table(comp, full_model, full_model_hc3, factors)
        self._print_throughput_table(threading, mode, throughput_values_df)

    def _print_residual_variance(self, comp: pd.DataFrame, full_model: RegressionResultsWrapper, factors: list[str]):
        diagnostics_comp_mt = comp.copy()
        diagnostics_comp_mt["residual"] = full_model.resid

        residual_sd_comp_mt = diagnostics_comp_mt.groupby(
            factors
        )["residual"].std()

        print(residual_sd_comp_mt.describe())
        sd_ratio = residual_sd_comp_mt.max() / residual_sd_comp_mt.min()
        print("Largest / smallest residual SD ratio: %f" % sd_ratio)

    def _extract_relevant_factors(self, ranked: pd.DataFrame, target: int) -> list:
        relevant = []

        first_interaction = None
        for idx, _ in ranked.iterrows():
            factors = self._extract_factors(str(idx))
            if isinstance(factors, tuple):
                first_interaction = factors
                break

        n = (ranked["omega_squared_percent"].cumsum() < target).sum() + 1
        relevant_rows = ranked.iloc[:n]
        for idx, _ in relevant_rows.iterrows():
            factors = self._extract_factors(str(idx))
            relevant.append(factors)

        if first_interaction not in relevant:
            relevant.append(first_interaction)

        return relevant

    def _extract_factors(self, entry: str) -> str | tuple:
        regex = r"C\(\s*([^,]+)\s*,"
        factors = []
        for part in entry.split(":"):
            search_result = re.search(regex, part)
            factor = search_result.group(1).strip()
            factors.append(factor)
        if len(factors) == 1:
            return factors[0]
        return tuple(factors)

    def _print_anova_table(self, ranked: pd.DataFrame):
        headers = ["Factor", "DF", "ω²", "ω² %"]
        table_entries = []
        for idx, row in ranked.iterrows():
            values = list(row.values[:])
            values[0] = int(values[0])
            values.insert(0, str(idx))
            table_entries.append(values)
        table_str = tabulate.tabulate(table_entries,
                                      headers=headers,
                                      tablefmt="simple",
                                      )
        print("")
        print("ANOVA rank table:")
        print(table_str)

    def _calculate_anova_table(self, full_model: RegressionResultsWrapper):
        # Rank factors and interactions by explained variation by omega^2.
        # Reason: RQ asks which effects explain the most variation, which \(p\)-values cannot determine.
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
        return effects, ranked

    def _print_anova_hc3(self, anova_hc3_df: pd.DataFrame):
        headers = ["factor", "sum_sq", "df", "F", "PR(>F)"]
        table_entries = []
        for idx, row in anova_hc3_df.iterrows():
            values = list(row.values[:])
            values.insert(0, str(idx))
            table_entries.append(values)
        table_str = tabulate.tabulate(table_entries,
                                      headers=headers,
                                      tablefmt="simple",
                                      #floatfmt=".2f",
                                      )
        print("")
        print("ANOVA HC3:")
        print(table_str)

    def _calculate_anova_hc3(self, full_model: RegressionResultsWrapper):
        # Create robust inference for decompression effects.
        # Reason: HC3 protects significance tests and confidence intervals against unequal
        # configuration variability and keeps the method consistent with compression.
        # Test which factors and interactions affect mean compression throughput
        anova_hc3 = anova_lm(
            full_model,
            typ=3,
            robust="hc3"
        )
        return anova_hc3

    def _print_observations_that_strongly_influence_model(self, diagnostics: pd.DataFrame, factors: list[str]):
        headers = factors + ["run", "cooks_d"]
        table_entries = []
        for _, row in diagnostics.iterrows():
            values = row.values[:]
            table_entries.append(values)
        table_str = tabulate.tabulate(table_entries,
                                      headers=headers,
                                      tablefmt="simple",
                                      #floatfmt=".2f",
                                      )
        print("Influence table")
        print(table_str)

    def _find_observations_that_strongly_influence_model(self, compression_model, factors: list[str]):
        # Screen for observations that strongly influence the decompression model.
        # Reason: A small number of influential trials could distort effect estimates.
        influence = compression_model.get_influence()
        cooks_d = influence.cooks_distance[0]
        threshold = 4 / compression_model.nobs

        print("")
        print("Cook's threshold:             %f" % threshold)
        print("Maximum Cook's distance:      %f" % cooks_d.max())
        print("Observations above threshold: %d" % (cooks_d > threshold).sum())

        run_factors = factors + ["run"]
        diagnostics = compression_model.model.data.frame[
            run_factors
        ].copy()
        diagnostics["cooks_d"] = cooks_d

        diagnostics = diagnostics.sort_values(
            by=["cooks_d"],
            ascending=False,
        )
        diagnostics = diagnostics.head(10)
        return diagnostics

    def _validate_data_quantity(self, df: pd.DataFrame, threading: str, mode: str, factors: list[str]):
        comp = self._extract_comp_dataframe(df, mode, threading)
        comp = comp.pint.quantify()
        cell_counts = comp.groupby(
            factors,
            observed=True
        ).size()

        print("Observations:                 %d" % len(comp))
        print("Configurations:               %d" % len(cell_counts))
        print("Min trials per configuration: %d" % cell_counts.min())
        print("Max trials per configuration: %d" % cell_counts.max())
        missing = comp["log_throughput"].isna().sum()
        print("Missing response values:      %d" % missing)
        if missing:
            raise ValueError("Missing response values: %d" % missing)
        print("Tools:                        %s" % ", ".join(comp["tool"].unique()))

    def _print_effect_ranking_table(self, threading: str, mode: str, table_df: pd.DataFrame):
        headers = ["Rank", "Effect", "Effect type", "df", "ω²", "Cumulative ω² (%)", "95% bootstrap CI", "HC3 F", "HC3 p"]
        table_entries = []
        for _, row in table_df.iterrows():
            values = row.values[:]
            table_entries.append(values)
        coalign = ["global"] * len(headers)
        coalign[2] = "right"
        table_str = tabulate.tabulate(table_entries,
                                      headers=headers,
                                      tablefmt="simple",
                                      #floatfmt=".2f",
                                      colalign=coalign
                                      )
        print("Effect ranking table for: %s %s" % (threading, mode))
        print(table_str)

    def _create_effect_ranking_table(self, effects: pd.DataFrame, confidence_intervals: pd.Series,
                                     anova_hc3: pd.DataFrame, factors: list[str]):
        effect_ranking = (
            effects[["df", "omega_squared"]]
            .join(confidence_intervals)
            .join(anova_hc3[["F", "PR(>F)"]])
        )

        def clean_effect(term):
            for factor in factors: #["host", "tool", "dataset", "strength"]:
                term = term.replace(f"C({factor}, Sum)", factor)
            return term.replace(":", " × ")

        type_names = {
            1: "Main effect",
            2: "Two-way interaction",
            3: "Three-way interaction",
            4: "Four-way interaction"
        }

        effect_ranking["Effect"] = [
            clean_effect(term) for term in effect_ranking.index
        ]

        effect_ranking["Effect type"] = [
            type_names[term.count(":") + 1]
            for term in effect_ranking.index
        ]

        effect_ranking["Rank"] = (
            effect_ranking["omega_squared"]
            .rank(ascending=False, method="min")
            .astype(int)
        )

        effect_ranking["95% bootstrap CI"] = (
                effect_ranking["ci_low"].map(lambda x: f"{x:.6f}")
                + "–"
                + effect_ranking["ci_high"].map(lambda x: f"{x:.6f}")
        )

        effect_ranking["HC3 p"] = effect_ranking["PR(>F)"].map(
            lambda p: "<0.001" if p < 0.001 else f"{p:.3f}"
        )

        effect_ranking = effect_ranking.rename(columns={
            "F": "HC3 F"
        })

        effect_ranking = effect_ranking[
            [
                "Rank",
                "Effect",
                "Effect type",
                "df",
                "omega_squared",
                "95% bootstrap CI",
                "HC3 F",
                "HC3 p"
            ]
        ].sort_values("Rank")

        effect_ranking["cumulative_omega_squared_percent"] = (
                effect_ranking["omega_squared"].cumsum() * 100
        )
        effect_ranking.insert(5, "cumulative_omega_squared_percent", effect_ranking.pop("cumulative_omega_squared_percent"))

        return effect_ranking

    def _print_throughput_table(self, threading: str, mode: str, throughput_values_df: pd.DataFrame):
        headers = ["Factor", "Level", "Marginal geometric mean throughput (MiB/s)", "95% HC3 CI"]
        table_entries = []
        for _, row in throughput_values_df.iterrows():
            values = row.values[:]
            table_entries.append(values)
        table_str = tabulate.tabulate(table_entries,
                                      headers=headers,
                                      tablefmt="simple",
                                      #floatfmt=".2f",
                                      )
        print("Throughput values for: %s %s" % (threading, mode))
        print(table_str)

    def _create_throughput_table(self, comp: pd.DataFrame,
                                 full_model: RegressionResultsWrapper, full_model_hc3: OLSResults, factors: list[str]):
        levels, grid, x_grid = self._calculate_x_grid(comp, full_model, factors)
        beta, cov = self._get_beta_cov(full_model, full_model_hc3)

        tables = []
        for factor in factors:
            table = self._marginal_mean_ci(factor, levels, grid, x_grid, beta, cov).rename(columns={
                factor: "Level",
                "mean_MiB_s": "Mean throughput (MiB/s)"
            })

            table.insert(0, "Factor", factor)
            tables.append(table)

        throughput_values = pd.concat(
            tables,
            ignore_index=True
        )

        throughput_values["95% HC3 CI"] = (
                throughput_values["ci_low"].map(lambda x: f"{x:.3f}")
                + "–"
                + throughput_values["ci_high"].map(lambda x: f"{x:.3f}")
        )

        throughput_values = throughput_values[
            [
                "Factor",
                "Level",
                "Mean throughput (MiB/s)",
                "95% HC3 CI"
            ]
        ]

        throughput_values["Factor"] = pd.Categorical(
            throughput_values["Factor"],
            categories=factors,
            ordered=True
        )
        throughput_values = throughput_values.sort_values(
            ["Factor", "Mean throughput (MiB/s)"],
            ascending=[True, False]
        )

        return throughput_values

    def _print_diagnostics(self, extremes_df: pd.DataFrame, factors: list[str]):
        total = extremes_df["count"].sum()

        extremes_df = extremes_df.head(10)
        headers = factors + ["count"]
        table_entries = []
        for _, row in extremes_df.iterrows():
            values = row.values[:]
            table_entries.append(values)
        table_str = tabulate.tabulate(table_entries,
                                      headers=headers,
                                      tablefmt="simple",
                                      )
        print("")
        print("Extreme observations:         %d" % total)
        print(table_str)

    def _get_diagnostics(self, full_model: RegressionResultsWrapper, factors: list[str]) -> pd.DataFrame:
        # Determine whether extreme residuals are isolated or concentrated in particular configurations.
        run_factors = factors + ["run"]
        diagnostics = full_model.model.data.frame[run_factors].copy()
        diagnostics["studentized_residual"] = (
            full_model.get_influence().resid_studentized_internal
        )
        extreme = diagnostics[
            diagnostics["studentized_residual"].abs() > 3
            ]
        extremes = extreme.groupby(
            factors
        ).size().sort_values(ascending=False)
        extremes_df = extremes.reset_index(name="count")
        return extremes_df

    def _show_factor_means(self, comp: pd.DataFrame, relevant_factors: list):
        print("")
        print("-" * 20 + " Factor means " + "-" * 20)
        print("")
        for relevant_factor in relevant_factors:
            if isinstance(relevant_factor, str):
                means_df = self._calculate_mean(comp, relevant_factor)
                self._print_means(relevant_factor, means_df)
            else:
                factor2_levels = get_factor_levels(relevant_factor[1], comp)
                means_df = self._calculate_mean_combined(comp, relevant_factor, factor2_levels)
                self._print_means(" / ".join(relevant_factor), means_df, factor2_levels)

    def _show_factor_means_ci(self, comp: pd.DataFrame,
                              full_model: RegressionResultsWrapper, full_model_hc3: OLSResults,
                              relevant_factors: list,
                              factors: list[str]):
        print("")
        print("-" * 20 + " Factor means CIs " + "-" * 20)
        print("")
        for relevant_factor in relevant_factors:
            if isinstance(relevant_factor, str):
                mean_cis = self._calculate_factor_ci(comp, full_model, full_model_hc3, relevant_factor, factors)
                self._print_mean_cis(relevant_factor, mean_cis)

                factor_ratios = self._calculate_throughput_ratios_between_factor_levels(comp,
                                                                                        full_model, full_model_hc3,
                                                                                        relevant_factor, factors)
                self._print_throughput_ratios_between_factor_levels(relevant_factor, factor_ratios)
            else:
                factor2_levels = get_factor_levels(relevant_factor[1], comp)
                combination_ci = self._calculate_factor_combination_ci(comp, full_model, full_model_hc3,
                                                                       relevant_factor, factors, factor2_levels)
                self._print_mean_cis(relevant_factor, combination_ci)

                combination_ratios = self._calculate_throughput_ratios_within_factor_levels(comp,
                                                                                            full_model, full_model_hc3,
                                                                                            relevant_factor,
                                                                                            factors,
                                                                                            factor2_levels)
                self._print_throughput_ratios_within_factor_levels(relevant_factor, combination_ratios)

    def _print_ci(self, sorted_ci_values: pd.DataFrame):
        headers = ["Factor", "ω²", "CI low", "CI high"]
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
        print("")
        print("95%% confidence interval table")
        print(table_str)

    def _calculate_ci(self, comp: pd.DataFrame, full_formula: str, effects: pd.DataFrame, factors: list[str]) \
            -> tuple[pd.DataFrame, pd.Series]:
        # Calculate bootstrap confidence intervals for the decompression \(\omega^2\) values.
        # Reason: This confirms whether the top-three ranking is stable.


        boot_data = comp.reset_index(drop=True)

        groups = list(
            boot_data.groupby(factors, observed=True).indices.values()
        )

        boot_results = []
        print("Calculating confidence intervals...")
        for _ in tqdm(range(self._n_boot), colour="green"):
            sampled_positions = np.concatenate([
                self._rng.choice(group, size=len(group), replace=True)
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
        ).T # transpose, swap rows and columns

        confidence_intervals.columns = ["ci_low", "ci_high"]

        results_with_ci = effects[["omega_squared"]].join(
            confidence_intervals
        )

        sorted_values = results_with_ci.sort_values("omega_squared", ascending=False)
        return sorted_values, confidence_intervals

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
                list(factor),
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
        levels, grid, x_grid = self._calculate_x_grid(comp, full_model, factors)
        beta, cov = self._get_beta_cov(full_model, full_model_hc3)

        mean_cis = self._marginal_mean_ci(factor, levels, grid, x_grid, beta, cov)

        mean_cis = mean_cis.sort_values(
            by=["mean_MiB_s"],
            ascending=False,
        )
        return mean_cis

    def _marginal_mean_ci(self, factor: str, levels: dict, grid, x_grid: np.ndarray, beta: np.ndarray, cov: np.ndarray):
        rows = []
        for level in levels[factor]:
            l = x_grid[grid[factor] == level].mean(axis=0)

            estimate = l @ beta
            se = np.sqrt(l @ cov @ l)

            rows.append({
                factor: level,
                "mean_MiB_s": np.exp(estimate) / 2 ** 20,
                "ci_low": np.exp(estimate - 1.96 * se) / 2 ** 20,
                "ci_high": np.exp(estimate + 1.96 * se) / 2 ** 20
            })

        return pd.DataFrame(rows)

    def _calculate_factor_combination_ci(self, comp: pd.DataFrame,
                                         full_model: RegressionResultsWrapper, full_model_hc3: OLSResults,
                                         factor_combination: list[str], factors: list[str], factor2_levels: list[str]):
        levels, grid, x_grid = self._calculate_x_grid(comp, full_model, factors)
        beta, cov = self._get_beta_cov(full_model, full_model_hc3)

        rows = []
        factor1 = factor_combination[0]
        factor2 = factor_combination[1]
        for factor1_level in levels[factor1]:
            for factor2_level in factor2_levels:
                mask = (
                        (grid[factor1] == factor1_level)
                        & (grid[factor2] == factor2_level)
                )

                l = x_grid[mask].mean(axis=0)
                estimate = l @ beta
                se = np.sqrt(l @ cov @ l)

                rows.append({
                    "factor1": factor1_level,
                    "factor2": factor2_level,
                    "mean_MiB_s": np.exp(estimate) / 2 ** 20,
                    "ci_low": np.exp(estimate - 1.96 * se) / 2 ** 20,
                    "ci_high": np.exp(estimate + 1.96 * se) / 2 ** 20
                })

        combination_ci = pd.DataFrame(rows)
        combination_ci = combination_ci.sort_values(
            by=["factor1", "mean_MiB_s"],
            ascending=[True, False],
        )

        return combination_ci

    def _calculate_x_grid(self, comp: pd.DataFrame, full_model: RegressionResultsWrapper, factors: list[str]) \
            -> tuple[dict, pd.DataFrame, np.ndarray]:
        levels = {
            factor: comp[factor].unique().tolist()
            for factor in factors
        }

        grid = pd.DataFrame(
            product(*(levels[f] for f in factors)),
            columns=factors
        )

        x_grid = np.asarray(
            build_design_matrices(
                [full_model.model.data.design_info],
                grid
            )[0]
        )
        return levels, grid, x_grid

    def _get_beta_cov(self, full_model: RegressionResultsWrapper, full_model_hc3: OLSResults):
        beta = np.asarray(full_model.params)
        cov = np.asarray(full_model_hc3.cov_params())
        return beta, cov

    def _print_mean_cis(self, factor: str | list[str], mean_cis: pd.DataFrame):
        if isinstance(factor, str):
            factor = [factor]
        headers = list(factor) + ["Mean MiB/s", "CI low", "CI high"]
        table_entries = []
        for _, row in mean_cis.iterrows():
            values = row.values[:]
            table_entries.append(values)
        table_str = tabulate.tabulate(table_entries,
                                      headers=headers,
                                      tablefmt="simple",
                                      #floatfmt=".2f",
                                      )
        print("95%% confidence interval of means for: %s" % list(factor))
        print(table_str)

    def _calculate_throughput_ratios_between_factor_levels(self, comp: pd.DataFrame,
                                                           full_model: RegressionResultsWrapper,
                                                           full_model_hc3: OLSResults,
                                                           factor: str, factors: list[str]):
        levels, grid, x_grid = self._calculate_x_grid(comp, full_model, factors)
        beta, cov = self._get_beta_cov(full_model, full_model_hc3)

        rows = []
        for first, second in combinations(levels[factor], 2):
            l_first = x_grid[grid[factor] == first].mean(axis=0)
            l_second = x_grid[grid[factor] == second].mean(axis=0)

            contrast = l_first - l_second
            estimate = contrast @ beta
            se = np.sqrt(contrast @ cov @ contrast)

            rows.append({
                "comparison": f"{first} / {second}",
                "ratio": np.exp(estimate),
                "ci_low": np.exp(estimate - 1.96 * se),
                "ci_high": np.exp(estimate + 1.96 * se)
            })

        dataset_ratios = pd.DataFrame(rows)

        dataset_ratios = dataset_ratios.sort_values(
            by=["ratio"],
            ascending=False,
        )
        return dataset_ratios

    def _print_throughput_ratios_between_factor_levels(self, factor: str, factor_ratios: pd.DataFrame):
        headers = ["Comparison", "Ratio", "CI low", "CI high"]
        table_entries = []
        for _, row in factor_ratios.iterrows():
            values = row.values[:]
            table_entries.append(values)
        table_str = tabulate.tabulate(table_entries,
                                      headers=headers,
                                      tablefmt="simple",
                                      #floatfmt=".2f",
                                      )
        print("Throughput ratios between factor levels of: %s" % factor)
        print(table_str)

    def _calculate_throughput_ratios_within_factor_levels(self, comp: pd.DataFrame,
                                                          full_model: RegressionResultsWrapper,
                                                          full_model_hc3: OLSResults,
                                                          factor_combination: list[str], factors: list[str],
                                                          factor2_levels: list[str]):
        levels, grid, x_grid = self._calculate_x_grid(comp, full_model, factors)
        beta, cov = self._get_beta_cov(full_model, full_model_hc3)

        factor2_pairs = list(combinations(factor2_levels, 2))

        rows = []
        factor1 = factor_combination[0]
        factor2 = factor_combination[1]
        for factor1_level in levels[factor1]:
            for first, second in factor2_pairs:
                l_first = x_grid[
                    (grid[factor1] == factor1_level)
                    & (grid[factor2] == first)
                    ].mean(axis=0)

                l_second = x_grid[
                    (grid[factor1] == factor1_level)
                    & (grid[factor2] == second)
                    ].mean(axis=0)

                contrast = l_first - l_second
                estimate = contrast @ beta
                se = np.sqrt(contrast @ cov @ contrast)

                rows.append({
                    factor1: factor1_level,
                    factor2: f"{first} / {second}",
                    "ratio": np.exp(estimate),
                    "ci_low": np.exp(estimate - 1.96 * se),
                    "ci_high": np.exp(estimate + 1.96 * se)
                })

        combination_ratios = pd.DataFrame(rows)

        return combination_ratios

    def _print_throughput_ratios_within_factor_levels(self, factor_combination: list[str], factor_ratios: pd.DataFrame):
        headers = list(factor_combination) + ["Ratio", "CI low", "CI high"]
        table_entries = []
        for _, row in factor_ratios.iterrows():
            values = row.values[:]
            table_entries.append(values)
        table_str = tabulate.tabulate(table_entries,
                                      headers=headers,
                                      tablefmt="simple",
                                      #floatfmt=".2f",
                                      )
        print("Throughput ratios within factor levels of: %s" % list(factor_combination))
        print(table_str)
