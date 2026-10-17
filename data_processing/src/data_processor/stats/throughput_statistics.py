import logging
from pathlib import Path

import tabulate
import pandas as pd
import numpy as np
from statsmodels.regression.linear_model import RegressionResultsWrapper
from statsmodels.regression.linear_model import OLSResults

from data_processor import ureg
from data_processor.constants import get_factor_levels
from data_processor.util import tabulate_pint
from .base_statistics import BaseStatistics
from .effect_names import get_effect_name
from .factor_means import FactorMeans
from .bootstrap_ci_calculator import BootstrapCiCalculator
from .drift_analyzer import DriftAnalyzer


class ThroughputStatistics(BaseStatistics):
    def __init__(self, resources: Path, n_boot: int, seed: int | None, threading: str, mode: str, create_tex: bool):
        super().__init__(resources, threading, mode, create_tex)
        self._logger = logging.getLogger(self.__class__.__name__)
        self._display_unit = "MiB/s"
        self._bootstrap_ci_calculator = BootstrapCiCalculator(self.factors, n_boot, seed)

    def process(self, tp_file: Path):
        self._log_section("Process %s %s " % (self._threading, self._mode))

        df = self._frameio.load(tp_file)
        comp = self._extract_data(df)
        comp = self._calculate_base(comp)

        self._validate_data_quantity(comp, "log_throughput")
        drift_analyzer = DriftAnalyzer(self.factors, self.DRIFT_THRESHOLD, self._threading, self._mode)
        self._log_section("Validate temporal drift")
        drift_analyzer.validate_temporal_drift(comp)

        effect_ranking_df, means_ci = self._calculate_statistics(comp, self._threading, self._mode)
        self._create_means_ci_files(means_ci, tp_file)

        if self._create_tex:
            base = tp_file.stem.removeprefix("tp_")
            filename = "tp_effect_ranking_%s_%s_%s_full" % (self._threading, self._mode, base) + ".tex"
            self._create_effect_ranking_tex(effect_ranking_df, filename)
            top_df = effect_ranking_df[
                effect_ranking_df["omega_squared"].cumsum().shift(fill_value=0).lt(0.95)
            ]
            filename = "tp_effect_ranking_%s_%s_%s" % (self._threading, self._mode, base) + ".tex"
            self._create_effect_ranking_tex(top_df, filename)

    def _create_means_ci_files(self, means_ci: dict, tp_file: Path):
        base = tp_file.stem.removeprefix("tp_")
        for effect, df in means_ci.items():
            filename = "tp_effect_%s_%s_%s_%s" % (self._threading, self._mode, effect, base) + ".csv"
            self._frameio.persist(df, self._resources / filename)

    def _calculate_base(self, df: pd.DataFrame) -> pd.DataFrame:
        reference = 1 * ureg.byte / ureg.second
        normalized = df["throughput"] / reference
        df["log_throughput"] = np.log(normalized).pint.magnitude
        return df

    def _calculate_statistics(self, comp: pd.DataFrame, threading: str, mode: str) -> tuple[pd.DataFrame, dict]:
        self._log_section("Calculate statistics")

        compression_model = self._fit_two_way_model(comp,"log_throughput")
        three_way_model, full_model = self._fit_higher_order_models(comp, "log_throughput")
        self._print_fitted_models("", compression_model, three_way_model, full_model)

        # Now check:
        # 1. residual variance across fitted values -> UI
        # 2. Check residual normality and extreme tails -> UI

        self._print_residual_variance(comp, full_model)

        print("")
        diagnostics = self._find_observations_that_strongly_influence_model(full_model)
        self._print_observations_that_strongly_influence_model(diagnostics)

        # Determine whether extreme residuals are isolated or concentrated in particular configurations.
        extremes = self._get_diagnostics(full_model)
        self._print_diagnostics(extremes)

        anova_hc3 = self._calculate_anova_hc3(full_model)
        self._print_anova_hc3("", anova_hc3)

        effects, ranked = self._calculate_anova_table(full_model)
        self._print_anova_table("", ranked)
        relevant_percent = 95
        relevant_factors = self._extract_relevant_factors(ranked, relevant_percent)
        print("")
        print("Relevant %d%% factors: %s" % (relevant_percent, " > ".join([str(f) for f in relevant_factors])))

        # Make later statistical inference robust to unequal residual variance.
        # Reason: HC3 protects significance tests and confidence intervals against unequal configuration
        # variability and keeps the method consistent with compression.
        # -> use HC3 version for confidence intervals and significance tests.
        full_model_hc3 = full_model.get_robustcov_results(
            cov_type="HC3"
        )

        #self._show_factor_means(comp, relevant_factors)
        self._log_section("Factor means CIs")
        factor_means = FactorMeans(self.factors, "throughput", self._display_unit)
        means_ci = factor_means.show_factor_means_ci(comp, full_model, full_model_hc3, relevant_factors)

        # Check that the top confidence intervals do not overlap
        _, confidence_intervals = self._bootstrap_ci_calculator.calculate_ci(comp, full_model.model.formula, effects)

        print("")
        effect_ranking = self._create_effect_ranking_table(ranked, confidence_intervals, anova_hc3)
        self._print_effect_ranking_table(threading, mode, effect_ranking)

        factor_means_values_df = self._create_factor_means_table(comp, full_model, full_model_hc3)
        self._print_factor_table(threading, mode, factor_means_values_df)
        return effect_ranking, means_ci

    def _print_residual_variance(self, comp: pd.DataFrame, full_model: RegressionResultsWrapper):
        diagnostics_comp_mt = comp.copy()
        diagnostics_comp_mt["residual"] = full_model.resid

        residual_sd_comp_mt = diagnostics_comp_mt.groupby(
            self.factors
        )["residual"].std()

        print(residual_sd_comp_mt.describe())
        if residual_sd_comp_mt.min():
            sd_ratio = residual_sd_comp_mt.max() / residual_sd_comp_mt.min()
        else:
            sd_ratio = np.inf
            zero_mask = np.isclose(
                residual_sd_comp_mt,
                0.0,
                atol=1e-12,
                rtol=0
            )
            zero_sd_configurations = (
                residual_sd_comp_mt[zero_mask]
                .rename("residual_sd")
                .reset_index()
            )

            print("Note: no variation found for these configurations")
            print(zero_sd_configurations)
        print("Largest / smallest residual SD ratio: %f" % sd_ratio)

    def _print_factor_table(self, threading: str, mode: str, throughput_values_df: pd.DataFrame):
        display_df = throughput_values_df.copy()
        display_df["Factor"] = display_df["Factor"].map(get_effect_name)
        for column in ["mean_throughput", "ci_low", "ci_high"]:
            display_df[column] = display_df[column].pint.to(self._display_unit)

        unit = display_df["ci_low"].pint.units

        display_df["95% CI"] = (
                display_df["ci_low"].pint.magnitude.map("{:.3f}".format)
                + " - "
                + display_df["ci_high"].pint.to(unit).pint.magnitude.map("{:.3f}".format)
        )
        display_df = display_df.drop(columns=["ci_low", "ci_high"])

        headers = ["Factor", "Level", "Marginal geometric mean throughput", "95% HC3 CI"]
        colalign = ["global"] * len(headers)
        colalign[3] = "center"
        table_str = tabulate_pint(
            display_df,
            showindex=False,
            headers=headers,
            colalign=colalign,
        )
        print("Throughput values for: %s %s" % (threading, mode))
        print(table_str)

    def _create_factor_means_table(self, comp: pd.DataFrame,
                                   full_model: RegressionResultsWrapper, full_model_hc3: OLSResults):
        factor_means = FactorMeans(self.factors, "throughput", self._display_unit)
        levels, grid, x_grid = factor_means.calculate_x_grid(comp, full_model)
        beta, cov = factor_means.get_beta_cov(full_model, full_model_hc3)

        tables = []
        for factor in self.factors:
            table = factor_means.marginal_mean_ci(factor, levels, grid, x_grid, beta, cov, comp).rename(columns={
                factor: "Level",
            })

            table.insert(0, "Factor", factor)
            tables.append(table)

        throughput_values = pd.concat(
            tables,
            ignore_index=True
        )

        throughput_values = throughput_values[
            [
                "Factor",
                "Level",
                "mean_throughput",
                "ci_low",
                "ci_high",
            ]
        ]

        throughput_values["Factor"] = pd.Categorical(
            throughput_values["Factor"],
            categories=self.factors,
            ordered=True
        )
        throughput_values = throughput_values.sort_values(
            ["Factor",
             "mean_throughput",
             ],
            ascending=[True, False]
        )

        return throughput_values

    def _print_diagnostics(self, extremes_df: pd.DataFrame):
        total = extremes_df["count"].sum()

        extremes_df = extremes_df.head(10)
        headers = self.factors + ["count"]
        table_str = tabulate.tabulate(extremes_df,
                                      showindex=False,
                                      headers=headers,
                                      )
        print("")
        print("Extreme observations:         %d" % total)
        print(table_str)

    def _get_diagnostics(self, full_model: RegressionResultsWrapper) -> pd.DataFrame:
        # Determine whether extreme residuals are isolated or concentrated in particular configurations.
        run_factors = self.factors + ["run"]
        diagnostics = full_model.model.data.frame[run_factors].copy()
        diagnostics["studentized_residual"] = (
            full_model.get_influence().resid_studentized_internal
        )
        extreme = diagnostics[
            diagnostics["studentized_residual"].abs() > 3
            ]
        extremes = extreme.groupby(
            self.factors
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

    def _print_means(self, factor: str, means: pd.DataFrame, extra_columns: list | None = None):
        headers = [factor]
        if extra_columns:
            headers += extra_columns
            for column in extra_columns:
                means[column] = means[column].pint.to(self._display_unit)
        else:
            headers += ["Mean throughput"]
            means["geometric_mean_throughput"] = means["geometric_mean_throughput"].pint.to(self._display_unit)
        table_str = tabulate_pint(
            means,
            headers=headers,
        )
        print("Throughput means for: %s" % factor)
        print(table_str)

    def _calculate_mean(self, comp: pd.DataFrame, factor: str, order: list | None = None):
        unit_ref = comp["throughput"].dtype
        means = (
            comp.groupby(factor, observed=True)["log_throughput"]
            .mean()
            .pipe(np.exp)
            .sort_values(ascending=False)
        )
        if order:
            means = means.reindex(order)
        mean_column = "geometric_mean_throughput"
        means = means.rename(mean_column).to_frame()
        means[mean_column] = means[mean_column].astype(unit_ref)
        return means

    def _calculate_mean_combined(self, comp: pd.DataFrame, factor: str | list[str], order: list):
        unit_ref = comp["throughput"].dtype
        means = (
            comp.groupby(
                list(factor),
                observed=True
            )["log_throughput"]
            .mean()
            .pipe(np.exp)
        )
        means = means.unstack(factor[-1])
        means = means.rename_axis(columns=factor[0])

        means = means[order]
        for column in order:
            means[column] = means[column].astype(unit_ref)

        return means

