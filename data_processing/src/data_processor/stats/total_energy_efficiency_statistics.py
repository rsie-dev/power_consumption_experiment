from pathlib import Path

import pandas as pd
import numpy as np

from .energy_efficiency_statistics import EnergyEfficiencyStatistics
from .factor_means import FactorMeans
from .diagnostic import Diagnostic
from .stat_model import StatModel


class TotalEnergyEfficiencyStatistics(EnergyEfficiencyStatistics):
    def __init__(self, resources: Path, logfile: bool, n_boot: int, seed: int | None, threading: str, mode: str, create_tex: bool):
        super().__init__(resources, logfile, n_boot, seed, threading, mode, create_tex)

    @property
    def base_column(self) -> str:
        return "energy_efficiency_total"

    @property
    def effect_name(self) -> str:
        return "total"

    def _calculate_base(self, df: pd.DataFrame) -> pd.DataFrame:
        df["ee_total_bpj"] = (
            df["energy_efficiency_total"]
            .pint.to("byte/joule")
            .pint.magnitude
        )

        # Dimensionless logarithms relative to 1 B/J
        df["log_energy_efficiency_total"] = np.log(df["ee_total_bpj"])

        #reference = 1 * ureg.byte / ureg.joule
        #normalized_total = df["energy_efficiency_total"] / reference
        #df["log_ee_total"] = np.log(normalized_total).pint.magnitude
        return df

    def _calculate_statistics(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        self._log_section("Check models")

        stat_model = StatModel()
        two_way_model = stat_model .fit_two_way_model(df, "log_energy_efficiency_total", self.factors)
        three_way_model, full_model = stat_model .fit_higher_order_models(df, "log_energy_efficiency_total", self.factors)
        stat_model .print_fitted_models("", two_way_model, three_way_model, full_model)

        # Now check:
        # 1. residual variance across fitted values -> UI
        # 2. Check residual normality and extreme tails -> UI

        #self._log_section("Residual variance")
        #residual_variance = ResidualVariance(self.factors)
        #zero_variance, configuration_details = residual_variance.calculate_residual_variance(df, full_model)
        #residual_variance.print_residual_variance(zero_variance, configuration_details)

        self._log_section("Diagnostic")
        diagnostic = Diagnostic(self.factors)
        diagnostics = diagnostic.find_observations_that_strongly_influence_model(full_model)
        diagnostic.print_observations_that_strongly_influence_model(diagnostics)

        # Determine whether extreme residuals are isolated or concentrated in particular configurations.
        extremes = diagnostic.get_diagnostics(full_model)
        diagnostic.print_diagnostics(extremes)

        self._log_section("Calculate ANOVA")
        anova_hc3, std_error_description_df = self._calculate_anova_hc3(full_model)
        self._print_anova_hc3(anova_hc3, std_error_description_df)

        ranked_effects_df, decompositions = self._calculate_anova_effect_rankings(full_model)
        self._print_anova_effect_rankings(ranked_effects_df)
        order_summary = self._calculate_order_summary(ranked_effects_df)
        self._print_order_summary(order_summary)
        self._print_anova_effect_decompositions(decompositions)

        relevant_percent = 95
        relevant_factors = self._extract_relevant_factors(ranked_effects_df, relevant_percent)
        self._logger.info("")
        self._logger.info("Relevant %d%% factors: %s", relevant_percent,
                          " > ".join([str(f) for f in relevant_factors]))

        # Make later statistical inference robust to unequal residual variance.
        # Reason: HC3 protects significance tests and confidence intervals against unequal configuration
        # variability and keeps the method consistent with compression.
        # -> use HC3 version for confidence intervals and significance tests.
        full_model_hc3 = full_model.get_robustcov_results(
            cov_type="HC3"
        )

        self._log_section("Factor mean CIs")
        total_factor_means = FactorMeans(self.factors, self.base_column, self._display_unit)
        self._logger.info("Factor mean CI total")
        means_ci = total_factor_means.show_factor_means_ci(df, full_model, full_model_hc3, relevant_factors)

        # Check that the top confidence intervals do not overlap
        self._bootstrap_ci_calculator.validate_data(df, full_model)
        confidence_intervals, bootstrap_results, bootstrap_omega, point_omega = (
            self._bootstrap_ci_calculator.calculate_ci(df)
        )
        self._bootstrap_ci_calculator.print_ci(bootstrap_results)

        self._log_section("Ranking")
        effect_rankings = self._effect_ranking.create_effect_ranking_table(ranked_effects_df,
                                                                           confidence_intervals, anova_hc3)
        self._effect_ranking.print_effect_ranking_table(effect_rankings)

        factor_means_values_df = self._create_factor_means_table(df, full_model, full_model_hc3)
        self._print_factor_table(factor_means_values_df)
        return effect_rankings, means_ci
