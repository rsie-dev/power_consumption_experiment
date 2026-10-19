from pathlib import Path

import pandas as pd
import numpy as np

from .energy_efficiency_statistics import EnergyEfficiencyStatistics
from .factor_means import FactorMeans
from .diagnostic import Diagnostic


class NetEnergyEfficiencyStatistics(EnergyEfficiencyStatistics):
    def __init__(self, resources: Path, logfile: bool, n_boot: int, seed: int | None,
                 threading: str, mode: str, create_tex: bool):
        super().__init__(resources, logfile, n_boot, seed, threading, mode, create_tex)

    @property
    def base_column(self) -> str:
        return "energy_efficiency_net"

    @property
    def effect_name(self) -> str:
        return "net"

    def _calculate_base(self, df: pd.DataFrame) -> pd.DataFrame:
        df["ee_net_bpj"] = (
            df["energy_efficiency_net"]
            .pint.to("byte/joule")
            .pint.magnitude
        )

        # Dimensionless logarithms relative to 1 B/J
        df["log_energy_efficiency_net"] = np.log(df["ee_net_bpj"])

        #reference = 1 * ureg.byte / ureg.joule
        #normalized_net = df["energy_efficiency_net"] / reference
        #df["log_ee_net"] = np.log(normalized_net).pint.magnitude
        return df

    def _calculate_statistics(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        self._log_section("Check models")

        two_way_model = self._fit_two_way_model(df, "log_energy_efficiency_net")
        three_way_model, full_model = self._fit_higher_order_models(df, "log_energy_efficiency_net")

        self._print_fitted_models("", two_way_model, three_way_model, full_model)

        # Now check:
        # 1. residual variance across fitted values -> UI
        # 2. Check residual normality and extreme tails -> UI

        self._log_section("Diagnostic")
        diagnostic = Diagnostic(self.factors)
        diagnostics = diagnostic.find_observations_that_strongly_influence_model(full_model)
        diagnostic.print_observations_that_strongly_influence_model(diagnostics)

        extremes = diagnostic.get_diagnostics(full_model)
        diagnostic.print_diagnostics(extremes)

        self._log_section("Calculate ANOVA")
        anova_hc3 = self._calculate_anova_hc3(full_model)
        self._print_anova_hc3("net", anova_hc3)

        effects, ranked = self._calculate_anova_table(full_model)
        self._print_anova_table("", ranked)
        relevant_percent = 95
        relevant_factors = self._extract_relevant_factors(ranked, relevant_percent)
        self._logger.info("")
        self._logger.info("Relevant %d%% factors:           %s", relevant_percent,
                          " > ".join([str(f) for f in relevant_factors]))

        full_model_hc3 = full_model.get_robustcov_results(
            cov_type="HC3"
        )

        self._log_section("Factor mean CIs")
        factor_means = FactorMeans(self.factors, self.base_column, self._display_unit)
        self._logger.info("Factor mean CI net")
        means_ci = factor_means.show_factor_means_ci(df, full_model,
                                                     full_model_hc3, relevant_factors)

        # Check that the top confidence intervals do not overlap
        _, confidence_intervals = self._bootstrap_ci_calculator.calculate_ci(df,
                                                                             full_model.model.formula,
                                                                             effects)
        self._logger.info("")
        effect_ranking = self._create_effect_ranking_table(ranked, confidence_intervals, anova_hc3)
        self._print_effect_ranking_table(effect_ranking)

        factor_means_values_df = self._create_factor_means_table(df, full_model, full_model_hc3)
        self._print_factor_table(factor_means_values_df)
        return effect_ranking, means_ci
