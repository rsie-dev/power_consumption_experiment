from pathlib import Path

import pandas as pd
import numpy as np

from .energy_efficiency_statistics import EnergyEfficiencyStatistics
from .factor_means import FactorMeans


class TotalEnergyEfficiencyStatistics(EnergyEfficiencyStatistics):
    def __init__(self, resources: Path, n_boot: int, seed: int | None, threading: str, mode: str, create_tex: bool):
        super().__init__(resources, n_boot, seed, threading, mode, create_tex)

    @property
    def base_column(self) -> str:
        return "energy_efficiency_total"

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

    def _validate_data(self, comp: pd.DataFrame):
        self._validate_data_quantity(comp, self.base_column)

    def _calculate_statistics(self, comp: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        self._log_section("Calculate statistics")

        two_way_model = self._fit_two_way_model(comp,"log_energy_efficiency_total")
        three_way_model, full_model = self._fit_higher_order_models(comp, "log_energy_efficiency_total")

        self._print_fitted_models("total", two_way_model, three_way_model, full_model)

        # Now check:
        # 1. residual variance across fitted values -> UI
        # 2. Check residual normality and extreme tails -> UI

        print()
        diagnostics = self._find_observations_that_strongly_influence_model(three_way_model)
        self._print_observations_that_strongly_influence_model(diagnostics)

        anova_hc3 = self._calculate_anova_hc3(three_way_model)
        self._print_anova_hc3("total", anova_hc3)

        effects, ranked = self._calculate_anova_table(three_way_model)
        self._print_anova_table("", ranked)
        relevant_percent = 95
        relevant_factors = self._extract_relevant_factors(ranked, relevant_percent)
        print("")
        print("Relevant %d%% factors: %s" % (relevant_percent, " > ".join([str(f) for f in relevant_factors])))

        three_way_model_hc3 = three_way_model.get_robustcov_results(
            cov_type="HC3"
        )

        self._log_section("Factor mean CIs")
        total_factor_means = FactorMeans(self.factors, self.base_column, self._display_unit)
        print("Factor mean CI total")
        means_ci = total_factor_means.show_factor_means_ci(comp, three_way_model, three_way_model_hc3, relevant_factors)

        # Check that the top confidence intervals do not overlap
        _, confidence_intervals = self._bootstrap_ci_calculator.calculate_ci(comp,
                                                                             three_way_model.model.formula,
                                                                             effects)
        print("")
        effect_ranking = self._create_effect_ranking_table(ranked, confidence_intervals, anova_hc3)
        self._print_effect_ranking_table(self._threading, self._mode, effect_ranking)

        factor_means_values_df = self._create_factor_means_table(comp, three_way_model, three_way_model_hc3)
        self._print_factor_table(factor_means_values_df)
        return effect_ranking, means_ci
