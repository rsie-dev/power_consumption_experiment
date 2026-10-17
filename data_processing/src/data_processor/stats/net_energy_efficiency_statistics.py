from pathlib import Path

import pandas as pd
import numpy as np

from .energy_efficiency_statistics import EnergyEfficiencyStatistics
from .factor_means import FactorMeans


class NetEnergyEfficiencyStatistics(EnergyEfficiencyStatistics):
    def __init__(self, resources: Path, n_boot: int, seed: int | None, threading: str, mode: str, create_tex: bool):
        super().__init__(resources, n_boot, seed, threading, mode, create_tex)

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

    def _validate_data(self, comp: pd.DataFrame):
        self._validate_data_quantity(comp, "energy_efficiency_net")

    def _calculate_statistics(self, comp: pd.DataFrame):
        self._log_section("Calculate statistics")

        two_way_model = self._fit_two_way_model(comp, "log_energy_efficiency_net")
        three_way_model, full_model = self._fit_higher_order_models(comp, "log_energy_efficiency_net")

        self._print_fitted_models("", two_way_model, three_way_model, full_model)

        # Now check:
        # 1. residual variance across fitted values -> UI
        # 2. Check residual normality and extreme tails -> UI

        print()
        diagnostics = self._find_observations_that_strongly_influence_model(three_way_model)
        self._print_observations_that_strongly_influence_model(diagnostics)

        anova_hc3 = self._calculate_anova_hc3(three_way_model)
        self._print_anova_hc3("net", anova_hc3)

        effects, net_ranked = self._calculate_anova_table(three_way_model)
        self._print_anova_table("", net_ranked)
        relevant_percent = 95
        relevant_factors = self._extract_relevant_factors(net_ranked, relevant_percent)
        print("")
        print("Relevant %d%% factors:           %s" % (relevant_percent, " > ".join([str(f) for f in relevant_factors])))

        three_way_model_hc3 = three_way_model.get_robustcov_results(
            cov_type="HC3"
        )

        self._log_section("Factor mean CIs")
        factor_means = FactorMeans(self.factors, "energy_efficiency_net", self._display_unit)
        print("Factor mean CI net")
        means_ci = factor_means.show_factor_means_ci(comp, three_way_model,
                                                     three_way_model_hc3, relevant_factors)

        # Check that the top confidence intervals do not overlap
        ci_values, confidence_intervals = self._bootstrap_ci_calculator.calculate_ci(comp,
                                                                                     three_way_model.model.formula,
                                                                                     effects)
        self._bootstrap_ci_calculator.print_ci(ci_values)
