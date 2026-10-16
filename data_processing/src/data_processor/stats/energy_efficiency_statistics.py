import logging
from pathlib import Path

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

from data_processor import ureg
from .base_statistics import BaseStatistics


class EnergyEfficiencyStatistics(BaseStatistics):
    def __init__(self, resources: Path, n_boot: int, seed: int | None, threading: str, mode: str, create_tex: bool):
        super().__init__(resources, n_boot, seed, threading, mode, create_tex)
        self._logger = logging.getLogger(self.__class__.__name__)

    def process(self, ee_file: Path):
        self._log_section("Process %s %s" % (self._threading, self._mode))

        df = self._frameio.load(ee_file)
        comp = self._extract_data(df)
        comp = self._calculate_base(comp)

        self._validate_data_quantity(comp, "energy_consumption_net")

        self._calculate_statistics(comp)

    def _calculate_base(self, df: pd.DataFrame) -> pd.DataFrame:
        df["ee_total_bpj"] = (
            df["energy_efficiency_total"]
            .pint.to("byte/joule")
            .pint.magnitude
        )

        df["ee_net_bpj"] = (
            df["energy_efficiency_net"]
            .pint.to("byte/joule")
            .pint.magnitude
        )

        # Dimensionless logarithms relative to 1 B/J
        df["log_ee_total"] = np.log(df["ee_total_bpj"])
        df["log_ee_net"] = np.log(df["ee_net_bpj"])

        #reference = 1 * ureg.byte / ureg.joule
        #normalized_total = df["energy_efficiency_total"] / reference
        #normalized_net = df["energy_efficiency_net"] / reference
        #df["log_ee_total"] = np.log(normalized_total).pint.magnitude
        #df["log_ee_net"] = np.log(normalized_net).pint.magnitude
        return df

    def _validate_data_quantity(self, comp: pd.DataFrame, base_column: str):
        super()._validate_data_quantity(comp, base_column)
        n_negative = (comp[base_column] < 0).sum()
        print("Negative net energy count     %d" % n_negative)

    def _calculate_statistics(self, comp: pd.DataFrame):
        self._log_section("Calculate statistics")

        total_two_way_model = self._fit_two_way_model(comp,"log_ee_total")
        total_three_way_model, total_full_model = self._fit_higher_order_models(comp, "log_ee_total")
        net_two_way_model = self._fit_two_way_model(comp, "log_ee_net")
        net_three_way_model, net_full_model = self._fit_higher_order_models(comp, "log_ee_net")

        self._print_fitted_models("total", total_two_way_model, total_three_way_model, total_full_model)
        self._print_fitted_models("net", net_two_way_model, net_three_way_model, net_full_model)

        # Now check:
        # 1. residual variance across fitted values -> UI
        # 2. Check residual normality and extreme tails -> UI

        print()
        print("Diagnostics for: total")
        total_diagnostics = self._find_observations_that_strongly_influence_model(total_three_way_model)
        self._print_observations_that_strongly_influence_model(total_diagnostics)
        print("Diagnostics for: net")
        net_diagnostics = self._find_observations_that_strongly_influence_model(net_three_way_model)
        self._print_observations_that_strongly_influence_model(net_diagnostics)

        total_anova_hc3 = self._calculate_anova_hc3(total_three_way_model)
        net_anova_hc3 = self._calculate_anova_hc3(net_three_way_model)
        self._print_anova_hc3("total", total_anova_hc3)
        self._print_anova_hc3("net", net_anova_hc3)

        total_effects, total_ranked = self._calculate_anova_table(total_three_way_model)
        net_effects, net_ranked = self._calculate_anova_table(net_three_way_model)
        self._print_anova_table("total", total_ranked)
        self._print_anova_table("net", net_ranked)
        relevant_percent = 95
        total_relevant_factors = self._extract_relevant_factors(total_ranked, relevant_percent)
        net_relevant_factors = self._extract_relevant_factors(net_ranked, relevant_percent)
        print("")
        print("Relevant %d%% factors for total: %s" % (relevant_percent, " > ".join([str(f) for f in total_relevant_factors])))
        print("Relevant %d%% factors for net:   %s" % (relevant_percent, " > ".join([str(f) for f in net_relevant_factors])))

        total_three_way_model_hc3 = total_three_way_model.get_robustcov_results(
            cov_type="HC3"
        )
        net_three_way_model_hc3 = net_three_way_model.get_robustcov_results(
            cov_type="HC3"
        )

        total_means_ci = self._show_factor_means_ci(comp, total_three_way_model, total_three_way_model_hc3, total_relevant_factors)
        #net_means_ci = self._show_factor_means_ci(comp, net_three_way_model, net_three_way_model_hc3, net_relevant_factors)
