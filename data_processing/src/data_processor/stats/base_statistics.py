import logging
from pathlib import Path
import re

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.regression.linear_model import RegressionResultsWrapper
import tabulate

from data_processor.processor import Processor


class BaseStatistics(Processor):
    DRIFT_THRESHOLD = 1.0  # percentage points across runs 1–30

    def __init__(self, resources: Path, n_boot: int, seed: int | None, threading: str, mode: str, create_tex: bool):
        super().__init__(resources)
        self._logger = logging.getLogger(self.__class__.__name__)
        self._n_boot = n_boot
        if seed is None:
            seed = 12345
        self._rng = np.random.default_rng(seed)
        self._threading = threading
        self._mode = mode
        self._create_tex = create_tex

    @property
    def factors(self):
        return ["host", "tool", "dataset", "strength"]

    def _log_section(self, title: str):
        print("")
        print("-" * 20 + f" {title} " + "-" * 20)
        print("")

    def _extract_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df[df["mode"] == self._mode]
        df = df[df["threading"] == self._threading]
        return df

    def _validate_data_quantity(self, comp: pd.DataFrame, base_column: str):
        cell_counts = comp.groupby(
            self.factors,
            observed=True
        ).size()

        print("Validate data quantity of:    %s %s" % (self._threading, self._mode))
        print("Observations:                 %d" % len(comp))
        print("Configurations:               %d" % len(cell_counts))
        print("Min trials per configuration: %d" % cell_counts.min())
        print("Max trials per configuration: %d" % cell_counts.max())
        missing = comp[base_column].isna().sum()
        print("Missing response values:      %d" % missing)
        if missing:
            raise ValueError("Missing response values: %d" % missing)
        print("Tools:                        %s" % ", ".join(comp["tool"].unique()))

    def _print_fitted_models(self, kind: str, two_way_model, three_way_model, full_model):
        # Check the fit has the expected dimensions.
        print()
        extra = f" {kind}:" if kind else ":"
        print("Two-way model%s" % extra)
        print("Observations:                 %d" % int(two_way_model.nobs))
        print("Model degrees of freedom:     %d" % int(two_way_model.df_model))
        print("Residual degrees of freedom:  %d" % int(two_way_model.df_resid))
        # This means the model explains about R^2 * 100% of observed log-throughput variation.
        print("R-squared:                    %f" % two_way_model.rsquared)

        ## Determine whether three-way interactions are sufficient
        ## or whether the four-way interaction is also needed.
        print(anova_lm(two_way_model, full_model))
        print("Full model:")
        print("R² increase:                  %f" % (full_model.rsquared - two_way_model.rsquared))

        print(anova_lm(two_way_model, three_way_model, full_model))
        print("Three-way R² gain:            %f" % (three_way_model.rsquared - two_way_model.rsquared))
        print("Four-way R² gain:             %f" % (full_model.rsquared - three_way_model.rsquared))

    def _fit_two_way_model(self, data: pd.DataFrame, response):
        formula = f"""
        {response} ~ (
            C(host, Sum)
            + C(tool, Sum)
            + C(dataset, Sum)
            + C(strength, Sum)
        )**2
        """

        return smf.ols(
            formula=formula,
            data=data
        ).fit()

    def _fit_higher_order_models(self, data: pd.DataFrame, response):
        three_way_formula = f"""
            {response} ~ (
                C(host, Sum)
                + C(tool, Sum)
                + C(dataset, Sum)
                + C(strength, Sum)
            )**3
        """

        full_formula = f"""
            {response} ~
            C(host, Sum) *
            C(tool, Sum) *
            C(dataset, Sum) *
            C(strength, Sum)
        """

        three_way_model = smf.ols(
            formula=three_way_formula,
            data=data
        ).fit()

        full_model = smf.ols(
            formula=full_formula,
            data=data
        ).fit()

        return three_way_model, full_model

    def _print_observations_that_strongly_influence_model(self, diagnostics: pd.DataFrame):
        headers = self.factors + ["run", "cooks_d"]
        table_str = tabulate.tabulate(diagnostics,
                                      showindex=False,
                                      headers=headers,
                                      )
        print("Influence table")
        print(table_str)

    def _find_observations_that_strongly_influence_model(self, model):
        # Screen for observations that strongly influence the decompression model.
        # Reason: A small number of influential trials could distort effect estimates.
        influence = model.get_influence()
        cooks_d = influence.cooks_distance[0]
        threshold = 4 / model.nobs

        print("Cook's threshold:             %f" % threshold)
        print("Maximum Cook's distance:      %f" % cooks_d.max())
        print("Observations above threshold: %d" % (cooks_d > threshold).sum())

        run_factors = self.factors + ["run"]
        diagnostics = model.model.data.frame[
            run_factors
        ].copy()
        diagnostics["cooks_d"] = cooks_d

        diagnostics = diagnostics.sort_values(
            by=["cooks_d"],
            ascending=False,
        )
        diagnostics = diagnostics.head(10)
        return diagnostics

    def _print_anova_hc3(self, kind: str, anova_hc3_df: pd.DataFrame):
        headers = ["factor", "sum_sq", "df", "F", "PR(>F)"]
        table_str = tabulate.tabulate(anova_hc3_df,
                                      headers=headers,
                                      )
        print()
        extra = f" {kind}:" if kind else ":"
        print("ANOVA HC3%s" % extra)
        print(table_str)

    def _calculate_anova_hc3(self, model: RegressionResultsWrapper):
        # Create robust inference for decompression effects.
        # Reason: HC3 protects significance tests and confidence intervals against unequal
        # configuration variability and keeps the method consistent with compression.
        # Test which factors and interactions affect mean compression throughput
        anova_hc3 = anova_lm(
            model,
            typ=3,
            robust="hc3"
        )
        return anova_hc3

    def _print_anova_table(self, kind: str, ranked: pd.DataFrame):
        headers = ["Factor", "DF", "ω²", "ω² %"]
        table_str = tabulate.tabulate(ranked,
                                      headers=headers,
                                      )
        print("")
        extra = f" {kind}:" if kind else ":"
        print("ANOVA rank table%s" % extra)
        print(table_str)

    def _calculate_anova_table(self, model: RegressionResultsWrapper):
        # Rank factors and interactions by explained variation by omega^2.
        # Reason: RQ asks which effects explain the most variation, which \(p\)-values cannot determine.
        anova_ss = anova_lm(model, typ=3)
        mse = (
                anova_ss.loc["Residual", "sum_sq"]
                / anova_ss.loc["Residual", "df"]
        )
        y = model.model.endog
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
