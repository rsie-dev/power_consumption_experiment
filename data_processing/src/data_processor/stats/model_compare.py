import logging

import numpy as np
import pandas as pd
from statsmodels.stats.anova import anova_lm
from statsmodels.regression.linear_model import RegressionResultsWrapper
import tabulate


class ModelCompare:
    def __init__(self):
        self._logger = logging.getLogger(self.__class__.__name__)

    def compare(self, sensitivity_model: RegressionResultsWrapper,
                full_model: RegressionResultsWrapper) -> pd.DataFrame:
        original = self._calculate_omega_squared(full_model)
        sensitivity = self._calculate_omega_squared(sensitivity_model)

        comparison = (
            original.rename(
                columns={
                    "omega2": "omega2_original",
                    "rank": "rank_original",
                }
            )
            .join(
                sensitivity.rename(
                    columns={
                        "omega2": "omega2_sensitivity",
                        "rank": "rank_sensitivity",
                    }
                ),
                how="outer",
            )
        )

        comparison["omega2_change"] = (
                comparison["omega2_sensitivity"]
                - comparison["omega2_original"]
        )

        # Change expressed in percentage points of total variation.
        comparison["omega2_change_pp"] = (
                100 * comparison["omega2_change"]
        )

        comparison["rank_change"] = (
                comparison["rank_sensitivity"]
                - comparison["rank_original"]
        )

        comparison = comparison.sort_values("rank_original")

        return comparison

    def _calculate_omega_squared(self, model: RegressionResultsWrapper):
        """Calculate omega² for factorial terms in an OLS model."""

        anova = anova_lm(model, typ=3)

        response = np.asarray(model.model.endog)
        total_ss = np.sum((response - response.mean()) ** 2)
        residual_ms = model.ssr / model.df_resid

        effects = anova.drop(
            index=["Intercept", "Residual"],
            errors="ignore",
        ).copy()

        # Run-order terms are nuisance terms, not ranked factors.
        effects = effects[
            ~effects.index.to_series().str.contains(
                "run_centered",
                regex=False,
            )
        ]

        effects["omega2"] = (
                            effects["sum_sq"]
                            - effects["df"] * residual_ms
                            ) / (total_ss + residual_ms)

        effects["rank"] = (
            effects["omega2"]
            .rank(method="min", ascending=False)
            .astype(int)
        )

        return effects[["omega2", "rank"]]

    def print_high_impact_terms(self, host: str, sensitivity_model: RegressionResultsWrapper,
                full_model: RegressionResultsWrapper, comparison: pd.DataFrame):
        self._logger.info("")
        self._print_comparison(host, comparison)

        original = self._calculate_omega_squared(full_model)
        sensitivity = self._calculate_omega_squared(sensitivity_model)

        original_95 = self._terms_to_95_percent(original)
        sensitivity_95 = self._terms_to_95_percent(sensitivity)

        self._logger.info("")
        self._print_terms("Original", original_95)
        self._print_terms("Sensitivity", sensitivity_95)

        original_set = set(original_95.index)
        sensitivity_set = set(sensitivity_95.index)

        added_terms = sensitivity_set - original_set
        removed_terms = original_set - sensitivity_set

        self._logger.info("")
        self._logger.info("Same 95%% term set: %s", original_set == sensitivity_set)
        self._logger.info("Added terms:       %s", added_terms)
        self._logger.info("Removed terms:     %s", removed_terms)

        important_terms = original_set | sensitivity_set
        important_comparison = comparison.loc[
            comparison.index.intersection(important_terms)
        ].sort_values("rank_original")

        important_df = important_comparison[[
                "omega2_original",
                "omega2_sensitivity",
                "omega2_change_pp",
                "rank_original",
                "rank_sensitivity",
        ]]

        headers = ["Factor", "ω² original" , "ω² sensitivity", "ω² change pp", "rank original", "rank sensitivity"]
        table_str = tabulate.tabulate(important_df,
                                      headers=headers,
                                      )
        self._logger.info("")
        self._logger.info("Change overview for %s:", host)
        self._logger.info(table_str)

    def _print_terms(self, kind: str, terms: pd.DataFrame):
        headers = ["Factor", "ω²", "Rank", "Cumulative ω²"]
        table_str = tabulate.tabulate(terms,
                                      headers=headers,
                                      )
        self._logger.info("%s 95%% terms:", kind)
        self._logger.info(table_str)

    def _print_comparison(self, host: str, comparison: pd.DataFrame):
        headers = ["Factor", "omega2_original", "rank_original", "omega2_sensitivity", "rank_sensitivity",
                   "omega2_change", "omega2_change_pp", "rank_change"]
        table_str = tabulate.tabulate(comparison,
                                      headers=headers,
                                      )
        self._logger.info("Model comparison for %s: ", host)
        self._logger.info(table_str)

    def _terms_to_95_percent(self, effect_table):
        ordered = effect_table.sort_values(
            "omega2",
            ascending=False,
        ).copy()

        # Negative omega² values are negligible.
        ordered["cumulative_omega2"] = (
            ordered["omega2"]
            .clip(lower=0)
            .cumsum()
        )

        reached = np.flatnonzero(
            ordered["cumulative_omega2"].to_numpy() >= 0.95
        )

        if len(reached) == 0:
            return ordered

        return ordered.iloc[: reached[0] + 1]
