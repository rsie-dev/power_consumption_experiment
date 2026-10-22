import logging

import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm


class StatModel:
    def __init__(self):
        self._logger = logging.getLogger(self.__class__.__name__)

    def print_fitted_model(self, kind: str, model):
        # Check the fit has the expected dimensions.
        self._logger.info("%s model", kind)
        self._logger.info("Observations:                 %d", int(model.nobs))
        self._logger.info("Model degrees of freedom:     %d", int(model.df_model))
        self._logger.info("Residual degrees of freedom:  %d", int(model.df_resid))
        # This means the model explains about R^2 * 100% of observed log-throughput variation.
        self._logger.info("R²:                           %f", model.rsquared)
        self._logger.info("Adjusted R²:                  %f", model.rsquared_adj)

    def print_fitted_models(self, kind: str, two_way_model, three_way_model, full_model):
        # Check the fit has the expected dimensions.
        extra = f" {kind}:" if kind else ":"
        self._logger.info("Two-way model%s", extra)
        self._logger.info("Observations:                 %d", int(two_way_model.nobs))
        self._logger.info("Model degrees of freedom:     %d", int(two_way_model.df_model))
        self._logger.info("Residual degrees of freedom:  %d", int(two_way_model.df_resid))
        # This means the model explains about R^2 * 100% of observed log-throughput variation.
        self._logger.info("R²:                           %f", two_way_model.rsquared)
        self._logger.info("Adjusted R²:                  %f", two_way_model.rsquared_adj)

        ## Determine whether three-way interactions are sufficient
        ## or whether the four-way interaction is also needed.
        self._logger.info(anova_lm(two_way_model, full_model))
        self._logger.info("Full model:")
        self._logger.info("R² gain:                      %f", (full_model.rsquared - two_way_model.rsquared))

        self._logger.info(anova_lm(two_way_model, three_way_model, full_model))
        self._logger.info("Three-way R² gain:            %f", (three_way_model.rsquared - two_way_model.rsquared))
        self._logger.info("Four-way R² gain:             %f", (full_model.rsquared - three_way_model.rsquared))

    def _build_interaction_formula(self, response: str, level: int, factors: list[str]):
        return self._build_formula(response, level, factors)

    def _build_full_formula(self, response: str, factors: list[str]):
        return self._build_formula(response, None, factors)

    def _build_formula(self, response: str, level: int | None, factors: list[str]):
        opening = f"{response} ~ (" if level else f"{response} ~"
        closing = f"\n)**{level}" if level else ""
        terms = [f"C({name}, Sum)" for name in factors]
        if level:
            terms = "\n    + ".join(terms)
        else:
            terms = " *\n    ".join(terms)
        formula = f"""
{opening}
    {terms}{closing}
"""
        return formula

    def fit_two_way_model(self, data: pd.DataFrame, response: str, factors: list[str]):
        formula = self._build_interaction_formula(response, 2, factors)

        return smf.ols(
            formula=formula,
            data=data
        ).fit()

    def fit_higher_order_models(self, data: pd.DataFrame, response: str, factors: list[str]):
        three_way_formula = self._build_interaction_formula(response, 3, factors)
        three_way_model = smf.ols(
            formula=three_way_formula,
            data=data
        ).fit()

        full_model = self.fit_full_model(data, response, factors)
        return three_way_model, full_model

    def fit_full_model(self, data: pd.DataFrame, response: str, factors: list[str]):
        full_formula = self._build_full_formula(response, factors)
        full_model = smf.ols(
            formula=full_formula,
            data=data
        ).fit()

        return full_model
