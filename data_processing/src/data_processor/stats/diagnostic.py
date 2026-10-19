import logging

import pandas as pd
from statsmodels.regression.linear_model import RegressionResultsWrapper
import tabulate


class Diagnostic:
    def __init__(self, factors: list[str]):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._factors = factors

    def print_observations_that_strongly_influence_model(self, diagnostics: pd.DataFrame):
        headers = self._factors + ["run", "cooks_d"]
        table_str = tabulate.tabulate(diagnostics,
                                      showindex=False,
                                      headers=headers,
                                      )
        self._logger.info("Influence table")
        self._logger.info(table_str)

    def find_observations_that_strongly_influence_model(self, model):
        # Screen for observations that strongly influence the decompression model.
        # Reason: A small number of influential trials could distort effect estimates.
        influence = model.get_influence()
        cooks_d = influence.cooks_distance[0]
        threshold = 4 / model.nobs

        self._logger.info("Cook's threshold:             %f", threshold)
        self._logger.info("Maximum Cook's distance:      %f", cooks_d.max())
        self._logger.info("Observations above threshold: %d", (cooks_d > threshold).sum())

        run_factors = self._factors + ["run"]
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

    def print_diagnostics(self, extremes_df: pd.DataFrame):
        total = extremes_df["count"].sum()

        extremes_df = extremes_df.head(10)
        headers = self._factors + ["count"]
        table_str = tabulate.tabulate(extremes_df,
                                      showindex=False,
                                      headers=headers,
                                      )
        self._logger.info("")
        self._logger.info("Extreme observations:         %d", total)
        self._logger.info(table_str)

    def get_diagnostics(self, full_model: RegressionResultsWrapper) -> pd.DataFrame:
        # Determine whether extreme residuals are isolated or concentrated in particular configurations.
        run_factors = self._factors + ["run"]
        diagnostics = full_model.model.data.frame[run_factors].copy()
        diagnostics["studentized_residual"] = (
            full_model.get_influence().resid_studentized_internal
        )
        extreme = diagnostics[
            diagnostics["studentized_residual"].abs() > 3
            ]
        extremes = extreme.groupby(
            self._factors
        ).size().sort_values(ascending=False)
        extremes_df = extremes.reset_index(name="count")
        return extremes_df

