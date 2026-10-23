import logging

import pandas as pd
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
