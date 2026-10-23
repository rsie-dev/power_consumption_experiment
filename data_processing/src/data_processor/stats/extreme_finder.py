import logging
from abc import ABC, abstractmethod

import pandas as pd
from statsmodels.regression.linear_model import RegressionResultsWrapper
import tabulate


class ExtremeFinder(ABC):
    def __init__(self, factors: list[str]):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._factors = factors

    def print_extremes(self, extremes_df: pd.DataFrame, criterion: str):
        total = extremes_df["count"].sum()
        configuration_count = len(extremes_df)

        extremes_df = extremes_df.head(10)
        headers = self._factors + ["count"]
        table_str = tabulate.tabulate(extremes_df,
                                      showindex=False,
                                      headers=headers,
                                      )
        self._logger.info("")
        self._logger.info("Potential extreme observations: %d (%s)", total, criterion)
        self._logger.info("Configurations affected:        %d", configuration_count)
        self._logger.info(table_str)

    @abstractmethod
    def find_extremes(self, df: pd.DataFrame, model: RegressionResultsWrapper) -> tuple[pd.DataFrame, str]:
        pass


class StudentizedResidualExtremeFinder(ExtremeFinder):
    def find_extremes(self, df: pd.DataFrame, model: RegressionResultsWrapper) -> tuple[pd.DataFrame, str]:
        # Determine whether extreme residuals are isolated or concentrated in particular configurations.
        run_factors = self._factors + ["run"]
        diagnostics = model.model.data.frame[run_factors].copy()
        diagnostics["studentized_residual"] = (
            model.get_influence().resid_studentized_internal
        )
        extreme = diagnostics[
            diagnostics["studentized_residual"].abs() > 3
            ]
        extremes = extreme.groupby(
            self._factors
        ).size().sort_values(ascending=False)
        extremes_df = extremes.reset_index(name="count")
        criterion = "|studentized residual| > 3"
        return extremes_df, criterion
