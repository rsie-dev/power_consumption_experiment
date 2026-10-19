import logging

import pandas as pd
import numpy as np
from statsmodels.regression.linear_model import RegressionResultsWrapper

from data_processor.util import tabulate_pint


class ResidualVariance:
    def __init__(self, factors: list[str]):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._factors = factors

    def print_residual_variance(self, zero_variance: pd.DataFrame, configuration_details: pd.DataFrame):
        self._logger.info("Zero-variance configurations: %s", len(zero_variance))
        headers = ["host", "tool", "dataset", "strength", "n", "residual_sd", "unique_duration_values",
                   "duration_min", "duration_max", "minimum_duration_step"]
        table_str = tabulate_pint(
            zero_variance,
            showindex=False,
            headers=headers,
        )
        self._logger.info(table_str)

        self._logger.info("")
        self._logger.info("Configurations with the highest residual dispersion:")
        largest = configuration_details.nlargest(15, "residual_sd")
        table_str = tabulate_pint(
            largest,
            showindex=False,
            headers=headers,
        )
        self._logger.info(table_str)

    def calculate_residual_variance(self, df: pd.DataFrame, model: RegressionResultsWrapper) -> tuple[
        pd.DataFrame, pd.DataFrame]:
        def minimum_positive_step(values):
            unique_values = np.sort(
                np.unique(np.asarray(values, dtype=float))
            )

            differences = np.diff(unique_values)
            positive_differences = differences[differences > 0]

            if positive_differences.size == 0:
                return np.nan

            return positive_differences.min()

        diagnostics = df.copy()
        model_rows = model.model.data.row_labels
        diagnostics = (
            diagnostics.loc[model_rows, self._factors]
            .reset_index(drop=True)
            .copy()
        )

        diagnostics["fitted"] = np.asarray(model.fittedvalues)
        diagnostics["residual"] = np.asarray(model.resid)

        diagnostics["duration_s"] = (
            df.loc[model_rows, "duration"]
            .pint.to("second")
            .pint.magnitude
            .to_numpy()
        )

        configuration_details = (
            diagnostics
            .groupby(self._factors, observed=True)
            .agg(
                n=("residual", "size"),
                residual_sd=("residual", "std"),
                unique_duration_values=("duration_s", "nunique"),
                duration_min=("duration_s", "min"),
                duration_max=("duration_s", "max"),
                minimum_duration_step=(
                    "duration_s",
                    minimum_positive_step
                ),
            )
            .reset_index()
        )

        for column in ["duration_min", "duration_max", "minimum_duration_step"]:
            configuration_details[column] = configuration_details[column].astype("pint[second]")

        zero_variance = configuration_details[
            np.isclose(
                configuration_details["residual_sd"],
                0.0,
                atol=1e-12
            )
        ]
        return zero_variance, configuration_details
