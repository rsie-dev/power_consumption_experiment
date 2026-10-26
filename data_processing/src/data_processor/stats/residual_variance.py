import logging

import pandas as pd
import numpy as np
from scipy.stats import levene, median_abs_deviation, spearmanr
from statsmodels.regression.linear_model import RegressionResultsWrapper
import tabulate

from data_processor.util import tabulate_pint


class ResidualVariance:
    def __init__(self, factors: list[str], base_column: str):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._factors = factors
        self._base_column = base_column

    def print_residual_variance(self, zero_variance: pd.DataFrame, configuration_details: pd.DataFrame) -> None:
        headers = ["host", "tool", "dataset", "strength", "n", "residual_sd",
                   "unique values", "min", "max", "minimum step"]
        table_str = tabulate_pint(
            zero_variance,
            showindex=False,
            headers=headers,
        )
        self._logger.info("")
        log_column = "log_%s" % self._base_column
        self._logger.info("Zero-variance configurations for %s: %s", log_column, len(zero_variance))
        self._logger.info(table_str)

        largest = configuration_details.nlargest(15, "residual_sd")
        table_str = tabulate_pint(
            largest,
            showindex=False,
            headers=headers,
        )
        self._logger.info("")
        self._logger.info("Configurations with the highest residual dispersion:")
        self._logger.info(table_str)

    def calculate_residual_variance(self, df: pd.DataFrame, model: RegressionResultsWrapper) -> tuple[
        pd.DataFrame, pd.DataFrame]:
        model_rows = model.model.data.row_labels

        diagnostics = (
            df.loc[model_rows, self._factors]
            .reset_index(drop=True)
            .copy()
        )

        diagnostics["fitted"] = np.asarray(model.fittedvalues)
        diagnostics["residual"] = np.asarray(model.resid)

        # Calculate residual dispersion for each configuration
        cell_variance = (
            diagnostics
            .groupby(self._factors, observed=True)
            .agg(
                n=("residual", "size"),
                fitted=("fitted", "mean"),
                residual_sd=("residual", "std"),
                residual_mad=(
                    "residual",
                    lambda x: median_abs_deviation(
                        x,
                        scale="normal"
                    )
                ),
                residual_rmse=(
                    "residual",
                    lambda x: np.sqrt(np.mean(np.square(x)))
                ),
            )
            .reset_index()
        )

        variance_df = cell_variance["residual_sd"].describe().to_frame()
        headers = ["", "residual_sd"]
        table_str = tabulate.tabulate(variance_df,
                                      headers=headers,
                                      )
        self._logger.info("Distribution of residual standard deviations:")
        self._logger.info(table_str)

        q10, q90 = cell_variance["residual_sd"].quantile([0.10, 0.90])

        self._logger.info("")
        self._logger.info("90th/10th percentile ratio:   %.3f", (q90 / q10))

        # Test equality of variances across configurations
        residual_groups = [
            group["residual"].to_numpy()
            for _, group in diagnostics.groupby(
                self._factors,
                observed=True
            )
        ]

        bf_statistic, bf_pvalue = levene(
            *residual_groups,
            center="median"
        )
        self._logger.info("")
        self._logger.info("Brown-Forsythe test:")
        self._logger.info("Statistic:                    %.6f", bf_statistic)
        self._logger.info("p-value:                      %.6g", bf_pvalue)

        # Examine the association between fitted values and dispersion
        rho, rho_pvalue = spearmanr(
            cell_variance["fitted"],
            cell_variance["residual_sd"]
        )
        self._logger.info("")
        self._logger.info("Spearman’s rank correlation between fitted log %s and "
                          "within-configuration residual standard deviation:", self._base_column)
        self._logger.info("rho:                          %.6f", rho)
        self._logger.info("p-value:                      %.6g", rho_pvalue)

        # Print numerical summaries by factor
        for factor in self._factors:
            summary = (
                cell_variance
                .groupby(factor, observed=True)["residual_sd"]
                .agg(["count", "median", "mean", "min", "max"])
                .sort_values("median", ascending=False)
            )

            headers = [factor, "count", "median", "mean", "min", "max"]
            table_str = tabulate.tabulate(summary,
                                          headers=headers,
                                          )
            self._logger.info("")
            self._logger.info("Residual dispersion by %s:", factor)
            self._logger.info(table_str)

        return None, None

    def calculate_zero_variance(self, df: pd.DataFrame, model: RegressionResultsWrapper) -> tuple[
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

        log_column = "log_%s" % self._base_column
        num_column = "%s_r" % log_column
        diagnostics[num_column] = (
            df.loc[model_rows, log_column]
            .to_numpy()
        )

        configuration_details = (
            diagnostics
            .groupby(self._factors, observed=True)
            .agg(
                n=("residual", "size"),
                residual_sd=("residual", "std"),
                unique_values=(num_column, "nunique"),
                min=(num_column, "min"),
                max=(num_column, "max"),
                minimum_step=(
                    num_column,
                    minimum_positive_step
                ),
            )
            .reset_index()
        )

        zero_variance = configuration_details[
            np.isclose(
                configuration_details["residual_sd"],
                0.0,
                atol=1e-12
            )
        ]
        return zero_variance, configuration_details
