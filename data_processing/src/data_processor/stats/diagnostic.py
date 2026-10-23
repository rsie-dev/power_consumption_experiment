import logging

import pandas as pd
import numpy as np
from statsmodels.regression.linear_model import RegressionResultsWrapper
from scipy.stats import median_abs_deviation
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

    def find_extremes(self, full_model: RegressionResultsWrapper) -> tuple[pd.DataFrame, str]:
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
        criterion = "|studentized residual| > 3"
        return extremes_df, criterion

    def find_extremes_by_residual_scores(self, df: pd.DataFrame, model: RegressionResultsWrapper,
                                         base_column: str, threshold: float = 3.5) -> tuple[pd.DataFrame, str]:
        # Globally studentized residuals use pooled residual variance.
        # Under the severe heteroskedasticity, they systematically flag observations
        # from high-variance configurations.
        # Thus Detect extremes using residuals standardized within each configuration
        model_rows = model.model.data.row_labels

        columns = self._factors + [
            "run",
            "log_%s" % base_column,
        ]

        diagnostics = df.loc[model_rows, columns].copy()

        diagnostics["fitted"] = np.asarray(
            model.fittedvalues
        )
        diagnostics["residual"] = np.asarray(
            model.resid
        )

        def score_cell(cell):
            # Save the group values because include_groups=False removes
            # the factor columns from cell.
            cell_key = cell.name

            if not isinstance(cell_key, tuple):
                cell_key = (cell_key,)

            residual = cell["residual"].to_numpy()

            center = np.median(residual)

            scale = median_abs_deviation(
                residual,
                scale="normal",
                nan_policy="omit",
            )

            scale_method = "MAD"

            # MAD can be zero because of timing quantization or ties.
            if (
                    not np.isfinite(scale)
                    or scale <= np.finfo(float).eps
            ):
                scale = np.std(residual, ddof=1)
                scale_method = "SD fallback"

            # A constant configuration has no within-cell extremes.
            if (
                    not np.isfinite(scale)
                    or scale <= np.finfo(float).eps
            ):
                scores = np.zeros(len(cell))
                scale_method = "constant"
            else:
                scores = (
                                 residual - center
                         ) / scale

            cell = cell.copy()

            # Restore the excluded factor columns.
            for factor, value in zip(self._factors, cell_key):
                cell[factor] = value

            cell["cell_residual_center"] = center
            cell["cell_residual_scale"] = scale
            cell["scale_method"] = scale_method
            cell["cell_robust_z"] = scores
            cell["extreme"] = np.abs(scores) > threshold

            return cell

        diagnostics = (
            diagnostics
            .groupby(
                self._factors,
                observed=True,
                group_keys=False,
            )
            .apply(
                score_cell,
                include_groups=False,
            )
            .reset_index()
        )

        extreme = (
            diagnostics.loc[diagnostics["extreme"]]
            .copy()
            .sort_values(
                "cell_robust_z",
                key=lambda values: values.abs(),
                ascending=False,
            )
        )

        #extreme_summary = (
        #    extreme
        #    .groupby(self._factors, observed=True)
        #    .size()
        #    .rename("extreme_count")
        #    .reset_index()
        #    .sort_values(
        #        "extreme_count",
        #        ascending=False,
        #    )
        #)

        extreme_counts = (
            extreme
            .groupby(
                self._factors,
                observed=True,
            )
            .size()
            .rename("count")
            .reset_index()
            .sort_values(
                "count",
                ascending=False,
            )
            .reset_index(drop=True)
        )

        criterion = f"Within-cell robust threshold: |z| > {threshold}"
        #print("")
        #self._logger.info(criterion)
        #self._logger.info("Extreme observations: %d", len(extreme))
        ##print(
        ##    f"Configurations affected:      "
        ##    f"{len(extreme_summary)}"
        ##)
        #self._logger.info(extreme)

        #print("extreme:")
        ##print(extreme.head(10))
        #print(extreme_counts)

        return extreme_counts, criterion
