import logging
import itertools

import pandas as pd
import numpy as np
from tqdm import tqdm
import tabulate

from statsmodels.regression.linear_model import RegressionResultsWrapper
from data_processor.util import tabulate_pint
from .effect_names import build_effect_names


class BootstrapCiCalculator:
    def __init__(self, factors: list[str], n_boot: int, seed: int | None, base_column: str):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._factors = factors
        self._n_boot = n_boot
        if seed is None:
            seed = 12345
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._base_column = base_column

    def print_bootstrap_infos(self):
        self._logger.info("")
        self._logger.info("Bootstrap iterations: %s", self._n_boot)
        self._logger.info("Bootstrap seed:       %s", self._seed)

    def print_ci(self, bootstrap_results: pd.DataFrame):
        headers = ["Factor", "Rank", "df", "eta²", "eta² low", "eta² high", "ω²", "ω² low", "ω² high"]
        table_str = tabulate.tabulate(bootstrap_results,
                                      headers=headers,
                                      )
        self._logger.info("")
        self._logger.info(r"95% confidence interval table, ordered by ω²")
        self._logger.info(table_str)
        self._logger.info("The bootstrap CIs were defined by the 2.5th and 97.5th"
                          "percentiles of the bootstrap distribution.")
        self._logger.info("A stratified nonparametric bootstrap was used.")
        self._logger.info("Trials were sampled with replacement within each configuration.")

    def calculate_ci(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, float]:
        # Prepare the balanced response array
        grouped = df.groupby(
            self._factors,
            observed=True,
            sort=True,
        )

        group_sizes = grouped.size()
        n_repeats = int(group_sizes.iloc[0])

        level_counts = [
            df[factor].nunique()
            for factor in self._factors
        ]

        cell_values = np.stack(
            [
                group["log_%s" % self._base_column].to_numpy(dtype=float)
                for _, group in grouped
            ]
        )

        response_array = cell_values.reshape(
            *level_counts,
            n_repeats,
        )

        factor_axes = range(len(self._factors))
        effect_names, effect_subsets = build_effect_names(self._factors, factor_axes)

        effect_degrees_of_freedom = np.array(
            [
                np.prod(
                    [
                        level_counts[axis] - 1
                        for axis in subset
                    ]
                )
                for subset in effect_subsets
            ],
            dtype=int,
        )

        point_eta, point_omega = self._calculate_effect_sizes(
            response_array, effect_subsets, factor_axes, n_repeats, level_counts, effect_degrees_of_freedom
        )

        # Run the stratified bootstrap
        bootstrap_eta = np.empty(
            (self._n_boot, len(effect_names))
        )
        bootstrap_omega = np.empty(
            (self._n_boot, len(effect_names))
        )

        self._logger.info("")
        self._logger.info("Calculating confidence intervals...")
        for bootstrap_index  in tqdm(range(self._n_boot), colour="green"):
            sample_indices = self._rng.integers(
                low=0,
                high=n_repeats,
                size=response_array.shape,
            )

            bootstrap_values = np.take_along_axis(
                response_array,
                sample_indices,
                axis=-1,
            )

            bootstrap_eta[bootstrap_index], bootstrap_omega[bootstrap_index] = (
                self._calculate_effect_sizes(bootstrap_values, effect_subsets, factor_axes,
                                             n_repeats, level_counts, effect_degrees_of_freedom)
            )

        # Calculate percentile confidence intervals
        eta_lower, eta_upper = np.quantile(
            bootstrap_eta,
            [0.025, 0.975],
            axis=0,
        )

        omega_lower, omega_upper = np.quantile(
            bootstrap_omega,
            [0.025, 0.975],
            axis=0,
        )

        bootstrap_results = pd.DataFrame(
            {
                "df": effect_degrees_of_freedom,
                "eta_squared": point_eta,
                "eta_ci_lower": eta_lower,
                "eta_ci_upper": eta_upper,
                "omega_squared": point_omega,
                "omega_ci_lower": omega_lower,
                "omega_ci_upper": omega_upper,
            },
            index=pd.Index(
                effect_names,
                name="effect",
            ),
        )

        bootstrap_results = bootstrap_results.sort_values(
            "omega_squared",
            ascending=False,
        )

        bootstrap_results.insert(
            0,
            "rank",
            np.arange(1, len(bootstrap_results) + 1),
        )

        confidence_intervals = bootstrap_results[[
            "eta_ci_lower", "eta_ci_upper",
            "omega_ci_lower", "omega_ci_upper",
        ]]
        return confidence_intervals, bootstrap_results, bootstrap_omega, point_omega

    def print_validation_results(self, validation_results: pd.DataFrame):
        headers = ["", "Value"]
        table_str = tabulate_pint(
            validation_results,
            showindex=False,
            headers=headers,
        )
        self._logger.info("Validation infos:")
        self._logger.info(table_str)

    def validate_data(self, df: pd.DataFrame, model: RegressionResultsWrapper) -> pd.DataFrame:
        # Prepare the balanced response array
        grouped = df.groupby(
            self._factors,
            observed=True,
            sort=True,
        )

        group_sizes = grouped.size()

        if group_sizes.nunique() != 1:
            raise ValueError("All configurations must contain the same number of observations")

        n_repeats = int(group_sizes.iloc[0])

        level_counts = [
            df[factor].nunique()
            for factor in self._factors
        ]

        expected_cells = int(np.prod(level_counts))

        if len(group_sizes) != expected_cells:
            raise ValueError(f"Expected {expected_cells} configurations, but found {len(group_sizes)}")

        cell_values = np.stack(
            [
                group["log_%s" % self._base_column].to_numpy(dtype=float)
                for _, group in grouped
            ]
        )

        response_array = cell_values.reshape(
            *level_counts,
            n_repeats,
        )

        factor_axes = range(len(self._factors))

        effect_subsets = [
            subset
            for order in range(1, len(self._factors) + 1)
            for subset in itertools.combinations(factor_axes, order)
        ]

        effect_degrees_of_freedom = np.array(
            [
                np.prod(
                    [
                        level_counts[axis] - 1
                        for axis in subset
                    ]
                )
                for subset in effect_subsets
            ],
            dtype=int,
        )

        # Verify the direct calculation against the original model
        point_eta, _ = self._calculate_effect_sizes(
            response_array, effect_subsets, factor_axes, n_repeats, level_counts, effect_degrees_of_freedom
        )

        if not np.isclose(
                point_eta.sum(),
                model.rsquared,
                rtol=1e-8,
                atol=1e-10,
        ):
            raise RuntimeError("The direct factorial decomposition does not match the model")

        validation_results = pd.DataFrame({
            "name": ["Sum of point eta²", "Model R²"],
            "value": [float(point_eta.sum()), float(model.rsquared)],
        })

        return validation_results

    def _calculate_effect_sizes(self, values: np.ndarray, effect_subsets: list, factor_axes: range,
                                n_repeats: int, level_counts: list, effect_degrees_of_freedom: np.ndarray):
        cell_means = values.mean(axis=-1)
        grand_mean = values.mean()

        ss_total = np.sum(
            np.square(values - grand_mean)
        )

        ss_residual = np.sum(
            np.square(
                values - cell_means[..., np.newaxis]
            )
        )

        residual_df = (
                values.size - cell_means.size
        )

        mse_residual = ss_residual / residual_df

        effect_sums_of_squares = []

        for subset in effect_subsets:
            component = cell_means.copy()

            for axis in factor_axes:
                if axis in subset:
                    component = (
                            component
                            - component.mean(
                        axis=axis,
                        keepdims=True,
                    )
                    )
                else:
                    component = component.mean(
                        axis=axis,
                        keepdims=True,
                    )

            replication_multiplier = (
                    n_repeats
                    * np.prod(
                [
                    level_counts[axis]
                    for axis in factor_axes
                    if axis not in subset
                ]
            )
            )

            effect_ss = (
                    replication_multiplier
                    * np.sum(np.square(component))
            )

            effect_sums_of_squares.append(effect_ss)

        effect_sums_of_squares = np.asarray(
            effect_sums_of_squares
        )

        eta_squared = (
                effect_sums_of_squares / ss_total
        )

        omega_squared = (
                (
                        effect_sums_of_squares
                        - effect_degrees_of_freedom
                        * mse_residual
                )
                / (ss_total + mse_residual)
        )

        return eta_squared, omega_squared
