import logging
from itertools import product, combinations
from dataclasses import dataclass

import pandas as pd
import numpy as np
from statsmodels.regression.linear_model import RegressionResultsWrapper
from statsmodels.regression.linear_model import OLSResults
from patsy.build import build_design_matrices
import tabulate

from data_processor.constants import get_factor_levels
from data_processor.util import tabulate_pint


@dataclass(frozen=True)
class FactorMeansCIResult:
    means: dict[str, pd.DataFrame]


class FactorMeans:
    def __init__(self, factors: list[str], base_column, display_unit):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._factors = factors
        self._base_column = base_column
        self._display_unit = display_unit

    @property
    def base_upper(self):
        return self._base_column.capitalize()

    def show_factor_means_ci(self, comp: pd.DataFrame,
                             model: RegressionResultsWrapper, model_hc3: OLSResults,
                             relevant_factors: list) -> FactorMeansCIResult:
        self._logger.info("")
        self._logger.info("Factor means CI")
        result = {}
        for relevant_factor in relevant_factors:
            self._logger.info("")
            if isinstance(relevant_factor, str):
                mean_cis = self._calculate_factor_ci(comp, model, model_hc3, relevant_factor)
                self._print_mean_cis(relevant_factor, mean_cis)
                result[relevant_factor] = mean_cis

                factor_ratios = self._calculate_ratios_between_factor_levels(comp,
                                                                             model, model_hc3,
                                                                             relevant_factor)
                self._print_ratios_between_factor_levels(relevant_factor, factor_ratios)
            else:
                factor2_levels = get_factor_levels(relevant_factor[1], comp)
                combination_ci = self._calculate_factor_combination_ci(comp, model, model_hc3,
                                                                       relevant_factor, factor2_levels)
                self._print_mean_cis(relevant_factor, combination_ci)
                result["_".join(relevant_factor)] = combination_ci

                combination_ratios = self._calculate_ratios_within_factor_levels(comp,
                                                                                 model, model_hc3,
                                                                                 relevant_factor,
                                                                                 factor2_levels)
                self._print_ratios_within_factor_levels(list(relevant_factor), combination_ratios)

        factor_means = FactorMeansCIResult(
            means=result
        )
        return factor_means

    def _print_mean_cis(self, factor: str | list[str], mean_cis: pd.DataFrame):
        if isinstance(factor, str):
            factor = [factor]
        headers = list(factor) + ["Mean", "CI low", "CI high"]
        for column in ["mean_%s" % self._base_column, "ci_low", "ci_high"]:
            mean_cis[column] = mean_cis[column].pint.to(self._display_unit)
        table_str = tabulate_pint(
            mean_cis,
            showindex=False,
            headers=headers,
        )
        self._logger.info("95%% confidence interval of geometric mean for: %s", list(factor))
        self._logger.info(table_str)

    def _print_ratios_between_factor_levels(self, factor: str, factor_ratios: pd.DataFrame):
        self._print_ratios_of_factor_levels("between ", ["Comparison"], factor, factor_ratios)

    def _print_ratios_within_factor_levels(self, factor_combination: list[str], factor_ratios: pd.DataFrame):
        self._print_ratios_of_factor_levels("within", factor_combination, str(factor_combination), factor_ratios)

    def _print_ratios_of_factor_levels(self, kind: str, header: list[str], factor_combination: str,
                                       factor_ratios: pd.DataFrame):
        headers = header + ["Ratio", "CI low", "CI high"]
        table_str = tabulate.tabulate(factor_ratios,
                                      showindex=False,
                                      headers=headers,
                                      )
        self._logger.info("%s ratios %s factor levels of: %s", self.base_upper, kind, factor_combination)
        self._logger.info(table_str)

    def _calculate_factor_ci(self, comp: pd.DataFrame, model: RegressionResultsWrapper, model_hc3: OLSResults,
                             factor: str) -> pd.DataFrame:
        levels, grid, x_grid = self.calculate_x_grid(comp, model)
        beta, cov = self.get_beta_cov(model, model_hc3)

        mean_cis = self.marginal_mean_ci(factor, levels, grid, x_grid, beta, cov, comp)

        mean_cis = mean_cis.sort_values(
            by=["mean_%s" % self._base_column],
            ascending=False,
        )
        return mean_cis

    def _calculate_factor_combination_ci(self, comp: pd.DataFrame,
                                         model: RegressionResultsWrapper, model_hc3: OLSResults,
                                         factor_combination: list[str], factor2_levels: list[str]) -> pd.DataFrame:
        levels, grid, x_grid = self.calculate_x_grid(comp, model)
        beta, cov = self.get_beta_cov(model, model_hc3)

        rows = []
        factor1 = factor_combination[0]
        factor2 = factor_combination[1]
        mean_column = "mean_%s" % self._base_column
        for factor1_level in levels[factor1]:
            for factor2_level in factor2_levels:
                mask = (
                        (grid[factor1] == factor1_level)
                        & (grid[factor2] == factor2_level)
                )

                l = x_grid[mask].mean(axis=0)
                estimate = l @ beta
                se = np.sqrt(l @ cov @ l)

                rows.append({
                    factor1: factor1_level,
                    factor2: factor2_level,
                    mean_column: np.exp(estimate),
                    "ci_low": np.exp(estimate - 1.96 * se),
                    "ci_high": np.exp(estimate + 1.96 * se),
                })

        combination_ci = pd.DataFrame(rows)
        combination_ci = combination_ci.sort_values(
            by=[factor1, mean_column],
            ascending=[True, False],
        )

        unit_ref = comp[self._base_column].dtype
        for column in [mean_column, "ci_low", "ci_high"]:
            combination_ci[column] = combination_ci[column].astype(unit_ref)

        return combination_ci

    def _calculate_ratios_between_factor_levels(self, comp: pd.DataFrame,
                                                model: RegressionResultsWrapper,
                                                model_hc3: OLSResults,
                                                factor: str):
        levels, grid, x_grid = self.calculate_x_grid(comp, model)
        beta, cov = self.get_beta_cov(model, model_hc3)

        rows = []
        for first, second in combinations(levels[factor], 2):
            l_first = x_grid[grid[factor] == first].mean(axis=0)
            l_second = x_grid[grid[factor] == second].mean(axis=0)

            contrast = l_first - l_second
            estimate = contrast @ beta
            se = np.sqrt(contrast @ cov @ contrast)

            rows.append({
                "comparison": f"{first} / {second}",
                "ratio": np.exp(estimate),
                "ci_low": np.exp(estimate - 1.96 * se),
                "ci_high": np.exp(estimate + 1.96 * se)
            })

        dataset_ratios = pd.DataFrame(rows)

        dataset_ratios = dataset_ratios.sort_values(
            by=["ratio"],
            ascending=False,
        )
        return dataset_ratios

    def _calculate_ratios_within_factor_levels(self, comp: pd.DataFrame,
                                               model: RegressionResultsWrapper,
                                               model_hc3: OLSResults,
                                               factor_combination: list[str],
                                               factor2_levels: list[str]):
        levels, grid, x_grid = self.calculate_x_grid(comp, model)
        beta, cov = self.get_beta_cov(model, model_hc3)

        factor2_pairs = list(combinations(factor2_levels, 2))

        rows = []
        factor1 = factor_combination[0]
        factor2 = factor_combination[1]
        for factor1_level in levels[factor1]:
            for first, second in factor2_pairs:
                l_first = x_grid[
                    (grid[factor1] == factor1_level)
                    & (grid[factor2] == first)
                    ].mean(axis=0)

                l_second = x_grid[
                    (grid[factor1] == factor1_level)
                    & (grid[factor2] == second)
                    ].mean(axis=0)

                contrast = l_first - l_second
                estimate = contrast @ beta
                se = np.sqrt(contrast @ cov @ contrast)

                rows.append({
                    "factor1": factor1_level,
                    "factor2": f"{first} / {second}",
                    "ratio": np.exp(estimate),
                    "ci_low": np.exp(estimate - 1.96 * se),
                    "ci_high": np.exp(estimate + 1.96 * se)
                })

        combination_ratios = pd.DataFrame(rows)

        return combination_ratios

    def marginal_mean_ci(self, factor: str, levels: dict, grid, x_grid: np.ndarray,
                         beta: np.ndarray, cov: np.ndarray, comp: pd.DataFrame) -> pd.DataFrame:
        rows = []
        mean_column = "mean_%s" % self._base_column
        for level in levels[factor]:
            l = x_grid[grid[factor] == level].mean(axis=0)

            estimate = l @ beta
            se = np.sqrt(l @ cov @ l)

            rows.append({
                factor: level,
                mean_column: np.exp(estimate),
                "ci_low": np.exp(estimate - 1.96 * se),
                "ci_high": np.exp(estimate + 1.96 * se)
            })

        marginal_means_df = pd.DataFrame(rows)
        unit_ref = comp[self._base_column].dtype
        for column in [mean_column, "ci_low", "ci_high"]:
            marginal_means_df[column] = marginal_means_df[column].astype(unit_ref)
        return marginal_means_df

    def calculate_x_grid(self, comp: pd.DataFrame, model: RegressionResultsWrapper) \
            -> tuple[dict, pd.DataFrame, np.ndarray]:
        levels = {
            factor: comp[factor].unique().tolist()
            for factor in self._factors
        }

        grid = pd.DataFrame(
            product(*(levels[f] for f in self._factors)),
            columns=self._factors
        )

        x_grid = np.asarray(
            build_design_matrices(
                [model.model.data.design_info],
                grid
            )[0]
        )
        return levels, grid, x_grid

    def get_beta_cov(self, model: RegressionResultsWrapper, model_hc3: OLSResults):
        beta = np.asarray(model.params)
        cov = np.asarray(model_hc3.cov_params())
        return beta, cov

    def show_factor_means(self, df: pd.DataFrame, relevant_factors: list):
        self._logger.info("")
        self._logger.info("Factor means")
        for relevant_factor in relevant_factors:
            if isinstance(relevant_factor, str):
                means_df = self._calculate_mean(df, relevant_factor)
                self._print_means(relevant_factor, means_df)
            else:
                factor2_levels = get_factor_levels(relevant_factor[1], df)
                means_df = self._calculate_mean_combined(df, relevant_factor, factor2_levels)
                self._print_means(" / ".join(relevant_factor), means_df, factor2_levels)

    def _print_means(self, factor: str, means: pd.DataFrame, extra_columns: list | None = None):
        headers = [factor]
        if extra_columns:
            headers += extra_columns
            for column in extra_columns:
                means[column] = means[column].pint.to(self._display_unit)
        else:
            headers += ["Mean %s" % self._base_column]
            mean_column = "geometric_mean_%s" % self._base_column
            means[mean_column] = means[mean_column].pint.to(self._display_unit)
        table_str = tabulate_pint(
            means,
            headers=headers,
        )
        self._logger.info("%s geometric means for: %s", self._base_column.capitalize(), factor)
        self._logger.info(table_str)

    def _calculate_mean(self, comp: pd.DataFrame, factor: str, order: list | None = None):
        unit_ref = comp[self._base_column].dtype
        means = (
            comp.groupby(factor, observed=True)["log_%s" % self._base_column]
            .mean()
            .pipe(np.exp)
            .sort_values(ascending=False)
        )
        if order:
            means = means.reindex(order)
        mean_column = "geometric_mean_%s" % self._base_column
        means = means.rename(mean_column).to_frame()
        means[mean_column] = means[mean_column].astype(unit_ref)
        return means

    def _calculate_mean_combined(self, comp: pd.DataFrame, factor: str | list[str], order: list):
        unit_ref = comp[self._base_column].dtype
        means = (
            comp.groupby(
                list(factor),
                observed=True
            )["log_%s" % self._base_column]
            .mean()
            .pipe(np.exp)
        )
        means = means.unstack(factor[-1])
        means = means.rename_axis(columns=factor[0])

        means = means[order]
        for column in order:
            means[column] = means[column].astype(unit_ref)

        return means
