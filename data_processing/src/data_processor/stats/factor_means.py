from itertools import product, combinations

import pandas as pd
import numpy as np
from statsmodels.regression.linear_model import RegressionResultsWrapper
from statsmodels.regression.linear_model import OLSResults
from patsy.build import build_design_matrices
import tabulate

from data_processor.constants import get_factor_levels
from data_processor.util import tabulate_pint


class FactorMeans:
    def __init__(self, factors: list[str]):
        self._factors = factors

    @property
    def factors(self):
        return self._factors

    def show_factor_means_ci(self, comp: pd.DataFrame,
                             full_model: RegressionResultsWrapper, full_model_hc3: OLSResults,
                             relevant_factors: list) -> dict:
        result = {}
        for relevant_factor in relevant_factors:
            print("")
            if isinstance(relevant_factor, str):
                mean_cis = self._calculate_factor_ci(comp, full_model, full_model_hc3, relevant_factor)
                self._print_mean_cis(relevant_factor, mean_cis)
                result[relevant_factor] = mean_cis

                factor_ratios = self._calculate_throughput_ratios_between_factor_levels(comp,
                                                                                        full_model, full_model_hc3,
                                                                                        relevant_factor)
                self._print_throughput_ratios_between_factor_levels(relevant_factor, factor_ratios)
            else:
                factor2_levels = get_factor_levels(relevant_factor[1], comp)
                combination_ci = self._calculate_factor_combination_ci(comp, full_model, full_model_hc3,
                                                                       relevant_factor, factor2_levels)
                self._print_mean_cis(relevant_factor, combination_ci)
                result["_".join(relevant_factor)] = combination_ci

                combination_ratios = self._calculate_throughput_ratios_within_factor_levels(comp,
                                                                                            full_model, full_model_hc3,
                                                                                            relevant_factor,
                                                                                            factor2_levels)
                self._print_throughput_ratios_within_factor_levels(list(relevant_factor), combination_ratios)
        return result

    def _print_mean_cis(self, factor: str | list[str], mean_cis: pd.DataFrame):
        if isinstance(factor, str):
            factor = [factor]
        headers = list(factor) + ["Mean", "CI low", "CI high"]
        for column in ["mean_throughput", "ci_low", "ci_high"]:
            mean_cis[column] = mean_cis[column].pint.to("MiB/s")
        table_str = tabulate_pint(
            mean_cis,
            showindex=False,
            headers=headers,
        )
        print("95%% confidence interval of means for: %s" % list(factor))
        print(table_str)

    def _print_throughput_ratios_between_factor_levels(self, factor: str, factor_ratios: pd.DataFrame):
        self._print_throughput_ratios_of_factor_levels("between ", ["Comparison"], factor, factor_ratios)

    def _print_throughput_ratios_within_factor_levels(self, factor_combination: list[str], factor_ratios: pd.DataFrame):
        self._print_throughput_ratios_of_factor_levels("within", factor_combination, str(factor_combination), factor_ratios)

    def _print_throughput_ratios_of_factor_levels(self, kind: str, header: list[str], factor_combination: str, factor_ratios: pd.DataFrame):
        headers = header + ["Ratio", "CI low", "CI high"]
        table_str = tabulate.tabulate(factor_ratios,
                                      showindex=False,
                                      headers=headers,
                                      )
        print(f"Throughput ratios {kind} factor levels of: {factor_combination}")
        print(table_str)

    def _calculate_factor_ci(self, comp: pd.DataFrame, full_model: RegressionResultsWrapper, full_model_hc3: OLSResults,
                             factor: str):
        levels, grid, x_grid = self.calculate_x_grid(comp, full_model)
        beta, cov = self.get_beta_cov(full_model, full_model_hc3)

        mean_cis = self.marginal_mean_ci(factor, levels, grid, x_grid, beta, cov, comp)

        mean_cis = mean_cis.sort_values(
            by=["mean_throughput"],
            ascending=False,
        )
        return mean_cis

    def _calculate_factor_combination_ci(self, comp: pd.DataFrame,
                                         full_model: RegressionResultsWrapper, full_model_hc3: OLSResults,
                                         factor_combination: list[str], factor2_levels: list[str]):
        levels, grid, x_grid = self.calculate_x_grid(comp, full_model)
        beta, cov = self.get_beta_cov(full_model, full_model_hc3)

        rows = []
        factor1 = factor_combination[0]
        factor2 = factor_combination[1]
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
                    "mean_throughput": np.exp(estimate),
                    "ci_low": np.exp(estimate - 1.96 * se),
                    "ci_high": np.exp(estimate + 1.96 * se),
                })

        combination_ci = pd.DataFrame(rows)
        combination_ci = combination_ci.sort_values(
            by=[factor1, "mean_throughput"],
            ascending=[True, False],
        )

        unit_ref = comp["throughput"].dtype
        for column in ["mean_throughput", "ci_low", "ci_high"]:
            combination_ci[column] = combination_ci[column].astype(unit_ref)

        return combination_ci

    def _calculate_throughput_ratios_between_factor_levels(self, comp: pd.DataFrame,
                                                           full_model: RegressionResultsWrapper,
                                                           full_model_hc3: OLSResults,
                                                           factor: str):
        levels, grid, x_grid = self.calculate_x_grid(comp, full_model)
        beta, cov = self.get_beta_cov(full_model, full_model_hc3)

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

    def _calculate_throughput_ratios_within_factor_levels(self, comp: pd.DataFrame,
                                                          full_model: RegressionResultsWrapper,
                                                          full_model_hc3: OLSResults,
                                                          factor_combination: list[str],
                                                          factor2_levels: list[str]):
        levels, grid, x_grid = self.calculate_x_grid(comp, full_model)
        beta, cov = self.get_beta_cov(full_model, full_model_hc3)

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
                         beta: np.ndarray, cov: np.ndarray,
                         comp: pd.DataFrame):
        rows = []
        for level in levels[factor]:
            l = x_grid[grid[factor] == level].mean(axis=0)

            estimate = l @ beta
            se = np.sqrt(l @ cov @ l)

            rows.append({
                factor: level,
                "mean_throughput": np.exp(estimate),
                "ci_low": np.exp(estimate - 1.96 * se),
                "ci_high": np.exp(estimate + 1.96 * se)
            })

        marginal_means_df = pd.DataFrame(rows)
        unit_ref = comp["throughput"].dtype
        for column in ["mean_throughput", "ci_low", "ci_high"]:
            marginal_means_df[column] = marginal_means_df[column].astype(unit_ref)
        return marginal_means_df

    def calculate_x_grid(self, comp: pd.DataFrame, model: RegressionResultsWrapper) -> tuple[dict, pd.DataFrame, np.ndarray]:
        levels = {
            factor: comp[factor].unique().tolist()
            for factor in self.factors
        }

        grid = pd.DataFrame(
            product(*(levels[f] for f in self.factors)),
            columns=self.factors
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
