import logging
from pathlib import Path
import re
from contextlib import contextmanager, nullcontext
from abc import abstractmethod

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.regression.linear_model import RegressionResultsWrapper
import tabulate

from data_processor.processor import Processor
from data_processor.constants import ORDER_STRENGTH
from data_processor.util import tabulate_pint
from .effect_names import get_effect_name, EFFECT_NAMES


class BaseStatistics(Processor):
    DRIFT_THRESHOLD = 1.0  # percentage points across runs 1–30

    def __init__(self, resources: Path, logfile: bool, threading: str, mode: str, create_tex: bool):
        super().__init__(resources)
        self._logger = logging.getLogger(self.__class__.__name__)
        self._logfile = logfile
        self._threading = threading
        self._mode = mode
        self._create_tex = create_tex

    def process(self, used_power_file: Path):
        if self._logfile:
            base = self.extra_log_base(used_power_file)
            logfile_path = self._resources / f"{base}_{self._threading}_{self._mode}.log"
            cm = self._add_logfile(logfile_path)
            self._logger.info("create extra logfile: %s", logfile_path)
        else:
            cm = nullcontext()
        with cm:
            self._do_process(used_power_file)

    @abstractmethod
    def extra_log_base(self, input_file: Path) -> str:
        pass

    @property
    @abstractmethod
    def base_column(self) -> str:
        pass

    @contextmanager
    def _add_logfile(self, logfile: Path):
        handler = logging.FileHandler(logfile, mode="w")
        file_logger = [""]
        try:
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(message)s')
            handler.setFormatter(formatter)
            for logger in file_logger:
                logging.getLogger(logger).addHandler(handler)
            yield
        finally:
            try:
                for logger in file_logger:
                    logging.getLogger(logger).removeHandler(handler)
            finally:
                handler.close()

    @property
    def factors(self):
        return ["host", "tool", "dataset", "strength"]

    def _log_section(self, title: str):
        self._logger.info("")
        self._logger.info("%s %s %s", "-" * 20, title, "-" * 20)
        self._logger.info("")

    def _do_process(self, input_file: Path):
        df = self._frameio.load(input_file)
        df = self._filter_data(df)
        df = self._calculate_base(df)

        self._validate_data(df)

        effect_ranking_df, means_ci = self._calculate_statistics(df)
        self._create_rank_file(effect_ranking_df, input_file)
        self._create_means_ci_files(means_ci, input_file)

        self._post_process(effect_ranking_df, input_file)

    def _filter_data(self, df: pd.DataFrame):
        df = df[df["mode"] == self._mode]
        df = df[df["threading"] == self._threading]
        return df

    def _validate_data(self, df: pd.DataFrame):
        self._log_section("Validate data")
        self._validate_factor_values(df)
        self._validate_tool_strength(df)
        self._validate_data_quantity(df)

    def _validate_factor_values(self, df: pd.DataFrame):
        for factor in self.factors:
            if df[factor].isna().any():
                missing = df[factor].isna().sum()
                raise ValueError(f"Factor {factor} has {missing} missing values")

    def _validate_tool_strength(self, df: pd.DataFrame):
        df = df.copy()
        observed_strengths = set(df["strength"].dropna().astype(str))

        unexpected_strengths = observed_strengths - set(ORDER_STRENGTH)
        if unexpected_strengths:
            raise ValueError(
                f"Unexpected strength values: {sorted(unexpected_strengths)}"
            )

        df["strength"] = pd.Categorical(
            df["strength"],
            categories=ORDER_STRENGTH,
            ordered=True
        )

        strength_coverage = (
            df.groupby(["tool", "strength"], observed=False)
            .size()
            .unstack(fill_value=0)
            .reindex(columns=ORDER_STRENGTH)
        )

        if (strength_coverage == 0).any().any():
            missing = (
                strength_coverage
                .eq(0)
                .stack()
            )
            missing = missing[missing].index.tolist()
            raise ValueError(f"Missing combinations from tool and strength: {missing}")

    def _validate_data_quantity(self, df: pd.DataFrame):
        cell_counts = df.groupby(
            self.factors,
            observed=True
        ).size()

        self._logger.info("Validate data quantity of:    %s %s", self._threading, self._mode)
        self._logger.info("Observations:                 %d", len(df))
        self._logger.info("Configurations:               %d", len(cell_counts))
        self._logger.info("Min trials per configuration: %d", cell_counts.min())
        self._logger.info("Max trials per configuration: %d", cell_counts.max())
        missing = df[self.base_column].isna().sum()
        self._logger.info("Missing response values:      %d", missing)
        if missing:
            raise ValueError("Missing response values: %d" % missing)
        self._logger.info("Tools:                        %s", ", ".join(df["tool"].unique()))

    @abstractmethod
    def _calculate_base(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def _calculate_statistics(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        pass

    def _create_rank_file(self, effect_ranking_df: pd.DataFrame, input_file: Path) -> None:
        pass

    @abstractmethod
    def _create_means_ci_files(self, means_ci: dict, input_file: Path) -> None:
        pass

    @abstractmethod
    def _post_process(self, effect_ranking_df: pd.DataFrame, input_file: Path) -> None:
        pass

    def _print_fitted_models(self, kind: str, two_way_model, three_way_model, full_model):
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

    def _fit_two_way_model(self, data: pd.DataFrame, response: str):
        formula = self._build_interaction_formula(response, 2, self.factors)

        return smf.ols(
            formula=formula,
            data=data
        ).fit()

    def _fit_higher_order_models(self, data: pd.DataFrame, response: str):
        three_way_formula = self._build_interaction_formula(response, 3, self.factors)
        three_way_model = smf.ols(
            formula=three_way_formula,
            data=data
        ).fit()

        full_formula = self._build_full_formula(response, self.factors)
        full_model = smf.ols(
            formula=full_formula,
            data=data
        ).fit()

        return three_way_model, full_model

    def _print_anova_hc3(self, anova_hc3_df: pd.DataFrame, std_error_description_df: pd.DataFrame):
        self._logger.info("HC3/conventional standard-error ratios:")
        headers = ["HC3 to conventional", "ratio"]
        table_str = tabulate_pint(
            std_error_description_df,
            headers=headers,
        )
        self._logger.info(table_str)

        headers = ["factor", "sum_sq", "df", "F", "PR(>F)"]
        table_str = tabulate.tabulate(anova_hc3_df,
                                      headers=headers,
                                      )
        self._logger.info("ANOVA HC3")
        self._logger.info(table_str)

    def _calculate_anova_hc3(self, model: RegressionResultsWrapper):
        # Create robust inference for decompression effects.
        # Reason: HC3 protects significance tests and confidence intervals against unequal
        # configuration variability and keeps the method consistent with compression.
        # Test which factors and interactions affect mean compression throughput
        anova_hc3 = anova_lm(
            model,
            typ=3,
            robust="hc3"
        )
        std_error_description_df = self._anova_hc3(model)
        return anova_hc3, std_error_description_df

    def _anova_hc3(self, model: RegressionResultsWrapper) -> pd.DataFrame:
        # Create an HC3-robust results object
        model_hc3 = (
            model
            .get_robustcov_results(cov_type="HC3")
        )

        # Compare conventional and HC3 standard errors
        standard_error_comparison = pd.DataFrame(
            {
                "conventional_se": np.asarray(
                    model.bse
                ),
                "hc3_se": np.asarray(
                    model_hc3.bse
                ),
            },
            index=model.params.index,
        )

        standard_error_comparison["hc3_to_conventional_ratio"] = (
                standard_error_comparison["hc3_se"]
                / standard_error_comparison["conventional_se"]
        )

        std_error_description = standard_error_comparison[
            "hc3_to_conventional_ratio"
        ].describe(
            percentiles=[0.10, 0.25, 0.50, 0.75, 0.90]
        )
        return std_error_description.to_frame()

    def _print_anova_effect_rankings(self, effect_table: pd.DataFrame):
        headers = ["Rank", "Effect", "effect type", "df", "sum²", "eta²", "ω²", "total variation (%)", "explained variation (%)"]
        colalign = ["global"] * len(headers)
        colalign[2] = "right"
        table_str = tabulate_pint(
            effect_table,
            showindex=False,
            headers=headers,
            colalign=colalign,
            floatfmt=["", "", "", "",  ".8f", ".8f", ".8f", ".2f", ".2f"]
        )
        self._logger.info("")
        self._logger.info("Effect-size ranking:")
        self._logger.info(table_str)

    def _print_anova_effect_decompositions(self, decompositions: pd.DataFrame):
        self._logger.info("")
        self._logger.info("Decomposition checks:")
        headers = ["Entry", "value"]
        table_str = tabulate_pint(
            decompositions,
            showindex=False,
            headers=headers,
        )
        self._logger.info(table_str)

    def _calculate_anova_effect_rankings(self, model: RegressionResultsWrapper):
        # Rank factors and interactions by explained variation by omega^2.
        # Reason: RQ asks which effects explain the most variation, which \(p\)-values cannot determine.
        # Calculate conventional Type III sums of squares
        # The F-tests and p-values from this table are not used
        anova_ss = anova_lm(
            model,
            typ=3,
        )

        effect_table = (
            anova_ss
            .drop(index=["Intercept", "Residual"], errors="ignore")
            [["sum_sq", "df"]]
            .copy()
        )

        # Calculate corrected total, model, and residual sums of squares
        response = np.asarray(
            model.model.endog,
            dtype=float,
        )

        ss_total = np.sum(
            np.square(response - response.mean())
        )

        ss_residual = np.sum(
            np.square(model.resid)
        )

        ss_model = ss_total - ss_residual
        mse_residual = model.mse_resid

        # Calculate additive and bias-adjusted effect sizes
        effect_table["eta_squared"] = (
                effect_table["sum_sq"] / ss_total
        )

        effect_table["omega_squared"] = (
                (
                        effect_table["sum_sq"]
                        - effect_table["df"] * mse_residual
                )
                / (ss_total + mse_residual)
        )

        effect_table["percent_total_variation"] = (
                100 * effect_table["eta_squared"]
        )

        effect_table["percent_explained_variation"] = (
                100 * effect_table["sum_sq"] / ss_model
        )

        effect_table["effect"] = [
            name
            for name in effect_table.index
        ]

        effect_table.index.name = "effect"

        # Determine interaction order
        effect_table["effect_order"] = [
            name.count(":") + 1
            for name in effect_table.index
        ]
        effect_table["effect_type"] = effect_table["effect_order"].map(lambda e: EFFECT_NAMES[e])

        # Rank effects using omega squared
        effect_table["rank"] = (
            effect_table["omega_squared"]
            .rank(method="min", ascending=False)
            .astype(int)
        )

        effect_table = effect_table.sort_values(
            "omega_squared",
            ascending=False,
        )

        ranked_effects_df = effect_table[
            [
                "rank",
                "effect",
                "effect_type",
                "df",
                "sum_sq",
                "eta_squared",
                "omega_squared",
                "percent_total_variation",
                "percent_explained_variation",
            ]
        ]
        ranked_effects_df['df'] = ranked_effects_df['df'].astype(int)

        # Verify that the effect sums of squares form an additive decomposition
        effect_ss_sum = effect_table["sum_sq"].sum()
        decomposition_error = (
                ss_total - effect_ss_sum - ss_residual
        )

        decompositions = pd.DataFrame(
            {
                "entry": [
                    "Corrected total sum of squares",
                    "Model sum of squares",
                    "Effect sum of squares",
                    "Residual sum of squares",
                    "Decomposition error",
                    "Sum of effect eta squared",
                    "Model R²",
                    "Residual variation",
                ],
                "value": [
                    ss_total,
                    ss_model,
                    effect_ss_sum,
                    ss_residual,
                    decomposition_error,
                    effect_table["eta_squared"].sum(),
                    model.rsquared,
                    ss_residual / ss_total,
                ],
            }
        )

        return ranked_effects_df, decompositions

    def _print_order_summary(self, order_summary: pd.DataFrame):
        self._logger.info("")
        self._logger.info("Variation by effect order:")
        headers = ["Type", "sum²", "eta²", "ω²", "total variation (%)"]
        table_str = tabulate_pint(
            order_summary,
            headers=headers,
            floatfmt=["", ".8f", ".8f", ".8f", ".2f"]
        )
        self._logger.info(table_str)

    def _calculate_order_summary(self, effect_table_df: pd.DataFrame) -> pd.DataFrame:
        # Summarize effects by interaction order
        order_summary = (
            effect_table_df
            .groupby("effect_type", observed=True)
            .agg(
                sum_sq=("sum_sq", "sum"),
                eta_squared=("eta_squared", "sum"),
                omega_squared=("omega_squared", "sum"),
            )
        )

        order_summary["percent_total_variation"] = (
                100 * order_summary["eta_squared"]
        )

        order_summary = order_summary.sort_values(
            "omega_squared",
            ascending=False,
        )

        return order_summary

    def _extract_relevant_factors(self, ranked: pd.DataFrame, target: int) -> list:
        relevant = []

        first_interaction = None
        for idx, _ in ranked.iterrows():
            factors = self._extract_factors(str(idx))
            if isinstance(factors, tuple):
                first_interaction = factors
                break

        n = (ranked["omega_squared"].cumsum() < (target / 100)).sum() + 1
        relevant_rows = ranked.iloc[:n]
        for idx, _ in relevant_rows.iterrows():
            factors = self._extract_factors(str(idx))
            relevant.append(factors)

        if first_interaction not in relevant:
            relevant.append(first_interaction)

        return relevant

    def _extract_factors(self, entry: str) -> str | tuple:
        regex = r"C\(\s*([^,]+)\s*,"
        factors = []
        for part in entry.split(":"):
            search_result = re.search(regex, part)
            factor = search_result.group(1).strip()
            factors.append(factor)
        if len(factors) == 1:
            return factors[0]
        return tuple(factors)

