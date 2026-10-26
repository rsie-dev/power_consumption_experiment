import logging
from pathlib import Path
import re
from contextlib import contextmanager, nullcontext
from abc import abstractmethod

import numpy as np
import pandas as pd
from statsmodels.stats.anova import anova_lm
from statsmodels.regression.linear_model import RegressionResultsWrapper
import tabulate

from data_processor.processor import Processor
from data_processor.constants import ORDER_STRENGTH
from data_processor.util import tabulate_pint
from .effect_names import EFFECT_NAMES
from .drift_analyzer import DriftAnalyzer
from .model_compare import ModelCompare
from .bootstrap_ci_calculator import BootstrapCiCalculator
from .effect_ranking import EffectRanking
from .stat_model import StatModel
from .factor_means import FactorMeans
from .diagnostic import Diagnostic
from .extreme_finder import StudentizedResidualExtremeFinder
from .residual_variance import ResidualVariance


class BaseStatistics(Processor):
    DRIFT_THRESHOLD = 1.0  # percentage points across runs 1–30

    def __init__(self, resources: Path, logfile: bool, n_boot: int, seed: int | None,
                 threading: str, mode: str, create_tex: bool):
        super().__init__(resources)
        self._logger = logging.getLogger(self.__class__.__name__)
        self._logfile = logfile
        self._threading = threading
        self._mode = mode
        self._create_tex = create_tex
        self._bootstrap_ci_calculator = BootstrapCiCalculator(self.factors, n_boot, seed, self.base_column)
        self._effect_ranking = EffectRanking(self.factors)
        self._stat_model = StatModel()

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

    @property
    @abstractmethod
    def display_unit (self) -> str:
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
        self._log_section("1. Preparation")
        self._logger.info("load data")
        df = self._frameio.load(input_file)

        self._log_section("1.1 Define subset")
        self._logger.info("Evaluate: %s %s", self._threading, self._mode)
        df = self._filter_data(df)

        self._log_section("1.2 Validate data")
        self._validate_data(df)

        self._log_section("1.3 Calculate log target")
        df = self._calculate_base(df)

        self._log_section("1.4 Check temporal drift")
        self._check_temporal_drift(df)

        effect_ranking_df, means_ci = self._calculate_statistics(df)
        self._create_rank_file(effect_ranking_df, input_file)
        self._create_means_ci_files(means_ci, input_file)

        self._post_process(effect_ranking_df, input_file)

    def _check_temporal_drift(self, df: pd.DataFrame):
        drift_analyzer = DriftAnalyzer(self.factors, self.DRIFT_THRESHOLD, self._threading, self._mode, self.base_column)
        inconclusive_hosts = drift_analyzer.validate_temporal_drift(df)
        for host in inconclusive_hosts:
            sensitivity_model = drift_analyzer.calculate_run_order_effect(host, df)
            full_model = self.create_full_model(df)
            model_compare = ModelCompare()
            comparison = model_compare.compare(sensitivity_model, full_model)
            model_compare.print_high_impact_terms(host, sensitivity_model, full_model, comparison)

    @abstractmethod
    def create_full_model(self, df: pd.DataFrame) -> RegressionResultsWrapper:
        pass

    def _filter_data(self, df: pd.DataFrame):
        df = df[df["mode"] == self._mode]
        df = df[df["threading"] == self._threading]
        return df

    def _validate_data(self, df: pd.DataFrame):
        df = df.copy()
        self._validate_factor_values(df)
        self._validate_tool_strength(df)
        self._validate_data_quantity(df)

    def _validate_factor_values(self, df: pd.DataFrame):
        trial_key = self.factors + ["run"]
        targets = [self.base_column]

        # Count duplicate rows after the first occurrence of each trial key.
        self._logger.info("Duplicate trial keys: %s", df.duplicated(subset=trial_key).sum())

        # Count missing factor cells, including blank strings.
        factor_values = df[self.factors].replace(r"^\s*$", pd.NA, regex=True)
        self._logger.info("Missing factor values: %s", factor_values.isna().sum().sum())

        # Check numeric magnitudes of the unit-aware target columns.
        for target in targets:
            values = df[target].pint.magnitude
            self._logger.info("Nonpositive %s: %s", target, (values <= 0).sum())

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

        self._logger.info("Observations:                 %d", len(df))
        self._logger.info("Configurations:               %d", len(cell_counts))
        self._logger.info("Min trials per configuration: %d", cell_counts.min())
        self._logger.info("Max trials per configuration: %d", cell_counts.max())
        missing = df[self.base_column].isna().sum()
        self._logger.info("Missing response values:      %d", missing)
        if missing:
            raise ValueError("Missing response values: %d" % missing)
        self._logger.info("Tools:                        %s", ", ".join(df["tool"].unique()))

    def _check_factorial_completeness(self, df: pd.DataFrame) -> None:
        df = df.copy()
        factor_values = df[self.factors].replace(r"^\s*$", pd.NA, regex=True)

        # Infer each factor's levels from the available data.
        levels = [factor_values[col].dropna().unique() for col in self.factors]

        # All possible combinations of factor levels.
        expected = pd.MultiIndex.from_product(levels, names=self.factors)

        # Unique combinations actually present; exclude incomplete factor rows.
        observed = pd.MultiIndex.from_frame(
            factor_values.dropna().drop_duplicates()
        )

        missing = expected.difference(observed)

        self._logger.info("Expected cells:               %d", len(expected))
        self._logger.info("Observed cells:               %d", len(observed))
        self._logger.info("Missing cells:                %d", len(missing))

        if len(missing):
            self._logger.info("")
            self._logger.info("Missing combinations:")
            missing_df = missing.to_frame()
            table_str = tabulate_pint(
                missing_df,
                showindex=False,
                headers=self.factors,
            )
            self._logger.info(table_str)

    @abstractmethod
    def _calculate_base(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    def _calculate_statistics(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        self._log_section("2. Modeling")

        self._log_section("2.1 Check factorial completeness")
        self._check_factorial_completeness(df)

        self._log_section("2.2 Fit full factorial model")
        full_model = self.create_full_model(df)
        self._stat_model.print_fitted_model("full", full_model)

        # Now check:
        # 1. residual variance across fitted values -> UI
        # 2. Check residual normality and extreme tails -> UI

        self._log_section("2.3 Check residuals")
        residual_variance = ResidualVariance(self.factors)
        residual_variance.calculate_residual_variance(df, full_model)
        zero_variance, configuration_details = residual_variance.calculate_zero_variance(df,
                                                                                         full_model, self.base_column)
        residual_variance.print_residual_variance(zero_variance, configuration_details, self.base_column)

        self._log_section("2.4 Investigate extreme observations")
        diagnostic = Diagnostic(self.factors)
        diagnostics = diagnostic.find_observations_that_strongly_influence_model(full_model)
        diagnostic.print_observations_that_strongly_influence_model(diagnostics)

        # Determine whether extreme residuals are isolated or concentrated in particular configurations.
        extreme_finder = StudentizedResidualExtremeFinder(self.factors)
        extremes, criterion = extreme_finder.find_extremes(df, full_model)
        extreme_finder.print_extremes(extremes, criterion)

        self._log_section("3. Inference")
        self._log_section("3.1 Calculate robust ANOVA")
        anova_hc3, std_error_description_df = self._calculate_anova_hc3(full_model)
        self._print_anova_hc3(anova_hc3, std_error_description_df)

        ranked_effects_df, decompositions = self._calculate_anova_effect_rankings(full_model)
        self._print_anova_effect_rankings(ranked_effects_df)
        self._print_anova_effect_decompositions(decompositions)

        relevant_percent = 95
        relevant_factors = self._extract_relevant_factors(ranked_effects_df, relevant_percent)
        self._logger.info("")
        self._logger.info("Terms cumulatively explaining at least %d%% of variation: %s",
                          relevant_percent, " > ".join([str(f) for f in relevant_factors]))

        # Make later statistical inference robust to unequal residual variance.
        # Reason: HC3 protects significance tests and confidence intervals against unequal configuration
        # variability and keeps the method consistent with compression.
        # -> use HC3 version for confidence intervals and significance tests.
        full_model_hc3 = full_model.get_robustcov_results(
            cov_type="HC3"
        )

        self._log_section("3.2 Estimate effects")
        factor_means = FactorMeans(self.factors, self.base_column, self.display_unit)
        #factor_means.show_factor_means(df, relevant_factors)
        means_ci = factor_means.show_factor_means_ci(df, full_model, full_model_hc3, relevant_factors)

        self._log_section("3.3 Quantify importance")
        order_summary = self._calculate_order_summary(ranked_effects_df)
        self._print_order_summary(order_summary)

        # Check that the top confidence intervals do not overlap
        self._log_section("3.4 Bootstrap importance")
        validation_results = self._bootstrap_ci_calculator.validate_data(df, full_model)
        self._bootstrap_ci_calculator.print_validation_results(validation_results )
        confidence_intervals, bootstrap_results, bootstrap_omega, point_omega = (
            self._bootstrap_ci_calculator.calculate_ci(df)
        )
        self._bootstrap_ci_calculator.print_bootstrap_infos()
        self._bootstrap_ci_calculator.print_ci(bootstrap_results)

        rank_stability, adjacent_comparisons = self._effect_ranking.create_rank_stability_tables(bootstrap_omega,
                                                                                                 point_omega)
        self._effect_ranking.print_effect_rank_stability_table(rank_stability)
        self._effect_ranking.print_effect_adjacent_rank_comparisons_table(adjacent_comparisons)

        self._log_section("4. Results")
        self._log_section("4.1 Rank terms")
        effect_rankings = self._effect_ranking.create_effect_ranking_table(ranked_effects_df,
                                                                           confidence_intervals, anova_hc3)
        self._effect_ranking.print_effect_ranking_table(effect_rankings)


        #factor_means_values_df = self._create_factor_means_table(df, full_model, full_model_hc3)
        #self._print_factor_table(factor_means_values_df)

        effect_rankings_stability = (
            effect_rankings
            .join(rank_stability[["rank_95_low", "rank_95_high", "probability_same_rank"]])
        )

        return effect_rankings_stability, means_ci
    def _create_rank_file(self, effect_ranking_df: pd.DataFrame, input_file: Path) -> None:
        pass

    @abstractmethod
    def _create_means_ci_files(self, means_ci: dict, input_file: Path) -> None:
        pass

    @abstractmethod
    def _post_process(self, effect_ranking_df: pd.DataFrame, input_file: Path) -> None:
        pass

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
        headers = ["Rank", "Effect", "effect type", "df", "sum²", "eta²", "ω²",
                   "total variation (%)", "explained variation (%)"]
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
