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
from .effect_names import get_effect_name


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
                f"Unerwartete strength-Werte: {sorted(unexpected_strengths)}"
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

    def _print_anova_table(self, kind: str, ranked: pd.DataFrame):
        headers = ["Factor", "DF", "ω²", "ω² %"]
        table_str = tabulate.tabulate(ranked,
                                      headers=headers,
                                      )
        self._logger.info("")
        extra = f" {kind}:" if kind else ":"
        self._logger.info("ANOVA rank table%s", extra)
        self._logger.info(table_str)

    def _calculate_anova_table(self, model: RegressionResultsWrapper):
        # Rank factors and interactions by explained variation by omega^2.
        # Reason: RQ asks which effects explain the most variation, which \(p\)-values cannot determine.
        anova_ss = anova_lm(model, typ=3)
        mse = (
                anova_ss.loc["Residual", "sum_sq"]
                / anova_ss.loc["Residual", "df"]
        )
        y = model.model.endog
        ss_total = np.sum((y - y.mean()) ** 2)
        effects = anova_ss.drop(
            index=["Intercept", "Residual"]
        ).copy()
        effects["omega_squared"] = (
                                           effects["sum_sq"] - effects["df"] * mse
                                   ) / (ss_total + mse)
        effects["omega_squared"] = effects["omega_squared"].clip(lower=0)
        ranked = effects[["df", "omega_squared"]].sort_values("omega_squared", ascending=False)
        ranked["omega_squared_percent"] = ranked["omega_squared"] * 100
        return effects, ranked

    def _extract_relevant_factors(self, ranked: pd.DataFrame, target: int) -> list:
        relevant = []

        first_interaction = None
        for idx, _ in ranked.iterrows():
            factors = self._extract_factors(str(idx))
            if isinstance(factors, tuple):
                first_interaction = factors
                break

        n = (ranked["omega_squared_percent"].cumsum() < target).sum() + 1
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

    def _print_effect_ranking_table(self, table_df: pd.DataFrame):
        table_df = table_df.drop(columns=["ci_low", "ci_high"])
        table_df["Effect"] = table_df["Effect"].map(get_effect_name)
        headers = ["Rank", "Effect", "Effect type", "df", "ω²", "Cumulative ω² (%)",
                   "95% bootstrap CI", "HC3 F", "HC3 p"]
        coalign = ["global"] * len(headers)
        coalign[2] = "right"
        table_str = tabulate.tabulate(table_df,
                                      showindex=False,
                                      headers=headers,
                                      colalign=coalign
                                      )
        self._logger.info("Effect ranking table for: %s %s", self._threading, self._mode)
        self._logger.info(table_str)

    def _create_effect_ranking_tex(self, df: pd.DataFrame, filename: str):
        tex_file = self._resources / filename
        self._logger.info("Generate: %s", tex_file)

        latex = self._build_effect_ranking_lines(df)
        with tex_file.open(mode="w", encoding="UTF_8") as f:
            f.write(latex)

    def _build_effect_ranking_lines(self, df: pd.DataFrame) -> str:
        df = df.copy()
        df['df'] = df['df'].astype(int)
        for column in ["omega_squared", "ci_low", "ci_high"]:
            df["%s_percent" % column] = df[column] * 100

        df["Effect"] = df["Effect"].map(get_effect_name)

        tex_df = df[["Rank", "Effect", "df",
                     "omega_squared_percent", "ci_low_percent", "ci_high_percent",
                     "cumulative_omega_squared_percent"]]

        tex_df.columns = pd.MultiIndex.from_tuples([
            ("Rank", ""),
            ("Effect", ""),
            ("df", ""),
            (r"$\omega^2$ (\%)", ""),
            (r"\SI{95}{\percent} bootstrap CI", "Low"),
            (r"\SI{95}{\percent} bootstrap CI", "High"),
            ("Cumulative", r"$\omega^2$ (\%)"),
        ])

        int_format = "S[table-format=2]"
        percent_format = "S[round-mode=places, round-precision=2, table-format=2.2]"
        ci_format = ("S[round-mode=places, round-precision=2, table-format=2.2, "
                     "table-fixed-width=true, table-column-width=0.9cm, table-text-alignment=center]")
        latex = (
            tex_df.style
            .hide(axis="index")
            .format_index(escape=None, axis=1)
            .to_latex(
                siunitx=True,
                hrules=True,
                multicol_align="c",
                column_format=(
                        int_format +
                        "l" +
                        int_format +
                        percent_format +
                        ci_format +
                        ci_format +
                        percent_format
                ),
            )
        )

        lines = latex.splitlines()
        toprule_index = next(
            i for i, line in enumerate(lines)
            if line.strip() == r"\toprule"
        )
        lines.insert(toprule_index + 2, r"\cmidrule(lr){5-6}")
        latex = "\n".join(lines)

        return latex

    def _create_effect_ranking_table(self, effects: pd.DataFrame, confidence_intervals: pd.Series,
                                     anova_hc3: pd.DataFrame):
        effect_ranking = (
            effects[["df", "omega_squared"]]
            .join(confidence_intervals)
            .join(anova_hc3[["F", "PR(>F)"]])
        )

        def clean_effect(term):
            for factor in self.factors:
                term = term.replace(f"C({factor}, Sum)", factor)
            return term.replace(":", " × ")

        type_names = {
            1: "Main effect",
            2: "Two-way interaction",
            3: "Three-way interaction",
            4: "Four-way interaction"
        }

        effect_ranking["Effect"] = [
            clean_effect(term) for term in effect_ranking.index
        ]

        effect_ranking["Effect type"] = [
            type_names[term.count(":") + 1]
            for term in effect_ranking.index
        ]

        effect_ranking["Rank"] = (
            effect_ranking["omega_squared"]
            .rank(ascending=False, method="min")
            .astype(int)
        )

        effect_ranking["95% bootstrap CI"] = (
                effect_ranking["ci_low"].map(lambda x: f"{x:.6f}")
                + "–"
                + effect_ranking["ci_high"].map(lambda x: f"{x:.6f}")
        )

        effect_ranking["HC3 p"] = effect_ranking["PR(>F)"].map(
            lambda p: "<0.001" if p < 0.001 else f"{p:.3f}"
        )

        effect_ranking = effect_ranking.rename(columns={
            "F": "HC3 F"
        })

        effect_ranking = effect_ranking[
            [
                "Rank",
                "Effect",
                "Effect type",
                "df",
                "omega_squared",
                "95% bootstrap CI",
                "ci_low",
                "ci_high",
                "HC3 F",
                "HC3 p"
            ]
        ].sort_values("Rank")

        effect_ranking["cumulative_omega_squared_percent"] = (
                effect_ranking["omega_squared"].cumsum() * 100
        )
        effect_ranking.insert(5, "cumulative_omega_squared_percent",
                              effect_ranking.pop("cumulative_omega_squared_percent"))

        return effect_ranking
