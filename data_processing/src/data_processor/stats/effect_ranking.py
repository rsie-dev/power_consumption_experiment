import logging
from pathlib import Path

import pandas as pd
import numpy as np
import tabulate

from data_processor.util import tabulate_pint
from .effect_names import get_effect_name, build_effect_names, EFFECT_NAMES


class EffectRanking:
    def __init__(self, factors: list[str]):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._factors = factors

    def print_effect_ranking_table(self, df: pd.DataFrame):
        table_df = df.copy()
        table_df["bootstrap_ci_eta"] = (
                table_df["eta_ci_lower"].map(lambda x: f"{x:.6f}")
                + "–"
                + table_df["eta_ci_upper"].map(lambda x: f"{x:.6f}")
        )
        table_df["bootstrap_ci_omega"] = (
                table_df["omega_ci_lower"].map(lambda x: f"{x:.6f}")
                + "–"
                + table_df["omega_ci_upper"].map(lambda x: f"{x:.6f}")
        )

        table_df = table_df[[
            "rank",
            "effect",
            "effect_type",
            "df",
            "eta_squared",
            "bootstrap_ci_eta",
            "omega_squared",
            "cumulative_omega_squared",
            "bootstrap_ci_omega",
            "HC3 F",
            "HC3 p"
        ]]

        #table_df["effect"] = table_df["effect"].map(get_effect_name)
        headers = ["Rank", "Effect", "Effect type", "df",
                   "eta²", "95% bootstrap CI eta²",
                   "ω²", "Cumulative ω²", "95% bootstrap CI ω²",
                   "HC3 F", "HC3 p"]
        coalign = ["global"] * len(headers)
        coalign[2] = "right"
        table_str = tabulate.tabulate(table_df,
                                      showindex=False,
                                      headers=headers,
                                      colalign=coalign
                                      )
        self._logger.info("Effect ranking table:")
        self._logger.info(table_str)

    def create_effect_rank_stability_tex(self, df: pd.DataFrame, filename: Path):
        self._logger.info("Generate: %s", filename)

        latex = self._build_effect_rank_stability_lines(df)
        with filename.open(mode="w", encoding="UTF_8") as f:
            f.write(latex)

    def _build_effect_rank_stability_lines(self, df: pd.DataFrame) -> str:
        df = df.copy()
        for column in ["eta_squared", "cumulative_eta_squared", "eta_ci_lower", "eta_ci_upper", "omega_squared"]:
            df["%s_percent" % column] = df[column] * 100

        #df["effect"] = df["effect"].map(get_effect_name)

        tex_df = df[["rank",
                     "effect",
                     "eta_squared_percent", "eta_ci_lower_percent", "eta_ci_upper_percent",
                     "omega_squared_percent",
                     "cumulative_eta_squared_percent",
        ]]

        tex_df.columns = pd.MultiIndex.from_tuples([
            ("Rank", ""),
            ("Effect", ""),
            (r"$\eta^2$ (\%)", ""),
            (r"\SI{95}{\percent} bootstrap CI", "Low"),
            (r"\SI{95}{\percent} bootstrap CI", "High"),
            (r"$\omega^2$ (\%)", ""),
            ("Cumulative", r"$\eta^2$ (\%)"),
        ])

        int_format = "S[table-format=2]"
        #float_format = "S[round-mode=places, round-precision=5, table-format=1.5, group-digits=false]"
        percent_format = "S[round-mode=places, round-precision=2, table-format=2.2]"
        ci_format = ("S[round-mode=places, round-precision=2, table-format=2.2, "
                     "table-fixed-width=true, table-column-width=0.9cm, table-text-alignment=center]")
        column_formats = [
            int_format,
            "l",
            percent_format,
            ci_format,
            ci_format,
            percent_format,
            percent_format,
        ]
        latex = (
            tex_df.style
            .hide(axis="index")
            .format_index(escape=None, axis=1)
            .to_latex(
                siunitx=True,
                hrules=True,
                multicol_align="c",
                column_format="".join(column_formats),
            )
        )

        lines = latex.splitlines()
        toprule_index = next(
            i for i, line in enumerate(lines)
            if line.strip() == r"\toprule"
        )
        lines.insert(toprule_index + 2, r"\cmidrule(lr){4-5}")
        latex = "\n".join(lines)

        return latex

    def create_effect_ranking_tex(self, df: pd.DataFrame, filename: Path):
        self._logger.info("Generate: %s", filename)

        latex = self._build_effect_ranking_lines(df)
        with filename.open(mode="w", encoding="UTF_8") as f:
            f.write(latex)

    def _build_effect_ranking_lines(self, df: pd.DataFrame) -> str:
        df = df.copy()
        df['df'] = df['df'].astype(int)
        for column in ["omega_squared", "omega_ci_lower", "omega_ci_upper"]:
            df["%s_percent" % column] = df[column] * 100

        df["Effect"] = df["Effect"].map(get_effect_name)

        tex_df = df[["Rank", "Effect", "df",
                     "omega_squared_percent", "omega_ci_lower_percent", "omega_ci_upper_percent",
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

    def create_effect_ranking_table(self, ranked: pd.DataFrame, confidence_intervals: pd.DataFrame,
                                     anova_hc3: pd.DataFrame) -> pd.DataFrame:
        effect_ranking = (
            ranked[["df", "eta_squared", "omega_squared"]]
            .join(confidence_intervals)
            .join(anova_hc3[["F", "PR(>F)"]])
        )

        # ToDo:
        def clean_effect(term):
            for factor in self._factors:
                term = term.replace(f"C({factor}, Sum)", factor)
            return term.replace(":", " × ")

        effect_ranking["effect"] = [
            clean_effect(term) for term in effect_ranking.index
        ]

        effect_ranking["effect_type"] = [
            EFFECT_NAMES[term.count(":") + 1]
            for term in effect_ranking.index
        ]

        effect_ranking["rank"] = (
            effect_ranking["omega_squared"]
            .rank(ascending=False, method="min")
            .astype(int)
        )

        effect_ranking["HC3 p"] = effect_ranking["PR(>F)"].map(
            lambda p: "<0.001" if p < 0.001 else f"{p:.3f}"
        )

        effect_ranking = effect_ranking.rename(columns={
            "F": "HC3 F"
        })

        effect_ranking = effect_ranking[
            [
                "rank",
                "effect",
                "effect_type",
                "df",
                "eta_squared",
                "eta_ci_lower",
                "eta_ci_upper",
                "omega_squared",
                "omega_ci_lower",
                "omega_ci_upper",
                "HC3 F",
                "HC3 p"
            ]
        ].sort_values("rank")

        effect_ranking["cumulative_eta_squared"] = effect_ranking["eta_squared"].cumsum()
        effect_ranking["cumulative_omega_squared"] = effect_ranking["omega_squared"].cumsum()

        return effect_ranking

    def print_effect_rank_stability_table(self, rank_stability: pd.DataFrame):
        headers = ["Effect", "Point rank", "Modal rank", "Median rank", "Rank 95 low", "Rank 95 high",
                   "Probability same rank", "Probability top 4", "Probability top 5"]
        table_str = tabulate_pint(
            rank_stability,
            headers=headers,
        )
        self._logger.info("Bootstrap rank stability:")
        self._logger.info(table_str)

    def print_effect_adjacent_rank_comparisons_table(self, adjacent_comparisons: pd.DataFrame):
        headers = ["Higher effect", "Lower effect", "Probability higher"]
        table_str = tabulate_pint(
            adjacent_comparisons,
            showindex=False,
            headers=headers,
        )
        self._logger.info("")
        self._logger.info("Adjacent-rank comparisons:")
        self._logger.info(table_str)

    def create_rank_stability_tables(self, bootstrap_omega,
                                     point_omega) -> tuple[pd.DataFrame, pd.DataFrame]:
        factor_axes = range(len(self._factors))
        effect_names, _ = build_effect_names(self._factors, factor_axes)

        n_bootstrap, n_effects = bootstrap_omega.shape
        if n_effects != len(effect_names):
            raise ValueError("The number of effects does not match the bootstrap results")

        # Calculate the point-estimate ranks
        point_order = np.argsort(
            -point_omega,
            kind="stable",
        )

        point_ranks = np.empty(
            n_effects,
            dtype=int,
        )

        point_ranks[point_order] = np.arange(
            1,
            n_effects + 1,
        )

        # Calculate ranks for every bootstrap sample
        bootstrap_order = np.argsort(
            -bootstrap_omega,
            axis=1,
            kind="stable",
        )

        bootstrap_ranks = np.empty_like(
            bootstrap_order,
            dtype=int,
        )

        bootstrap_ranks[
            np.arange(n_bootstrap)[:, np.newaxis],
            bootstrap_order
        ] = np.arange(1, n_effects + 1)

        # Calculate modal ranks
        modal_ranks = np.array(
            [
                np.bincount(
                    bootstrap_ranks[:, effect_index],
                    minlength=n_effects + 1,
                )[1:].argmax() + 1
                for effect_index in range(n_effects)
            ]
        )

        # Calculate central 95% bootstrap rank ranges
        rank_lower, rank_upper = np.quantile(
            bootstrap_ranks,
            [0.025, 0.975],
            axis=0,
            method="nearest",
        )

        # Create the rank-stability summary
        rank_stability = pd.DataFrame(
            {
                "point_rank": point_ranks,
                "modal_rank": modal_ranks,
                "median_rank": np.median(
                    bootstrap_ranks,
                    axis=0,
                ),
                "rank_95_low": rank_lower.astype(int),
                "rank_95_high": rank_upper.astype(int),
                "probability_same_rank": np.mean(
                    bootstrap_ranks
                    == point_ranks[np.newaxis, :],
                    axis=0,
                ),
                "probability_top_4": np.mean(
                    bootstrap_ranks <= 4,
                    axis=0,
                ),
                "probability_top_5": np.mean(
                    bootstrap_ranks <= 5,
                    axis=0,
                ),
            },
            index=pd.Index(
                effect_names,
                name="effect",
            ),
        )

        rank_stability = rank_stability.sort_values(
            "point_rank"
        )

        # Compare adjacent effects in the point-estimate ranking
        adjacent_comparisons = []
        for position in range(n_effects - 1):
            higher_index = point_order[position]
            lower_index = point_order[position + 1]

            probability_higher = np.mean(
                bootstrap_omega[:, higher_index]
                > bootstrap_omega[:, lower_index]
            )

            adjacent_comparisons.append(
                {
                    "higher_effect": effect_names[higher_index],
                    "lower_effect": effect_names[lower_index],
                    "probability_higher": probability_higher,
                }
            )

        adjacent_comparisons = pd.DataFrame(
            adjacent_comparisons
        )

        return rank_stability, adjacent_comparisons
