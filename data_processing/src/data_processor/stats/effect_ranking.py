import logging
from pathlib import Path

import pandas as pd
import tabulate

from .effect_names import get_effect_name, EFFECT_NAMES


class EffectRanking:
    def __init__(self, factors: list[str]):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._factors = factors

    def print_effect_ranking_table(self, table_df: pd.DataFrame):
        table_df = table_df.drop(columns=["omega_ci_lower", "omega_ci_upper"])
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
        self._logger.info("Effect ranking table:")
        self._logger.info(table_str)

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
            ranked[["df", "omega_squared"]]
            .join(confidence_intervals)
            .join(anova_hc3[["F", "PR(>F)"]])
        )

        def clean_effect(term):
            for factor in self._factors:
                term = term.replace(f"C({factor}, Sum)", factor)
            return term.replace(":", " × ")

        effect_ranking["Effect"] = [
            clean_effect(term) for term in effect_ranking.index
        ]

        effect_ranking["Effect type"] = [
            EFFECT_NAMES[term.count(":") + 1]
            for term in effect_ranking.index
        ]

        effect_ranking["Rank"] = (
            effect_ranking["omega_squared"]
            .rank(ascending=False, method="min")
            .astype(int)
        )

        effect_ranking["95% bootstrap CI"] = (
                effect_ranking["omega_ci_lower"].map(lambda x: f"{x:.6f}")
                + "–"
                + effect_ranking["omega_ci_upper"].map(lambda x: f"{x:.6f}")
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
                "omega_ci_lower",
                "omega_ci_upper",
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
