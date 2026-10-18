import logging
from pathlib import Path

import pandas as pd

from data_processor.processor import Processor
from .effect_names import get_effect_name


class EnergyEfficiencyMerger(Processor):
    def __init__(self, resources: Path):
        super().__init__(resources)
        self._logger = logging.getLogger(self.__class__.__name__)

    def merge(self, rank_files: list[Path]):
        rank1_df = self._load_rank_file(rank_files[0])
        rank2_df = self._load_rank_file(rank_files[1])

        merged_df = self._merge_rank_files(rank1_df, rank2_df)
        # ToDo: limit to the top 95%
        self._create_effect_ranking_tex(merged_df, "merged.tex")

    def _load_rank_file(self, file_name: Path) -> pd.DataFrame:
        df = self._frameio.load(file_name)
        df['df'] = df['df'].astype(int)
        for column in ["omega_squared", "ci_low", "ci_high"]:
            df["%s_percent" % column] = df[column] * 100
        df["cumulative_omega_squared_percent"] = df["omega_squared"].cumsum() * 100
        df = df[["Effect", "df",
                 "Rank",
                 "omega_squared_percent", "ci_low_percent", "ci_high_percent",
                 "cumulative_omega_squared_percent"]]
        return df

    def _merge_rank_files(self, rank1_df: pd.DataFrame, rank2_df: pd.DataFrame) -> pd.DataFrame:
        result = rank1_df.merge(
            rank2_df,
            on=["Effect", "df"],
            how="left",
            suffixes=("", "2"),
        )
        return result

    def _create_effect_ranking_tex(self, df: pd.DataFrame, filename: str):
        tex_file = self._resources / filename
        self._logger.info("Generate: %s", tex_file)

        latex = self._build_effect_ranking_lines(df)
        with tex_file.open(mode="w", encoding="UTF_8") as f:
            f.write(latex)

    def _build_effect_ranking_lines(self, df: pd.DataFrame) -> str:
        df = df.copy()
        df["Effect"] = df["Effect"].map(get_effect_name)

        tex_df = df[["Effect", "df",
                     "Rank", "omega_squared_percent",
                     "ci_low_percent", "ci_high_percent",
                     "Rank2", "omega_squared_percent2",
                     "ci_low_percent2", "ci_high_percent2",
                     ]]

        group1 = "$EE_{total}$"
        group2 = "$EE_{net}$"
        bootstrap = r"\SI{95}{\percent} bootstrap CI"
        tex_df.columns = pd.MultiIndex.from_tuples([
            ("", "", "Effect"),
            ("", "", "df"),

            (group1, "", "Rank"),
            (group1, "", r"$\omega^2$ (\%)"),
            (group1, bootstrap, "Low"),
            (group1, bootstrap, "High"),

            (group2, "", "Rank"),
            (group2, "", r"$\omega^2$ (\%)"),
            (group2, bootstrap, "Low"),
            (group2, bootstrap, "High"),
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
                        "l" +
                        int_format +
                        int_format +
                        percent_format +
                        ci_format +
                        ci_format +
                        int_format +
                        percent_format +
                        ci_format +
                        ci_format
                ),
            )
        )

        lines = latex.splitlines()
        toprule_index = next(
            i for i, line in enumerate(lines)
            if line.strip() == r"\toprule"
        )
        lines.insert(toprule_index + 2, r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}")
        lines.insert(toprule_index + 4, r"\cmidrule(lr){5-6}\cmidrule(lr){9-10}")
        latex = "\n".join(lines)

        return latex
