import logging
from pathlib import Path

import pandas as pd

from data_processor.processor import Processor


class RankingMerger(Processor):
    def __init__(self, resources: Path):
        super().__init__(resources)
        self._logger = logging.getLogger(self.__class__.__name__)

    def merge(self, rank_files: list[Path]):
        frames = []
        for rank_file in rank_files:
            df = self._frameio.load(rank_file)
            frames.append(df)
        merged_df = pd.concat(frames)
        print(merged_df)
        base = rank_files[0].name[:2]
        self._create_ranking_tex(merged_df, f"{base}_ranking_stability_merged.tex")

    def _create_ranking_tex(self, df: pd.DataFrame, filename: str):
        tex_file = self._resources / filename
        self._logger.info("Generate: %s", tex_file)
        latex = self._build_ranking_lines(df)
        with tex_file.open(mode="w", encoding="UTF_8") as f:
            f.write(latex)

    def _build_ranking_lines(self, df: pd.DataFrame) -> str:
        df.columns = pd.MultiIndex.from_tuples([
            ("", "Mode"),
            ("", "Threading"),
            ("", "Utilities "),
            ("", "Comparison"),
            ("", "Pairs"),
            (r"$\tau_b$", "Median"),
            (r"$\tau_b$", "Min"),
            (r"$\tau_b$", "Max"),
            ("Reversals", "Count"),
            ("Reversals", r"\%"),
        ])

        int_format = "S[table-format=2]"
        float_format = "S[round-mode=places, round-precision=3, table-format=1.3]"
        percent_format = "S[round-mode=places, round-precision=2, table-format=2.2]"
        range_format = "S[table-format=1.3, table-auto-round]"
        range_format2 = "S[table-format=1, table-auto-round]"
        #range_format = "S[round-mode=places, round-precision=3], table-format=1.3]"
                     #"table-fixed-width=true, table-column-width=0.9cm, table-text-alignment=center]")
        latex = (
            df.style
            .hide(axis="index")
            .format_index(escape=None, axis=1)
            .to_latex(
                siunitx=True,
                hrules=True,
                multicol_align="c",
                column_format=(
                        "l" +
                        "c" +
                        int_format +
                        "c" +
                        int_format +
                        float_format +
                        range_format +
                        range_format2 +
                        int_format +
                        percent_format
                ),
            )
        )

        lines = latex.splitlines()
        toprule_index = next(
            i for i, line in enumerate(lines)
            if line.strip() == r"\toprule"
        )
        lines.insert(toprule_index + 2, r"\cmidrule(lr){6-8}\cmidrule(lr){9-10}")
        latex = "\n".join(lines)

        return latex
