import logging
from pathlib import Path

import pandas as pd

from data_processor.processor import Processor


def get_host_entry(host: str) -> str:
    if host == "raspi5":
        return r"\gls{pi5}"
    if host == "visionfive2":
        return r"\gls{visionfive2}"
    if host == "radxax4":
        return r"\gls{radxax4}"
    raise RuntimeError("unknown host: %s" % host)


def get_mode(mode: str) -> str:
    if mode == "compress":
        return "Compression"
    if mode == "decompress":
        return "Decompression"
    raise RuntimeError("unknown mode: %s" % mode)


class RankingMerger(Processor):
    def __init__(self, resources: Path):
        super().__init__(resources)
        self._logger = logging.getLogger(self.__class__.__name__)

    def merge(self, rank_files: list[Path]):
        frames = []
        for rank_file in rank_files:
            df = self._frameio.load(rank_file)
            frames.append(df)

        base = self._merge_paths(rank_files[0], rank_files[1])
        modes = self._last_parts(*rank_files)
        pairs = list(zip(frames, modes))
        filename = base.with_suffix(".tex")
        self._create_merged_ranking_report_tex(pairs, filename)

    def _create_merged_ranking_report_tex(self, pairs: list[tuple[pd.DataFrame, str]], filename: Path) -> None:
        self._logger.info("Generate: %s", filename)
        latex = self._build_ranking_pairs(pairs)
        with filename.open(mode="w", encoding="UTF_8") as f:
            f.write("\n".join(latex))

    def _build_ranking_pairs(self, pairs: list[tuple[pd.DataFrame, str]]):
        entries = []
        for df, mode in pairs:
            lines = self._build_ranking_lines(df)
            entries.append((lines, mode))
        first_lines, first_mode = entries[0]
        second_lines, second_mode = entries[1]

        header = first_lines[:4]
        footer = first_lines[-2:]
        tool_count = len(pairs[0][0].columns)

        body1 = first_lines[4:-2]
        body1.insert(0, "\\multicolumn{%d}{l}{\\textit{%s}} \\\\" % (tool_count, get_mode(first_mode)))
        body1.append("\\midrule")

        body2 = second_lines[4:-2]
        body2.insert(0, "\\multicolumn{%d}{l}{\\textit{%s}} \\\\" % (tool_count, get_mode(second_mode)))

        result = header + body1 + body2 + footer
        return result

    def _build_ranking_lines(self, df: pd.DataFrame):
        host_column = "Platform / Rank"
        df = df.rename(columns={"platform_rank": host_column})
        df[host_column] = df[host_column].map(get_host_entry)

        column_formats = ["l"] + ["c"] * 9
        latex = (
            df.style
            .hide(axis="index")
            .to_latex(
                siunitx=True,
                hrules=True,
                column_format="".join(column_formats),
            )
        )

        lines = latex.splitlines()
        return lines

    def _last_parts(self, *paths: Path) -> list[str]:
        return [p.stem.rsplit("_", 1)[-1] for p in paths]

    def _merge_paths(self, a: Path, b: Path) -> Path:
        a_parts = a.stem.split("_")
        b_parts = b.stem.split("_")

        if a.suffix != b.suffix:
            raise ValueError(f"Suffix mismatch: {a.suffix!r} vs {b.suffix!r}")
        if a.parent != b.parent:
            raise ValueError(f"Directory mismatch: {a.parent} vs {b.parent}")
        if a_parts[:-1] != b_parts[:-1]:
            raise ValueError(f"Common prefix differs: {a_parts[:-1]} vs {b_parts[:-1]}")
        if a_parts[-1] == b_parts[-1]:
            raise ValueError(f"Last parts are identical: {a_parts[-1]!r}")

        return a.with_name("_".join(a_parts[:-1]) + a.suffix)
