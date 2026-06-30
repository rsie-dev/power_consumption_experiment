import logging
from pathlib import Path

import tabulate
import pandas as pd

from data_processor.util import FrameIO
from data_processor.data_set import DataSet, dataset_from_str


class CompressionRatio:
    TOOL_ORDER = ["gzip", "pigz", "bzip2", "lbzip2", "bzip3", "xz", "lz4", "lzop", "zstd", "brotli"]

    def __init__(self, resources: Path):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._resources = resources

    def process(self, used_energy_file: Path, create_tex: bool, no_tool: list, no_dataset: list):
        frameio = FrameIO()
        self._logger.debug("loading %s", used_energy_file)
        df = frameio.load(used_energy_file)
        first_host = df.loc[0, "host"]
        df = df[df["host"] == first_host]
        df = df[df["run"] == 1]
        df = df[~df["tool"].isin(no_tool)]
        df = df[~df["dataset"].isin(no_dataset)]
        #self._validate_multi(df)
        df["compression_ratio"] = df.apply(
            lambda row: dataset_from_str(row["dataset"]).value / row["size"],
            axis=1,
        )
        df["compression_ratio"] = df["compression_ratio"].astype(float)

        self._process_threading(used_energy_file, df, "single")
        self._process_threading(used_energy_file, df, "multi")

        if create_tex:
            self._process_tex(used_energy_file, df)

    def _process_threading(self, used_energy_file: Path, df: pd.DataFrame, threading: str):
        df = df[df["threading"] == threading]

        result_df = (
            df.pivot(
                index=["dataset", "strength"],
                columns="tool",
                values="compression_ratio"
            )
            .reset_index()
        )
        result_df.columns.name = None

        def f_map(str_ds):
            return self._get_data_file(dataset_from_str(str_ds))

        result_df = result_df.sort_values("dataset", key=lambda s: s.map(f_map))

        table_entries = []
        tool_names = result_df.columns.drop(["dataset", "strength"]).tolist()
        tool_names = sorted(tool_names, key=lambda x: self.TOOL_ORDER.index(x))
        for _, row in result_df.iterrows():
            dataset = row["dataset"]
            strength = row["strength"]
            values = [row[tool] for tool in tool_names]
            table_entries.append([dataset, strength] + values)

        headers = ["dataset", "strength"] + tool_names
        table_str = tabulate.tabulate(table_entries,
                                      headers=headers,
                                      tablefmt="simple"
                                      )
        print("%s entries:" % threading)
        print(table_str)
        cr_file = self._resources / ("cr_%s_%s" % (threading, used_energy_file.stem.removeprefix("used_energy_")) + ".csv")
        frameio = FrameIO()
        frameio.persist(result_df, cr_file)

    def _validate_multi(self, df):
        print(df)
        group_cols = ["host", "tool", "dataset", "mode", "strength"]
        valid_groups = (
            df.groupby(group_cols)["threading"]
            .nunique()
            .loc[lambda x: x == 2]
            .index
        )
        df = df.set_index(group_cols).loc[valid_groups].reset_index()
        df = df.drop(columns=["run", "real", "duration", "average_power", "energy"])
        print(df)

        size_mismatch = (
            df.groupby(group_cols)["size"]
            .nunique()
            .loc[lambda x: x > 1]
            .index
        )

        result = (
            df.set_index(group_cols)
            .loc[size_mismatch]
            .reset_index()
        )

        unique_tools = df["tool"].unique()
        mismatch_tools = result["tool"].unique()
        print("multi:          %s" % ", ".join(unique_tools))
        print("mismatch tools: %s" % ", ".join(mismatch_tools))
        with pd.option_context(
                "display.max_rows", None,
                "display.max_columns", None,
                "display.width", None
        ):
            print(result)

    def _process_tex(self, used_energy_file: Path, df: pd.DataFrame):
        self._process_tex_threading(used_energy_file, df, "single")
        self._process_tex_threading(used_energy_file, df, "multi")

    def _process_tex_threading(self, used_energy_file: Path, df: pd.DataFrame, threading: str):
        df = df[df["threading"] == threading]

        result_df = (
            df.pivot(
                index=["dataset", "strength"],
                columns="tool",
                values="compression_ratio"
            )
            .reset_index()
        )
        result_df.columns.name = None

        def f_map(str_ds):
            return self._get_data_file(dataset_from_str(str_ds))

        result_df = result_df.sort_values("dataset", key=lambda s: s.map(f_map))
        tool_names = result_df.columns.drop(["dataset", "strength"]).tolist()
        tool_names = sorted(tool_names, key=lambda x: self.TOOL_ORDER.index(x))
        self._create_tex(used_energy_file, result_df, threading, tool_names)

    def _create_tex(self, used_energy_file: Path, result_df, threading, tool_names):
        tex_file = self._resources / ("cr_%s_%s" % (threading, used_energy_file.stem.removeprefix("used_energy_")) + ".tex")
        self._logger.info("Generate: %s", tex_file)
        lines = []
        lines.append("\\begin{tabular}")
        lines.append("{")
        lines.append("l")
        lines.append("c")
        for _ in tool_names:
            lines.append("S[round-mode=places, round-precision=2, table-format=1.2]")
        lines.append("}")
        lines.append("\\toprule")
        header_entries = ["Dataset", "Strength"] + ["{%s}" % tool for tool in tool_names]
        lines.append(" & ".join(header_entries) + "\\\\")
        lines.append("\\midrule")
        used_datasets = set()
        for _, row in result_df.iterrows():
            dataset = self._get_data_file(dataset_from_str(row["dataset"]))
            strength = row["strength"]
            values = ["%f" % row[tool] for tool in tool_names]
            entries = []
            if dataset in used_datasets:
                entries.append("")
            else:
                entries.append(dataset)
                used_datasets.add(dataset)
            entries.append(strength)
            entries.extend(values)
            lines.append(" & ".join(entries) + "\\\\")
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        with tex_file.open(mode="w", encoding="UTF_8") as f:
            for line in lines:
                f.write(line + "\n")

    def _get_data_file(self, dataset: DataSet) -> str:
        files = {
            DataSet.TEXT: "dickens",
            DataSet.XML: "xml",
            DataSet.XML2: "xml2",
            DataSet.WEBSTER: "webster",
            DataSet.IMAGE: "x-ray",
            DataSet.SENSOR: "data.txt",
        }
        return files[dataset]