import logging
from pathlib import Path

import tabulate
from tabulate import SEPARATING_LINE
import pandas as pd

from data_processor.util import FrameIO
from data_processor.data_set import dataset_from_str, get_data_file


class CompressionRatio:
    ORDER_TOOL = ["gzip", "pigz", "bzip2", "lbzip2", "bzip3", "xz", "lz4", "lzop", "zstd", "brotli"]
    ORDER_STRENGTH = ["min", "default", "max"]

    def __init__(self, resources: Path):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._resources = resources

    def process(self, used_energy_file: Path, create_tex: bool, no_tool: list, no_dataset: list):
        frameio = FrameIO()
        self._logger.debug("loading %s", used_energy_file)
        df = frameio.load(used_energy_file)
        first_host = df.loc[0, "host"]
        df = df[df["mode"] == "compress"]
        df = df[df["host"] == first_host]
        df = df[df["run"] == 1]
        df = df[~df["tool"].isin(no_tool)]
        df = df[~df["dataset"].isin(no_dataset)]
        self._validate_multi(df)
        df["compression_ratio"] = df.apply(
            lambda row: dataset_from_str(row["dataset"]).value / row["size"],
            axis=1,
        )
        df["compression_ratio"] = df["compression_ratio"].astype(float)

        self._show_tables(df)
        self._create_csv(used_energy_file, df)
        if create_tex:
            self._process_tex(used_energy_file, df)

    def _create_csv(self, used_energy_file: Path, df: pd.DataFrame):
        result_df, _, _ = self._restructure_data(df)
        cr_file = self._resources / ("cr_%s" % used_energy_file.stem.removeprefix("used_energy_") + ".csv")
        frameio = FrameIO()
        # Reshape to long format
        csv_df = (
            result_df.melt(
                id_vars=["dataset", "strength", "threading"],
                var_name="tool",
                value_name="cr",
            )
            .reset_index(drop=True)
        )
        csv_df = csv_df[csv_df["cr"].notna()]
        frameio.persist(csv_df, cr_file)

    def _show_tables(self, df: pd.DataFrame):
        result_df, fixed_columns, tool_names = self._restructure_data(df)
        table_entries = []
        for _, row in result_df.iterrows():
            dataset = row["dataset"]
            strength = row["strength"]
            threading = row["threading"]
            values = [None if pd.isna(row[tool]) else row[tool] for tool in tool_names]
            table_entries.append([dataset, strength, threading] + values)

        headers = fixed_columns + tool_names
        table_str = tabulate.tabulate(table_entries,
                                      headers=headers,
                                      tablefmt="simple"
                                      )

        print("entries:")
        print(table_str)
        self._show_mode_deviations(result_df)

    def _show_mode_deviations(self, df: pd.DataFrame):
        multi = df["threading"] == "multi"
        cols_to_drop = df.columns[df.loc[multi].isna().any()]
        df = df.drop(columns=cols_to_drop)

        # Remove tools where CR multi == CR single
        single = df[df["threading"] == "single"].set_index(["dataset", "strength"])
        multi = df[df["threading"] == "multi"].set_index(["dataset", "strength"])
        single, multi = single.align(multi, join="inner")
        tool_cols = [c for c in df.columns if c not in ["dataset", "strength", "threading"]]
        equal_cols = single[tool_cols].eq(multi[tool_cols]).all()
        cols_to_drop = equal_cols[equal_cols].index
        df = df.drop(columns=cols_to_drop)

        # Calculate the difference between multi and single
        keys = ["dataset", "strength"]
        single = df[df["threading"] == "single"].set_index(keys)
        multi = df[df["threading"] == "multi"].set_index(keys)
        single, multi = single.align(multi, join="inner")
        tool_cols = [c for c in df.columns if c not in keys + ["threading"]]
        diff = multi[tool_cols] - single[tool_cols]
        diff_df = diff.reset_index()

        table_entries = []
        for _, row in diff_df.iterrows():
            table_entries.append(row.values[:])

        table_entries.append(SEPARATING_LINE)
        mean_diff = diff.mean(axis=0)
        std_diff = diff.std(axis=0)
        table_entries.append(["mean", None] + mean_diff.tolist())
        table_entries.append(["std", None] + std_diff.tolist())

        table_str = tabulate.tabulate(table_entries,
                                      headers=list(diff_df.columns),
                                      tablefmt="simple"
                                      )

        print("differences:")
        print(table_str)
        #print(diff.describe())

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
        result_df, fixed_columns, _ = self._restructure_data(df)
        self._create_tex(used_energy_file, result_df, threading, fixed_columns)

    def _restructure_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list, list]:
        fixed_columns = ["dataset", "strength", "threading"]
        result_df = (
            df.pivot(
                index=fixed_columns,
                columns="tool",
                values="compression_ratio"
            )
            .reset_index()
        )
        result_df.columns.name = None

        def dataset_map(str_ds):
            return get_data_file(dataset_from_str(str_ds))

        result_df["_dataset_key"] = result_df["dataset"].apply(dataset_map)
        result_df["_strength_key"] = result_df["strength"].apply(self.ORDER_STRENGTH.index)
        result_df = result_df.sort_values(
            by=["_dataset_key", "_strength_key", "threading"],
            ascending=[True, True, False]
        ).drop(columns=["_dataset_key", "_strength_key"])

        cols = list(result_df.columns)
        tool_names = result_df.columns.drop(fixed_columns).tolist()
        tool_names = sorted(tool_names, key=self.ORDER_TOOL.index)
        result_df = result_df[cols[:len(fixed_columns)] + tool_names]
        return result_df, fixed_columns, tool_names

    def _create_tex(self, used_energy_file: Path, result_df, threading, fixed_columns):
        filename = "cr_%s_%s" % (threading, used_energy_file.stem.removeprefix("used_energy_")) + ".tex"
        tex_file = self._resources / filename
        self._logger.info("Generate: %s", tex_file)

        lines = self._build_lines(result_df, fixed_columns)
        with tex_file.open(mode="w", encoding="UTF_8") as f:
            for line in lines:
                f.write(line + "\n")

    def _build_lines(self, df: pd.DataFrame, fixed_columns: list) -> list[str]:
        lines = []
        lines.append("\\begin{tabular}")
        lines.append("{")
        lines.append("l")
        lines.append("c")
        tool_names = df.columns.drop(fixed_columns).tolist()
        for _ in tool_names:
            lines.append("S[round-mode=places, round-precision=2, table-format=1.2]")
        lines.append("}")
        lines.append("\\toprule")
        header_entries = ["Dataset", "Strength"] + ["{%s}" % tool for tool in tool_names]
        lines.append(" & ".join(header_entries) + "\\\\")
        lines.append("\\midrule")
        used_datasets = set()
        for _, row in df.iterrows():
            dataset = get_data_file(dataset_from_str(row["dataset"]))
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
        return lines
