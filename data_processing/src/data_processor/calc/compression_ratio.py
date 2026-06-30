import logging
from pathlib import Path

import tabulate

from data_processor.util import FrameIO
from data_processor.data_set import dataset_from_str


class CompressionRatio:
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
        df = df[~df["dataset"].isin(no_dataset)]
        # ToDo: validate single vs multi
        df = df[df["threading"] == "single"]

        df["compression_ratio"] = df.apply(
            lambda row: dataset_from_str(row["dataset"]).value / row["size"],
            axis=1,
        )
        df["compression_ratio"] = df["compression_ratio"].astype(float)

        result_df = (
            df.pivot(
                index=["dataset", "strength"],
                columns="tool",
                values="compression_ratio"
            )
            .reset_index()
        )
        result_df.columns.name = None
        for tool in no_tool:
            result_df = result_df.drop(tool, axis=1)

        table_entries = []
        tool_names = result_df.columns.drop(["dataset", "strength"]).tolist()
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
        print(table_str)
        cr_file = self._resources / ("cr_" + used_energy_file.stem.removeprefix("used_energy_") + ".csv")
        frameio.persist(result_df, cr_file)

        if create_tex:
            self._create_tex(used_energy_file, result_df, tool_names)

    def _create_tex(self, used_energy_file: Path, result_df, tool_names):
        tex_file = self._resources / ("cr_" + used_energy_file.stem.removeprefix("used_energy_") + ".tex")
        self._logger.info("Generate: %s", tex_file)
        lines = []
        lines.append("\\begin{tabular}")
        lines.append("{")
        lines.append("l")
        lines.append("c")
        for _ in tool_names:
            lines.append("S[round-mode=places, round-precision=2, table-format=2.2]")
        lines.append("}")
        lines.append("\\toprule")
        header_entries = ["Dataset", "Strength"] + ["{%s}" % tool for tool in tool_names]
        lines.append(" & ".join(header_entries) + "\\\\")
        lines.append("\\midrule")
        for _, row in result_df.iterrows():
            dataset = row["dataset"]
            strength = row["strength"]
            values = ["%f" % row[tool] for tool in tool_names]
            entries = [dataset, strength] + values
            lines.append(" & ".join(entries) + "\\\\")
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        with tex_file.open(mode="w", encoding="UTF_8") as f:
            for line in lines:
                f.write(line + "\n")
