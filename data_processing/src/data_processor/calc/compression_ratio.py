import logging
from pathlib import Path

import tabulate

from data_processor.util import FrameIO
from data_processor.data_set import dataset_from_str


class CompressionRatio:
    def __init__(self, resources: Path):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._resources = resources

    def process(self, used_energy_file: Path):
        frameio = FrameIO()
        self._logger.debug("loading %s", used_energy_file)
        df = frameio.load(used_energy_file)
        first_host = df.loc[0, "host"]
        df = df[df["host"] == first_host]
        df = df[df["run"] == 1]
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
