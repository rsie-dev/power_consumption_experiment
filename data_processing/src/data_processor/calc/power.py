import logging
from pathlib import Path

import tabulate
import pandas as pd

from data_processor.util import FrameIO
from data_processor.data_set import dataset_from_str


class Power:
    ORDER_TOOL = ["gzip", "pigz", "bzip2", "lbzip2", "bzip3", "xz", "lz4", "lzop", "zstd", "brotli"]
    ORDER_STRENGTH = ["min", "default", "max"]

    def __init__(self, resources: Path):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._frameio = FrameIO()
        self._resources = resources

    def process(self, used_energy_file: Path, create_tex: bool, no_tool: list, no_dataset: list):
        self._logger.debug("loading %s", used_energy_file)
        df = self._frameio.load(used_energy_file)
        df = df[~df["tool"].isin(no_tool)]
        df = df[~df["dataset"].isin(no_dataset)]
