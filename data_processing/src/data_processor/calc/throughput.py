import logging
from pathlib import Path

import tabulate
import pandas as pd

from data_processor.util import FrameIO


class Throughput:
    ORDER_TOOL = ["gzip", "pigz", "bzip2", "lbzip2", "bzip3", "xz", "lz4", "lzop", "zstd", "brotli"]
    ORDER_STRENGTH = ["min", "default", "max"]

    def __init__(self, resources: Path):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._resources = resources

    def process(self, used_energy_file: Path, create_tex: bool, no_tool: list, no_dataset: list):
        frameio = FrameIO()
