import logging
from pathlib import Path

import pandas as pd

from data_processor.processor import Processor


class BaseRanking(Processor):
    def __init__(self, resources: Path, threading: str, mode: str):
        super().__init__(resources)
        self._logger = logging.getLogger(self.__class__.__name__)
        self._threading = threading
        self._mode = mode

    def process(self, input_file: Path) -> None:
        self._rank(input_file)

    def _rank(self, input_file: Path) -> None:
        df = self._preparation(input_file)

    def _preparation(self, input_file: Path) -> pd.DataFrame:
        self._log_section("1. Preparation")
        self._logger.info("load data")
        df = self._frameio.load(input_file)

        self._log_section("1.1 Define subset")
        self._logger.info("Evaluate: %s %s", self._threading, self._mode)
        df = self._filter_data(df)

        return df

    def _filter_data(self, df: pd.DataFrame):
        df = df[df["mode"] == self._mode]
        df = df[df["threading"] == self._threading]
        return df

    def _log_section(self, title: str):
        self._logger.info("")
        self._logger.info("%s %s %s", "-" * 20, title, "-" * 20)
        self._logger.info("")
