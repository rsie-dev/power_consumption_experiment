import logging
from pathlib import Path
from abc import ABC

import pandas as pd

from data_processor.util import FrameIO


class Processor(ABC):
    def __init__(self, resources: Path):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._frameio = FrameIO()
        self._resources = resources

    def _create_csv(self, file_name: str|Path, df: pd.DataFrame) -> None:
        csv_file = self._resources / Path(file_name)
        self._frameio.persist(df, csv_file)
