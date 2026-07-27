import logging
from pathlib import Path

import pandas as pd

from data_processor.processor import Processor
from .calc_params import CalcParams


class Calculator(Processor):
    def __init__(self, resources: Path):
        super().__init__(resources)
        self._logger = logging.getLogger(self.__class__.__name__)

    def _load(self, params: CalcParams) -> pd.DataFrame:
        df = self._frameio.load(params.used_energy_file)
        df = df[~df["tool"].isin(params.no_tool)]
        df = df[~df["dataset"].isin(params.no_dataset)]
        return df
