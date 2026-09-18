import logging
from pathlib import Path

import pandas as pd

from data_processor.processor import Processor
from data_processor.constants import ORDER_TOOL, ORDER_STRENGTH
from .calc_params import EnergyParams


class Calculator(Processor):
    def __init__(self, resources: Path):
        super().__init__(resources)
        self._logger = logging.getLogger(self.__class__.__name__)

    def _load(self, params: EnergyParams) -> pd.DataFrame:
        df = self._frameio.load(params.used_energy_file)
        df = df[~df["tool"].isin(params.no_tool)]
        df = df[~df["dataset"].isin(params.no_dataset)]
        return df

    def _order_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df["_tool_key"] = df["tool"].apply(ORDER_TOOL.index)
        df["_strength_key"] = df["strength"].apply(ORDER_STRENGTH.index)
        df = df.sort_values(
            by=["host", "_tool_key", "dataset", "mode", "_strength_key"],
        ).drop(columns=["_tool_key", "_strength_key"])
        return df
