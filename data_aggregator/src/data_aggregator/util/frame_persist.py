import logging
from pathlib import Path

import pandas as pd


class FramePersist:
    def __init__(self):
        self._logger = logging.getLogger(self.__class__.__name__)

    def persist(self, df: pd.DataFrame, target_path: Path) -> None:
        self._logger.info("Generate: %s", target_path)
        df = df.pint.dequantify()
        df.to_csv(target_path, encoding='UTF_8', index=False, header=True)
