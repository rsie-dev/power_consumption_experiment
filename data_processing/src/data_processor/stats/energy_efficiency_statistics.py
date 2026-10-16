import logging
from pathlib import Path

import tabulate
import pandas as pd
import numpy as np

from data_processor.processor import Processor


class EnergyEfficiencyStatistics(Processor):
    DRIFT_THRESHOLD = 1.0  # percentage points across runs 1–30

    def __init__(self, resources: Path, n_boot: int, seed: int | None, threading: str, mode: str, create_tex: bool):
        super().__init__(resources)
        self._logger = logging.getLogger(self.__class__.__name__)
        self._n_boot = n_boot
        if seed is None:
            seed = 12345
        self._rng = np.random.default_rng(seed)
        self._threading = threading
        self._mode = mode
        self._create_tex = create_tex

    def process(self, tp_file: Path):
        pass
