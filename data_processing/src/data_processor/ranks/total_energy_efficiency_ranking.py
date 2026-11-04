import logging
from pathlib import Path

from .energy_efficiency_ranking import EnergyEfficiencyRanking


class TotalEnergyEfficiencyRanking(EnergyEfficiencyRanking):
    def __init__(self, resources: Path, logfile: bool, n_boot: int, seed: int | None, threading: str, mode: str):
        super().__init__(resources, logfile, n_boot, seed, threading, mode)
        self._logger = logging.getLogger(self.__class__.__name__)

    def extra_log_name(self, input_file: Path) -> str:
        base = self._get_base_from_input_file(input_file)
        return f"ee_total_ranking_{base}_{self._threading}_{self._mode}"

    @property
    def base_column(self) -> str:
        return "energy_efficiency_total"
