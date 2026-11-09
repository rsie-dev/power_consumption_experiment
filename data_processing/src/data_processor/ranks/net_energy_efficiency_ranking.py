import logging
from pathlib import Path

from .energy_efficiency_ranking import EnergyEfficiencyRanking


class NetEnergyEfficiencyRanking(EnergyEfficiencyRanking):
    def __init__(self, resources: Path, logfile: bool, n_boot: int, seed: int | None,
                 threading: str, mode: str, create_tex: bool):
        super().__init__(resources, logfile, n_boot, seed, threading, mode, create_tex)
        self._logger = logging.getLogger(self.__class__.__name__)

    @property
    def base_column(self) -> str:
        return "energy_efficiency_net"

    @property
    def kind(self) -> str:
        return "ee_net"
