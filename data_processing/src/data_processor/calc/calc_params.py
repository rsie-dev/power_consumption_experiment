from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CalcParams:
    no_tool: list[str]
    no_dataset: list[str]


@dataclass(frozen=True)
class EnergyParams(CalcParams):
    used_energy_file: Path
