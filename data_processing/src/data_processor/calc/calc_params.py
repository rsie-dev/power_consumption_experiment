from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CalcParams:
    used_energy_file: Path
    no_tool: list[str]
    no_dataset: list[str]
