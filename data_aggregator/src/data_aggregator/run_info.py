from dataclasses import dataclass

from .measurement import Measurement


@dataclass(frozen=True)
class RunInfo:
    run: int
    measurement: Measurement
