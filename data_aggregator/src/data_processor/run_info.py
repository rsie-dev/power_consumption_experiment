from dataclasses import dataclass

from data_processor.measurement import Measurement


@dataclass(frozen=True)
class RunInfo:
    run: int
    measurement: Measurement
