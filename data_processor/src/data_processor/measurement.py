from dataclasses import dataclass
import datetime

import pint


ureg = pint.UnitRegistry()
Q_ = ureg.Quantity


@dataclass(frozen=True)
class Timings:
    real: datetime.timedelta
    user: datetime.timedelta
    sys: datetime.timedelta


@dataclass(frozen=True)
class ElectricalMeasurement:
    timestamp: datetime.datetime
    relative_time: datetime.timedelta
    voltage: pint.Quantity
    current: pint.Quantity

    def __post_init__(self):
        object.__setattr__(self, "voltage", self.voltage.to("volt"))
        object.__setattr__(self, "current", self.current.to("ampere"))


@dataclass(frozen=True)
class Measurement:
    start: datetime.datetime
    end: datetime.datetime
    timings: Timings
    count: int | None
    readings: list[ElectricalMeasurement]
