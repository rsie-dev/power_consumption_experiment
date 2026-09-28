from dataclasses import dataclass
import datetime

import pandas as pd


@dataclass(frozen=True)
class Timings:
    real: datetime.timedelta
    user: datetime.timedelta
    sys: datetime.timedelta


@dataclass(frozen=True)
class Marker:
    start: datetime.datetime
    end: datetime.datetime


@dataclass(frozen=True)
class Measurement:
    marker_host: Marker
    marker_device: Marker
    readings: pd.DataFrame
    timings: Timings | None
    count: int | None
