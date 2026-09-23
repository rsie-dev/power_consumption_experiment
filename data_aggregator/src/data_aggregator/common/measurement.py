from dataclasses import dataclass
import datetime

import pandas as pd


@dataclass(frozen=True)
class Timings:
    real: datetime.timedelta
    user: datetime.timedelta
    sys: datetime.timedelta


@dataclass(frozen=True)
class Measurement:
    start: datetime.datetime
    end: datetime.datetime
    readings: pd.DataFrame
    timings: Timings | None
    count: int | None
