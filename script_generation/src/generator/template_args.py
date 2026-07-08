from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TemplateArgs:  # pylint: disable=too-many-instance-attributes
    host: str
    ip: str
    runs: int
    data_folder: Path
    multimeter: str
    with_timers: bool
    with_caches: bool
    warmup: int | None
    mon_temp: float | None
    head_delay: int | None
    tail_delay: int | None
