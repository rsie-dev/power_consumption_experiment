from dataclasses import dataclass
from enum import Enum
from pathlib import Path


@dataclass(frozen=True)
class DataSetDefinition:
    set_name: str
    data_file: Path


class DataSet(DataSetDefinition, Enum):
    TEXT = ("text", Path("dickens"))
    XML = ("xml", Path("xml"))
    IMAGE = ("image", Path("x-ray"))
    SENSOR = ("sensor", Path("sensor_data.txt"))
