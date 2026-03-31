from enum import Enum
from pathlib import Path


class DataSet(Enum):
    TEXT = Path("dickens")
    XML = Path("xml")
    IMAGE = Path("x-ray")
    SENSOR = Path("data.txt")
