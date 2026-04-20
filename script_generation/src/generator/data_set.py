from enum import Enum
from pathlib import Path


class DataSet(Enum):
    TEXT = Path("dickens")
    XML = Path("xml")
    XML2 = Path("xml2")
    IMAGE = Path("x-ray")
    SENSOR = Path("data.txt")
