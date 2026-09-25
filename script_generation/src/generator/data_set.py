from enum import Enum
from pathlib import Path


class DataSet(Enum):
    TEXT = Path("dickens")
    TEXTLARGE = Path("enwik8")
    XML = Path("xml")
    XML2 = Path("xml2")
    WEBSTER = Path("webster")
    IMAGE = Path("x-ray")
    IMAGELARGE = Path("images.tar")
    SENSOR = Path("data.txt")
