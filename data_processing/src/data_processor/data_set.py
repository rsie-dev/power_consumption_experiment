from enum import Enum
from dataclasses import dataclass

import pint

from data_processor import ureg


@dataclass(frozen=True)
class Entry:
    size: pint.Quantity[int]
    file: str


class DataSet(Entry, Enum):
    TEXT = 10192446 * ureg.byte, "dickens"
    TEXTLARGE = 100000000 * ureg.byte, "enwik8"
    XML = 5345280 * ureg.byte, "xml"
    XML2 = 10690560 * ureg.byte, "xml2"
    WEBSTER = 41458703 * ureg.byte, "webster"
    IMAGE = 8474240 * ureg.byte, "x-ray"
    IMAGELARGE = 20715520 * ureg.byte, "images.tar"
    SENSOR = 150910946 * ureg.byte, "data.txt"


def dataset_from_str(s: str) -> DataSet:
    try:
        return DataSet[s.strip().upper()]
    except KeyError as e:
        raise ValueError(f"Unknown dataset: {s}") from e


def dataset_map(str_ds) -> pint.Quantity[int]:
    ds = dataset_from_str(str_ds)
    return ds.file
