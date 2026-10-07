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


def _dataset_from_str(ds_str: str) -> DataSet:
    try:
        return DataSet[ds_str.strip().upper()]
    except KeyError as e:
        raise ValueError(f"Unknown dataset: {ds_str}") from e


def get_dataset_file(ds_str) -> str:
    ds = _dataset_from_str(ds_str)
    return ds.file


def get_dataset_size(ds_str: str):
    return _dataset_from_str(ds_str).size
