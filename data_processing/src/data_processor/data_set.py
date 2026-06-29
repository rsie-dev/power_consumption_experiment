from enum import Enum

from data_processor import ureg


class DataSet(Enum):
    TEXT = 10192446 * ureg.byte
    XML = 5345280 * ureg.byte
    XML2 = 10690560 * ureg.byte
    WEBSTER = 41458703 * ureg.byte
    IMAGE = 8474240 * ureg.byte
    SENSOR = 150910946 * ureg.byte


def dataset_from_str(s: str) -> DataSet:
    try:
        return DataSet[s.strip().upper()]
    except KeyError:
        raise ValueError(f"Unknown dataset: {s}")
