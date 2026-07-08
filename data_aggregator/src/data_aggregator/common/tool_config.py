from enum import Enum, auto
from dataclasses import dataclass


class OperationMode(Enum):
    BASELINE = auto()
    COMPRESS = auto()
    DECOMPRESS = auto()


class CompressionStrength(Enum):
    MIN = auto()
    DEFAULT = auto()
    MAX = auto()


class Threading(Enum):
    SINGLE = auto()
    MULTI = auto()
    NONE = SINGLE


@dataclass(frozen=True)
class ToolConfig:
    mode: OperationMode
    strength: CompressionStrength
    threading: Threading
