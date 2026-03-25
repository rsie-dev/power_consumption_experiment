from enum import Enum, auto
from dataclasses import dataclass


class OperationMode(Enum):
    COMPRESS = auto()
    DECOMPRESS = auto()


class CompressionStrength(Enum):
    MIN = auto()
    DEFAULT = auto()
    MAX = auto()


class Threading(Enum):
    SINGLE = auto()
    MULTI = auto()


@dataclass(frozen=True)
class ToolConfig:
    mode: OperationMode
    strength: CompressionStrength
    threading: Threading
