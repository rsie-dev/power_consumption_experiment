from enum import Enum, auto
from dataclasses import dataclass

from generator.threading import Threading


class OperationMode(Enum):
    COMPRESS = auto()
    DECOMPRESS = auto()


class CompressionStrength(Enum):
    MIN = auto()
    DEFAULT = auto()
    MAX = auto()


@dataclass(frozen=True)
class ToolConfig:
    mode: OperationMode
    strength: CompressionStrength
    threading: Threading
