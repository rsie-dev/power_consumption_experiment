from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class ToolConfig:
    compress: str
    decompress: str
    min: str
    max: str
    keep: str
    to_stdout: str
    single_thread: str
    multi_thread: str


class Tool(ToolConfig, Enum):
    gzip = ("-z", "", "", "", "-k", "-c", "", "")
    bzip2 = ("-z", "", "", "", "-k", "-c", "", "")
    xz = ("-z", "", "", "", "-k", "-c", "-T 1", "-T 0")
    lz4 = ("-z", "", "", "", "-k", "-c", "-T1", "")
    lzop = ("", "", "", "", "-k", "-c", "", "")
