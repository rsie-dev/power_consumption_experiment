from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class ToolDefinition:
    binary: str
    compress: str
    decompress: str
    min: str
    max: str
    keep: str
    to_stdout: str
    single_thread: str
    multi_thread: str


class Tool(ToolDefinition, Enum):
    gzip = ("gzip", "-z", "", "", "", "-k", "-c", "", "")
    bzip2 = ("bzip2", "-z", "", "", "", "-k", "-c", "", "")
    xz = ("xz", "-z", "", "", "", "-k", "-c", "-T 1", "-T 0")
    lz4 = ("lz4", "-z", "", "", "", "-k", "-c", "-T1", "")
    lzop = ("lzop", "", "", "", "", "-k", "-c", "", "")
