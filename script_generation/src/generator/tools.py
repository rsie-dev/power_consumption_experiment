from dataclasses import dataclass
from enum import Enum

from generator.threading import Threading


@dataclass(frozen=True,kw_only=True)
class ToolDefinition:
    binary: str
    extension: str
    compress: str
    decompress: str
    min: str
    max: str
    keep: str
    to_stdout: str
    threading: Threading
    single_thread: str
    multi_thread: str


class Tool(Enum):
    gzip = ToolDefinition(binary="gzip", extension=".gz", compress="-z", decompress="", min="", max="",
                          keep="-k", to_stdout="-c",
                          threading=Threading.SINGLE, single_thread="", multi_thread="")
    bzip2 = ToolDefinition(binary="bzip2", extension=".bz2", compress="-z", decompress="", min="", max="",
                           keep="-k", to_stdout="-c",
                           threading=Threading.SINGLE, single_thread="", multi_thread="")
    xz = ToolDefinition(binary="xz", extension=".xz", compress="-z", decompress="", min="", max="",
                        keep="-k", to_stdout="-c",
                        threading=Threading.MULTI, single_thread="-T 1", multi_thread="-T 0")
    lz4 = ToolDefinition(binary="lz4", extension=".lz4", compress="-z", decompress="", min="", max="",
                         keep="-k", to_stdout="-c",
                         threading=Threading.MULTI, single_thread="-T1", multi_thread="")
    lzop = ToolDefinition(binary="lzop", extension=".lzo", compress="", decompress="", min="", max="",
                          keep="-k", to_stdout="-c",
                          threading=Threading.SINGLE, single_thread="", multi_thread="")
    zstd = ToolDefinition(binary="zstd", extension=".zst", compress="-z", decompress="-d", min="--fast", max="--ultra",
                         keep="-k", to_stdout="-c",
                         threading=Threading.MULTI, single_thread="--single-thread", multi_thread="-T0")
