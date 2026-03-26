from dataclasses import dataclass
from enum import Enum

from generator.threading import Threading


@dataclass(frozen=True, kw_only=True)
class ToolDefinition:   # pylint: disable=too-many-instance-attributes
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
    GZIP = ToolDefinition(binary="gzip", extension=".gz", compress="-z", decompress="-d", min="--fast", max="--best",
                          keep="-k", to_stdout="-c",
                          threading=Threading.SINGLE, single_thread="", multi_thread="")
    BZIP2 = ToolDefinition(binary="bzip2", extension=".bz2", compress="-z", decompress="-d", min="--fast", max="--best",
                           keep="-k", to_stdout="-c",
                           threading=Threading.SINGLE, single_thread="", multi_thread="")
    BZIP3 = ToolDefinition(binary="bzip3", extension=".bz3", compress="-z", decompress="-d", min="-b 1", max="-b 511",
                           keep="-k", to_stdout="-c",
                           threading=Threading.MULTI, single_thread="-j 1", multi_thread="-j 4")
    XZ = ToolDefinition(binary="xz", extension=".xz", compress="-z", decompress="-d", min="-0", max="-9",
                        keep="-k", to_stdout="-c",
                        threading=Threading.MULTI, single_thread="-T 1", multi_thread="-T 0")
    LZ4 = ToolDefinition(binary="lz4", extension=".lz4", compress="-z", decompress="-d", min="--fast", max="--best",
                         keep="-k", to_stdout="-c",
                         threading=Threading.MULTI, single_thread="-T1", multi_thread="")
    LZOP = ToolDefinition(binary="lzop", extension=".lzo", compress="", decompress="-d", min="--fast", max="--best",
                          keep="-k", to_stdout="-c",
                          threading=Threading.SINGLE, single_thread="", multi_thread="")
    ZSTD = ToolDefinition(binary="zstd", extension=".zst", compress="-z", decompress="-d", min="--fast", max="--ultra",
                          keep="-k", to_stdout="-c",
                          threading=Threading.MULTI, single_thread="--single-thread", multi_thread="-T0")
    BROTLI = ToolDefinition(binary="brotli", extension=".br", compress="", decompress="-d", min="-q 0", max="-q 11",
                            keep="-k", to_stdout="-c",
                            threading=Threading.SINGLE, single_thread="", multi_thread="")
