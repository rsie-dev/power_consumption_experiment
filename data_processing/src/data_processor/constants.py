import pandas as pd

GROUP_COLS = ["host", "tool", "dataset", "mode", "strength", "threading"]

ORDER_TOOL = ["gzip", "pigz", "bzip2", "lbzip2", "bzip3", "xz", "lz4", "lzop", "zstd", "brotli", "sleep"]

ORDER_STRENGTH = ["min", "default", "max"]

ORDER_THREADING = ["single", "multi"]

ORDER_HOST = ["raspi5", "visionfive2", "radxax4"]

ORDER_DATASET = ["textlarge", "webster", "imagelarge", "sensor"]


def get_factor_levels(factor, df: pd.DataFrame) -> list[str]:
    if factor == "strength":
        return ORDER_STRENGTH
    if factor == "dataset":
        return ORDER_DATASET
    if factor == "tool":
        tools = df["tool"].dropna().unique().tolist()
        return tools
    raise RuntimeError("invalid factor: %s" % factor)
