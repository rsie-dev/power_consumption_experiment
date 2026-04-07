from dataclasses import dataclass

from .tool_config import ToolConfig


@dataclass(frozen=True)
class MeasurementInfo:
    host: str
    tool: str
    dataset: str
    tool_config: ToolConfig
