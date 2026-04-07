from dataclasses import dataclass

from .tool_config import ToolConfig


@dataclass(frozen=True)
class MeasurementInfo:
    tool: str
    dataset: str
    tool_config: ToolConfig
