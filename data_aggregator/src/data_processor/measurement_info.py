from dataclasses import dataclass

from data_processor.tool_config import ToolConfig


@dataclass(frozen=True)
class MeasurementInfo:
    tool: str
    dataset: str
    tool_config: ToolConfig
