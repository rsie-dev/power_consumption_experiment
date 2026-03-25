import logging
from pathlib import Path

from generator.script_generator import ScriptGenerator
from generator.tools import Tool
from generator.tool_config import ToolConfig, OperationMode, CompressionStrength, Threading
from generator.data_set import DataSet


class SingleScriptGenerator(ScriptGenerator):
    def __init__(self):
        super().__init__()
        self._logger = logging.getLogger(self.__class__.__name__)

    def _get_template_name(self) -> str:
        return "single_experiment.jinja"

    def _write_scripts(self, template, script_folder: Path, args) -> None:
        tool_config = ToolConfig(mode=OperationMode.COMPRESS, strength=CompressionStrength.DEFAULT,
                                 threading=Threading.SINGLE)
        tools = [Tool.lzop, Tool.gzip]

        data_sets = []
        data_sets.append(self._build_data_set_entry(DataSet.TEXT, tools, tool_config))
        data_sets.append(self._build_data_set_entry(DataSet.TEXT, tools, tool_config))

        data = {
            "args": args,
            "multimeter": "07D1A5642160",
            "data_sets": data_sets,
        }

        output = template.render(data)
        print(output)

    def _build_data_set_entry(self, data_set: DataSet, tools: list[Tool], tool_config: ToolConfig):
        tool_entries = []
        for tool in tools:
            tool_entries.append(self._build_tool_entry(tool, tool_config, data_set))
        entry = {
            "data_set": data_set,
            "tools": tool_entries,
        }
        return entry

