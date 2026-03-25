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
        data_set = DataSet.TEXT

        tools = []
        tools.append(self._build_tool_entry(Tool.lzop, tool_config, data_set))
        tools.append(self._build_tool_entry(Tool.gzip, tool_config, data_set))

        data = {
            "args": args,
            "multimeter": "07D1A5642160",
            "tools": tools,
            "data_set": data_set,
        }

        output = template.render(data)
        print(output)
