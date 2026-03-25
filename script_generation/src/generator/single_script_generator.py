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
        tools = [tool for tool in Tool]
        tool_configs = self._build_tool_configs()

        data_sets = []
        for data_set in DataSet:
            data_sets.append(self._build_data_set_entry(data_set, tools, tool_configs))

        measurement_sets = len(tools) * len(tool_configs) * len(DataSet)
        self._logger.info("Generating %d measurement sets", measurement_sets)

        data = {
            "args": args,
            "data_sets": data_sets,
        }

        host_script = script_folder / f"{args.host}.py"
        self._logger.info("Generate: %s", host_script.relative_to(Path.cwd()))
        with host_script.open("wt", encoding="UTF_8") as script:
            output = template.render(data)
            script.write(output)

    def _build_data_set_entry(self, data_set: DataSet, tools: list[Tool], tool_configs: list[ToolConfig]):
        tool_entries = []
        for tool in tools:
            for tool_config in tool_configs:
                tool_entry = self._build_tool_entry(tool, tool_config, data_set)
                tool_entries.append(tool_entry)
        entry = {
            "data_set": data_set,
            "tools": tool_entries,
        }
        return entry

    def _build_tool_configs(self):
        tool_configs = []
        for mode in OperationMode:
            for threading in Threading:
                if mode == OperationMode.COMPRESS:
                    for strength in CompressionStrength:
                        tool_config = ToolConfig(mode=mode, strength=strength, threading=threading)
                        tool_configs.append(tool_config)
                else:
                    tool_config = ToolConfig(mode=mode, strength=CompressionStrength.DEFAULT, threading=threading)
                    tool_configs.append(tool_config)
        return tool_configs
