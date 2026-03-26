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
        tool_configs_decompress = self._build_tool_configs(OperationMode.DECOMPRESS)

        data_sets_compress, measurement_sets_compress = self._get_measurement_sets_compress(tools)
        data_sets_decompress, measurement_sets_decompress = self._get_measurement_sets_decompress(tools)

        data_sets = data_sets_compress + data_sets_decompress
        measurement_sets = measurement_sets_compress + measurement_sets_decompress
        self._logger.info("Generating %d measurement sets", measurement_sets)

        data = {
            "args": args,
            "data_sets": data_sets,
        }

        host_script = script_folder / f"{args.host}.py"
        self._generate_script(host_script, template, data)

    def _get_measurement_sets_compress(self, tools: list[Tool]):
        tool_configs = self._build_tool_configs(OperationMode.COMPRESS)
        data_sets = []
        for data_set in DataSet:
            data_sets.append(self._build_data_set_entry(data_set, tools, tool_configs))
        measurement_sets = len(tools) * len(tool_configs) * len(data_sets)
        return data_sets, measurement_sets

    def _get_measurement_sets_decompress(self, tools: list[Tool]):
        tool_configs = self._build_tool_configs(OperationMode.DECOMPRESS)
        data_sets = []
        for tool in tools:
            for data_set in DataSet:
                entries = self._build_data_set_entry_decompress(data_set, tool, tool_configs)
                data_sets.extend(entries)
        measurement_sets = len(data_sets)
        return data_sets, measurement_sets

    def _generate_script(self, script_file: Path, template, data):
        self._logger.info("Generate: %s", script_file.relative_to(Path.cwd()))
        with script_file.open("wt", encoding="UTF_8") as script:
            output = template.render(data)
            script.write(output)

    def _build_data_set_entry(self, data_set: DataSet, tools: list[Tool], tool_configs: list[ToolConfig]):
        tool_entries = []
        for tool in tools:
            for tool_config in tool_configs:
                tool_entry = self._build_tool_entry(tool, tool_config, data_set)
                tool_entries.append(tool_entry)
        entry = {
            "data_set_name": data_set.set_name,
            "data_set_file": data_set.data_file,
            "tools": tool_entries,
        }
        return entry

    def _build_data_set_entry_decompress(self, data_set: DataSet, tool: Tool, tool_configs: list[ToolConfig]):
        entries = []

        for tool_config in tool_configs:
            tool_entry = self._build_tool_entry(tool, tool_config, data_set)
            decompress_file = data_set.data_file.with_stem(f"{data_set.data_file.stem}_{tool_config.threading.name.lower()}")
            decompress_file = decompress_file.with_suffix(tool.value.extension)
            entry = {
                "data_set_name": f"{data_set.set_name}_{tool.name}_{tool_config.threading.name.lower()}",
                "data_set_file": f"{decompress_file}",
                "tools": [tool_entry],
            }
            entries.append(entry)

        return entries

    def _build_tool_configs(self, mode: OperationMode):
        tool_configs = []
        for threading in Threading:
            if mode == OperationMode.COMPRESS:
                for strength in CompressionStrength:
                    tool_config = ToolConfig(mode=mode, strength=strength, threading=threading)
                    tool_configs.append(tool_config)
            else:
                tool_config = ToolConfig(mode=mode, strength=CompressionStrength.DEFAULT, threading=threading)
                tool_configs.append(tool_config)
        return tool_configs
