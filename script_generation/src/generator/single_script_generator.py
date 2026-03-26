import logging
from pathlib import Path

from generator.script_generator import ScriptGenerator
from generator.tools import Tool
from generator.tool_config import ToolConfig, OperationMode, CompressionStrength, Threading
from generator.data_set import DataSet


class SingleScriptGenerator(ScriptGenerator):
    def __init__(self, script_folder: Path):
        super().__init__(script_folder)
        self._logger = logging.getLogger(self.__class__.__name__)

    def _get_template_name(self) -> str:
        return "experiment.jinja"

    def _write_scripts(self, tools: list[Tool], data_sets: list[DataSet], template, args) -> None:
        data_sets_compress, measurement_sets_compress = self._get_measurement_sets_compress(tools, data_sets)
        data_sets_decompress, measurement_sets_decompress = self._get_measurement_sets_decompress(tools, data_sets)

        data_sets = data_sets_compress + data_sets_decompress
        measurement_sets = measurement_sets_compress + measurement_sets_decompress
        self._logger.info("Generating %d measurement sets", measurement_sets)

        data = {
            "args": args,
            "data_sets": data_sets,
        }

        host_script = self._script_folder / f"{args.host}.py"
        self._generate_script(host_script, template, data)

    def _get_measurement_sets_compress(self, tools: list[Tool], data_sets: list[DataSet]):
        measurement_sets = 0
        data_set_entries = []
        for data_set in data_sets:
            entry, count = self._build_data_set_entry_compress(data_set, tools)
            data_set_entries.append(entry)
            measurement_sets += count
        return data_set_entries, measurement_sets

    def _get_measurement_sets_decompress(self, tools: list[Tool], data_sets: list[DataSet]):
        measurement_sets = 0
        data_set_entries = []
        for tool in tools:
            for data_set in data_sets:
                entries, count = self._build_data_set_entry_decompress(data_set, tool)
                data_set_entries.extend(entries)
                measurement_sets += count
        return data_set_entries, measurement_sets

    def _generate_script(self, script_file: Path, template, data):
        self._logger.info("Generate: %s", script_file.relative_to(Path.cwd()))
        with script_file.open("wt", encoding="UTF_8") as script:
            output = template.render(data)
            script.write(output)

    def _build_data_set_entry_compress(self, data_set: DataSet, tools: list[Tool]):
        count = 0
        tool_entries = []
        for tool in tools:
            tool_configs = self._build_tool_configs(tool, OperationMode.COMPRESS)
            for tool_config in tool_configs:
                tool_entry = self._build_tool_entry(tool, tool_config, data_set)
                tool_entries.append(tool_entry)
            count += len(tool_configs)
        entry = {
            "data_set_name": data_set.set_name,
            "data_set_file": data_set.data_file,
            "tools": tool_entries,
        }
        return entry, count

    def _build_data_set_entry_decompress(self, data_set: DataSet, tool: Tool):
        entries = []

        tool_configs = self._build_tool_configs(tool, OperationMode.DECOMPRESS)
        for tool_config in tool_configs:
            tool_entry = self._build_tool_entry(tool, tool_config, data_set)
            decompress_file = data_set.data_file
            data_set_name = f"{data_set.set_name}_{tool.name}"
            if tool.value.threading == Threading.MULTI:
                threading_name = tool_config.threading.name.lower()
                decompress_file = decompress_file.with_stem(f"{data_set.data_file.stem}_{threading_name}")
                data_set_name = f"{data_set.set_name}_{tool.name}_{threading_name}"
            suffixes = decompress_file.suffixes + [tool.value.extension]
            decompress_file = decompress_file.with_suffix("".join(suffixes))
            entry = {
                "data_set_name": data_set_name,
                "data_set_file": f"{decompress_file}",
                "tools": [tool_entry],
            }
            entries.append(entry)

        return entries, len(entries)

    def _build_tool_configs(self, tool: Tool, mode: OperationMode):
        tool_configs = []
        if tool.value.threading == Threading.SINGLE:
            threadings = [Threading.SINGLE]
        else:
            threadings = list(Threading)
        for threading in threadings:
            if mode == OperationMode.COMPRESS:
                for strength in CompressionStrength:
                    tool_config = ToolConfig(mode=mode, strength=strength, threading=threading)
                    tool_configs.append(tool_config)
            else:
                tool_config = ToolConfig(mode=mode, strength=CompressionStrength.DEFAULT, threading=threading)
                tool_configs.append(tool_config)
        return tool_configs
