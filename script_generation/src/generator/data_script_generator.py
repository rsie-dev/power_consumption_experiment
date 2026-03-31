import logging
from pathlib import Path

from generator.script_generator import ScriptGenerator
from generator.tools import Tool
from generator.tool_config import ToolConfig, OperationMode, CompressionStrength, Threading
from generator.data_set import DataSet


class DataScriptGenerator(ScriptGenerator):
    def __init__(self, script_folder: Path):
        super().__init__(script_folder)
        self._logger = logging.getLogger(self.__class__.__name__)

    def _get_template_name(self) -> str:
        return "data_gen.jinja"

    def _write_scripts(self, tools: list[Tool], data_sets: list[DataSet], template, args) -> None:
        data_sets_in, data_sets_out, measurement_sets_decompress = self._get_data_sets(tools, data_sets[:2])

        measurement_sets = measurement_sets_decompress
        self._logger.info("Generating %d data sets", measurement_sets)

        data = {
            "args": args,
            "data_sets_in": data_sets_in,
            "data_sets": data_sets_out,
        }

        host_script = self._script_folder / f"{args.host}_data_gen.py"
        self._generate_script(host_script, template, data)

    def _get_data_sets(self, tools: list[Tool], data_sets: list[DataSet]):
        data_sets_in = []
        for data_set in data_sets:
            data_set_name = f"{data_set.name.lower()}"
            entry = {
                "data_set_name": data_set_name,
                "data_set_file": data_set.value,
            }
            data_sets_in.append(entry)

        measurement_sets = 0
        data_sets_out = []
        for tool in tools:
            for data_set in data_sets:
                entries, count = self._build_data_set_entry(data_set, tool)
                data_sets_out.extend(entries)
                measurement_sets += count
        return data_sets_in, data_sets_out, measurement_sets

    def _build_data_set_entry(self, data_set: DataSet, tool: Tool):
        entries = []

        tool_configs = self._build_tool_configs(tool)
        for tool_config in tool_configs:
            tool_entry = self._build_tool_entry(tool, tool_config, data_set)
            decompress_file = data_set.value
            decompress_file = decompress_file.with_stem(f"{data_set.value.stem}_{tool_config.strength.name.lower()}")
            data_set_name = f"{data_set.name.lower()}"
            target_name = f"{data_set.name.lower()}_{tool.name.lower()}"
            if tool.value.threading == Threading.MULTI:
                threading_name = tool_config.threading.name.lower()
                decompress_file = decompress_file.with_stem(f"{decompress_file.stem}_{threading_name}")
                target_name= f"{target_name}_{threading_name}"
            suffixes = decompress_file.suffixes + [tool.value.extension]
            decompress_file = decompress_file.with_suffix("".join(suffixes))
            entry = {
                "data_set_name": data_set_name,
                "data_set_file": data_set.value,
                "target_name": target_name,
                "target_file": f"{decompress_file}",
                "tools": [tool_entry],
            }
            entries.append(entry)

        return entries, len(entries)

    def _build_tool_configs(self, tool: Tool):
        tool_configs = []
        if tool.value.threading == Threading.SINGLE:
            threadings = [Threading.SINGLE]
        else:
            threadings = list(Threading)
        for threading in threadings:
            tool_config = ToolConfig(mode=OperationMode.COMPRESS, strength=CompressionStrength.DEFAULT,
                                     threading=threading)
            tool_configs.append(tool_config)
        return tool_configs
