import logging
from pathlib import Path

from generator.single_script_generator import SingleScriptGenerator
from generator.tools import Tool
from generator.tool_config import ToolConfig, OperationMode, CompressionStrength, Threading
from generator.data_set import DataSet


class ModeScriptGenerator(SingleScriptGenerator):
    def __init__(self):
        super().__init__()
        self._logger = logging.getLogger(self.__class__.__name__)

    def _get_template_name(self) -> str:
        return "single_experiment.jinja"

    def _write_scripts(self, template, script_folder: Path, args) -> None:
        tools = [tool for tool in Tool]

        data_sets_compress, measurement_sets_compress = self._get_measurement_sets_compress(tools)
        self._write_measurement_sets(template, script_folder, args, data_sets_compress,
                                     measurement_sets_compress, OperationMode.COMPRESS)

        data_sets_decompress, measurement_sets_decompress = self._get_measurement_sets_decompress(tools)
        self._write_measurement_sets(template, script_folder, args, data_sets_decompress,
                                     measurement_sets_decompress, OperationMode.DECOMPRESS)

    def _write_measurement_sets(self, template, script_folder: Path, args, data_sets, measurement_sets, mode: OperationMode):
        self._logger.info("Generating %d %s measurement sets", measurement_sets, mode.name.lower())
        data = {
            "args": args,
            "data_sets": data_sets,
        }
        host_script = script_folder / f"{args.host}_{mode.name.lower()}.py"
        self._generate_script(host_script, template, data)

    def _get_measurement_sets_compress(self, tools: list[Tool]):
        tool_configs_compress = self._build_tool_configs(OperationMode.COMPRESS)
        data_sets_compress = []
        for data_set in DataSet:
            data_sets_compress.append(self._build_data_set_entry(data_set, tools, tool_configs_compress))
        measurement_sets_compress = len(tools) * len(tool_configs_compress) * len(data_sets_compress)
        return data_sets_compress, measurement_sets_compress

    def _get_measurement_sets_decompress(self, tools: list[Tool]):
        tool_configs_decompress = self._build_tool_configs(OperationMode.DECOMPRESS)
        data_sets_decompress = []
        for tool in tools:
            for data_set in DataSet:
                entries = self._build_data_set_entry_decompress(data_set, tool, tool_configs_decompress)
                data_sets_decompress.extend(entries)
        measurement_sets_decompress = len(data_sets_decompress)
        return data_sets_decompress, measurement_sets_decompress
