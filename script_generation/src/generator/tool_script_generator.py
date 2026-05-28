import logging
from pathlib import Path

from generator.host_script_generator import HostScriptGenerator
from generator.tools import Tool
from generator.data_set import DataSet
from generator.tool_config import CompressionStrength, OperationMode


class ToolScriptGenerator(HostScriptGenerator):
    def __init__(self, script_folder: Path):
        super().__init__(script_folder)
        self._logger = logging.getLogger(self.__class__.__name__)

    def _write_scripts(self, tools: list[Tool], data_sets: list[DataSet],
                       compression_strengths: list[CompressionStrength], modes: list[OperationMode],
                       template, args) -> None:
        base_data = {
            "args": args,
            "input_sets": [ds.name for ds in data_sets],
            "strengths": [s.name for s in compression_strengths],
            "modes": [mode.name for mode in modes],
        }
        for tool in tools:
            all_data_sets = []
            measurement_sets = 0
            if OperationMode.COMPRESS in modes:
                data_sets_compress, measurement_sets_compress = self._get_measurement_sets_compress([tool],
                                                                                                    data_sets,
                                                                                                    compression_strengths)
                all_data_sets += data_sets_compress
                measurement_sets += measurement_sets_compress
            if OperationMode.DECOMPRESS in modes:
                data_sets_decompress, measurement_sets_decompress = self._get_measurement_sets_decompress([tool],
                                                                                                          data_sets)
                all_data_sets += data_sets_decompress
                measurement_sets += measurement_sets_decompress
            self._logger.info("Generating %d %s measurement sets", measurement_sets,
                              tool.name.lower())
            self._write_measurement_sets(base_data, template, args, all_data_sets, tool)

    def _write_measurement_sets(self, base_data, template, args, all_data_sets, tool: Tool):
        data = base_data | {
            "tools": [tool.name],
            "data_sets": all_data_sets,
        }
        host_script = self._script_folder / f"{args.host}_{tool.name.lower()}.py"
        self._generate_script(host_script, template, data)
