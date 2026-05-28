import logging
from pathlib import Path

from generator.host_script_generator import HostScriptGenerator
from generator.tools import Tool
from generator.data_set import DataSet
from generator.tool_config import OperationMode, CompressionStrength


class ModeScriptGenerator(HostScriptGenerator):
    def __init__(self, script_folder: Path):
        super().__init__(script_folder)
        self._logger = logging.getLogger(self.__class__.__name__)

    def _write_scripts(self, tools: list[Tool], data_sets: list[DataSet],
                       compression_strengths: list[CompressionStrength], modes: list[OperationMode],
                       template, args) -> None:
        base_data = {
            "args": args,
            "tools": [tool.name for tool in tools],
            "input_sets": [ds.name for ds in data_sets],
            "strengths": [s.name for s in compression_strengths],
        }
        if OperationMode.COMPRESS in modes:
            data_sets_compress, measurement_sets_compress = self._get_measurement_sets_compress(tools,
                                                                                                data_sets,
                                                                                                compression_strengths)
            self._logger.info("Generating %d %s measurement sets", measurement_sets_compress,
                              OperationMode.COMPRESS.name.lower())
            self._write_measurement_sets(base_data, template, args, data_sets_compress, OperationMode.COMPRESS)

        if OperationMode.DECOMPRESS in modes:
            data_sets_decompress, measurement_sets_decompress = self._get_measurement_sets_decompress(tools, data_sets)
            self._logger.info("Generating %d %s measurement sets", measurement_sets_decompress,
                              OperationMode.DECOMPRESS.name.lower())
            self._write_measurement_sets(base_data, template, args, data_sets_decompress, OperationMode.DECOMPRESS)

    def _write_measurement_sets(self, base_data, template, args, data_sets, mode: OperationMode):
        data = base_data | {
            "modes": [mode.name],
            "data_sets": data_sets,
        }
        host_script = self._script_folder / f"{args.host}_{mode.name.lower()}.py"
        self._generate_script(host_script, template, data)
