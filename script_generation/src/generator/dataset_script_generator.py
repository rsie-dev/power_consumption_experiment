import logging
from pathlib import Path

from generator.host_script_generator import HostScriptGenerator
from generator.tools import Tool
from generator.data_set import DataSet
from generator.tool_config import CompressionStrength, OperationMode


class DataSetScriptGenerator(HostScriptGenerator):
    def __init__(self, script_folder: Path, prefix: str):
        super().__init__(script_folder, prefix)
        self._logger = logging.getLogger(self.__class__.__name__)

    def _write_scripts(self, tools: list[Tool], data_sets: list[DataSet],
                       compression_strengths: list[CompressionStrength], modes: list[OperationMode],
                       template, args) -> None:
        for data_set in data_sets:
            data_sets_compress, measurement_sets_compress = self._get_measurement_sets_compress(tools,
                                                                                                [data_set],
                                                                                                compression_strengths)
            data_sets_decompress, measurement_sets_decompress = self._get_measurement_sets_decompress(tools, [data_set])
            all_data_sets = data_sets_compress + data_sets_decompress
            measurement_sets = measurement_sets_compress + measurement_sets_decompress

            self._logger.info("Generating %d %s measurement sets", measurement_sets,
                              data_set.name.lower())
            self._write_measurement_sets(template, args, all_data_sets, data_set)

    def _write_measurement_sets(self, template, args, data_sets, data_set: DataSet):
        data = {
            "args": args,
            "data_sets": data_sets,
        }
        host_script = self._script_folder / f"{self._prefix}{args.host}_{data_set.name.lower()}.py"
        self._generate_script(host_script, template, data)
