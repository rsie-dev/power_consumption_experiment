import logging
from pathlib import Path

from generator.single_script_generator import SingleScriptGenerator
from generator.tools import Tool
from generator.data_set import DataSet
from generator.tool_config import OperationMode


class ModeScriptGenerator(SingleScriptGenerator):
    def __init__(self, script_folder: Path):
        super().__init__(script_folder)
        self._logger = logging.getLogger(self.__class__.__name__)

    def _write_scripts(self, tools: list[Tool], data_sets: list[DataSet], template, args) -> None:
        data_sets_compress, measurement_sets_compress = self._get_measurement_sets_compress(tools, data_sets)
        self._write_measurement_sets(template, args, data_sets_compress,
                                     measurement_sets_compress, OperationMode.COMPRESS)

        data_sets_decompress, measurement_sets_decompress = self._get_measurement_sets_decompress(tools, data_sets)
        self._write_measurement_sets(template, args, data_sets_decompress,
                                     measurement_sets_decompress, OperationMode.DECOMPRESS)

    def _write_measurement_sets(self, template, args, data_sets, measurement_sets, mode: OperationMode):
        self._logger.info("Generating %d %s measurement sets", measurement_sets, mode.name.lower())
        data = {
            "args": args,
            "data_sets": data_sets,
        }
        host_script = self._script_folder / f"{args.host}_{mode.name.lower()}.py"
        self._generate_script(host_script, template, data)
