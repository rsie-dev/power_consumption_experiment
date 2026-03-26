import logging
from pathlib import Path

from generator.single_script_generator import SingleScriptGenerator
from generator.tools import Tool
from generator.tool_config import OperationMode


class ModeScriptGenerator(SingleScriptGenerator):
    def __init__(self):
        super().__init__()
        self._logger = logging.getLogger(self.__class__.__name__)

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

