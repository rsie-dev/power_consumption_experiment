import logging
from pathlib import Path

from generator.script_generator import ScriptGenerator
from generator.tools import Tool
from generator.data_set import DataSet
from generator.tool_config import CompressionStrength, OperationMode
from generator.template_args import TemplateArgs


class BaselineScriptGenerator(ScriptGenerator):
    def __init__(self, script_folder: Path, prefix: str, template_args: TemplateArgs):
        super().__init__(script_folder, prefix, template_args)
        self._logger = logging.getLogger(self.__class__.__name__)

    def _get_template_name(self) -> str:
        return "baseline.jinja"

    def _build_data(self, tools: list[Tool], data_sets: list[DataSet],
                    compression_strengths: list[CompressionStrength], modes: list[OperationMode],
                    template) -> dict:
        data = {
            "template_args": self._template_args,
            "sleep_time": 15,
        }
        return data

    def _build_script_name(self, tools: list[Tool], data_sets: list[DataSet],
                               compression_strengths: list[CompressionStrength], modes: list[OperationMode]) -> Path:
        host_script = self._script_folder / f"{self._prefix}{self._template_args.host}_baseline.py"
        return host_script
