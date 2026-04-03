from enum import Enum
from typing import Type
from pathlib import Path

from generator.script_generator import ScriptGenerator
from generator.host_script_generator import HostScriptGenerator
from generator.mode_script_generator import ModeScriptGenerator
from generator.tool_script_generator import ToolScriptGenerator
from generator.dataset_script_generator import DataSetScriptGenerator
from generator.data_gen_script_generator import DataGenScriptGenerator
from generator.baseline_script_generator import BaselineScriptGenerator


class GeneratorType(Enum):
    HOST = HostScriptGenerator
    MODE = ModeScriptGenerator
    TOOL = ToolScriptGenerator
    DATASET = DataSetScriptGenerator
    DATAGEN = DataGenScriptGenerator
    BASELINE = BaselineScriptGenerator

    def create(self, script_folder: Path) -> ScriptGenerator:
        generator_class: Type[ScriptGenerator] = self.value
        return generator_class(script_folder)
