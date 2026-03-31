from enum import Enum
from typing import Type
from pathlib import Path

from generator.script_generator import ScriptGenerator
from generator.single_script_generator import SingleScriptGenerator
from generator.mode_script_generator import ModeScriptGenerator
from generator.tool_script_generator import ToolScriptGenerator
from generator.dataset_script_generator import DataSetScriptGenerator
from generator.data_script_generator import DataScriptGenerator


class GeneratorType(Enum):
    SINGLE = SingleScriptGenerator
    MODE = ModeScriptGenerator
    TOOL = ToolScriptGenerator
    DATASET = DataSetScriptGenerator
    DATA = DataScriptGenerator

    def create(self, script_folder: Path) -> ScriptGenerator:
        generator_class: Type[ScriptGenerator] = self.value
        return generator_class(script_folder)
