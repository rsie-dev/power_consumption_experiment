from enum import Enum
from typing import Type

from generator.script_generator import ScriptGenerator
from generator.single_script_generator import SingleScriptGenerator
from generator.mode_script_generator import ModeScriptGenerator


class GeneratorType(Enum):
    SINGLE = SingleScriptGenerator
    MODE = ModeScriptGenerator

    def create(self) -> ScriptGenerator:
        generator_class: Type[ScriptGenerator] = self.value
        return generator_class()
