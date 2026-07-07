from enum import Enum
from typing import Type
from pathlib import Path

from generator.script_generator import ScriptGenerator
from generator.host_script_generator import HostScriptGenerator
from generator.data_gen_script_generator import DataGenScriptGenerator
from generator.baseline_script_generator import BaselineScriptGenerator
from generator.template_args import TemplateArgs


class GeneratorType(Enum):
    HOST = HostScriptGenerator
    DATAGEN = DataGenScriptGenerator
    BASELINE = BaselineScriptGenerator

    def create(self, script_folder: Path, prefix: str, template_args: TemplateArgs) -> ScriptGenerator:
        generator_class: Type[ScriptGenerator] = self.value
        return generator_class(script_folder, prefix, template_args)
