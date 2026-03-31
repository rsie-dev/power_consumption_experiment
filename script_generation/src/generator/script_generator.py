import logging
from pathlib import Path
from abc import abstractmethod

from jinja2 import Environment, PackageLoader, StrictUndefined

from generator.tools import Tool
from generator.tool_config import ToolConfig, OperationMode, CompressionStrength, Threading
from generator.data_set import DataSet


class ScriptGenerator:
    def __init__(self, script_folder: Path):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._script_folder = script_folder

    def generate(self, tools: list[Tool], data_sets: list[DataSet], args):
        env = Environment(
            loader=PackageLoader("generator"),
            trim_blocks=True,
            lstrip_blocks=True,
            undefined=StrictUndefined,
        )

        self._logger.info("Generate scripts for: %s", args.host)
        self._logger.info("Using tools:          %s", ", ".join([tool.name for tool in tools]))
        self._logger.info("Using data sets:      %s", ", ".join([data_set.name for data_set in data_sets]))
        template_name = self._get_template_name()
        template = env.get_template(template_name)
        self._write_scripts(tools, data_sets, template, args)

    @abstractmethod
    def _get_template_name(self) -> str:
        pass

    @abstractmethod
    def _write_scripts(self, tools: list[Tool], data_sets: list[DataSet], template, args) -> None:
        pass

    def _build_tool_entry(self, tool: Tool, tool_config: ToolConfig, data_set: DataSet):
        entry = {
            "tool": tool,
            "measurement_tags": self._get_measurement_tags(tool, tool_config, data_set),
            "tool_args": self._get_tool_config(tool, tool_config),
        }
        return entry

    def _get_tool_config(self, tool: Tool, config: ToolConfig):
        tool_args = []
        if config.mode == OperationMode.COMPRESS:
            tool_args.append(tool.value.compress)
        else:
            tool_args.append(tool.value.decompress)

        tool_args.extend([tool.value.keep, tool.value.to_stdout])

        if config.strength == CompressionStrength.MIN:
            tool_args.append(tool.value.min)
        elif config.strength == CompressionStrength.MAX:
            tool_args.append(tool.value.max)

        if config.threading == Threading.SINGLE:
            tool_args.append(tool.value.single_thread)
        else:
            tool_args.append(tool.value.multi_thread)

        return " ".join(tool_args)

    def _get_measurement_tags(self, tool: Tool, config: ToolConfig, data_set: DataSet):
        tags = []
        tags.append(tool.name.lower())
        tags.append(config.mode.name.lower())
        tags.append(data_set.name.lower())
        if config.mode == OperationMode.COMPRESS:
            tags.append(config.strength.name.lower())
        if tool.value.threading == Threading.MULTI:
            tags.append(config.threading.name.lower())
        return tags

    def _generate_script(self, script_file: Path, template, data):
        self._logger.info("Generate: %s", script_file.relative_to(Path.cwd()))
        with script_file.open("wt", encoding="UTF_8") as script:
            output = template.render(data)
            script.write(output)
