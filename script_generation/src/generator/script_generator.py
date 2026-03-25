import logging
from pathlib import Path

from jinja2 import Environment, PackageLoader

from generator.tools import Tool
from generator.tool_config import ToolConfig, OperationMode, CompressionStrength, Threading


class ScriptGenerator:
    def __init__(self):
        self._logger = logging.getLogger(self.__class__.__name__)

    def generate(self, script_folder: Path, args):
        env = Environment(
            loader=PackageLoader("generator")
        )

        tool: Tool = Tool.bzip2
        print("tool: %s" % tool.name)

        template = env.get_template('experiment.jinja')
        tool_config = ToolConfig(mode=OperationMode.COMPRESS, strength=CompressionStrength.DEFAULT,
                                 threading=Threading.SINGLE)
        data = {
            "runs": 10,
            "host": "raspi5",
            "ip": "192.168.1.102",
            "multimeter": "07D1A5642160",
            #"head_delay": 2,
            "tool": tool.name,
            "tool_tags": self._get_tool_tags(tool, tool_config),
            "tool_args": self._get_tool_config(tool, tool_config),
            "input_file": "test.data",
        }
        output = template.render(data)
        print(output)

    def _get_tool_config(self, tool: Tool, config: ToolConfig):
        tool_args = []
        if config.mode == OperationMode.COMPRESS:
            tool_args.append(tool.compress)
        else:
            tool_args.append(tool.decompress)

        tool_args.extend([tool.keep, tool.to_stdout])

        if config.strength == CompressionStrength.MIN:
            tool_args.append(tool.min)
        elif config.strength == CompressionStrength.MAX:
            tool_args.append(tool.max)

        if config.threading == Threading.SINGLE:
            tool_args.append(tool.single_thread)
        else:
            tool_args.append(tool.multi_thread)

        return " ".join(tool_args)

    def _get_tool_tags(self, tool: Tool, config: ToolConfig):
        tags = []
        tags.append(config.mode.name.lower())
        if config.mode == OperationMode.COMPRESS:
            tags.append(config.strength.name.lower())
        tags.append(config.threading.name.lower())
        return tags
