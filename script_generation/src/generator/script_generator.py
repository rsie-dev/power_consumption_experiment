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

        template = env.get_template('experiment.jinja')

        tool_config = ToolConfig(mode=OperationMode.COMPRESS, strength=CompressionStrength.DEFAULT,
                                 threading=Threading.SINGLE)

        tools = []
        tools.append(_build_tool_entry(Tool.lzop, tool_config))
        tools.append(_build_tool_entry(Tool.gzip, tool_config))

        data = {
            "runs": 10,
            "host": "raspi5",
            "ip": "192.168.1.102",
            "multimeter": "07D1A5642160",
            #"head_delay": 2,
            "tools": tools,
            "input_file": "test.data",
        }
        output = template.render(data)
        print(output)


def _build_tool_entry(tool: Tool, tool_config: ToolConfig):
    entry = {
        "tool": tool,
        "tool_tags": _get_tool_tags(tool_config),
        "tool_args": _get_tool_config(tool, tool_config),
    }
    return entry


def _get_tool_config(tool: Tool, config: ToolConfig):
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


def _get_tool_tags(config: ToolConfig):
    tags = []
    tags.append(config.mode.name.lower())
    if config.mode == OperationMode.COMPRESS:
        tags.append(config.strength.name.lower())
    tags.append(config.threading.name.lower())
    return tags
