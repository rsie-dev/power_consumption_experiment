import logging

from jinja2 import Environment, PackageLoader, select_autoescape

from generator.tools import Tool


class ScriptGenerator:
    def __init__(self):
        self._logger = logging.getLogger(self.__class__.__name__)

    def generate(self, args):
        env = Environment(
            loader=PackageLoader("generator")
        )

        tool: Tool = Tool.bzip2
        print("tool: %s" % tool.name)

        template = env.get_template('experiment.jinja')
        data = {
            "runs": 10,
            "host": "raspi5",
            "ip": "192.168.1.102",
            "multimeter": "07D1A5642160",
            #"head_delay": 2,
            "tool": tool.name,
            "tool_tags": ["default"],
            "tool_args": self._get_tool_config(tool, {}),
            "input_file": "test.data",
        }
        output = template.render(data)
        print(output)

    def _get_tool_config(self, tool: Tool, config: dict):
        config = []
        config.extend([tool.compress, tool.keep, tool.to_stdout])
        return " ".join(config)