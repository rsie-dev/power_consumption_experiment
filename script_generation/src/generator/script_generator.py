import logging

from jinja2 import Environment, PackageLoader, select_autoescape


class ScriptGenerator:
    def __init__(self):
        self._logger = logging.getLogger(self.__class__.__name__)

    def generate(self, args):
        env = Environment(
            loader=PackageLoader("generator"),
            #autoescape=select_autoescape()
        )

        template = env.get_template('experiment.jinja')
        data = {
            "runs": 10,
            "host": "raspi5",
            "ip": "192.168.1.102",
            "multimeter": "07D1A5642160",
            #"head_delay": 2,
            "tool": "gzip",
            "tool_args": "-zkc",
            "input_file": "test.data",
        }
        output = template.render(data)
        print(output)