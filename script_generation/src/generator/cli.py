import logging.config
import argparse
from pathlib import Path

from ruamel.yaml import YAML

from generator.generator_type import GeneratorType
from generator.tools import Tool
from generator.data_set import DataSet


class Generator:
    def __init__(self):
        self._logger = logging.getLogger(self.__class__.__name__)

    def _get_app_folder(self):
        script = Path(__file__).resolve()
        folder = script.parent
        return folder

    def _start_logging(self, args):
        log_file_name = args.logFile
        num_log_level = 50 - min(4, 2 + args.verbose) * 10
        log_level = logging.getLevelName(num_log_level)

        yaml_config = self._get_logging_config()
        yaml_config['handlers']['console']['level'] = log_level
        if log_file_name:
            yaml_config['handlers']['file']['filename'] = log_file_name
        logging.config.dictConfig(yaml_config)

    def _get_logging_config(self):
        folder = self._get_app_folder()
        config = folder / 'logging.yaml'
        with open(config, "rt", encoding="UTF_8") as f:
            yaml = YAML(typ="safe")
            return yaml.load(f)

    def _generate(self, args):
        generator_type = GeneratorType[args.type.upper()]
        script_folder = Path.cwd() / "scripts"
        sg = generator_type.create(script_folder)
        tools = self._get_tools(args)
        data_sets = self._get_data_sets(args)
        script_folder.mkdir(parents=True, exist_ok=True)
        sg.generate(tools, data_sets, args)

    def _get_tools(self, args) -> list[Tool]:
        if args.no_tool:
            tools = list(Tool)
            skip_tools = [Tool[tool.upper()] for tool in args.no_tool]
            tools = [t for t in tools if t not in skip_tools]
        else:
            tools = [Tool[tool.upper()] for tool in args.tool]
        return tools

    def _get_data_sets(self, args) -> list[DataSet]:
        if args.no_data_set:
            data_sets = list(DataSet)
            skip_ds = [DataSet[ds.upper()] for ds in args.no_data_set]
            data_sets = [ds for ds in data_sets if ds not in skip_ds]
        else:
            data_sets = [DataSet[ds.upper()] for ds in args.data_set]
        return data_sets

    def main(self):
        parser = argparse.ArgumentParser()
        default = ' (default: %(default)s)'
        parser.add_argument('-v', '--verbose', action='count', default=1, help="set the verbosity level" + default)
        parser.add_argument('-l', '--logFile', help="logfile name")
        parser.add_argument('--runs', default=30, help="amount of runs" + default)
        parser.add_argument('--head-delay', type=int, help="head delay per measurement")
        parser.add_argument('--tail-delay', type=int, help="tail delay per measurement")
        parser.add_argument('--warmup', type=int, default=120, help="warmup task time in S, 0 to disable" + default)
        parser.add_argument('--mon-temp', type=float, help="activate temperature monitoring with max MON_TEMP delta")
        parser.add_argument('--data-folder', default=Path("data"), help="data folder" + default)
        parser.add_argument('--host', required=True, help="DUT host name")
        parser.add_argument('--ip', required=True, help="DUT ip address")
        parser.add_argument('--multimeter', required=True, help="multimeter serial number")
        parser.add_argument('-t', '--type',
                            choices=[type.name.lower() for type in GeneratorType],
                            default=GeneratorType.HOST.name.lower(), help="generator type" + default)
        tool_group = parser.add_mutually_exclusive_group()
        tool_group.add_argument('--tool', nargs="+",
                                choices=[tool.name.lower() for tool in Tool],
                                default=[tool.name.lower() for tool in Tool],
                                help="tools to use")
        tool_group.add_argument('--no-tool', nargs="*",
                                choices=[tool.name.lower() for tool in Tool],
                                help="tools to skip")
        data_set_group = parser.add_mutually_exclusive_group()
        data_set_group.add_argument('--data-set', nargs="*",
                                    choices=[ds.name.lower() for ds in DataSet],
                                    default=[ds.name.lower() for ds in DataSet],
                                    help="data sets to skip")
        data_set_group.add_argument('--no-data-set', nargs="*",
                                    choices=[ds.name.lower() for ds in DataSet],
                                    help="data sets to skip")

        args = parser.parse_args()

        self._start_logging(args)
        try:
            self._generate(args)
            return 0
        except KeyboardInterrupt:
            self._logger.warning("User cancel")
        except Exception as e:  # pylint: disable=broad-exception-caught
            self._logger.exception("Error: %s", e)
        return 1


def app():
    generator = Generator()
    return generator.main()
