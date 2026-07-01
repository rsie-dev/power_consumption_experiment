import logging.config
import argparse
from pathlib import Path

from ruamel.yaml import YAML

from .multimeter import MultimeterValidate
from .stats import Statistics
from .calc import CompressionRatio, Throughput


class Processor:
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

    def main(self):
        parser = argparse.ArgumentParser()
        default = ' (default: %(default)s)'
        parser.add_argument('-v', '--verbose', action='count', default=1, help="set the verbosity level" + default)
        parser.add_argument('-l', '--logFile', help="logfile name")

        subparsers = parser.add_subparsers(required=True, dest="subcommand", title='subcommands',
                                           description='valid subcommands', help='sub-command help')

        parser_stats = subparsers.add_parser('stats', help="basic statistics")
        parser_stats.add_argument('used_power_file', type=Path)
        parser_stats.add_argument('-r', '--resources', type=Path, default=Path("resources"),
                                  help="resource output folder")
        parser_stats.set_defaults(func=self._stats)

        parser_calc = subparsers.add_parser('calc', help="calculate subcommands")
        subparsers_calc = parser_calc.add_subparsers(required=True, dest="subcommand", title='subcommands',
                                                     description='valid subcommands', help='sub-command help')

        parser_calc_cr = subparsers_calc.add_parser('cr', help="calculate compression ratio")
        parser_calc_cr.add_argument('used_power_file', type=Path)
        parser_calc_cr.add_argument('-r', '--resources', type=Path, default=Path("resources"),
                                    help="resource output folder")
        parser_calc_cr.add_argument('--tex', action='store_true', help="create latex table")
        parser_calc_cr.add_argument('--no-tool', nargs="*", help="tools to skip")
        parser_calc_cr.add_argument('--no-data-set', nargs="*", help="data sets to skip")
        parser_calc_cr.set_defaults(func=self._calc_cr)

        parser_calc_trough = subparsers_calc.add_parser('tp', help="calculate throughput")
        parser_calc_trough.add_argument('used_power_file', type=Path)
        parser_calc_trough.add_argument('-r', '--resources', type=Path, default=Path("resources"),
                                        help="resource output folder")
        parser_calc_trough.add_argument('--tex', action='store_true', help="create latex table")
        parser_calc_trough.add_argument('--no-tool', nargs="*", help="tools to skip")
        parser_calc_trough.add_argument('--no-data-set', nargs="*", help="data sets to skip")
        parser_calc_trough.set_defaults(func=self._calc_through)

        parser_multimeter = subparsers.add_parser('multimeter')
        subparsers_multimeter = parser_multimeter.add_subparsers(required=True, dest="subcommand",
                                                                title='multimeter subcommands',
                                                                description='valid subcommands',
                                                                help='sub-command help')

        parser_multimeter_validate = subparsers_multimeter.add_parser('validate',
                                                                      help="calculate average power usage")
        parser_multimeter_validate.set_defaults(func=self._multimeter_validate)

        args = parser.parse_args()

        self._start_logging(args)
        try:
            args.func(args)
            return 0
        except KeyboardInterrupt:
            self._logger.warning("User cancel")
        except Exception as e:  # pylint: disable=broad-exception-caught
            self._logger.exception("Error: %s", e)
        return 1

    def _multimeter_validate(self, args):
        validate = MultimeterValidate()
        validate.validate()

    def _stats(self, args):
        resources_folder = args.resources
        resources_folder.mkdir(parents=True, exist_ok=True)
        statistics = Statistics(resources_folder)
        statistics.process(args.used_power_file)

    def _calc_cr(self, args):
        resources_folder = args.resources
        resources_folder.mkdir(parents=True, exist_ok=True)
        cr = CompressionRatio(resources_folder)
        cr.process(args.used_power_file, args.tex,
                   args.no_tool if args.no_tool else [],
                   args.no_data_set if args.no_data_set else []
                   )

    def _calc_through(self, args):
        resources_folder = args.resources
        resources_folder.mkdir(parents=True, exist_ok=True)
        tp = Throughput(resources_folder)
        tp.process(args.used_power_file, args.tex,
                   args.no_tool if args.no_tool else [],
                   args.no_data_set if args.no_data_set else []
                   )

def app():
    processor = Processor()
    return processor.main()
