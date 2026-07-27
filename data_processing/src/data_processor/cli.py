import logging.config
import argparse
from pathlib import Path

from ruamel.yaml import YAML

from .multimeter import MultimeterValidate
from .stats import Statistics
from .calc import CompressionRatio, Throughput, Power, EnergyConsumption


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

        common_parser = argparse.ArgumentParser(add_help=False)
        common_parser.add_argument('used_energy_file', type=Path)
        common_parser.add_argument('-r', '--resources', type=Path, default=Path("resources"),
                                   help="resource output folder")

        parser_stats = subparsers.add_parser('stats', help="basic statistics", parents=[common_parser])
        parser_stats.set_defaults(func=self._stats)

        parser_calc = subparsers.add_parser('calc', help="calculate subcommands")
        subparsers_calc = parser_calc.add_subparsers(required=True, dest="subcommand", title='subcommands',
                                                     description='valid subcommands', help='sub-command help')

        common_calc_parser = argparse.ArgumentParser(add_help=False, parents=[common_parser])
        common_calc_parser.add_argument('--tex', action='store_true', help="create latex table")
        common_calc_parser.add_argument('--no-tool', nargs="*", help="tools to skip")
        common_calc_parser.add_argument('--no-data-set', nargs="*", help="data sets to skip")

        parser_calc_cr = subparsers_calc.add_parser('cr', help="calculate compression ratio",
                                                    parents=[common_calc_parser])
        parser_calc_cr.set_defaults(func=self._calc_cr)

        parser_calc_trough = subparsers_calc.add_parser('tp', help="calculate throughput",
                                                        parents=[common_calc_parser])
        parser_calc_trough.set_defaults(func=self._calc_through)

        parser_calc_power = subparsers_calc.add_parser('power', help="calculate power",
                                                       parents=[common_calc_parser])
        parser_calc_power.set_defaults(func=self._calc_power)

        parser_energy = subparsers_calc.add_parser('energy', help="energy subcommands")
        subparsers_energy = parser_energy.add_subparsers(required=True, dest="subcommand", title='subcommands',
                                                         description='valid subcommands', help='sub-command help')

        parser_energy_consumption = subparsers_energy.add_parser('consumption', help="calculate energy consumption",
                                                                 parents=[common_calc_parser])
        parser_energy_consumption.add_argument('--idle-power', type=Path, required=True, help="idle power CSV file")
        parser_energy_consumption.set_defaults(func=self._calc_energy_consumption)

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
        statistics.process(args.used_energy_file)

    def _calc_cr(self, args):
        resources_folder = args.resources
        resources_folder.mkdir(parents=True, exist_ok=True)
        cr = CompressionRatio(resources_folder)
        cr.process(args.used_energy_file, args.tex,
                   args.no_tool if args.no_tool else [],
                   args.no_data_set if args.no_data_set else []
                   )

    def _calc_through(self, args):
        resources_folder = args.resources
        resources_folder.mkdir(parents=True, exist_ok=True)
        tp = Throughput(resources_folder)
        tp.process(args.used_energy_file, args.tex,
                   args.no_tool if args.no_tool else [],
                   args.no_data_set if args.no_data_set else []
                   )

    def _calc_power(self, args):
        resources_folder = args.resources
        resources_folder.mkdir(parents=True, exist_ok=True)
        tp = Power(resources_folder)
        tp.process(args.used_energy_file, args.tex,
                   args.no_tool if args.no_tool else [],
                   args.no_data_set if args.no_data_set else []
                   )

    def _calc_energy_consumption(self, args):
        resources_folder = args.resources
        resources_folder.mkdir(parents=True, exist_ok=True)
        ec = EnergyConsumption(resources_folder)
        params = EnergyConsumption.Params(
            used_energy_file=args.used_energy_file,
            no_tool=args.no_tool if args.no_tool else [],
            no_dataset=args.no_data_set if args.no_data_set else [],
            idle_power=args.idle_power,
        )
        ec.process(params)

def app():
    processor = Processor()
    return processor.main()
