import logging.config
import argparse
from pathlib import Path

from ruamel.yaml import YAML

from .aggregator import RunAggregator, PowerAggregator
from .calculate import PowerCalculator, AverageCalculator


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

    def _aggregate_runs(self, args):
        if not self._valid_input_folder(args.raw_data):
            raise RuntimeError("not a valid resource folder: %s" % args.raw_data)
        host_folders = self._collect_host_folder(args.raw_data)
        resources_folder = args.resources
        resources_folder.mkdir(parents=True, exist_ok=True)
        aggregator = RunAggregator(resources_folder)
        for host_folder in host_folders:
            aggregator.aggregate(host_folder.stem, host_folder)

    def _collect_host_folder(self, resources: Path) -> list[Path]:
        self._logger.info("Collecting host folder in: %s", resources)
        hosts = []
        for child in resources.iterdir():
            if child.is_dir():
                self._logger.debug("found host folder: %s", child)
                hosts.append(child)
        return hosts

    def _valid_input_folder(self, folder: Path) -> bool:
        log = folder / "experiment.log"
        if not log.exists():
            return False
        script = folder / f"{folder.stem}.py"
        if not script.exists():
            return False
        return True

    def _aggregate_power(self, args):
        resources_folder = args.resources
        resources_folder.mkdir(parents=True, exist_ok=True)
        aggregator = PowerAggregator(resources_folder)
        aggregator.aggregate(args.power_data)

    def _calculate_power(self, args):
        resources_folder = args.resources
        resources_folder.mkdir(parents=True, exist_ok=True)
        calculator = PowerCalculator(resources_folder)
        calculator.calculate(args.preprocessed_data)

    def _calculate_averages(self, args):
        resources_folder = args.resources
        resources_folder.mkdir(parents=True, exist_ok=True)
        calculator = AverageCalculator(resources_folder)
        calculator.calculate(args.power_data)

    def main(self):
        parser = argparse.ArgumentParser()
        default = ' (default: %(default)s)'
        parser.add_argument('-v', '--verbose', action='count', default=1, help="set the verbosity level" + default)
        parser.add_argument('-l', '--logFile', help="logfile name")
        parser.add_argument('-r', '--resources', type=Path, default=Path("resources"),
                            help="resource output folder")

        subparsers = parser.add_subparsers(required=True, dest="subcommand", title='subcommands',
                                           description='valid subcommands', help='sub-command help')

        parser_aggregate = subparsers.add_parser('aggregate')
        subparsers_aggregate = parser_aggregate.add_subparsers(required=True, dest="subcommand",
                                                               title='aggregate subcommands',
                                                               description='valid subcommands', help='sub-command help')

        parser_aggregate_runs = subparsers_aggregate.add_parser('runs', help="aggregate raw measurement data")
        parser_aggregate_runs.add_argument('-d', '--raw-data', type=Path, required=True,
                                       help="raw data measurement folder")
        parser_aggregate_runs.set_defaults(func=self._aggregate_runs)

        parser_aggregate_power = subparsers_aggregate.add_parser('power', help="aggregate power of runs")
        parser_aggregate_power.add_argument('-d', '--power-data', type=Path, required=True,
                                            help="power usage file")
        parser_aggregate_power.set_defaults(func=self._aggregate_power)

        parser_calculate = subparsers.add_parser('calculate')
        subparsers_calculate = parser_calculate.add_subparsers(required=True, dest="subcommand",
                                                               title='calculate subcommands',
                                                               description='valid subcommands', help='sub-command help')

        parser_calculate_power = subparsers_calculate.add_parser('power',
                                                 help="calculate used power from preprocessed measurement data")
        parser_calculate_power.add_argument('-d', '--preprocessed-data', type=Path, required=True,
                                      help="preprocessed file")
        parser_calculate_power.set_defaults(func=self._calculate_power)

        parser_calculate_averages = subparsers_calculate.add_parser('averages',
                                                                 help="calculate average power usages")
        parser_calculate_averages.add_argument('-d', '--power-data', type=Path, required=True,
                                               help="power usage file")
        parser_calculate_averages.set_defaults(func=self._calculate_averages)

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


def app():
    processor = Processor()
    return processor.main()
