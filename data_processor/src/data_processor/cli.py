import logging.config
import argparse
from pathlib import Path

from ruamel.yaml import YAML

from .measurement_collector import MeasurementCollector
from .tool_aggregator import ToolAggregator


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

    def _process_raw_data(self, args):
        if not self._valid_input_folder(args.resources):
            raise RuntimeError("not a valid resource folder: %s" % args.resources)
        host_folder = self._collect_host_folders(args.resources)
        measurement_folders = list(host_folder.iterdir())
        measurement_collector = MeasurementCollector()
        for measurement_folder in measurement_folders[:1]:
            tool_aggregator = ToolAggregator()
            measurement_info, runs = measurement_collector.collect_measurements(measurement_folder)
            tool_aggregator.aggregate_runs(measurement_info, runs)

    def _collect_host_folders(self, resources: Path) -> Path:
        self._logger.info("Collecting input folders in: %s", resources)
        for child in resources.iterdir():
            if child.is_dir():
                self._logger.debug("found host folder: %s", child)
                return child
        raise RuntimeError("no host folder found")

    def _valid_input_folder(self, folder: Path) -> bool:
        log = folder / "experiment.log"
        if not log.exists():
            return False
        script = folder / f"{folder.stem}.py"
        if not script.exists():
            return False
        return True

    def main(self):
        parser = argparse.ArgumentParser()
        default = ' (default: %(default)s)'
        parser.add_argument('-v', '--verbose', action='count', default=1, help="set the verbosity level" + default)
        parser.add_argument('-l', '--logFile', help="logfile name")
        parser.add_argument('-r', '--resources', type=Path, required=True, help="resource folder")
        args = parser.parse_args()

        self._start_logging(args)
        try:
            self._process_raw_data(args)
            return 0
        except KeyboardInterrupt:
            self._logger.warning("User cancel")
        except Exception as e:  # pylint: disable=broad-exception-caught
            self._logger.exception("Error: %s", e)
        return 1


def app():
    processor = Processor()
    return processor.main()
