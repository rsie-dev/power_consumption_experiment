import logging.config
import argparse
from pathlib import Path

from ruamel.yaml import YAML

from generator.script_generator import ScriptGenerator
from generator.generator_type import GeneratorType


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
        sg = generator_type.create()
        script_folder = Path.cwd() / "scripts"
        self._logger.info("Generate scripts for: %s", args.host)
        script_folder.mkdir(parents=True, exist_ok=True)
        sg.generate(script_folder, args)

    def main(self):
        parser = argparse.ArgumentParser()
        default = ' (default: %(default)s)'
        parser.add_argument('-v', '--verbose', action='count', default=1, help="set the verbosity level" + default)
        parser.add_argument('-l', '--logFile', help="logfile name")
        parser.add_argument('--runs', default=30, help="amount of runs" + default)
        parser.add_argument('--head-delay', type=int, help="head delay per measurement")
        parser.add_argument('--tail-delay', type=int, help="tail delay per measurement")
        parser.add_argument('--data-folder', default=Path("data"), help="data folder" + default)
        parser.add_argument('--host', required=True, help="DUT host name")
        parser.add_argument('--ip', required=True, help="DUT ip address")
        parser.add_argument('--multimeter', required=True, help="multimeter serial number")
        parser.add_argument('-t', '--type',
                            choices=[type.name.lower() for type in GeneratorType],
                            default=GeneratorType.SINGLE.name.lower(), help="generator type" + default)

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
