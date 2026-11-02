import logging
from pathlib import Path
from abc import ABC, abstractmethod
from contextlib import contextmanager, nullcontext

import pandas as pd

from data_processor.util import FrameIO


class Processor(ABC):
    def __init__(self, resources: Path):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._frameio = FrameIO()
        self._resources = resources

    def _create_csv(self, file_name: str|Path, df: pd.DataFrame) -> None:
        csv_file = self._resources / Path(file_name)
        self._frameio.persist(df, csv_file)


class LoggingProcessor(Processor):
    def __init__(self, resources: Path, logfile: bool):
        super().__init__(resources)
        self._logger = logging.getLogger(self.__class__.__name__)
        self._logfile = logfile

    def process(self, input_file: Path):
        if self._logfile:
            name = self.extra_log_name(input_file)
            logfile_path = self._resources / f"{name}.log"
            cm = self._add_logfile(logfile_path)
            self._logger.info("create extra logfile: %s", logfile_path)
        else:
            cm = nullcontext()
        with cm:
            self._analyze(input_file)

    @abstractmethod
    def extra_log_name(self, input_file: Path) -> str:
        pass

    @abstractmethod
    def _analyze(self, input_file: Path) -> None:
        pass

    @contextmanager
    def _add_logfile(self, logfile: Path):
        handler = logging.FileHandler(logfile, mode="w")
        file_logger = [""]
        try:
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(message)s')
            handler.setFormatter(formatter)
            for logger in file_logger:
                logging.getLogger(logger).addHandler(handler)
            yield
        finally:
            try:
                for logger in file_logger:
                    logging.getLogger(logger).removeHandler(handler)
            finally:
                handler.close()
