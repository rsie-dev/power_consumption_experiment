import logging
from pathlib import Path

from data_processor.util import FrameIO


class Calculator:
    def __init__(self, resources: Path):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._frameio = FrameIO()
        self._resources = resources
