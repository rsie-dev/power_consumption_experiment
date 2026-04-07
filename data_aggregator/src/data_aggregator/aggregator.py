import logging
from pathlib import Path

from .measurement_collector import MeasurementCollector
from .tool_aggregator import ToolAggregator


class Aggregator:
    def __init__(self):
        self._logger = logging.getLogger(self.__class__.__name__)

    def aggregate_raw_data(self, host_folder: Path):
        measurement_folders = list(host_folder.iterdir())
        measurement_collector = MeasurementCollector()
        for measurement_folder in measurement_folders[:1]:
            tool_aggregator = ToolAggregator()
            measurement_info, runs = measurement_collector.collect_measurements(measurement_folder)
            tool_aggregator.aggregate_runs(measurement_info, runs)
