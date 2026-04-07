import logging
from pathlib import Path

from data_aggregator.common import OperationMode, CompressionStrength, Threading, ToolConfig
from data_aggregator.common import MeasurementInfo
from data_aggregator.ingest import RunCollector
from .tool_aggregator import ToolAggregator


class Aggregator:
    def __init__(self):
        self._logger = logging.getLogger(self.__class__.__name__)

    def aggregate_raw_data(self, host: str, host_folder: Path):
        self._logger.info("Aggregate measurements of: %s", host)
        measurement_folders = list(host_folder.iterdir())
        self._logger.info("Found %d measurements", len(measurement_folders))
        resources_folder = Path("resources")
        for measurement_folder in measurement_folders[:1]:
            measurement_info = self._get_measurement_info(host, measurement_folder.stem)
            run_collector = RunCollector()
            runs = run_collector.collect_runs(measurement_info, measurement_folder)
            tool_aggregator = ToolAggregator()
            tool_aggregator.aggregate_runs(resources_folder, measurement_info, runs)

    def _get_measurement_info(self, host: str, tags: str) -> MeasurementInfo:
        tokens = tags.split("_")
        tool = tokens[0]
        mode = OperationMode[tokens[1].upper()]
        dataset = tokens[2]
        threading = Threading.NONE
        if mode == OperationMode.COMPRESS:
            strength = CompressionStrength[tokens[3].upper()]
            if len(tokens) > 4:
                threading = Threading[tokens[4].upper()]
        else:
            strength = CompressionStrength.DEFAULT
            if len(tokens) > 3:
                threading = Threading[tokens[3].upper()]

        tool_config = ToolConfig(mode=mode, strength=strength, threading=threading)
        measurement_info = MeasurementInfo(host=host, tool=tool, dataset=dataset, tool_config=tool_config)
        return measurement_info
