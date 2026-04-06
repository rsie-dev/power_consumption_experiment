import logging
from pathlib import Path

from data_processor.tool_config import OperationMode, CompressionStrength, Threading, ToolConfig
from data_processor.measurement_info import MeasurementInfo
from data_processor.run_info import RunInfo
from .run_collector import RunCollector


class MeasurementCollector:
    def __init__(self):
        self._logger = logging.getLogger(self.__class__.__name__)

    def collect_measurements(self, measurement_folder: Path):
        measurement_info = self._get_measurement_info(measurement_folder.stem)
        self._logger.info("Collecting measurements of: %s", measurement_info)
        #run_folders = list(measurement_folder.iterdir())
        #self._logger.info("Found %d runs", len(run_folders))

        run_collector = RunCollector()
        runs = run_collector.collect_runs(measurement_folder, measurement_info.tool_config.mode)
        return runs

    def _get_measurement_info(self, tags: str) -> MeasurementInfo:
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
        measurement_info = MeasurementInfo(tool=tool, dataset=dataset, tool_config=tool_config)
        return measurement_info
