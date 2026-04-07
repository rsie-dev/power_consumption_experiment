import logging
from pathlib import Path
from typing import Generator

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

        resources_folder = Path("resources")
        resources_folder.mkdir(parents=True, exist_ok=True)

        run_collector = RunCollector()
        run_folders = list(measurement_folder.iterdir())
        self._logger.info("Found %d runs", len(run_folders))
        runs = []
        for run_folder in run_folders[:1]:
            run_info = run_collector.collect_run(run_folder, measurement_info.tool_config.mode)
            #df = run_info.measurement.readings
            #csv_file = resources_folder / Path(measurement_folder.stem).with_suffix(".csv")
            #self._logger.info("Generate: %s", csv_file)
            #df = df.pint.dequantify()
            #df.to_csv(csv_file, encoding='UTF_8', index=False, header=True)
            runs.append(run_info)
        return measurement_info, runs

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
