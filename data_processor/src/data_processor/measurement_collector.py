import logging
from pathlib import Path
import csv
import datetime

from data_processor.tool_config import OperationMode, CompressionStrength, Threading, ToolConfig
from data_processor.measurement_info import MeasurementInfo
from data_processor.run_info import RunInfo
from data_processor.measurement import Timings, ElectricalMeasurement, Measurement, Q_


class MeasurementCollector:
    def __init__(self):
        self._logger = logging.getLogger(self.__class__.__name__)

    def collect_measurements(self, measurement_folder: Path):
        measurement_info = self._get_measurement_info(measurement_folder.stem)
        self._logger.info("Collecting measurements of: %s", measurement_info)
        run_folders = list(measurement_folder.iterdir())
        self._logger.info("Found %d runs", len(run_folders))
        runs: list[RunInfo]= []
        for run_folder in run_folders:
            run_info = self._process_run_folder(run_folder, measurement_info.tool_config.mode)
            runs.append(run_info)
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

    def _process_run_folder(self, run_folder, mode: OperationMode) -> RunInfo:
        self._logger.debug("processing run folder: %s", run_folder)
        run = int(run_folder.stem[4:])
        count = None
        count_file = run_folder / 'count_stdout.csv'
        if mode == OperationMode.COMPRESS:
            with open(count_file, encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)
                first = next(iter(reader))
                count = int(first["count_B"])

        end, start = self._read_markers(run_folder)

        timings = self._read_timings(run_folder)

        readings = self._read_measurement(run_folder)
        measurement = Measurement(start=start, end=end, count=count, timings=timings, readings=readings)
        return RunInfo(run=run, measurement=measurement )

    def _read_timings(self, run_folder):
        with open(run_folder / 'timings.csv', encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            it = iter(reader)
            start_entry = next(it)
            real = datetime.timedelta(seconds=float(start_entry["real_S"]))
            user = datetime.timedelta(seconds=float(start_entry["user_S"]))
            sys = datetime.timedelta(seconds=float(start_entry["sys_S"]))
            timings = Timings(real=real, user=user, sys=sys)
        return timings

    def _read_markers(self, run_folder):
        with open(run_folder / 'markers.csv', encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            it = iter(reader)
            start_entry = next(it)
            start = datetime.datetime.fromisoformat(start_entry["timestamp"])
            end_entry = next(it)
            end = datetime.datetime.fromisoformat(end_entry["timestamp"])
        return end, start

    def _read_measurement(self, run_folder: Path) -> list[ElectricalMeasurement]:
        readings = []
        with open(run_folder / 'multimeter.csv', encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                timestamp = datetime.datetime.fromisoformat(row["timestamp"])
                rel_time = datetime.timedelta(seconds=float(row["rel_time_S"]))
                voltage = Q_(float(row["voltage_V"]), "volt")
                current = Q_(float(row["current_A"]), "ampere")
                measurement = ElectricalMeasurement(timestamp=timestamp, relative_time=rel_time,
                                                    voltage=voltage, current=current)
                readings.append(measurement)
        return readings
