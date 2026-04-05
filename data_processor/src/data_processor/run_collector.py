import logging
from pathlib import Path
import csv
import datetime
from collections.abc import Generator

import pint

from data_processor.tool_config import OperationMode
from data_processor.run_info import RunInfo
from data_processor.measurement import Timings, ElectricalMeasurement, Measurement


class RunCollector:
    def __init__(self):
        self._logger = logging.getLogger(self.__class__.__name__)

    def collect_runs(self, runs_folder: Path, mode: OperationMode) -> Generator[RunInfo]:
        run_folders = list(runs_folder.iterdir())
        self._logger.info("Found %d runs", len(run_folders))
        for run_folder in run_folders:
            run_info = self._process_run_folder(run_folder, mode)
            yield run_info

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
        ureg = pint.get_application_registry()
        Q_ = ureg.Quantity
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
