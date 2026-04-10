import logging
from pathlib import Path
import csv
import datetime
from typing import Generator

import pandas as pd

from data_aggregator.common import OperationMode
from data_aggregator.common import RunInfo
from data_aggregator.common import Timings, Measurement
from data_aggregator.common import MeasurementInfo


class RunCollector:
    def __init__(self):
        self._logger = logging.getLogger(self.__class__.__name__)

    def collect_runs(self, measurement_info: MeasurementInfo,
                     measurement_folder: Path) -> Generator[RunInfo, None, None]:
        self._logger.debug("Collecting runs of: %s", measurement_info)

        run_folders = list(measurement_folder.iterdir())
        self._logger.info("Found %d runs", len(run_folders))
        for run_folder in run_folders:
            run_info = self.collect_run(measurement_info.tool_config.mode, run_folder)
            yield run_info

    def collect_run(self, mode: OperationMode, run_folder: Path) -> RunInfo:
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

        readings = self._read_measurement(run_folder, run)
        measurement = Measurement(start=start, end=end, count=count, timings=timings, readings=readings)
        return RunInfo(run=run, measurement=measurement)

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

    def _read_measurement(self, run_folder: Path, run: int) -> pd.DataFrame:
        #self._logger.debug("read: %s", (run_folder / 'multimeter.csv'))
        df = pd.read_csv(run_folder / 'multimeter.csv', parse_dates=["timestamp"], date_format="ISO8601",
                         skipinitialspace=True)
        df["voltage_V"] = df["voltage_V"].astype("pint[volt]")
        df["current_A"] = df["current_A"].astype("pint[ampere]")
        df = df.rename(columns={'voltage_V': 'voltage', 'current_A': 'current'})
        df = df.drop('temperature_C', axis=1)
        df['run'] = run
        df = df[['run', 'timestamp', "voltage", "current"]]
        return df
