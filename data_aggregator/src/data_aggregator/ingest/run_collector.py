import logging
from pathlib import Path
import datetime
from typing import Generator

from data_aggregator.common import OperationMode
from data_aggregator.common import RunInfo
from data_aggregator.common import Timings, Measurement
from data_aggregator.common import MeasurementInfo
from data_aggregator.util import FrameIO
from data_aggregator import ureg


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
        frame_io = FrameIO()
        run = int(run_folder.stem[4:])
        count = None
        count_file = run_folder / 'count_stdout.csv'
        if mode == OperationMode.COMPRESS:
            df = frame_io.load(count_file)
            count_quantity = df["count"].iloc[0]
            count = int(count_quantity.to(ureg.byte).magnitude)

        end, start = self._read_markers(run_folder)
        timings = self._read_timings(run_folder)

        readings = frame_io.load(run_folder / 'multimeter.csv')
        readings['run'] = run
        measurement = Measurement(start=start, end=end, count=count, timings=timings, readings=readings)
        return RunInfo(run=run, measurement=measurement)

    def _read_timings(self, run_folder):
        frame_io = FrameIO()
        df = frame_io.load(run_folder / 'timings.csv')
        real = datetime.timedelta(seconds=df["real"].iloc[0].to(ureg.second).magnitude)
        user = datetime.timedelta(seconds=df["user"].iloc[0].to(ureg.second).magnitude)
        sys = datetime.timedelta(seconds=df["sys"].iloc[0].to(ureg.second).magnitude)
        timings = Timings(real=real, user=user, sys=sys)
        return timings

    def _read_markers(self, run_folder):
        frame_io = FrameIO()
        df = frame_io.load(run_folder / 'markers.csv')
        start = df.loc[df["kind"] == "START", "timestamp"].iloc[0]
        end = df.loc[df["kind"] == "END", "timestamp"].iloc[0]
        return end, start
