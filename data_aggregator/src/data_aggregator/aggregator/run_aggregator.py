import logging
from pathlib import Path

import pandas as pd

from data_aggregator.common import OperationMode, CompressionStrength, Threading, ToolConfig
from data_aggregator.common import MeasurementInfo
from data_aggregator.ingest import RunCollector
from data_aggregator.calculate import TrapezoidEnergyCalculator, PowerCalculator
from data_aggregator.util import FrameIO
from data_aggregator import ureg


class RunAggregator:
    def __init__(self, resources_folder: Path):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._resources_folder = resources_folder

    def aggregate(self, host: str, host_folder: Path):
        frame_io = FrameIO()
        for measurement_info, df in self.collect_runs(host, host_folder):
            preprocessed_name = self._build_preprocessed_path(measurement_info)
            csv_file = self._resources_folder / preprocessed_name
            frame_io.persist(df, csv_file)

    def collect_runs(self, host: str, host_folder: Path):
        # ToDo:
        #log run duration, real time, measurement count per run and after clean

        self._logger.info("Aggregate measurements of: %s", host)
        measurement_folders = list(host_folder.iterdir())
        self._logger.info("Found %d measurements", len(measurement_folders))
        calculator = TrapezoidEnergyCalculator()
        for measurement_folder in measurement_folders:
            if not measurement_folder.is_dir():
                continue
            measurement_info = self._get_measurement_info(host, measurement_folder.stem)
            run_collector = RunCollector()
            runs = run_collector.collect_runs(measurement_info, measurement_folder)
            df = self._preprocess_runs(measurement_info, runs)
            df = calculator.calculate_energy(df)
            df.insert(loc=0, column='host', value=measurement_info.host)
            df.insert(loc=1, column='tool', value=measurement_info.tool)
            df.insert(loc=2, column='dataset', value=measurement_info.dataset)
            df.insert(loc=3, column='mode', value=measurement_info.tool_config.mode.name.lower())
            df.insert(loc=4, column='strength', value=measurement_info.tool_config.strength.name.lower())
            df.insert(loc=5, column='threading', value=measurement_info.tool_config.threading.name.lower())
            yield measurement_info, df

    def _get_measurement_info(self, host: str, tags: str) -> MeasurementInfo:
        if tags == "baseline":
            tool_config = ToolConfig(mode=OperationMode.BASELINE, strength=CompressionStrength.DEFAULT, threading=Threading.NONE)
            measurement_info = MeasurementInfo(host=host, tool="sleep", dataset="none", tool_config=tool_config)
            return measurement_info

        try:
            tokens = tags.split("_")
            tool = tokens[0]
            mode = OperationMode[tokens[1].upper()]
            dataset = tokens[2]
            strength = CompressionStrength[tokens[3].upper()]
            threading = Threading.NONE
            if len(tokens) > 4:
                threading = Threading[tokens[4].upper()]

            tool_config = ToolConfig(mode=mode, strength=strength, threading=threading)
            measurement_info = MeasurementInfo(host=host, tool=tool, dataset=dataset, tool_config=tool_config)
            return measurement_info
        except IndexError as e:
            raise ValueError(f"host {host} invalid tags: {tags}") from e

    def _preprocess_runs(self, measurement_info: MeasurementInfo, runs):
        name = self._build_name(measurement_info)
        self._logger.info("Preprocess runs for: %s", name)
        all_runs = []
        entries_count = 0
        power_calculator = PowerCalculator()
        for run in runs:
            entries_count += len(run.measurement.readings)
            cut_run = self._cut_lead_tail(run)
            if len(cut_run.index) < 3:
                raise ValueError("no or too few samples after cutting: %s" % len(cut_run.index))
            power_df = power_calculator.calculate_power(cut_run)
            power_df["real"] = run.measurement.timings.real.total_seconds() * ureg.second
            power_df["real"] = power_df["real"].astype("pint[second]")
            power_df["size"] = run.measurement.count
            power_df["size"] = power_df["size"].astype("pint[byte]")
            all_runs.append(power_df)

        df_all = pd.concat(all_runs)
        df_all = df_all.sort_values('run', kind="stable")
        self._logger.debug("Raw entries: %d, after cut: %d", entries_count, len(df_all))
        return df_all

    def _cut_lead_tail(self, run) -> pd.DataFrame:
        measurement = run.measurement
        df = measurement.readings
        filtered_df = df[(df['timestamp'] >= measurement.start) & (df['timestamp'] <= measurement.end)]
        return filtered_df

    def _build_name(self, measurement_info: MeasurementInfo) -> str:
        tokens = self._build_name_tokens(measurement_info)
        return "_".join(tokens[1:])

    def _build_preprocessed_path(self, measurement_info: MeasurementInfo) -> Path:
        tokens = self._build_name_tokens(measurement_info)
        return Path("_".join(tokens)).with_suffix(".csv")

    def _build_name_tokens(self, measurement_info: MeasurementInfo) -> list[str]:
        tokens = []
        tokens.append("preprocessed")
        tokens.append(measurement_info.host)
        tokens.append(measurement_info.tool)
        tokens.append(measurement_info.tool_config.mode.name.lower())
        tokens.append(measurement_info.dataset)
        if measurement_info.tool_config.mode == OperationMode.COMPRESS:
            tokens.append(measurement_info.tool_config.strength.name.lower())
        if measurement_info.tool_config.threading != Threading.NONE:
            tokens.append(measurement_info.tool_config.threading.name.lower())
        return tokens
