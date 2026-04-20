import logging
from pathlib import Path

import pandas as pd

from data_aggregator.common import OperationMode, CompressionStrength, Threading, ToolConfig
from data_aggregator.common import MeasurementInfo
from data_aggregator.ingest import RunCollector
from data_aggregator.calculate import PowerCalculator
from data_aggregator.util import FrameIO
from data_aggregator import ureg


class RunAggregator:
    def __init__(self, resources_folder: Path):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._resources_folder = resources_folder

    def aggregate(self, host: str, host_folder: Path):
        self._logger.info("Aggregate measurements of: %s", host)
        measurement_folders = list(host_folder.iterdir())
        self._logger.info("Found %d measurements", len(measurement_folders))
        calculator = PowerCalculator()
        for measurement_folder in measurement_folders:
            if not measurement_folder.is_dir():
                continue
            measurement_info = self._get_measurement_info(host, measurement_folder.stem)
            run_collector = RunCollector()
            runs = run_collector.collect_runs(measurement_info, measurement_folder)
            df = self._preprocess_runs(measurement_info, runs)
            df = calculator.calculate_power(df)

            preprocessed_name = self._build_preprocessed_path(measurement_info)
            csv_file = self._resources_folder / preprocessed_name
            frame_io = FrameIO()
            frame_io.persist(df, csv_file)

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

    def _preprocess_runs(self, measurement_info: MeasurementInfo, runs):
        name = self._build_name(measurement_info)
        self._logger.info("Preprocess runs for: %s", name)
        all_runs = []
        entries_count = 0
        for run in runs:
            entries_count += len(run.measurement.readings)
            cut_run = self._cut_lead_tail(run)
            energy_df = self._calculate_energy(cut_run)
            energy_df["real"] = run.measurement.timings.real.total_seconds() * ureg.second
            energy_df["real"] = energy_df["real"].astype("pint[second]")
            energy_df["size"] = run.measurement.count
            energy_df["size"] = energy_df["size"].astype("pint[byte]")
            all_runs.append(energy_df)

        df_all = pd.concat(all_runs)
        df_all = df_all.sort_values('run', kind="stable")
        self._logger.info("Raw entries: %d, after cut: %d", entries_count, len(df_all))
        return df_all

    def _calculate_energy(self, df_run: pd.DataFrame) -> pd.DataFrame:
        df_run["energy"] = df_run.voltage * df_run.current
        return df_run

    def _cut_lead_tail(self, run) -> pd.DataFrame:
        measurement = run.measurement
        df = measurement.readings
        filtered_df = df[(df['timestamp'] >= measurement.start) & (df['timestamp'] <= measurement.end)]
        return  filtered_df

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
