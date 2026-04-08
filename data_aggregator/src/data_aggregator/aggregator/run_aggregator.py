import logging
from pathlib import Path

import pandas as pd

from data_aggregator.common import OperationMode, CompressionStrength, Threading, ToolConfig
from data_aggregator.common import MeasurementInfo
from data_aggregator.ingest import RunCollector


class RunAggregator:
    def __init__(self, resources_folder: Path):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._resources_folder = resources_folder

    def aggregate(self, host: str, host_folder: Path):
        self._logger.info("Aggregate measurements of: %s", host)
        measurement_folders = list(host_folder.iterdir())
        self._logger.info("Found %d measurements", len(measurement_folders))
        for measurement_folder in measurement_folders[:1]:
            measurement_info = self._get_measurement_info(host, measurement_folder.stem)
            run_collector = RunCollector()
            runs = run_collector.collect_runs(measurement_info, measurement_folder)
            self._preprocess_runs(self._resources_folder, measurement_info, runs)

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

    def _preprocess_runs(self, resources_folder: Path, measurement_info: MeasurementInfo, runs):
        name = self._build_name(measurement_info)
        self._logger.info("Preprocess runs for: %s", name)
        all_runs = []
        entries_count = 0
        for run in runs:
            entries_count += len(run.measurement.readings)
            cut_run = self._cut_lead_tail(run)
            energy_df = self._calculate_energy(cut_run)
            all_runs.append(energy_df)

        df_all = pd.concat(all_runs)
        self._logger.info("Raw entries: %d, cut: %d", entries_count, len(df_all))

        preprocessed_name = self._build_preprocessed_path(measurement_info)
        csv_file = resources_folder / preprocessed_name
        self._logger.info("Generate: %s", csv_file)
        df = df_all.pint.dequantify()
        df.to_csv(csv_file, encoding='UTF_8', index=False, header=True)

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
