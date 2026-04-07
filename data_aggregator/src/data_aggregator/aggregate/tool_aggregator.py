import logging
from pathlib import Path

import pandas as pd

from .measurement_info import MeasurementInfo
from .tool_config import OperationMode, Threading


class ToolAggregator:
    def __init__(self):
        self._logger = logging.getLogger(self.__class__.__name__)

    def aggregate_runs(self, resources_folder: Path, measurement_info: MeasurementInfo, runs):
        name = self._build_aggregated_name(measurement_info)
        self._logger.info("Aggregate runs for: %s", name)
        all_runs = []
        entries_count = 0
        for run in runs:
            entries_count += len(run.measurement.readings)
            cut_run = self._cut_lead_tail(run)
            energy_df = self._calculate_energy(cut_run)
            all_runs.append(energy_df)

        df_all = pd.concat(all_runs)
        self._logger.info("Raw entries: %d, cut: %d", entries_count, len(df_all))

        aggregated_name = self._build_aggregated_path(measurement_info)
        csv_file = resources_folder / aggregated_name
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

    def _build_aggregated_name(self, measurement_info: MeasurementInfo) -> str:
        tokens = self._build_aggregated_tokens(measurement_info)
        return "_".join(tokens[:-1])

    def _build_aggregated_path(self, measurement_info: MeasurementInfo) -> Path:
        tokens = self._build_aggregated_tokens(measurement_info)
        return Path("_".join(tokens)).with_suffix(".csv")

    def _build_aggregated_tokens(self, measurement_info: MeasurementInfo) -> list[str]:
        tokens = []
        tokens.append(measurement_info.host)
        tokens.append(measurement_info.tool)
        tokens.append(measurement_info.tool_config.mode.name.lower())
        tokens.append(measurement_info.dataset)
        if measurement_info.tool_config.mode == OperationMode.COMPRESS:
            tokens.append(measurement_info.tool_config.strength.name.lower())
        if measurement_info.tool_config.threading != Threading.NONE:
            tokens.append(measurement_info.tool_config.threading.name.lower())
        tokens.append("aggregated")
        return tokens
