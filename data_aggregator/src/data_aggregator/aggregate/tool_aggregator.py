import logging
from pathlib import Path

import pandas as pd

from .measurement_info import MeasurementInfo
from .tool_config import OperationMode, Threading


class ToolAggregator:
    def __init__(self):
        self._logger = logging.getLogger(self.__class__.__name__)

    def aggregate_runs(self, resources_folder: Path, measurement_info: MeasurementInfo, runs):
        self._logger.info("Aggregate runs for %s", measurement_info)
        all_runs = []
        entries_count = 0
        for run in runs:
            entries_count += len(run.measurement.readings)
            cut_run = self._cut_lead_tail(run)
            all_runs.append(cut_run)

        df_all = pd.concat(all_runs)
        self._logger.info("Raw entries: %d, cut: %d", entries_count, len(df_all))

        aggregated_name = self._build_aggregated_name(measurement_info)
        csv_file = resources_folder / aggregated_name
        self._logger.info("Generate: %s", csv_file)
        df = df_all.pint.dequantify()
        df.to_csv(csv_file, encoding='UTF_8', index=False, header=True)

    def _cut_lead_tail(self, run) -> pd.DataFrame:
        df = run.measurement.readings
        filtered_df = df[(df['timestamp'] >= run.measurement.start) & (df['timestamp'] <= run.measurement.end)]
        return  filtered_df

    def _build_aggregated_name(self, measurement_info: MeasurementInfo) -> Path:
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
        return Path("_".join(tokens)).with_suffix(".csv")
