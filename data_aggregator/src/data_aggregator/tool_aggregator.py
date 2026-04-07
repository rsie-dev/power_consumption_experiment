import logging

from .measurement_info import MeasurementInfo


class ToolAggregator:
    def __init__(self):
        self._logger = logging.getLogger(self.__class__.__name__)

    def aggregate_runs(self, measurement_info: MeasurementInfo, runs):
        for run in runs:
            self._cut_lead_tail(run)

    def _cut_lead_tail(self, run):
        pass
