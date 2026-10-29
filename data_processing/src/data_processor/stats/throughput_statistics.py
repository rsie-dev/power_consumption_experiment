import logging
from pathlib import Path

import pandas as pd
import numpy as np
from statsmodels.regression.linear_model import RegressionResultsWrapper
from statsmodels.regression.linear_model import OLSResults

from data_processor import ureg
from data_processor.util import tabulate_pint
from .base_statistics import BaseStatistics
from .effect_names import get_effect_name
from .factor_means import FactorMeans, FactorMeansCIResult


class ThroughputStatistics(BaseStatistics):
    def __init__(self, resources: Path, logfile: bool, n_boot: int, seed: int | None,
                 threading: str, mode: str, create_tex: bool):
        super().__init__(resources, logfile, n_boot, seed, threading, mode, create_tex)
        self._logger = logging.getLogger(self.__class__.__name__)

    def extra_log_base(self, input_file: Path) -> str:
        base = input_file.stem.removeprefix("tp_")
        return "tp_stat_%s" % base

    @property
    def base_column(self) -> str:
        return "throughput"

    @property
    def display_unit (self) -> str:
        return "MiB/s"

    def _validate_data(self, df: pd.DataFrame):
        super()._validate_data(df)

        throughput_bps = (
            df["throughput"]
            .pint.to("byte / second")
            .pint.magnitude
            .astype(float)
        )
        invalid = ~np.isfinite(throughput_bps) | (throughput_bps <= 0)
        if invalid.any():
            raise ValueError(f"Found {invalid.sum()} invalid throughput values")
        self._logger.info("Invalid throughput values:    %d", invalid.sum())

    def _create_means_ci_files(self, factor_means: FactorMeansCIResult, input_file: Path):
        base = input_file.stem.removeprefix("tp_")
        for effect, df in factor_means.means.items():
            filename = "tp_effect_%s_%s_%s_%s" % (self._threading, self._mode, effect, base) + ".csv"
            self._frameio.persist(df, self._resources / filename)

    def _calculate_base(self, df: pd.DataFrame) -> pd.DataFrame:
        self._logger.info("Log-transform throughput")
        reference = 1 * ureg.byte / ureg.second
        normalized = df["throughput"] / reference
        df["log_throughput"] = np.log(normalized).pint.magnitude
        return df

    def create_full_model(self, df: pd.DataFrame) -> RegressionResultsWrapper:
        full_model = self._stat_model.fit_full_model(df, "log_throughput", self.factors)
        return full_model

    def _print_factor_table(self, throughput_values_df: pd.DataFrame):
        display_df = throughput_values_df.copy()
        display_df["Factor"] = display_df["Factor"].map(get_effect_name)
        for column in ["mean_throughput", "ci_low", "ci_high"]:
            display_df[column] = display_df[column].pint.to(self.display_unit)

        unit = display_df["ci_low"].pint.units

        display_df["95% CI"] = (
                display_df["ci_low"].pint.magnitude.map("{:.3f}".format)
                + " - "
                + display_df["ci_high"].pint.to(unit).pint.magnitude.map("{:.3f}".format)
        )
        display_df = display_df.drop(columns=["ci_low", "ci_high"])

        headers = ["Factor", "Level", "Marginal geometric mean throughput", "95% HC3 CI"]
        colalign = ["global"] * len(headers)
        colalign[3] = "center"
        table_str = tabulate_pint(
            display_df,
            showindex=False,
            headers=headers,
            colalign=colalign,
        )
        self._logger.info("Throughput values for: %s %s", self._threading, self._mode)
        self._logger.info(table_str)

    def _create_factor_means_table(self, comp: pd.DataFrame,
                                   full_model: RegressionResultsWrapper, full_model_hc3: OLSResults):
        factor_means = FactorMeans(self.factors, self.base_column, self.display_unit)
        levels, grid, x_grid = factor_means.calculate_x_grid(comp, full_model)
        beta, cov = factor_means.get_beta_cov(full_model, full_model_hc3)

        tables = []
        for factor in self.factors:
            table = factor_means.marginal_mean_ci(factor, levels, grid, x_grid, beta, cov, comp).rename(columns={
                factor: "Level",
            })

            table.insert(0, "Factor", factor)
            tables.append(table)

        mean_values = pd.concat(
            tables,
            ignore_index=True
        )

        mean_values = mean_values[
            [
                "Factor",
                "Level",
                "mean_throughput",
                "ci_low",
                "ci_high",
            ]
        ]

        mean_values["Factor"] = pd.Categorical(
            mean_values["Factor"],
            categories=self.factors,
            ordered=True
        )
        mean_values = mean_values.sort_values(
            ["Factor",
             "mean_throughput",
             ],
            ascending=[True, False]
        )

        return mean_values

    def _post_process(self, effect_ranking_df: pd.DataFrame, input_file: Path) -> None:
        if not self._create_tex:
            return

        base = input_file.stem.removeprefix("tp_")
        filename = "tp_effect_ranking_%s_%s_%s_full" % (self._threading, self._mode, base) + ".tex"
        self._effect_ranking.create_effect_rank_stability_tex(effect_ranking_df, self._resources / filename)
        top_df = effect_ranking_df[
            effect_ranking_df["eta_squared"].cumsum().shift(fill_value=0).lt(0.95)
        ]
        filename = "tp_effect_ranking_%s_%s_%s" % (self._threading, self._mode, base) + ".tex"
        self._effect_ranking.create_effect_rank_stability_tex(top_df, self._resources / filename)
