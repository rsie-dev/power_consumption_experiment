import logging
from pathlib import Path
from abc import abstractmethod

import pandas as pd
from statsmodels.regression.linear_model import RegressionResultsWrapper
from statsmodels.regression.linear_model import OLSResults

from data_processor.util import tabulate_pint
from .base_statistics import BaseStatistics
from .factor_means import FactorMeans, FactorMeansCIResult
from .effect_names import get_effect_name


class EnergyEfficiencyStatistics(BaseStatistics):
    def __init__(self, resources: Path, logfile: bool, n_boot: int, seed: int | None,
                 threading: str, mode: str, create_tex: bool):
        super().__init__(resources, logfile, n_boot, seed, threading, mode, create_tex)
        self._logger = logging.getLogger(self.__class__.__name__)

    def extra_log_base(self, input_file: Path) -> str:
        base = input_file.stem.removeprefix("ee_")
        return "ee_stat_%s_%s" % (self.effect_name, base)

    def _validate_data_quantity(self, df: pd.DataFrame):
        super()._validate_data_quantity(df)
        n_negative = (df[self.base_column] < 0).sum()
        self._logger.info("Negative energy count:        %d", n_negative)

    @property
    @abstractmethod
    def effect_name(self) -> str:
        pass

    @property
    def display_unit (self) -> str:
        return "MiB/joule"

    def _create_rank_file(self, effect_ranking_df: pd.DataFrame, input_file: Path) -> None:
        base = input_file.stem.removeprefix("ee_")
        rank_filename = "ee_%s_effect_ranking_%s_%s_%s.csv" % (self.effect_name, self._threading, self._mode, base)
        self._create_csv(rank_filename, effect_ranking_df)

    def _create_means_ci_files(self, factor_means: FactorMeansCIResult, input_file: Path):
        base = input_file.stem.removeprefix("ee_")
        for effect, df in factor_means.means.items():
            filename = "ee_%s_effect_%s_%s_%s_%s.csv" % (self.effect_name, self._threading, self._mode, effect, base)
            self._frameio.persist(df, self._resources / filename)

    def _print_factor_table(self, factor_means_values_df: pd.DataFrame):
        display_df = factor_means_values_df.copy()
        display_df["Factor"] = display_df["Factor"].map(get_effect_name)
        mean_column = "mean_%s" % self.base_column
        for column in [mean_column, "ci_low", "ci_high"]:
            display_df[column] = display_df[column].pint.to(self.display_unit)

        unit = display_df["ci_low"].pint.units

        display_df["95% CI"] = (
                display_df["ci_low"].pint.magnitude.map("{:.3f}".format)
                + " - "
                + display_df["ci_high"].pint.to(unit).pint.magnitude.map("{:.3f}".format)
        )
        display_df = display_df.drop(columns=["ci_low", "ci_high"])

        headers = ["Factor", "Level", "Marginal geometric mean %s" % self.base_column, "95% HC3 CI"]
        colalign = ["global"] * len(headers)
        colalign[3] = "center"
        table_str = tabulate_pint(
            display_df,
            showindex=False,
            headers=headers,
            colalign=colalign,
        )
        self._logger.info("%s values for: %s %s", self.base_column, self._threading, self._mode)
        self._logger.info(table_str)

    def _create_factor_means_table(self, comp: pd.DataFrame,
                                   model: RegressionResultsWrapper, model_hc3: OLSResults):
        factor_means = FactorMeans(self.factors, self.base_column, self.display_unit)
        levels, grid, x_grid = factor_means.calculate_x_grid(comp, model)
        beta, cov = factor_means.get_beta_cov(model, model_hc3)

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

        mean_column = "mean_%s" % self.base_column
        mean_values = mean_values[
            [
                "Factor",
                "Level",
                mean_column,
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
             mean_column,
             ],
            ascending=[True, False]
        )

        return mean_values

    def _post_process(self, effect_ranking_df: pd.DataFrame, input_file: Path) -> None:
        if not self._create_tex:
            return

        base = input_file.stem.removeprefix("ee_")
        filename = "ee_%s_effect_ranking_%s_%s_%s_full.tex" % (self.effect_name, self._threading, self._mode, base)
        self._effect_ranking.create_effect_rank_stability_tex(effect_ranking_df, self._resources / filename)
        top_df = effect_ranking_df[
            effect_ranking_df["omega_squared"].cumsum().shift(fill_value=0).lt(0.95)
        ]
        filename = "ee_%s_effect_ranking_%s_%s_%s.tex" % (self.effect_name, self._threading, self._mode, base)
        self._effect_ranking.create_effect_rank_stability_tex(top_df, self._resources / filename)
