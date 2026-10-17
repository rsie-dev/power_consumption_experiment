import logging
from pathlib import Path
from abc import abstractmethod

import pandas as pd
from statsmodels.regression.linear_model import RegressionResultsWrapper
from statsmodels.regression.linear_model import OLSResults

from data_processor.util import tabulate_pint
from .base_statistics import BaseStatistics
from .bootstrap_ci_calculator import BootstrapCiCalculator
from .factor_means import FactorMeans
from .effect_names import get_effect_name


class EnergyEfficiencyStatistics(BaseStatistics):
    def __init__(self, resources: Path, n_boot: int, seed: int | None, threading: str, mode: str, create_tex: bool):
        super().__init__(resources, threading, mode, create_tex)
        self._logger = logging.getLogger(self.__class__.__name__)
        self._display_unit = "MiB/joule"
        self._bootstrap_ci_calculator = BootstrapCiCalculator(self.factors, n_boot, seed)

    def process(self, ee_file: Path):
        self._log_section("Process %s %s" % (self._threading, self._mode))

        df = self._frameio.load(ee_file)
        comp = self._extract_data(df)
        comp = self._calculate_base(comp)

        self._validate_data(comp)
        self._calculate_statistics(comp)

    @property
    @abstractmethod
    def base_column(self) -> str:
        pass

    @abstractmethod
    def _calculate_base(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    def _validate_data(self, comp: pd.DataFrame):
        self._validate_data_quantity(comp, self.base_column)

    def _validate_data_quantity(self, comp: pd.DataFrame, base_column: str):
        super()._validate_data_quantity(comp, base_column)
        n_negative = (comp[base_column] < 0).sum()
        print("Negative energy count         %d" % n_negative)

    @abstractmethod
    def _calculate_statistics(self, comp: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        pass

    def _print_factor_table(self, factor_means_values_df: pd.DataFrame):
        display_df = factor_means_values_df.copy()
        display_df["Factor"] = display_df["Factor"].map(get_effect_name)
        mean_column = "mean_%s" % self.base_column
        for column in [mean_column, "ci_low", "ci_high"]:
            display_df[column] = display_df[column].pint.to(self._display_unit)

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
        print("%s values for: %s %s" % (self.base_column, self._threading, self._mode))
        print(table_str)

    def _create_factor_means_table(self, comp: pd.DataFrame,
                                   model: RegressionResultsWrapper, model_hc3: OLSResults):
        factor_means = FactorMeans(self.factors, self.base_column, self._display_unit)
        levels, grid, x_grid = factor_means.calculate_x_grid(comp, model)
        beta, cov = factor_means.get_beta_cov(model, model_hc3)

        tables = []
        for factor in self.factors:
            table = factor_means.marginal_mean_ci(factor, levels, grid, x_grid, beta, cov, comp).rename(columns={
                factor: "Level",
            })

            table.insert(0, "Factor", factor)
            tables.append(table)

        throughput_values = pd.concat(
            tables,
            ignore_index=True
        )

        mean_column = "mean_%s" % self.base_column
        throughput_values = throughput_values[
            [
                "Factor",
                "Level",
                mean_column,
                "ci_low",
                "ci_high",
            ]
        ]

        throughput_values["Factor"] = pd.Categorical(
            throughput_values["Factor"],
            categories=self.factors,
            ordered=True
        )
        throughput_values = throughput_values.sort_values(
            ["Factor",
             mean_column,
             ],
            ascending=[True, False]
        )

        return throughput_values
