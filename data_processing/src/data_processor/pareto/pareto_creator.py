import logging
from pathlib import Path
from dataclasses import dataclass, astuple, fields

import pandas as pd
import numpy as np
import tabulate
from paretoset import paretoset

from data_processor.processor import Processor

@dataclass(frozen=True)
class Data:
    cr_df: pd.DataFrame
    tp_df: pd.DataFrame
    ee_df: pd.DataFrame

    def __iter__(self):
        return iter(astuple(self))


class ParetoCreator(Processor):
    CR_KEY = ["tool", "dataset", "strength"]
    TRIAL_KEY = [
        "host", "tool", "dataset", "mode",
        "strength", "threading", "run",
    ]
    STRATUM_KEY = ["host", "dataset"]
    CONFIGURATION_KEY = ["tool", "strength", "threading"]
    GROUP_KEY = STRATUM_KEY + CONFIGURATION_KEY

    def __init__(self, resources: Path, mode: str):
        super().__init__(resources)
        self._logger = logging.getLogger(self.__class__.__name__)
        self._mode = mode
        self._n_boot = 10000
        self._rng = np.random.default_rng(20261105)

    def process(self, cr_file: Path, tp_file: Path, ee_file: Path):
        self._log_section("1. Data preparation")
        self._logger.info("load data")
        data = self._load_data(cr_file, tp_file, ee_file)

        self._log_section("1.3 Define subset")
        self._logger.info("Evaluate: %s", self._mode)
        data = self._filter_data(data)

        #for field in fields(data):
        #    value = getattr(data, field.name)
        #    self._logger.info("")
        #    self._logger.info("frame: %s", field.name)
        #    self._logger.info(value.dtypes)

        self._validate_data(data)
        analysis_df = self._integrate_data(data)
        config_df = self._configuration_level_estimation(analysis_df)
        self._pareto_front_identification(config_df)

    def _pareto_front_identification(self, config_df: pd.DataFrame):
        self._log_section("5 Pareto fronts")

        self._log_section("5.1 Identify deterministic Pareto fronts")
        config_df["pareto_optimal"] = False

        for _, indices in config_df.groupby(self.STRATUM_KEY, observed=True).groups.items():
            group = config_df.loc[indices]

            # Temporary canonical-unit values; Pint columns remain unchanged.
            objectives = pd.DataFrame({
                "throughput": group["throughput_mean"].pint.to(
                    "byte/second"
                ).pint.magnitude,
                "ee_net": group["ee_net_mean"].pint.to(
                    "byte/joule"
                ).pint.magnitude,
                "cr": group["cr"],
            })

            config_df.loc[indices, "pareto_optimal"] = paretoset(
                objectives,
                sense=["max", "max", "max"],
            )

        pareto_counts = (
            config_df.groupby(self.STRATUM_KEY, observed=True)["pareto_optimal"]
            .sum()
            .rename("pareto_configurations")
        )

        pareto_counts_df = pareto_counts.copy()
        pareto_counts_df = pareto_counts_df.reset_index()
        pareto_counts_df["host"] = pareto_counts_df["host"].where(
            pareto_counts_df["host"].ne(pareto_counts_df["host"].shift()), ""
        )
        table_str = tabulate.tabulate(
            pareto_counts_df,
            headers=["Host", "Dataset", "Count"],
            showindex=False,
        )
        self._logger.info(table_str)

        self._logger.info("")
        self._logger.info("Total Pareto-optimal configurations: %d", config_df["pareto_optimal"].sum())

    def _configuration_level_estimation(self, analysis_df: pd.DataFrame) -> pd.DataFrame:
        self._log_section("4 Configuration-level estimation")

        self._log_section("4.1 Define configurations and strata")
        self._define_configurations(analysis_df)

        self._log_section("4.2 Calculate configuration-level means")
        config_df = self._calculate_configuration_level_means(analysis_df)

        self._log_section("4.3 Calculate bootstrap confidence intervals")
        self._calculate_bootstrap_ci(analysis_df, config_df)
        return config_df

    def _calculate_bootstrap_ci(self, analysis_df: pd.DataFrame, config_df: pd.DataFrame):
        records = []
        for key, group in analysis_df.groupby(self.GROUP_KEY, observed=True, sort=False):
            indices = self._rng.integers(
                0, len(group),
                size=(self._n_boot, len(group)),
            )

            tp = group["throughput"].array.quantity
            ee = group["energy_efficiency_net"].array.quantity

            tp_boot = tp[indices].mean(axis=1)
            ee_boot = ee[indices].mean(axis=1)

            tp_ci = np.quantile(tp_boot, [0.025, 0.975])
            ee_ci = np.quantile(ee_boot, [0.025, 0.975])

            records.append(
                dict(
                    zip(self.GROUP_KEY, key),
                    throughput_ci_low=tp_ci[0],
                    throughput_ci_high=tp_ci[1],
                    ee_net_ci_low=ee_ci[0],
                    ee_net_ci_high=ee_ci[1],
                )
            )

        ci_df = pd.DataFrame(records)

        # Convert quantity-object columns into Pint columns.
        units = {
            "throughput_ci_low": "byte/second",
            "throughput_ci_high": "byte/second",
            "ee_net_ci_low": "byte/joule",
            "ee_net_ci_high": "byte/joule",
        }

        for column, unit in units.items():
            ci_df[column] = pd.Series(
                [value.to(unit).magnitude for value in ci_df[column]],
                dtype=f"pint[{unit}][Float64]",
            )

        config_df = config_df.merge(
            ci_df,
            on=self.GROUP_KEY,
            how="left",
            validate="one_to_one",
        )

        tp_valid = (
                (config_df["throughput_ci_low"] <= config_df["throughput_mean"])
                & (config_df["throughput_mean"] <= config_df["throughput_ci_high"])
        )

        ee_valid = (
                (config_df["ee_net_ci_low"] <= config_df["ee_net_mean"])
                & (config_df["ee_net_mean"] <= config_df["ee_net_ci_high"])
        )

        self._logger.info("CI rows:                      %d", len(ci_df))
        self._logger.info("Invalid throughput intervals: %d", (~tp_valid).sum())
        self._logger.info("Invalid EE_net intervals:     %d", (~ee_valid).sum())
        self._logger.info("")
        self._logger.info("CI dtypes:")
        table_str = tabulate.tabulate(
            config_df[list(units)].dtypes.to_frame(),
            headers=["Column", "Type"],
        )
        self._logger.info(table_str)

    def _calculate_configuration_level_means(self, analysis_df: pd.DataFrame) -> pd.DataFrame:
        grouped = analysis_df.groupby(self.GROUP_KEY, observed=True)

        frames = [
            grouped["throughput"].mean().rename("throughput_mean"),
            grouped["energy_efficiency_net"].mean().rename("ee_net_mean"),
            grouped["cr"].first().rename("cr"),
            grouped.size().rename("n_trials"),
        ]
        config_df = pd.concat(
            frames,
            axis=1,
        ).reset_index()

        self._logger.info("Configuration rows: %d", len(config_df))
        trials = [str(s) for s in sorted(config_df["n_trials"].unique())]
        self._logger.info("Trial-count values: %s", ", ".join(trials))

        table_str = tabulate.tabulate(
            config_df[["throughput_mean", "ee_net_mean", "cr"]].dtypes.to_frame(),
            headers=["Column", "Type"],
        )
        self._logger.info("")
        self._logger.info("Measurement dtypes:")
        self._logger.info(table_str)

        table_str = tabulate.tabulate(
            config_df[
                ["throughput_mean", "ee_net_mean", "cr"]
            ].isna().sum().to_frame(),
            headers=["Value", "Count"]
        )
        self._logger.info("")
        self._logger.info("Missing values:")
        self._logger.info(table_str)
        return config_df

    def _define_configurations(self, analysis_df: pd.DataFrame):
        unique_configs = analysis_df[self.GROUP_KEY].drop_duplicates()

        configs_per_stratum = (
            unique_configs.groupby(self.STRATUM_KEY, observed=True)
            .size()
            .rename("configurations")
        )

        self._logger.info("Number of strata: %d", len(configs_per_stratum))
        self._logger.info("")
        self._logger.info("Configurations per stratum:")
        configs_per_stratum = configs_per_stratum.to_frame()

        configs_per_stratum_df = configs_per_stratum.copy()
        configs_per_stratum_df = configs_per_stratum_df .reset_index()
        configs_per_stratum_df["host"] = configs_per_stratum_df["host"].where(
            configs_per_stratum_df["host"].ne(configs_per_stratum_df["host"].shift()), ""
        )

        table_str = tabulate.tabulate(
            configs_per_stratum_df,
            headers=["Host", "Dataset", "Configurations"],
            showindex=False,
        )
        self._logger.info(table_str)

    def _integrate_data(self, data: Data) -> pd.DataFrame:
        self._log_section("3 Integrate data")

        self._log_section("3.1 Merge throughput and energy-efficiency")
        analysis_df = self._merge_throughput_energy_efficiency(data)

        self._log_section("3.2 Merge compression ratio")
        analysis_df = self._merge_compression_ratio(data, analysis_df)

        self._log_section("3.3 Validate the integrated dataset")
        self._validate_integrated_dataset(analysis_df)
        return analysis_df

    def _validate_integrated_dataset(self, analysis_df: pd.DataFrame):
        duplicate_trials = analysis_df.duplicated(self.TRIAL_KEY, keep=False)
        cr_variation = (
            analysis_df.groupby(self.CR_KEY, observed=True)["cr"]
            .nunique(dropna=False)
        )

        self._logger.info("Rows:                            %d", len(analysis_df))
        self._logger.info("Unique trial keys:               %d", analysis_df[self.TRIAL_KEY].drop_duplicates().shape[0])
        self._logger.info("Rows with duplicated trial keys: %d", duplicate_trials.sum())
        self._logger.info("CR keys with multiple values:    %d", (cr_variation != 1).sum())

    def _merge_compression_ratio(self, data: Data, analysis_df: pd.DataFrame) -> pd.DataFrame:
        cr_lookup = data.cr_df[self.CR_KEY+ ["cr"]].copy()

        analysis_df = analysis_df.merge(
            cr_lookup,
            on=self.CR_KEY,
            how="left",
            validate="many_to_one",
            indicator=True,
        )

        self._logger.info("Rows: %d", len(analysis_df))

        self._logger.info("")
        self._logger.info("Merge status:")
        value_counts = analysis_df["_merge"].value_counts()
        table_str = tabulate.tabulate(
            value_counts.to_frame(),
        )
        self._logger.info(table_str)

        self._logger.info("")
        self._logger.info("Missing CR values: %d", analysis_df["cr"].isna().sum())

        analysis_df = analysis_df.drop(columns="_merge")
        return analysis_df

    def _merge_throughput_energy_efficiency(self, data: Data) -> pd.DataFrame:
        analysis_df = data.tp_df.merge(
            data.ee_df,
            on=self.TRIAL_KEY,
            how="outer",
            validate="one_to_one",
            indicator=True,
        )

        self._logger.info("Rows: %d", len(analysis_df))

        self._logger.info("")
        self._logger.info("Merge status:")
        value_counts = analysis_df["_merge"].value_counts()
        table_str = tabulate.tabulate(
            value_counts.to_frame(),
        )
        self._logger.info(table_str)

        analysis_df = analysis_df.drop(columns="_merge")

        analysis_df_info = analysis_df[[
            "duration",
            "throughput",
            "energy_consumption_net",
            "energy_efficiency_net",
        ]]

        table_str = tabulate.tabulate(
            analysis_df_info.dtypes.to_frame(),
        )
        self._logger.info("")
        self._logger.info("Measurement dtypes:")
        self._logger.info(table_str)
        return analysis_df

    def _validate_data(self, data: Data):
        self._log_section("2 Validate data")
        self._log_section("2.1 Validate net energy and efficiency")
        self._validate_data_energy(data.ee_df)

        self._log_section("2.2 Validate trial-key uniqueness")
        self._validate_trial_key_uniqueness(data)

        self._log_section("2.3 Validate compression-ratio uniqueness")
        self._validate_compression_ratio_uniqueness(data)

        self._log_section("2.4 Validate 30 trials per configuration")
        self._validate_trials(data)

        self._log_section("2.5 Validate feasible threading modes")
        self._validate_feasible_threading_modes(data)


    def _validate_feasible_threading_modes(self, data: Data):
        threading_by_tool = (
            data.tp_df.groupby("tool", observed=True)["threading"]
            .agg(lambda values: set(values))
            .to_dict()
        )

        for tool in sorted(threading_by_tool):
            self._logger.info("%6s: %s", tool, sorted(threading_by_tool[tool]))

    def _validate_trials(self, data: Data):
        config_key = [
            "host", "tool", "dataset", "mode",
            "strength", "threading",
        ]

        for field in fields(data):
            if field.name == "cr_df":
                continue

            df = getattr(data, field.name)
            counts = df.groupby(config_key, observed=True).size()
            invalid_counts = counts[counts != 30]

            self._logger.info("")
            self._logger.info("%s:", field.name)
            self._logger.info("Configurations:                   %d", len(counts))
            self._logger.info("Minimum trials:                   %d", counts.min())
            self._logger.info("Maximum trials:                   %d", counts.max())
            self._logger.info("Configurations without 30 trials: %d", len(invalid_counts))

            if not invalid_counts.empty:
                self._logger.info(invalid_counts.to_string())

    def _validate_compression_ratio_uniqueness(self, data: Data):
        cr_lookup = data.cr_df[self.CR_KEY + ["cr"]].copy()
        duplicate_mask = cr_lookup.duplicated(self.CR_KEY, keep=False)
        self._logger.info("Rows:                      %d", len(cr_lookup))
        self._logger.info("Unique configuration keys: %d", cr_lookup[self.CR_KEY].drop_duplicates().shape[0])
        self._logger.info("Rows with duplicated keys: %d", duplicate_mask.sum())
        self._logger.info("Missing CR values:         %d", data.cr_df["cr"].isna().sum())
        self._logger.info("Nonpositive CR values:     %d", (data.cr_df["cr"] <= 0).sum())

    def _validate_trial_key_uniqueness(self, data: Data):
        for field in fields(data):
            if field.name == "cr_df":
                continue
            df = getattr(data, field.name)
            duplicate_mask = df.duplicated(self.TRIAL_KEY, keep=False)

            self._logger.info("")
            self._logger.info("%s:", field.name)
            self._logger.info("Rows:                       %d", len(df))
            self._logger.info("Unique trial keys:          %d", df[self.TRIAL_KEY].drop_duplicates().shape[0])
            self._logger.info("Rows with duplicated keys:  %d", duplicate_mask.sum())

    def _validate_data_energy(self, ee_df: pd.DataFrame):
        net_energy = pd.Series(
            np.asarray(ee_df["energy_consumption_net"].pint.magnitude, dtype=float),
            index=ee_df.index,
        )
        net_ee = pd.Series(
            np.asarray(ee_df["energy_efficiency_net"].pint.magnitude, dtype=float),
            index=ee_df.index,
        )
        invalid = (
                (net_energy <= 0)
                | ~np.isfinite(net_energy)
                | ~np.isfinite(net_ee)
        )

        self._logger.info("Validate net energy and efficiency")
        self._logger.info("Total rows:             %d", len(ee_df))
        self._logger.info("Nonpositive net energy: %d", (net_energy <= 0).sum())
        self._logger.info("Nonfinite net energy:   %d", (~np.isfinite(net_energy)).sum())
        self._logger.info("Nonfinite EE_net:       %d", (~np.isfinite(net_ee)).sum())
        self._logger.info("Invalid rows:           %d", invalid.sum())

    def _filter_data(self, data: Data) -> Data:
        return Data(
            cr_df=data.cr_df, # [data.cr_df["mode"] == self._mode],
            tp_df=data.tp_df[data.tp_df["mode"] == self._mode],
            ee_df=data.ee_df[data.ee_df["mode"] == self._mode],
        )

    def _load_data(self, cr_file: Path, tp_file: Path, ee_file: Path) -> Data:
        cr_df = self._frameio.load(cr_file)
        tp_df = self._frameio.load(tp_file)
        ee_df = self._frameio.load(ee_file)
        return Data(
            cr_df=cr_df,
            tp_df=tp_df,
            ee_df=ee_df,
        )

    def _log_section(self, title: str):
        self._logger.info("")
        self._logger.info("%s %s %s", "-" * 20, title, "-" * 20)
        self._logger.info("")
