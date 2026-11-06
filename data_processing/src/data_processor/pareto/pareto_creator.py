import logging
from pathlib import Path
from dataclasses import dataclass, astuple

import pandas as pd
import numpy as np
import tabulate
from paretoset import paretoset

from data_processor.processor import Processor
from data_processor.constants import ORDER_TOOL, ORDER_STRENGTH

@dataclass(frozen=True)
class Data:
    cr_df: pd.DataFrame
    tp_df: pd.DataFrame
    ee_df: pd.DataFrame

    def __iter__(self):
        return iter(astuple(self))


class ParetoCreator(Processor):
    CONFIG_COLS = [
        "host", "tool", "dataset", "mode", "strength", "threading"
    ]
    TRIAL_COLS = CONFIG_COLS + ["run"]
    CONTEXT_COLS = ["host", "dataset"]
    OBJECTIVE_COLS = [
        "throughput",
        "energy_efficiency_net",
        "cr",
    ]

    def __init__(self, resources: Path, mode: str):
        super().__init__(resources)
        self._logger = logging.getLogger(self.__class__.__name__)
        self._mode = mode
        self._n_boot = 2000
        seed = None
        if seed is None:
            self._seed = 12345
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def process(self, cr_file: Path, tp_file: Path, ee_file: Path):
        self._log_section("1 Data preparation")
        self._logger.info("load data")
        data = self._load_data(cr_file, tp_file, ee_file)

        self._log_section("1.1 Filter data")
        self._logger.info("Evaluate: %s", self._mode)
        data = self._filter_data(data)
        self._present_data(data)

        self._log_section("1.2 Validate data")
        self._validate_data(data)

        analysis_df, cr_lookup  = self._outcome_integration(data)
        config_df = self._configuration_level_estimates(analysis_df, cr_lookup)
        contexts, configuration_results, pareto_configurations = self._pareto_front_identification(config_df)
        self._cross_context_consistency(contexts, configuration_results, pareto_configurations)
        self._robustness_analysis()

    def _robustness_analysis(self):
        self._log_section("6 Robustness analysis")

        self._log_section("6.1 Bootstrap Pareto membership")


    def _cross_context_consistency(self, contexts: dict, configuration_results, pareto_configurations: pd.DataFrame):
        self._log_section("5 Cross-context consistency")

        self._log_section("5.1 Calculate utility coverage")
        all_tools, utility_context_presence = self._calculate_utility_coverage(contexts, configuration_results, pareto_configurations)

        self._log_section("5.2 Verify configuration-count bias is avoided")
        self._verify_configuration_count_bias_is_avoided(pareto_configurations, all_tools, utility_context_presence)

        self._log_section("5.3 Describe contributing configurations")
        self._describe_contributing_configurations(contexts, pareto_configurations)

    def _describe_contributing_configurations(self, contexts: dict, pareto_configurations: pd.DataFrame):
        setting_context_presence = (
            pareto_configurations[
                ["tool", "strength", "threading", "host", "dataset"]
            ]
            .drop_duplicates()
        )

        setting_summary = (
            setting_context_presence
            .groupby(
                ["tool", "strength", "threading"],
                observed=True,
            )
            .size()
            .rename("pareto_contexts")
            .reset_index()
        )

        total_contexts = len(contexts)
        setting_summary["coverage_percent"] = (
                100 * setting_summary["pareto_contexts"] / total_contexts
        )

        setting_summary = setting_summary.sort_values(
            ["tool", "pareto_contexts", "strength", "threading"],
            ascending=[True, False, True, True],
            ignore_index=True,
        )

        setting_summary_df = setting_summary.copy()
        setting_summary_df["_tool_key"] = setting_summary_df["tool"].apply(ORDER_TOOL.index)
        setting_summary_df["_strength_key"] = setting_summary_df["strength"].apply(ORDER_STRENGTH.index)
        setting_summary_df = setting_summary_df.sort_values(
            by=["_tool_key", "pareto_contexts", "_strength_key", "threading"],
            ascending=[True, False, True, True],
            ignore_index=True,
        ).drop(columns=["_tool_key", "_strength_key"])
        table_str = tabulate.tabulate(
            setting_summary_df,
            headers="keys",
            showindex=False,
            floatfmt=".1f",
        )
        self._logger.info(table_str)

    def _verify_configuration_count_bias_is_avoided(self, pareto_configurations: pd.DataFrame,
                                                    all_tools: list, utility_context_presence: pd.DataFrame):
        tool_context_counts = (
            pareto_configurations
            .groupby(["tool", "host", "dataset"], observed=True)
            .size()
            .rename("pareto_configurations")
            .reset_index()
        )

        bias_check = (
            tool_context_counts
            .groupby("tool", observed=True)
            .agg(
                pareto_contexts=("pareto_configurations", "size"),
                total_pareto_configurations=("pareto_configurations", "sum"),
                max_configurations_in_one_context=(
                    "pareto_configurations", "max"
                ),
            )
            .reindex(all_tools, fill_value=0)
            .reset_index()
            .sort_values(
                ["pareto_contexts", "tool"],
                ascending=[False, True],
            )
        )

        self._logger.info("Duplicate utility-context rows: %d",
              int(utility_context_presence.duplicated(
                  ["tool", "host", "dataset"]
              ).sum()))

        table_str = tabulate.tabulate(
            bias_check,
            headers=["Tool", "Pareto contexts", "Total pareto configurations", "Max configurations in one context"],
            showindex=False,
        )
        self._logger.info(table_str)

    def _calculate_utility_coverage(self, contexts: dict,
                                    configuration_results: pd.DataFrame,
                                    pareto_configurations: pd.DataFrame) -> tuple[list, pd.DataFrame]:
        total_contexts = len(contexts)

        utility_context_presence = (
            pareto_configurations[
                ["host", "dataset", "tool"]
            ]
            .drop_duplicates()
        )

        all_tools = sorted(configuration_results["tool"].unique())

        utility_coverage = (
            utility_context_presence
            .groupby("tool", observed=True)
            .size()
            .reindex(all_tools, fill_value=0)
            .rename("pareto_contexts")
            .reset_index()
        )

        utility_coverage["coverage_percent"] = (
                100
                * utility_coverage["pareto_contexts"]
                / total_contexts
        )

        utility_coverage = utility_coverage.sort_values(
            ["pareto_contexts", "tool"],
            ascending=[False, True],
            ignore_index=True,
        )

        table_str = tabulate.tabulate(
            utility_coverage,
            headers="keys",
            showindex=False,
            floatfmt=".1f",
        )
        self._logger.info(table_str)

        return all_tools, utility_context_presence

    def _pareto_front_identification(self, config_df: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
        self._log_section("4 Context-specific Pareto analysis")

        self._log_section("4.1 Separate contexts")
        contexts = self._separate_contexts(config_df)

        self._log_section("4.2 Determine dominance")
        self._determine_dominance(contexts)

        self._log_section("4.3 Identify Pareto-optimal configurations")

        configuration_results, pareto_configurations= self._identify_pareto_optimal_configurations(contexts)
        return contexts, configuration_results, pareto_configurations

    def _identify_pareto_optimal_configurations(self, contexts: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
        marked_contexts = []
        for frame in contexts.values():
            frame = frame.copy()
            frame["pareto_optimal"] = self._pareto_mask_maximize(frame)
            marked_contexts.append(frame)

        configuration_results = pd.concat(
            marked_contexts,
            ignore_index=True,
        )

        pareto_configurations = (
            configuration_results.loc[
                configuration_results["pareto_optimal"]
            ]
            .copy()
        )

        pareto_counts = (
            pareto_configurations
            .groupby(self.CONTEXT_COLS, observed=True)
            .size()
        )

        self._logger.info("Pareto-optimal configurations: %d", len(pareto_configurations))
        self._logger.info("Dominated configurations:      %d", int((~configuration_results["pareto_optimal"]).sum()))
        self._logger.info("Pareto-count range:            %d - %d",
              int(pareto_counts.min()),
              int(pareto_counts.max()))
        self._logger.info("Duplicate configurations:      %d",
              int(pareto_configurations.duplicated(self.CONFIG_COLS).sum()))
        return configuration_results, pareto_configurations

    def _determine_dominance(self, contexts: dict):
        dominance_rows = []
        for (host, dataset), frame in contexts.items():
            pareto_mask = self._pareto_mask_maximize(frame)

            dominance_rows.append({
                "host": host,
                "dataset": dataset,
                "dominated": int((~pareto_mask).sum()),
                "nondominated": int(pareto_mask.sum()),
                "total": len(frame),
            })

        dominance_summary = pd.DataFrame(dominance_rows)

        dominance_summary_df = dominance_summary.copy()
        dominance_summary_df["host"] = dominance_summary_df["host"].mask(dominance_summary_df["host"].eq(dominance_summary_df["host"].shift()), "")
        table_str = tabulate.tabulate(
            dominance_summary_df,
            headers="keys",
            showindex=False,
        )
        self._logger.info(table_str)

    def _pareto_mask_maximize(self, frame):
        numeric_objectives = pd.DataFrame({
            column: self._magnitude(frame[column]).to_numpy(dtype=float)
            for column in self.OBJECTIVE_COLS
        })

        if not np.isfinite(numeric_objectives.to_numpy()).all():
            raise ValueError("Objective values contain non-finite values.")

        return np.asarray(
            paretoset(
                numeric_objectives,
                sense=["max", "max", "max"],
            ),
            dtype=bool,
        )

    def _separate_contexts(self, configuration_metrics: pd.DataFrame) -> dict:
        contexts = {
            key: group.copy()
            for key, group in configuration_metrics.groupby(
                self.CONTEXT_COLS,
                observed=True,
                sort=True,
            )
        }

        context_counts = (
            configuration_metrics
            .groupby(self.CONTEXT_COLS, observed=True)
            .size()
            .rename("configurations")
        )

        self._logger.info("Contexts:                  %d", len(contexts))
        self._logger.info("Configuration-count range: %d - %d",
              int(context_counts.min()),
              int(context_counts.max()))

        context_counts_df = context_counts.to_frame().copy()
        context_counts_df = context_counts_df.reset_index()
        context_counts_df["host"] = context_counts_df["host"].where(
            context_counts_df["host"].ne(context_counts_df["host"].shift()), ""
        )
        table_str = tabulate.tabulate(
            context_counts_df,
            headers=["Host", "Dataset", "Count"],
            showindex=False,
        )
        self._logger.info(table_str)
        return contexts

    def _configuration_level_estimates(self, analysis_df: pd.DataFrame, cr_lookup: pd.DataFrame) -> pd.DataFrame:
        self._log_section("3 Configuration-level estimates")

        self._log_section("3.1 Aggregate trials")
        config_df = self._aggregate_trials(analysis_df)

        self._log_section("3.2 Assign compression ratio")
        config_df = self._assign_compression_ratio(config_df, cr_lookup)

        return config_df

    def _assign_compression_ratio(self, config_df: pd.DataFrame, cr_lookup: pd.DataFrame) -> pd.DataFrame:
        config_df = config_df.merge(
            cr_lookup,
            on=["dataset", "tool", "strength"],
            how="left",
            validate="many_to_one",
            indicator=True,
        )

        self._logger.info("Configurations:        %d", len(config_df))
        self._logger.info("Merge status:")
        table_str = tabulate.tabulate(
            config_df["_merge"].value_counts().to_frame(),
        )
        self._logger.info(table_str)

        self._logger.info("Missing CR:            %d", int(config_df["cr"].isna().sum()))
        self._logger.info("Duplicate configurations: %d", int(config_df.duplicated(self.CONFIG_COLS).sum()))
        self._logger.info("CR dtype:              %s", config_df["cr"].dtype)

        config_df = config_df.drop(columns="_merge")
        return config_df

    def _aggregate_trials(self, analysis_trials: pd.DataFrame) -> pd.DataFrame:
        configuration_metrics = (
            analysis_trials
            .groupby(self.CONFIG_COLS, observed=True)
            .agg(
                throughput=("throughput", "mean"),
                energy_efficiency_net=("energy_efficiency_net", "mean"),
                trials=("run", "nunique"),
            )
            .reset_index()
        )

        self._logger.info("Configurations:        %d", len(configuration_metrics))
        self._logger.info("Trial-count range:     %d %d",
              int(configuration_metrics["trials"].min()),
              int(configuration_metrics["trials"].max()))
        self._logger.info("Missing values:        %d", int(configuration_metrics.isna().sum().sum()))
        self._logger.info("Throughput dtype:      %s",
              configuration_metrics["throughput"].dtype)
        self._logger.info("Net-efficiency dtype:  %s",
              configuration_metrics["energy_efficiency_net"].dtype)

        return configuration_metrics

    def _outcome_integration(self, data: Data) -> tuple[pd.DataFrame, pd.DataFrame]:
        self._log_section("2 Outcome integration")

        self._log_section("2.1 Merge outcomes")
        analysis_df = self._merge_trial_level_outcomes(data)

        self._log_section("2.2 Merge compression ratio")
        analysis_df, cr_lookup  = self._merge_compression_ratio(data, analysis_df)

        self._log_section("2.3 Validate merged configurations")
        self._validate_integrated_dataset(analysis_df)
        return analysis_df, cr_lookup

    def _merge_trial_level_outcomes(self, data: Data) -> pd.DataFrame:
        analysis_trials = data.tp_df[
            self.TRIAL_COLS + ["throughput"]
            ].merge(
            data.ee_df[
                self.TRIAL_COLS
                + ["energy_consumption_net", "energy_efficiency_net"]
                ],
            on=self.TRIAL_COLS,
            how="inner",
            validate="one_to_one",
            indicator=True
        )

        self._logger.info("Rows:                  %d", len(analysis_trials))
        table_str = tabulate.tabulate(
            analysis_trials["_merge"].value_counts().to_frame(),
        )
        self._logger.info("Merge status:")
        self._logger.info(table_str)

        self._logger.info("Missing values:        %d", int(analysis_trials.isna().sum().sum()))

        analysis_trials = analysis_trials.drop(columns="_merge")
        return analysis_trials

    def _validate_integrated_dataset(self, analysis_df: pd.DataFrame):
        required_outcomes = [
            "throughput",
            "energy_consumption_net",
            "energy_efficiency_net",
            "cr",
        ]

        duplicates = analysis_df.duplicated(self.TRIAL_COLS).sum()

        config_summary = (
            analysis_df
            .groupby(self.CONFIG_COLS, observed=True)
            .agg(
                trials=("run", "nunique"),
                throughput_values=("throughput", "count"),
                ee_values=("energy_efficiency_net", "count"),
                cr_values=("cr", "count"),
                distinct_cr=("cr", "nunique"),
            )
        )

        invalid_configs = config_summary[
            (config_summary["trials"] != 30)
            | (config_summary["throughput_values"] != 30)
            | (config_summary["ee_values"] != 30)
            | (config_summary["cr_values"] != 30)
            | (config_summary["distinct_cr"] != 1)
            ]

        configs = analysis_df[self.CONFIG_COLS].drop_duplicates()

        self._logger.info("Rows:                  %d", len(analysis_df))
        self._logger.info("Configurations:        %d", len(configs))
        self._logger.info("Duplicate trials:      %d", int(duplicates))
        self._logger.info("Missing outcomes:      %d",
                          int(analysis_df[required_outcomes].isna().sum().sum()))
        self._logger.info("Invalid configurations:%d", len(invalid_configs))

        by_utility = configs.groupby("tool", observed=True).size().sort_index()
        table_str = tabulate.tabulate(
            by_utility.to_frame(),
            headers=["Utility", "Configurations"]
        )
        self._logger.info("")
        self._logger.info("Configurations by utility:")
        self._logger.info(table_str)

    def _merge_compression_ratio(self, data: Data, analysis_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        cr_lookup = data.cr_df[
            ["dataset", "tool", "strength", "cr"]
        ].copy()

        analysis_df = analysis_df.merge(
            cr_lookup,
            on=["dataset", "tool", "strength"],
            how="left",
            validate="many_to_one",
            indicator=True
        )

        self._logger.info("Rows:                  %d", len(analysis_df))
        self._logger.info("Merge status:")

        table_str = tabulate.tabulate(
            analysis_df["_merge"].value_counts().to_frame(),
        )
        self._logger.info(table_str)
        self._logger.info("Missing CR:            %d", int(analysis_df["cr"].isna().sum()))

        cr_counts = (
            analysis_df
            .groupby(
                ["host", "tool", "dataset", "strength"],
                observed=True
            )["cr"]
            .nunique()
        )

        self._logger.info("Maximum CR values per base configuration: %d", int(cr_counts.max()))

        analysis_df = analysis_df.drop(columns="_merge")
        return analysis_df, cr_lookup

    def _validate_data(self, data: Data):
        cr_cols = ["dataset", "tool", "strength"]

        single_only = {"gzip", "bzip2", "lzop"}

        # Missing values
        self._logger.info("Missing values:")
        self._logger.info("CR:                    %d", int(data.cr_df.isna().sum().sum()))
        self._logger.info("Throughput:            %d", int(data.tp_df.isna().sum().sum()))
        self._logger.info("Energy efficiency:     %d", int(data.ee_df.isna().sum().sum()))

        # Duplicate observations
        self._logger.info("")
        self._logger.info("Duplicate keys:")
        self._logger.info("CR:                    %d", int(data.cr_df.duplicated(cr_cols).sum()))
        self._logger.info("Throughput:            %d", int(data.tp_df.duplicated(self.TRIAL_COLS).sum()))
        self._logger.info("Energy efficiency:     %d", int(data.ee_df.duplicated(self.TRIAL_COLS).sum()))

        # Replications
        tp_reps = data.tp_df.groupby(self.CONFIG_COLS, observed=True).size()
        ee_reps = data.ee_df.groupby(self.CONFIG_COLS, observed=True).size()

        self._logger.info("")
        self._logger.info("Configurations and repetitions:")
        self._logger.info("Throughput:            %d %d %d", tp_reps.size, tp_reps.min(), tp_reps.max())
        self._logger.info("Energy efficiency:     %d %d %d", ee_reps.size, ee_reps.min(), ee_reps.max())

        # Matching trial keys
        tp_keys = pd.MultiIndex.from_frame(data.tp_df[self.TRIAL_COLS])
        ee_keys = pd.MultiIndex.from_frame(data.ee_df[self.TRIAL_COLS])

        self._logger.info("")
        self._logger.info("Unmatched trial keys:  %d",
              len(tp_keys.symmetric_difference(ee_keys)))

        # Threading feasibility and coverage
        configs = data.tp_df[self.CONFIG_COLS].drop_duplicates()

        single_violations = configs[
            configs["tool"].isin(single_only)
            & configs["threading"].ne("single")
            ]

        thread_counts = (
            configs.groupby(
                ["host", "tool", "dataset", "strength"],
                observed=True
            )["threading"]
            .nunique()
        )

        expected_threads = (
            thread_counts.index.get_level_values("tool")
            .map(lambda tool: 1 if tool in single_only else 2)
        )

        self._logger.info("Threading violations:  %d",
              len(single_violations) + int((thread_counts != expected_threads).sum()))

        # Invalid measurements
        invalid_cr = self._magnitude(data.cr_df["cr"]).le(0).sum()
        invalid_tp = self._magnitude(data.tp_df["throughput"]).le(0).sum()
        invalid_duration = self._magnitude(data.tp_df["duration"]).le(0).sum()
        invalid_net_energy = self._magnitude(data.ee_df["energy_consumption_net"]).le(0).sum()
        invalid_net_ee = self._magnitude(data.ee_df["energy_efficiency_net"]).le(0).sum()

        self._logger.info("")
        self._logger.info("Nonpositive values:")
        self._logger.info("CR:                    %d", int(invalid_cr))
        self._logger.info("Throughput:            %d", int(invalid_tp))
        self._logger.info("Duration:              %d", int(invalid_duration))
        self._logger.info("Net energy:            %d", int(invalid_net_energy))
        self._logger.info("Net energy efficiency: %d", int(invalid_net_ee))

    def _magnitude(self, series):
        try:
            return series.pint.magnitude
        except (AttributeError, TypeError):
            return pd.to_numeric(series, errors="coerce")

    def _present_data(self, data: Data):
        self._logger.info("Rows:")
        self._logger.info("CR:                 %d", len(data.cr_df))
        self._logger.info("Throughput:         %d", len(data.tp_df))
        self._logger.info("Energy efficiency:  %d", len(data.ee_df))

        self._logger.info("")
        self._logger.info("Modes:")
        self._logger.info("Throughput:         %s", ", ".join(data.tp_df["mode"].unique().tolist()))
        self._logger.info("Energy efficiency:  %s", ", ".join(data.ee_df["mode"].unique().tolist()))

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
