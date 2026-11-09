import logging
from pathlib import Path
from abc import abstractmethod

import pandas as pd
import numpy as np
import tabulate

from data_processor.processor import LoggingProcessor
from data_processor.util import tabulate_pint
from data_processor.constants import ORDER_TOOL, ORDER_HOST, SINGLE_THREADED_TOOLS


class BaseRanking(LoggingProcessor):
    def __init__(self, resources: Path, logfile: bool, n_boot: int, seed: int | None,
                 threading: str, mode: str, create_tex: bool):
        super().__init__(resources, logfile)
        self._logger = logging.getLogger(self.__class__.__name__)
        self._threading = threading
        self._mode = mode
        self._n_boot = n_boot
        if seed is None:
            seed = 12345
        self._rng = np.random.default_rng(seed)
        self._create_tex = create_tex

    @property
    def factors(self):
        return ["host", "tool", "dataset", "strength"]

    @property
    @abstractmethod
    def base_column(self) -> str:
        pass

    @property
    @abstractmethod
    def display_unit (self) -> str:
        pass

    @abstractmethod
    def _get_base_from_input_file(self, input_file: Path) -> str:
        pass

    def _analyze(self, input_file: Path) -> None:
        df = self._preparation(input_file)
        utility_metric  = self._calculate(df)
        ranked_metric, boot_estimates  = self._construct_rankings(df, utility_metric)
        ranking_report = self._report_results(ranked_metric, utility_metric, boot_estimates)
        if self._create_tex:
            base = self._get_base_from_input_file(input_file)
            filename = f"tp_ranking_{base}_{self._threading}_{self._mode}.tex"
            self._create_ranking_report_table(ranking_report, self._resources / filename)

    def _create_ranking_report_table(self, ranking_report: pd.DataFrame, filename: Path) -> None:
        self._logger.info("Generate: %s", filename)
        latex = self._build_ranking_lines(ranking_report)
        with filename.open(mode="w", encoding="UTF_8") as f:
            f.write(latex)

    def _build_ranking_lines(self, df: pd.DataFrame):
        df = df.copy()

        def get_host_entry(host: str) -> str:
            if host == "raspi5":
                return r"\gls{pi5}"
            if host == "visionfive2":
                return r"\gls{visionfive2}"
            if host == "radxax4":
                return r"\gls{radxax4}"
            raise RuntimeError("unknown host: %s" % host)

        host_column = "Platform / Rank"
        df[host_column] = df.index
        df.insert(0, host_column, df.pop(host_column))

        df[host_column] = df[host_column].map(get_host_entry)

        column_formats = ["l"] + ["c"] * 9
        latex = (
            df.style
            .hide(axis="index")
            .to_latex(
                siunitx=True,
                hrules=True,
                column_format="".join(column_formats),
            )
        )

        return latex

    def _report_results(self, ranked_metric: pd.DataFrame, utility_metric: pd.DataFrame,
                        boot_estimates: dict) -> pd.DataFrame:
        self._log_section("4 Report results")

        self._log_section("4.1 Present the ranking table")
        ranking_report = self._present_ranking_table(ranked_metric)

        self._log_section("4.2 Report estimates")
        self._report_estimates(utility_metric, ranked_metric, boot_estimates)

        self._log_section("4.3 Summarize platform differences")
        self._summarize_platform_differences(ranked_metric)
        return ranking_report

    def _summarize_platform_differences(self, ranked_metric: pd.DataFrame):
        excluded_tools = ["brotli", "sleep"]
        if self._threading == "multi":
            excluded_tools += SINGLE_THREADED_TOOLS
        effective_tool_order = [item for item in ORDER_TOOL if item not in excluded_tools]
        rank_comparison = (
            ranked_metric.pivot(
                index="tool",
                columns="host",
                values="rank",
            )
            .reindex(effective_tool_order)
            .loc[:, ORDER_HOST]
        )

        rank_comparison["best_rank"] = rank_comparison.min(axis=1)
        rank_comparison["worst_rank"] = rank_comparison.max(axis=1)
        rank_comparison["rank_range"] = (
                rank_comparison["worst_rank"]
                - rank_comparison["best_rank"]
        )

        headers = ["tool / host"] + ORDER_HOST + ["best rank", "worst rank", "rank range"]
        table_str = tabulate_pint(
            rank_comparison,
            headers=headers,
        )
        self._logger.info(table_str)

    def _report_estimates(self, utility_metric: pd.DataFrame,
                          ranked_metric: pd.DataFrame, boot_estimates: dict):
        point_lookup = utility_metric.set_index(
            ["host", "tool"]
        )["metric_mean"]

        rank_lookup = ranked_metric.set_index(
            ["host", "tool"]
        )["rank"]

        records = []
        for key, bootstrap_values in boot_estimates.items():
            host, tool = key
            ci_low, ci_high = np.quantile(
                bootstrap_values,
                [0.025, 0.975],
            )

            records.append({
                "host": host,
                "tool": tool,
                "rank": rank_lookup.loc[key],
                self.base_column: point_lookup.loc[key],
                "ci_low": ci_low,
                "ci_high": ci_high,
            })

        metric_report = pd.DataFrame(records)

        metric_report = metric_report.pint.convert_object_dtype()
        for column in [self.base_column, "ci_low", "ci_high"]:
            metric_report[column] = metric_report[column].pint.to(self.display_unit)

        metric_report["host"] = pd.Categorical(
            metric_report["host"],
            categories=ORDER_HOST,
            ordered=True,
        )
        metric_report["tool"] = pd.Categorical(
            metric_report["tool"],
            categories=ORDER_TOOL,
            ordered=True,
        )

        metric_report = metric_report.sort_values(
            ["host", "tool"]
        )

        headers = ["host", "tool", "rank", self.base_column, "ci_low", "ci_high"]
        table_str = tabulate_pint(
            metric_report,
            headers=headers,
            showindex=False,
            floatfmt=".2f",
        )
        self._logger.info(table_str)

    def _present_ranking_table(self, ranked_metric: pd.DataFrame) -> pd.DataFrame:
        ranking_report = (
            ranked_metric.pivot(
                index="host",
                columns="rank",
                values="tool",
            )
            .reindex(ORDER_HOST)
        )

        ranking_report.index.name = "Platform / Rank"
        ranking_report.columns.name = None

        table_str = tabulate.tabulate(
            ranking_report,
            headers="keys",
        )
        self._logger.info(table_str)
        return ranking_report

    def _construct_rankings(self, df: pd.DataFrame, utility_metric: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        self._log_section("3 Construct rankings")

        self._log_section("3.1 Rank utilities")
        ranked_metric = self._rank_utilities(utility_metric)

        self._log_section("3.2 Quantify uncertainty")
        boot_estimates = self._quantify_uncertainty(df, ranked_metric)
        return ranked_metric, boot_estimates

    def _quantify_uncertainty(self, df: pd.DataFrame, ranked_metric: pd.DataFrame) -> dict:
        col = df[self.base_column]
        tp_unit = col.pint.units

        boot_sums = {}
        cell_counts = {}
        for key, group in df.groupby(
                self.factors,
                observed=True,
        ):
            host, tool, _, _ = key
            utility_key = (host, tool)

            values = (
                    tp_unit
                    * group[self.base_column].pint.magnitude.to_numpy()
            )

            indices = self._rng.integers(
                0,
                len(values),
                size=(self._n_boot, len(values)),
            )

            cell_means = values[indices].mean(axis=1)

            if utility_key not in boot_sums:
                boot_sums[utility_key] = tp_unit * np.zeros(self._n_boot)
                cell_counts[utility_key] = 0

            boot_sums[utility_key] += cell_means
            cell_counts[utility_key] += 1

        boot_estimates = {
            key: values / cell_counts[key]
            for key, values in boot_sums.items()
        }

        point_ranks = ranked_metric.set_index(
            ["host", "tool"]
        )["rank"]

        records = []
        for host in sorted(df["host"].unique()):
            tools = sorted(df["tool"].unique())

            estimates = np.column_stack([
                boot_estimates[(host, tool)].magnitude
                for tool in tools
            ])

            order = np.argsort(-estimates, axis=1, kind="stable")
            ranks = np.empty_like(order)
            ranks[
                np.arange(self._n_boot)[:, None],
                order
            ] = np.arange(1, len(tools) + 1)

            for column, tool in enumerate(tools):
                tool_ranks = ranks[:, column]
                frequencies = np.bincount(
                    tool_ranks,
                    minlength=len(tools) + 1,
                )

                records.append({
                    "host": host,
                    "tool": tool,
                    "point_rank": point_ranks.loc[(host, tool)],
                    "modal_bootstrap_rank": frequencies[1:].argmax() + 1,
                    "rank_95_low": np.quantile(
                        tool_ranks, 0.025, method="nearest"
                    ),
                    "rank_95_high": np.quantile(
                        tool_ranks, 0.975, method="nearest"
                    ),
                    "point_rank_probability": np.mean(
                        tool_ranks == point_ranks.loc[(host, tool)]
                    ),
                })

        rank_uncertainty = pd.DataFrame(records)

        rank_uncertainty["_host_key"] = rank_uncertainty["host"].apply(ORDER_HOST.index)
        rank_uncertainty["_tool_key"] = rank_uncertainty["tool"].apply(ORDER_TOOL.index)
        rank_uncertainty = rank_uncertainty.sort_values(
            by=["_host_key", "_tool_key"],
            ignore_index=True,
        ).drop(columns=["_host_key", "_tool_key"])

        table_str = tabulate.tabulate(
            rank_uncertainty,
            showindex=False,
            headers="keys",
        )
        self._logger.info(table_str)
        return boot_estimates

    def _rank_utilities(self, utility_metric: pd.DataFrame) -> pd.DataFrame:
        ranked_metric = (
            utility_metric
            .sort_values(
                ["host", "metric_mean", "tool"],
                ascending=[True, False, True],
            )
            .copy()
        )

        ranked_metric["rank"] = (
            ranked_metric.groupby("host", observed=True)
            .cumcount()
            .add(1)
        )

        ranking_table = ranked_metric.pivot(
            index="host",
            columns="rank",
            values="tool",
        )

        table_str = tabulate.tabulate(
            ranking_table,
            headers="keys",
        )
        self._logger.info(table_str)
        return ranked_metric

    def _calculate_utility_level_values(self, configuration_means: pd.DataFrame) -> pd.DataFrame:
        utility_metric = (
            configuration_means.groupby(
                ["host", "tool"],
                observed=True,
                as_index=False,
            )
            .agg(
                metric_mean=("metric_mean", "mean"),
                configurations=("metric_mean", "size"),
            )
        )

        self._logger.info("Utility-platform estimates:    %d", len(utility_metric))
        self._logger.info("Unit:                          %s", utility_metric["metric_mean"].pint.units)
        self._logger.info("Configurations per estimate:   %s", ", ".join([str(i) for i in sorted(utility_metric["configurations"].unique())]))
        return utility_metric

    def _calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        self._log_section("2 Aggregate metric")

        self._log_section("2.1 Average repetitions")
        configuration_means = self._calculate_medians(df)

        self._log_section("2.2 Calculate utility-level values")
        utility_metric = self._calculate_utility_level_values(configuration_means )
        return utility_metric

    def _calculate_medians(self, df: pd.DataFrame) -> pd.DataFrame:
        configuration_means = (
            df.groupby(
                self.factors,
                observed=True,
                as_index=False,
            )
            .agg(
                metric_mean=(self.base_column, "mean"),
                trials=("run", "size"),
            )
        )

        self._logger.info("Configuration means:           %d", len(configuration_means))
        self._logger.info("Unit:                          %s", configuration_means["metric_mean"].pint.units)
        self._logger.info("Trials per mean:               %s", ", ".join([str(i) for i in sorted(configuration_means["trials"].unique())]))
        return configuration_means

    def _preparation(self, input_file: Path) -> pd.DataFrame:
        self._log_section("1. Prepare the data")
        self._logger.info("load data")
        df = self._frameio.load(input_file)

        self._log_section("1.1 Select observations")
        self._logger.info("Evaluate: %s %s", self._threading, self._mode)
        df = self._filter_data(df)

        self._log_section("1.2 Validate data")
        self._validate_data(df)

        self._log_section("1.3 Check measurements")
        self._check_measurements(df)

        return df

    def _check_measurements(self, df: pd.DataFrame):
        col = df[self.base_column]
        tp_unit = col.pint.units

        missing = col.isna()
        nonfinite = pd.Series(
            ~np.isfinite(col.pint.magnitude),
            index=df.index,
        )
        nonpositive = col <= 0 * tp_unit

        group_cols = ["host", "tool", "dataset", "strength"]
        grouped = df.groupby(group_cols, observed=True)[self.base_column]

        q1 = grouped.transform(lambda x: x.quantile(0.25))
        q3 = grouped.transform(lambda x: x.quantile(0.75))
        iqr = q3 - q1

        potential_outlier = (
                (col < q1 - 3 * iqr)
                | (col > q3 + 3 * iqr)
        )

        self._logger.info("Unit:                          %s", tp_unit)
        self._logger.info("Missing:                       %d", missing.sum())
        self._logger.info("Non-finite:                    %d", nonfinite.sum())
        self._logger.info("Nonpositive:                   %d", nonpositive.sum())
        self._logger.info("Potential outliers:            %d", potential_outlier.sum())

    def _validate_data(self, df: pd.DataFrame):
        df = df.copy()

        coverage = (
            df.groupby(self.factors, observed=True)
            .agg(
                rows=("run", "size"),
                unique_runs=("run", "nunique"),
            )
        )

        incomplete = coverage.loc[
            (coverage["rows"] != 30)
            | (coverage["unique_runs"] != 30)
            ]

        self._logger.info("Configurations:                %d", len(coverage))
        self._logger.info("Trials per configuration:      %s", "%, ".join([str(i) for i in sorted(coverage["rows"].unique())]))
        self._logger.info("Unique runs per configuration: %s", "%, ".join([str(i) for i in sorted(coverage["unique_runs"].unique())]))
        self._logger.info("Incomplete configurations:     %d", len(incomplete))

    def _filter_data(self, df: pd.DataFrame):
        df = df[df["mode"] == self._mode]
        df = df[df["threading"] == self._threading]

        self._logger.info("Rows:      %d", len(df))
        self._logger.info("Modes:     %s", list(df["mode"].unique()))
        self._logger.info("Threading: %s", list(df["threading"].unique()))
        self._logger.info("Tools:     %s", ", ".join(df["tool"].unique()))
        return df

    def _log_section(self, title: str):
        self._logger.info("")
        self._logger.info("%s %s %s", "-" * 20, title, "-" * 20)
        self._logger.info("")
