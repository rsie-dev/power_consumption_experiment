import logging
from pathlib import Path
from abc import abstractmethod
from itertools import combinations
from math import comb

import pandas as pd
import numpy as np
from scipy.stats import kendalltau
from tqdm import tqdm
import tabulate

from data_processor.processor import LoggingProcessor
from data_processor.util import tabulate_pint


class BaseRanking(LoggingProcessor):
    def __init__(self, resources: Path, logfile: bool, n_boot: int, seed: int | None, threading: str, mode: str):
        super().__init__(resources, logfile)
        self._logger = logging.getLogger(self.__class__.__name__)
        self._threading = threading
        self._mode = mode
        self._n_boot = n_boot
        if seed is None:
            seed = 12345
        self._rng = np.random.default_rng(seed)

    @property
    def factors(self):
        return ["host", "tool", "dataset", "strength"]

    @property
    @abstractmethod
    def base_column(self) -> str:
        pass

    def _analyze(self, input_file: Path) -> None:
        df = self._preparation(input_file)
        stability = self._calculate(df)
        self._create_stability_file(stability, input_file)

    def _create_stability_file(self, stability: pd.DataFrame, input_file: Path) -> None:
        base = self.extra_log_name(input_file)
        filename = "%s_stability.csv" % base
        self._frameio.persist(stability, self._resources / filename)

    def _preparation(self, input_file: Path) -> pd.DataFrame:
        self._log_section("1. Preparation")
        self._logger.info("load data")
        df = self._frameio.load(input_file)

        self._log_section("1.1 Define subset")
        self._logger.info("Evaluate: %s %s", self._threading, self._mode)
        df = self._filter_data(df)

        self._log_section("1.2 Validate data")
        self._validate_data(df)

        return df

    def _calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        self._log_section("2. Calculate medians")
        summary = self._calculate_medians(df)
        self._print_medians(summary)

        self._log_section("3 Ranks")
        ranked, ties = self._rank_utilities(summary)
        self._print_ranked_utilities(ranked)
        self._print_ties_utilities(ties)

        self._log_section("4 Quantify consistency")

        self._log_section("4.1 Pairwise Kendall’s")
        platform_tau, dataset_tau = self._pair_utilities(ranked)
        self._print_pair_tau("Across platforms", platform_tau)
        self._logger.info("")
        self._print_pair_tau("Across datasets", dataset_tau)

        self._log_section("4.2 Bootstrap")
        platform_boot, dataset_boot = self._bootstrap(df)
        self._print_bootstrap(platform_boot, dataset_boot)

        self._log_section("4.3 CI intervals")
        platform_results, dataset_results = self._calculate_ci(platform_tau, dataset_tau, platform_boot, dataset_boot)
        self._print_ci(platform_results, dataset_results)

        self._log_section("5 Ranking")

        self._log_section("5.1 Rank ranges")
        utility_ranks = self._rank_ranges(ranked)
        self._print_rank_ranges(utility_ranks)

        self._log_section("5.2 Rank reversals")
        platform_reversals, dataset_reversals = self._rank_reversals(ranked)
        utility_count = len(df["tool"].unique())
        utility_pairs = int(utility_count * (utility_count -1) / 2)
        self._print_rank_reversals(platform_reversals, dataset_reversals, utility_pairs)

        self._log_section("6 Stability summary")
        stability = self._stability_summary(ranked, platform_results, dataset_results,
                                            platform_reversals, dataset_reversals)
        self._print_stability_summary(stability)

        stability_df = self._create_stability_frame(df, stability, platform_tau, dataset_tau)
        return stability_df

    def _create_stability_frame(self, df: pd.DataFrame, stability: pd.DataFrame,
                                platform_tau: pd.DataFrame, dataset_tau: pd.DataFrame) -> pd.DataFrame:
        stability = stability.copy()
        stability.insert(0, "mode", self._mode)
        stability.insert(1, "threading", self._threading)
        tools = len(df["tool"].unique())
        stability.insert(2, "tools", tools)
        platform_pairs = len(platform_tau)
        dataset_pairs = len(dataset_tau)
        stability["pairs"] = stability["comparison"].map({
            "Platforms": platform_pairs,
            "Datasets": dataset_pairs,
        })
        stability.insert(4, "pairs", stability.pop("pairs"))
        return stability

    def _filter_data(self, df: pd.DataFrame):
        df = df[df["mode"] == self._mode]
        df = df[df["threading"] == self._threading]
        return df

    def _validate_data(self, df: pd.DataFrame):
        df = df.copy()
        self._validate_data_quantity(df)

    def _validate_data_quantity(self, df: pd.DataFrame):
        keys = self.factors + ["run"]

        cell_counts = df.groupby(
            self.factors,
            observed=True
        ).size()

        self._logger.info("Observations:                 %d", len(df))
        self._logger.info("Configurations:               %d", len(cell_counts))
        self._logger.info("Min trials per configuration: %d", cell_counts.min())
        self._logger.info("Max trials per configuration: %d", cell_counts.max())
        missing = df[self.base_column].isna().sum()
        self._logger.info("Missing response values:      %d", missing)
        self._logger.info("Hosts:                        %s", ", ".join(df["host"].unique()))
        self._logger.info("Tools:                        %s", ", ".join(df["tool"].unique()))
        self._logger.info("Datasets:                     %s", ", ".join(df["dataset"].unique()))
        self._logger.info("Strengths:                    %s", ", ".join(df["strength"].unique()))
        self._logger.info("Base unit:                    %s", df[self.base_column].pint.units)
        self._logger.info("Duplicate trials:             %d", df.duplicated(keys).sum())

        if missing:
            raise ValueError("Missing response values: %d" % missing)

    def _print_medians(self, summary: pd.DataFrame):
        display_summary = summary[self.factors + ["trials"]].copy()
        median_column = "median_%s" % self.base_column
        display_summary[median_column] = (
            summary[median_column]
            .pint.to("MiB/s")
        )

        self._logger.info("Summary rows:                 %d", len(summary))
        self._logger.info("Trials per summary:           %d to %d", summary["trials"].min(), summary["trials"].max())

        display_summary = display_summary.head(12)

        headers = self.factors + ["Trials", "Median"]
        table_str = tabulate_pint(
            display_summary,
            showindex=False,
            headers=headers,
        )
        self._logger.info(table_str)

    def _calculate_medians(self, df: pd.DataFrame) -> pd.DataFrame:
        summary = (
            df.groupby(self.factors, observed=True)[self.base_column]
            .agg(
                median_throughput="median",
                trials="size"
            )
            .reset_index()
        )
        return summary

    def _print_ranked_utilities(self, ranked: pd.DataFrame):
        context = ["host", "dataset", "strength"]
        ranked = ranked.copy()
        group_sizes = ranked.groupby(context, observed=True).size()

        self._logger.info("Contexts:                     %d", group_sizes.size)
        self._logger.info("Utilities per context:        %d to %d", group_sizes.min(), group_sizes.max())

        ranked["rank"] = ranked["rank"].pint.magnitude.astype("int64")

        hosts = list(ranked["host"].unique())
        for host in hosts:
            dataset = "sensor"
            strength = "default"
            example = ranked[
                (ranked["host"] == host)
                & (ranked["dataset"] == dataset)
                & (ranked["strength"] == strength)
                ].sort_values("rank")

            example_display = example[["tool", "rank"]].copy()
            median_column = "median_%s" % self.base_column
            example_display[median_column] = (
                example[median_column]
                .pint.to("MiB/s")
            )

            self._logger.info("")
            self._logger.info("Ranks on %s for %s with %s:", host, dataset, strength)
            headers = ["Tool", "Rank", "Median"]
            table_str = tabulate_pint(
                example_display,
                showindex=False,
                headers=headers,
            )
            self._logger.info(table_str)

    def _print_ties_utilities(self, ties: pd.DataFrame):
        tie_display = ties[
            self.factors + ["rank"]
        ].copy()

        tie_display["rank"] = tie_display["rank"].pint.magnitude

        median_column = "median_%s" % self.base_column
        tie_display[median_column] = (
            ties[median_column]
            .pint.to("MiB/s")
        )

        tie_display = tie_display.sort_values(["host", "dataset", "strength", "rank"])

        self._logger.info("")
        self._logger.info("Tied utility rows:            %d", len(ties))
        headers = [f.capitalize() for f in self.factors] + ["Rank", "Median"]
        table_str = tabulate_pint(
            tie_display,
            showindex=False,
            headers=headers,
        )
        self._logger.info(table_str)

    def _rank_utilities(self, summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        context = ["host", "dataset", "strength"]

        median_column = "median_%s" % self.base_column
        ranked = summary.copy()
        ranked["rank"] = (
            ranked.groupby(context, observed=True)[median_column]
            .rank(method="average", ascending=False)
        )

        ties = ranked[
            ranked.duplicated(
                context + [median_column],
                keep=False,
            )
        ]

        return ranked, ties

    def _print_pair_tau(self, label: str, result: pd.DataFrame):
        self._logger.info(label)
        self._logger.info("Comparisons:                  %d", len(result))
        self._logger.info("Utilities:                    %d to %d", result["utilities"].min(),
                          result["utilities"].max())
        description = result["tau_b"].describe()[["min", "50%", "max"]]
        description = description.to_frame()

        headers = ["Item", "Tau_b"]
        table_str = tabulate_pint(
            description,
            headers=headers,
        )
        self._logger.info(table_str)

    def _pair_utilities(self, ranked: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        platform_tau = self._pairwise_tau(
            ranked,
            compare="host",
            fixed=["dataset", "strength"],
        )

        dataset_tau = self._pairwise_tau(
            ranked,
            compare="dataset",
            fixed=["host", "strength"],
        )
        return platform_tau, dataset_tau

    def _pairwise_tau(self, frame, compare, fixed) -> pd.DataFrame:
        rows = []

        for fixed_values, group in frame.groupby(fixed, observed=True):
            fixed_values = (
                fixed_values
                if isinstance(fixed_values, tuple)
                else (fixed_values,)
            )
            fixed_values = dict(zip(fixed, fixed_values))

            values = sorted(group[compare].unique())
            for first, second in combinations(values, 2):
                pair = (
                    group[group[compare].isin([first, second])]
                    .pivot(index="tool", columns=compare, values="rank")
                    .dropna()
                )

                tau = kendalltau(
                    pair[first],
                    pair[second],
                    variant="b",
                ).statistic

                rows.append({
                    **fixed_values,
                    "first": first,
                    "second": second,
                    "utilities": len(pair),
                    "tau_b": tau,
                })

        return pd.DataFrame(rows)

    def _print_bootstrap(self, platform_boot: pd.DataFrame, dataset_boot: pd.DataFrame):
        self._logger.info("Platform bootstrap rows:     %d", len(platform_boot))
        self._logger.info("Dataset bootstrap rows:      %d", len(dataset_boot))
        self._logger.info("Missing platform tau:        %f", platform_boot["tau_b"].isna().sum())
        self._logger.info("Missing dataset tau:         %f", dataset_boot["tau_b"].isna().sum())

    def _bootstrap(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        platform_boot = []
        dataset_boot = []
        for iteration in tqdm(range(self._n_boot), colour="green"):
            boot = (
                df.groupby(self.factors, group_keys=False, observed=True)
                .sample(
                    frac=1,
                    replace=True,
                    random_state=int(self._rng.integers(0, 2 ** 32 - 1)),
                )
            )

            median_column = "median_%s" % self.base_column
            boot_summary = (
                boot.groupby(self.factors, observed=True)[self.base_column]
                .median()
                .rename(median_column)
                .reset_index()
            )

            boot_summary["rank"] = (
                boot_summary.groupby(
                    ["host", "dataset", "strength"],
                    observed=True,
                )[median_column]
                .rank(ascending=False, method="average")
            )

            p = self._pairwise_tau(
                boot_summary, "host", ["dataset", "strength"]
            )
            p["iteration"] = iteration
            platform_boot.append(p)

            d = self._pairwise_tau(
                boot_summary, "dataset", ["host", "strength"]
            )
            d["iteration"] = iteration
            dataset_boot.append(d)

        platform_boot = pd.concat(platform_boot, ignore_index=True)
        dataset_boot = pd.concat(dataset_boot, ignore_index=True)

        return platform_boot, dataset_boot

    def _print_ci(self, platform_results: pd.DataFrame, dataset_results: pd.DataFrame):
        for label, result in [
            ("Platforms", platform_results),
            ("Datasets", dataset_results),
        ]:
            self._logger.info("")
            self._logger.info(label)
            self._logger.info("Rows:                         %d", len(result))
            self._logger.info("Missing intervals:            %d",
                              result[["ci_low", "ci_high"]].isna().sum().sum())
            self._logger.info("CI width:                     %f to %f", result["ci_width"].min(),
                              result["ci_width"].max())

        self._logger.info("")
        self._logger.info("Weakest platform comparisons:")
        platform_results = platform_results.sort_values("tau_b")
        platform_results = platform_results[["dataset", "strength", "first", "second",
                                             "tau_b", "ci_low", "ci_high"]]
        platform_results = platform_results.head(5)
        headers = ["Dataset", "Strength", "First", "Second", "Tau_b", "CI low", "CI high"]
        table_str = tabulate.tabulate(
            platform_results,
            showindex=False,
            headers=headers,
        )
        self._logger.info(table_str)

        dataset_results = dataset_results.sort_values("tau_b")
        dataset_results = dataset_results[["host", "strength", "first", "second",
                                           "tau_b", "ci_low", "ci_high"]]
        dataset_results = dataset_results.head(5)

        self._logger.info("")
        self._logger.info("Weakest dataset comparisons:")
        headers = ["Host", "Strength", "First", "Second", "Tau_b", "CI low", "CI high"]
        table_str = tabulate.tabulate(
            dataset_results,
            showindex=False,
            headers=headers,
        )
        self._logger.info(table_str)

    def _calculate_ci(self, platform_tau: pd.DataFrame, dataset_tau: pd.DataFrame,
                      platform_boot: pd.DataFrame, dataset_boot: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        platform_results = self._attach_ci(
            platform_tau,
            platform_boot,
            ["dataset", "strength", "first", "second"],
        )

        dataset_results = self._attach_ci(
            dataset_tau,
            dataset_boot,
            ["host", "strength", "first", "second"],
        )

        return platform_results, dataset_results

    def _attach_ci(self, point, bootstrap, keys):
        ci = (
            bootstrap.groupby(keys, observed=True)["tau_b"]
            .quantile([0.025, 0.975])
            .unstack()
            .reset_index()
            .rename(columns={0.025: "ci_low", 0.975: "ci_high"})
        )

        result = point.merge(
            ci,
            on=keys,
            how="left",
            validate="one_to_one",
        )
        result["ci_width"] = result["ci_high"] - result["ci_low"]
        return result

    def _print_rank_ranges(self, utility_ranks: pd.DataFrame):
        utility_ranks = utility_ranks.copy()
        headers = ["Tool", "Contexts", "Best rank", "Median rank", "Worst rank", "Top count", "Rank range", "Top (%)"]
        for column in ["best_rank", "median_rank", "worst_rank", "rank_range"]:
            utility_ranks[column] = utility_ranks[column].pint.magnitude.astype("int64")

        table_str = tabulate.tabulate(
            utility_ranks,
            showindex=False,
            headers=headers,
        )
        self._logger.info(table_str)

    def _rank_ranges(self, ranked: pd.DataFrame) -> pd.DataFrame:
        context = ["host", "dataset", "strength"]
        median_column = "median_%s" % self.base_column
        ranked["top_ranked"] = (
                ranked[median_column]
                == ranked.groupby(context, observed=True)[median_column]
                .transform("max")
        )

        utility_ranks = (
            ranked.groupby("tool", observed=True)
            .agg(
                contexts=("rank", "size"),
                best_rank=("rank", "min"),
                median_rank=("rank", "median"),
                worst_rank=("rank", "max"),
                top_count=("top_ranked", "sum"),
            )
            .reset_index()
        )

        utility_ranks["rank_range"] = (
                utility_ranks["worst_rank"] - utility_ranks["best_rank"]
        )
        utility_ranks["top_percent"] = (
                100 * utility_ranks["top_count"] / utility_ranks["contexts"]
        )

        utility_ranks = utility_ranks.sort_values(
            ["median_rank", "worst_rank", "tool"]
        )
        return utility_ranks

    def _print_rank_reversals(self, platform_reversals: pd.DataFrame, dataset_reversals: pd.DataFrame, utility_pairs: int):
        self._summarize_reversals("Across platforms", platform_reversals, opportunities=36, pair_count=utility_pairs)
        self._summarize_reversals("Across datasets", dataset_reversals, opportunities=54, pair_count=utility_pairs)

    def _summarize_reversals(self, label, reversals, opportunities: int, pair_count: int):
        counts = (
            reversals.groupby(["tool_a", "tool_b"])
            .size()
            .rename("reversals")
            .reset_index()
            .sort_values("reversals", ascending=False)
        )
        counts["percent"] = 100 * counts["reversals"] / opportunities

        self._logger.info("")
        self._logger.info(label)
        self._logger.info("Reversal events:        %d", len(reversals))
        self._logger.info("Affected utility pairs: %d of %d", len(counts), pair_count)
        counts = counts.head(10)
        headers = ["Tool A", "Tool B", "Reversals", "%"]
        table_str = tabulate_pint(
            counts,
            showindex=False,
            headers=headers,
        )
        self._logger.info(table_str)

    def _rank_reversals(self, ranked: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        platform_reversals = self._find_reversals(
            ranked, "host", ["dataset", "strength"]
        )
        dataset_reversals = self._find_reversals(
            ranked, "dataset", ["host", "strength"]
        )
        return platform_reversals, dataset_reversals

    def _find_reversals(self, frame, compare, fixed):
        rows = []

        for fixed_values, group in frame.groupby(fixed, observed=True):
            if not isinstance(fixed_values, tuple):
                fixed_values = (fixed_values,)
            fixed_values = dict(zip(fixed, fixed_values))

            for first, second in combinations(
                sorted(group[compare].unique()), 2
            ):
                ranks = (
                    group[group[compare].isin([first, second])]
                    .pivot(index="tool", columns=compare, values="rank")
                )

                for tool_a, tool_b in combinations(sorted(ranks.index), 2):
                    difference_first = (
                        ranks.loc[tool_a, first] - ranks.loc[tool_b, first]
                    )
                    difference_second = (
                        ranks.loc[tool_a, second] - ranks.loc[tool_b, second]
                    )

                    if difference_first * difference_second < 0:
                        rows.append({
                            **fixed_values,
                            "first": first,
                            "second": second,
                            "tool_a": tool_a,
                            "tool_b": tool_b,
                        })

        return pd.DataFrame(rows)

    def _print_stability_summary(self, stability: pd.DataFrame):
        headers = ["comparison", "Median tau b", "Minimum tau b", "Maximum tau b", "Reversal events", "reversal %"]
        table_str = tabulate_pint(
            stability,
            showindex=False,
            headers=headers,
        )
        self._logger.info(table_str)

    def _stability_summary(self, ranked: pd.DataFrame, platform_results: pd.DataFrame, dataset_results: pd.DataFrame,
                           platform_reversals: pd.DataFrame, dataset_reversals: pd.DataFrame) -> pd.DataFrame:
        utility_pairs = comb(ranked["tool"].nunique(), 2)  # 36

        stability = pd.DataFrame({
            "comparison": ["Platforms", "Datasets"],
            "median_tau_b": [
                platform_results["tau_b"].median(),
                dataset_results["tau_b"].median(),
            ],
            "minimum_tau_b": [
                platform_results["tau_b"].min(),
                dataset_results["tau_b"].min(),
            ],
            "maximum_tau_b": [
                platform_results["tau_b"].max(),
                dataset_results["tau_b"].max(),
            ],
            "reversal_events": [
                len(platform_reversals),
                len(dataset_reversals),
            ],
            "reversal_percent": [
                100 * len(platform_reversals)
                / (len(platform_results) * utility_pairs),
                100 * len(dataset_reversals)
                / (len(dataset_results) * utility_pairs),
            ],
        })
        return stability

    def _log_section(self, title: str):
        self._logger.info("")
        self._logger.info("%s %s %s", "-" * 20, title, "-" * 20)
        self._logger.info("")
