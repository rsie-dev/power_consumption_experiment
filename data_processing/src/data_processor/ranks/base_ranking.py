import logging
from pathlib import Path
from abc import abstractmethod

import pandas as pd
from itertools import combinations
from scipy.stats import kendalltau
import tabulate

from data_processor.processor import Processor
from data_processor.util import tabulate_pint


class BaseRanking(Processor):
    def __init__(self, resources: Path, threading: str, mode: str):
        super().__init__(resources)
        self._logger = logging.getLogger(self.__class__.__name__)
        self._threading = threading
        self._mode = mode

    @property
    def factors(self):
        return ["host", "tool", "dataset", "strength"]

    @property
    @abstractmethod
    def base_column(self) -> str:
        pass

    def process(self, input_file: Path) -> None:
        self._rank(input_file)

    def _rank(self, input_file: Path) -> None:
        df = self._preparation(input_file)
        self._calculate(df)

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

    def _calculate(self, df: pd.DataFrame):
        self._log_section("2. Calculate medians")

        #self._log_section("2.1 Medians")
        summary = self._calculate_medians(df)
        self._print_medians(summary)

        self._log_section("3 Ranks")
        ranked, ties = self._rank_utilities(summary)
        self._print_ranked_utilities(ranked)
        self._print_ties_utilities(ties)

        self._log_section("4 Pairwise Kendall’s")
        platform_tau, dataset_tau = self._pair_utilities(ranked)
        self._print_pair_tau("Across platforms", platform_tau)
        self._logger.info("")
        self._print_pair_tau("Across datasets", dataset_tau)

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

        host = "radxax4"
        dataset = "imagelarge"
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
        self._logger.info("Utilities:                    %d to %d", result["utilities"].min(), result["utilities"].max())
        description = result["tau_b"].describe()[["min", "50%", "max"]]
        description = description.to_frame()

        headers = ["Item", "Tau_b"]
        table_str = tabulate_pint(
            description,
            headers=headers,
        )
        self._logger.info(table_str)

    def _pair_utilities(self, ranked: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        def pairwise_tau(frame, compare, fixed) -> pd.DataFrame:
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

        platform_tau = pairwise_tau(
            ranked,
            compare="host",
            fixed=["dataset", "strength"],
        )

        dataset_tau = pairwise_tau(
            ranked,
            compare="dataset",
            fixed=["host", "strength"],
        )
        return platform_tau, dataset_tau

    def _log_section(self, title: str):
        self._logger.info("")
        self._logger.info("%s %s %s", "-" * 20, title, "-" * 20)
        self._logger.info("")
