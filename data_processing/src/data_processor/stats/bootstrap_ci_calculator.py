import pandas as pd
import numpy as np
from tqdm import tqdm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
import tabulate


class BootstrapCiCalculator:
    def __init__(self, factors: list[str], n_boot: int, rng):
        self._factors = factors
        self._n_boot = n_boot
        self._rng = rng

    def print_ci(self, sorted_ci_values: pd.DataFrame):
        headers = ["Factor", "ω²", "CI low", "CI high"]
        table_str = tabulate.tabulate(sorted_ci_values,
                                      headers=headers,
                                      )
        print("")
        print("95%% confidence interval table")
        print(table_str)

    def calculate_ci(self, comp: pd.DataFrame, full_formula: str, effects: pd.DataFrame) \
            -> tuple[pd.DataFrame, pd.Series]:
        # Calculate bootstrap confidence intervals for the decompression \(\omega^2\) values.
        # Reason: This confirms whether the top-three ranking is stable.
        boot_data = comp.reset_index(drop=True)

        groups = list(
            boot_data.groupby(self._factors, observed=True).indices.values()
        )

        boot_results = []
        print("Calculating confidence intervals...")
        for _ in tqdm(range(self._n_boot), colour="green"):
            sampled_positions = np.concatenate([
                self._rng.choice(group, size=len(group), replace=True)
                for group in groups
            ])

            sample = boot_data.iloc[sampled_positions]

            model = smf.ols(full_formula, data=sample).fit()
            table = anova_lm(model, typ=3)

            mse_boot = (
                    table.loc["Residual", "sum_sq"]
                    / table.loc["Residual", "df"]
            )

            y_boot = model.model.endog
            ss_total_boot = np.sum((y_boot - y_boot.mean()) ** 2)

            omega = (
                            table["sum_sq"] - table["df"] * mse_boot
                    ) / (ss_total_boot + mse_boot)

            boot_results.append(
                omega.drop(["Intercept", "Residual"]).clip(lower=0)
            )

        bootstrap_omega = pd.DataFrame(boot_results)

        confidence_intervals = bootstrap_omega.quantile(
            [0.025, 0.975]
        ).T # transpose, swap rows and columns

        confidence_intervals.columns = ["ci_low", "ci_high"]

        results_with_ci = effects[["omega_squared"]].join(
            confidence_intervals
        )

        sorted_values = results_with_ci.sort_values("omega_squared", ascending=False)
        return sorted_values, confidence_intervals
