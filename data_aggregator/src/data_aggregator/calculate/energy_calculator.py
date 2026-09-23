from abc import ABC, abstractmethod

import pandas as pd


class EnergyCalculator(ABC):

    @abstractmethod
    def calculate_energy(self, df: pd.DataFrame) -> pd.DataFrame:
        pass
