import logging

import pandas as pd
import numpy as np


class TrapezoidEnergyCalculator:
    def __init__(self):
        self._logger = logging.getLogger(self.__class__.__name__)

    def calculate_energy(self, df: pd.DataFrame) -> pd.DataFrame:
        df["power_duration"] = df.groupby("run")["timestamp"].diff().dt.total_seconds().astype("pint[second]")
        #df["energy_used"] = df["power"] * df["power_duration"]
        #df["energy_used"] = df["energy_used"].pint.to("joule")
        #power = df["power"]
        #time = df["power_duration"]

        ## 1. Extract magnitudes
        #p_mag = power.pint.magnitude
        #t_mag = time.pint.magnitude

        # 2. Extract units
        #p_unit = power.dtype.units
        #t_unit = time.dtype.units

        # 3. Integrate magnitudes
        #energy_mag = np.trapezoid(p_mag, t_mag)

        # 4. Reattach units
        #energy = energy_mag * (p_unit * t_unit)

        #df["energy_used"] = energy
        #df["energy_used"] = self._trapezoid(df["power"], df["power_duration"])

        # average power between previous and current row
        df["power_avg"] = (df["power"].shift() + df["power"]) / 2
        # trapezoid energy per row
        df["energy_used"] = df["power_avg"] * df["power_duration"]
        df["energy_used"] = df["energy_used"].pint.to("joule")
        del df['power_avg']

        print(df.dtypes)
        print(df.head())
        return df

    def _trapezoid(self, y, x):
        #y_mag = y.pint.magnitude
        #x_mag = x.pint.magnitude
        #unit = y.dtype.units * x.dtype.units
        #print("------")
        #print("y:\n%s" % y_mag)
        #print("x:\n%s" % x_mag)
        #print("unit: %s" % unit)
        #np_array = np.trapezoid(y_mag, x_mag)
        #print("np array:\n%r" % np_array)
        #print("------")
        #np_array = np.asarray(np_array, dtype=float)
        #return np_array * unit

        p = y.pint.magnitude
        t = x.pint.magnitude

        dx = np.diff(t)
        avg_p = (p[:-1] + p[1:]) / 2
        segment_energy_mag = avg_p * dx

        #energy_unit = df["power"].dtype.units * df["timestamp"].dtype.units
        energy_unit = y.dtype.units * x.dtype.units
        segment_energy = segment_energy_mag * energy_unit
        #cumulative_energy = np.cumsum(segment_energy_mag) * energy_unit
