import pandas as pd
import pint_pandas
import tabulate


def tabulate_pint(
               df: pd.DataFrame,
               headers: list[str],
               tablefmt="simple",
               showindex="default",
               colalign=None,
               ):
    display_df = df.copy()
    header_names = []


    o = 0
    if showindex == "default" or showindex is True:
        header_names.append(headers[0])
        o = 1

    for i, column in enumerate(display_df.columns):
        series = display_df[column]

        if isinstance(series.dtype, pint_pandas.PintType):
            unit = series.pint.units
            display_df[column] = series.pint.magnitude
            header_names.append(f"{headers[i + o]} ({unit:~P})")
        else:
            header_names.append(headers[i + o])

    table_str = tabulate.tabulate(display_df,
                                  headers=header_names,
                                  tablefmt=tablefmt,
                                  showindex=showindex,
                                  colalign=colalign,
                                  )
    return table_str
