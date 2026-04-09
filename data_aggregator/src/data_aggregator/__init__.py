from pint import UnitRegistry, set_application_registry
import pint_pandas  # needed for initialization


ureg = UnitRegistry()
ureg.formatter.default_format = "P"

set_application_registry(ureg)
