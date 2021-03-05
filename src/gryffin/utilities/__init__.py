#!/usr/bin/env python

from .decorators      import safe_execute
from .defaults        import default_general_configurations
from .defaults        import default_database_configurations
from .defaults        import default_model_configurations

from .exceptions      import GryffinParseError
from .exceptions      import GryffinModuleError
from .exceptions      import GryffinNotFoundError
from .exceptions      import GryffinUnknownSettingsError
from .exceptions      import GryffinValueError
from .exceptions      import GryffinVersionError

from .logger          import Logger

from .json_parser     import ParserJSON
from .pickle_parser   import ParserPickle
from .category_parser import CategoryParser
from .config_parser   import ConfigParser


def parse_time(start, end):
    elapsed = end - start  # elapsed time in seconds
    if elapsed <= 1.0:
        ms = elapsed * 1000.
        time_string = f"{ms:.4f} ms"
    elif 1.0 < elapsed < 60.0:
        time_string = f"{elapsed:.2f} s"
    else:
        m, s = divmod(elapsed, 60)
        time_string = f"{m} min {s:.2f} s"
    return time_string

