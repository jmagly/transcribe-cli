"""Shared test setup.

CI runners set variables such as FORCE_COLOR, which make Rich emit ANSI
styling even when output is captured. That splits option names like
``--output-dir`` and wraps paths, so plain-text assertions fail only in CI.
Clear them before the CLI module (and its module-level Console) is imported.
"""

import os

for _var in ("FORCE_COLOR", "PY_COLORS", "CLICOLOR_FORCE", "TTY_COMPATIBLE"):
    os.environ.pop(_var, None)
os.environ["NO_COLOR"] = "1"
os.environ["TERM"] = "dumb"
os.environ.setdefault("COLUMNS", "200")
