"""Stdlib-only ``.env`` file loader.

python-dotenv happens to be installed in some environments this project runs
in, but it must never become a required runtime dependency, so this is a
small hand-rolled parser instead of ``import dotenv``. Deliberately named
``env`` (not ``dotenv``) so it never shadows the real package.

Format supported (KEY=VALUE per line):
  - blank lines and lines starting with ``#`` are ignored
  - an optional leading ``export `` is stripped (``export FOO=bar``)
  - the value may be wrapped in matching single or double quotes, which are
    stripped
  - existing ``os.environ`` entries are never overridden: a var already set
    in the environment wins over the same key in the file
"""

from __future__ import annotations

import os
from pathlib import Path


def load_env(path: Path | str = ".env") -> dict[str, str]:
    """Parse ``path`` as a ``.env`` file and load it into ``os.environ``.

    Returns the dict of KEY -> VALUE pairs parsed from the file (regardless
    of whether each one ended up being applied to os.environ, i.e. even keys
    skipped because they were already set are included in the return value).
    A missing file is a silent no-op that returns {}.
    """
    file_path = Path(path)
    parsed: dict[str, str] = {}
    if not file_path.exists():
        return parsed

    for raw_line in file_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        parsed[key] = value
        if key not in os.environ:
            os.environ[key] = value

    return parsed
