from __future__ import annotations

import os
import inspect
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def is_linux() -> bool:
    return os.name == 'posix'


def get_current_dir() -> str:
    return os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))  # type: ignore[arg-type]  # currentframe never returns none on cpython


def get_project_root() -> Path:
    from pathlib import Path
    return Path(__file__).parent.parent
