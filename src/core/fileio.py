"""Small file-system helpers shared across packages."""

from __future__ import annotations

import os
from pathlib import Path


def atomic_write_text(path: Path, content: str) -> None:
    """Write UTF-8 text through a sibling temporary file and atomically replace the destination."""
    tmp_path = path.with_name(f"{path.name}.tmp")
    try:
        tmp_path.write_text(content, encoding="utf-8")
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
