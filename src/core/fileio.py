"""
Small file-system helpers shared across packages.

Responsibilities:
  - Write files atomically so readers never see a partial file
"""

from __future__ import annotations

import os
from pathlib import Path


def atomic_write_text(path: Path, content: str) -> None:
    """Write text to a path so an interrupted write leaves the old file.

    The content goes to a sibling temp file first, then os.replace swaps
    it in. A crash mid-write leaves the previous file untouched.

    Args:
        path: Destination file path. Its parent directory must exist.
        content: UTF-8 text to write.
    """
    tmp_path = path.with_name(f"{path.name}.tmp")
    try:
        tmp_path.write_text(content, encoding="utf-8")
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
