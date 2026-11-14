"""Convention checks for the source tree."""

from __future__ import annotations

import ast
import io
import tokenize
from collections.abc import Iterator
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE_FILES = sorted([*(ROOT / "src").rglob("*.py"), *(ROOT / "app").rglob("*.py")])
LOG_METHODS = {"debug", "info", "warning", "error", "exception", "critical"}


def _has_non_ascii(text: str) -> bool:
    return not text.isascii()


def _docstrings(tree: ast.AST) -> Iterator[tuple[int, str]]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            text = ast.get_docstring(node, clean=False)
            if text is not None:
                body = node.body[0]
                yield getattr(body, "lineno", 1), text


def _log_messages(tree: ast.AST) -> Iterator[tuple[int, str]]:
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr not in LOG_METHODS or not node.args:
            continue
        message = node.args[0]
        if isinstance(message, ast.Constant) and isinstance(message.value, str):
            yield message.lineno, message.value
        elif isinstance(message, ast.JoinedStr):
            for part in message.values:
                if isinstance(part, ast.Constant) and isinstance(part.value, str):
                    yield part.lineno, part.value


def _comments(source: str) -> Iterator[tuple[int, str]]:
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            yield token.start[0], token.string


def _offenders(find) -> list[str]:
    found = []
    for path in SOURCE_FILES:
        source = path.read_text(encoding="utf-8")
        for line, text in find(source):
            if _has_non_ascii(text):
                found.append(f"{path.relative_to(ROOT)}:{line}")
    return found


def test_comments_are_ascii():
    assert _offenders(_comments) == []


def test_docstrings_are_ascii():
    assert _offenders(lambda source: _docstrings(ast.parse(source))) == []


def test_log_messages_are_ascii():
    assert _offenders(lambda source: _log_messages(ast.parse(source))) == []
