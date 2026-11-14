"""Thread-safe lazy singletons."""

from __future__ import annotations

import threading
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

T = TypeVar("T")


def locked_singleton(factory: Callable[[], T]) -> Callable[[], T]:
    """Cache a zero-argument factory behind a lock and expose cache_clear() for tests."""
    lock = threading.Lock()
    holder: list[T] = []

    @wraps(factory)
    def wrapper() -> T:
        if not holder:
            with lock:
                if not holder:
                    holder.append(factory())
        return holder[0]

    def cache_clear() -> None:
        with lock:
            holder.clear()

    wrapper_any: Any = wrapper
    wrapper_any.cache_clear = cache_clear
    return wrapper
