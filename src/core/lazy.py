"""
Thread-safe lazy singletons.

Responsibilities:
  - Provide a decorator that builds a value once and shares it
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

T = TypeVar("T")


def locked_singleton(factory: Callable[[], T]) -> Callable[[], T]:
    """Cache the result of a zero-argument factory behind a lock.

    The factory runs on the first call only. Concurrent first calls wait
    on the lock instead of building the value twice (double-checked
    locking). If the factory raises, nothing is cached and the next call
    retries. The wrapper gains a cache_clear() method for tests.

    Args:
        factory: A function that takes no arguments and builds the value.

    Returns:
        A function with the same name that returns the shared value.
    """
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
