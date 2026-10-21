from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._entries: dict[str, T] = {}

    def register(self, name: str) -> Callable[[T], T]:
        def decorate(obj: T) -> T:
            if name in self._entries:
                raise KeyError(f"{self._kind} '{name}' already registered")
            self._entries[name] = obj
            return obj

        return decorate

    def add(self, name: str, obj: T) -> None:
        self._entries[name] = obj

    def get(self, name: str) -> T:
        if name not in self._entries:
            known = ", ".join(sorted(self._entries)) or "<empty>"
            raise KeyError(f"unknown {self._kind} '{name}'; known: {known}")
        return self._entries[name]

    def names(self) -> list[str]:
        return sorted(self._entries)
