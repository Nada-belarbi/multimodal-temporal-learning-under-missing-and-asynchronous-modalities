"""Explicit, injectable registries for stable cross-package component names."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from .applications import MosiAdapter, Pamap2Adapter
from .evaluation.report import render_report
from .evaluation.runner import evaluate
from .models.registry import MODEL_NAMES, build_majority_reference, build_model
from .runtime.engine import InferenceEngine
from .runtime.replay import replay

T = TypeVar("T")
Factory = Callable[..., T]


class RegistryError(LookupError):
    """Raised when a component key is duplicate or unknown."""


class ComponentRegistry(Generic[T]):
    def __init__(self, component_kind: str) -> None:
        if not component_kind:
            raise ValueError("component_kind must be nonempty")
        self.component_kind = component_kind
        self._factories: dict[str, Factory[T]] = {}

    def register(self, key: str, factory: Factory[T]) -> None:
        if not key or not key.strip():
            raise ValueError("registry keys must be nonempty")
        if key in self._factories:
            raise RegistryError(f"{self.component_kind} key {key!r} is already registered")
        self._factories[key] = factory

    def resolve(self, key: str) -> Factory[T]:
        try:
            return self._factories[key]
        except KeyError as error:
            raise RegistryError(
                f"unknown {self.component_kind} {key!r}; available={sorted(self._factories)}"
            ) from error

    def create(self, key: str, *args: Any, **kwargs: Any) -> T:
        return self.resolve(key)(*args, **kwargs)

    def keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))

    def __contains__(self, key: object) -> bool:
        return key in self._factories

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())


@dataclass(frozen=True)
class ServiceRegistries:
    adapters: ComponentRegistry[Any]
    models: ComponentRegistry[Any]
    evaluators: ComponentRegistry[Any]
    runtime: ComponentRegistry[Any]
    reports: ComponentRegistry[Any]


def build_default_registries() -> ServiceRegistries:
    """Create fresh built-in registries; callers can extend their own instance in tests/plugins."""
    adapters: ComponentRegistry[Any] = ComponentRegistry("adapter")
    adapters.register("pamap2", Pamap2Adapter)
    adapters.register("mosi", MosiAdapter)

    models: ComponentRegistry[Any] = ComponentRegistry("model")
    for model_name in MODEL_NAMES:
        if model_name == "majority":
            models.register(model_name, build_majority_reference)
        else:
            models.register(
                model_name,
                lambda source_dimensions, class_map, config=None, *, _name=model_name: build_model(
                    _name, source_dimensions, class_map, config
                ),
            )

    evaluators: ComponentRegistry[Any] = ComponentRegistry("evaluator")
    evaluators.register("default", evaluate)
    runtime: ComponentRegistry[Any] = ComponentRegistry("runtime")
    runtime.register("predictor", InferenceEngine)
    runtime.register("replay", replay)
    reports: ComponentRegistry[Any] = ComponentRegistry("report renderer")
    reports.register("html", render_report)
    return ServiceRegistries(adapters, models, evaluators, runtime, reports)
