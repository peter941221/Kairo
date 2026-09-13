"""Cache-aware, correctness-first runtime dispatch for Kairo templates."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from .blueprint import make_compile_plan, validate_blueprint
from .cache import KernelCacheKey, RuntimeKernelCache


@dataclass(frozen=True)
class DispatchResult:
    """Result plus explainability metadata for one dispatch attempt."""

    output: Any
    backend: str
    cache_hit: bool
    plan: dict[str, Any]
    fallback_reason: str | None = None
    artifact_ms: float | None = None
    launch_ms: float | None = None


class DispatchError(RuntimeError):
    """Compilation or launch failed and no safe fallback was provided."""


class RuntimeDispatcher:
    """Execute a validated blueprint through an injected compiler/launcher.

    The workbench does not hard-code a CUDA compiler. Callers inject a builder
    and launcher, which keeps this layer usable for PTX, Cubin, graph packages,
    and deterministic test doubles while preserving one dispatch contract.
    """

    def __init__(
        self,
        cache: RuntimeKernelCache,
        builder: Callable[[dict[str, Any], tuple[int, ...]], bytes],
        launcher: Callable[[bytes, dict[str, Any]], Any],
    ):
        self.cache = cache
        self.builder = builder
        self.launcher = launcher

    def dispatch(
        self,
        blueprint: dict[str, Any],
        shape: tuple[int, ...],
        *,
        driver_version: str,
        gpu_capability: str,
        template_version: str = "v1",
        fallback: Callable[[Exception], Any] | None = None,
    ) -> DispatchResult:
        report = validate_blueprint(blueprint)
        plan = make_compile_plan(
            report,
            shape,
            driver_version=driver_version,
            gpu_capability=gpu_capability,
            template_version=template_version,
        )
        key = KernelCacheKey(
            blueprint_hash=report["blueprint_hash"],
            shape=tuple(shape),
            driver_version=driver_version,
            gpu_capability=gpu_capability,
            template_version=template_version,
        )
        artifact_started = time.perf_counter()
        try:
            artifact, cache_hit = self.cache.get_or_build(
                key, lambda: self.builder(report, tuple(shape))
            )
        except Exception as exc:
            artifact_ms = (time.perf_counter() - artifact_started) * 1000.0
            if fallback is None:
                raise DispatchError(f"artifact build failed: {exc}") from exc
            return DispatchResult(
                output=fallback(exc),
                backend="fallback",
                cache_hit=False,
                plan=plan,
                fallback_reason=f"build:{type(exc).__name__}: {exc}",
                artifact_ms=round(artifact_ms, 3),
            )
        artifact_ms = (time.perf_counter() - artifact_started) * 1000.0
        launch_started = time.perf_counter()
        try:
            output = self.launcher(artifact, report)
        except Exception as exc:
            launch_ms = (time.perf_counter() - launch_started) * 1000.0
            if fallback is None:
                raise DispatchError(f"artifact launch failed: {exc}") from exc
            return DispatchResult(
                output=fallback(exc),
                backend="fallback",
                cache_hit=cache_hit,
                plan=plan,
                fallback_reason=f"launch:{type(exc).__name__}: {exc}",
                artifact_ms=round(artifact_ms, 3),
                launch_ms=round(launch_ms, 3),
            )
        launch_ms = (time.perf_counter() - launch_started) * 1000.0
        return DispatchResult(
            output=output,
            backend="kairo",
            cache_hit=cache_hit,
            plan=plan,
            artifact_ms=round(artifact_ms, 3),
            launch_ms=round(launch_ms, 3),
        )
