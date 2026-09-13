"""Declarative kernel blueprints and fail-closed topology constraints.

The validator intentionally accepts a small, explicit template vocabulary. A
blueprint is a planning artifact, not permission to generate arbitrary CUDA:
unknown template/precision combinations fail before compilation or execution.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any


class BlueprintValidationError(ValueError):
    """A blueprint violates a named constraint."""

    def __init__(self, path: str, message: str):
        self.path = path
        self.message = message
        super().__init__(f"{path}: {message}")


_REQUIRED = {
    "architecture_target",
    "precision",
    "template",
    "tile",
    "pipeline",
    "layout",
    "epilogue",
    "scheduling",
}
_PRECISIONS = {"fp16", "bf16", "fp8_scaled", "nvfp4_scaled"}
_ARCHITECTURES = {"sm90", "sm100", "sm120"}
_TOPOLOGIES = {
    ("fp16", "fp16_gemm"),
    ("bf16", "fp16_gemm"),
    ("fp16", "tma_wmma"),
    ("bf16", "tma_wmma"),
    ("fp8_scaled", "fp8_scaled_gemm"),
    ("nvfp4_scaled", "nvfp4_scaled_gemm"),
}
_LAYOUTS = {"row_major", "col_major", "interleaved", "swizzled"}
_EPILOGUES = {"identity", "relu", "silu", "scale", "bias_scale"}
_SCHEDULES = {"cta", "persistent", "stream_k"}


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BlueprintValidationError(path, "must be an object")
    return value


def _positive_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise BlueprintValidationError(path, "must be a positive integer")
    return value


def _choice(value: Any, path: str, choices: set[str]) -> str:
    if not isinstance(value, str) or value not in choices:
        raise BlueprintValidationError(path, f"must be one of {sorted(choices)}")
    return value


def _blueprint_hash(data: dict[str, Any]) -> str:
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_blueprint(blueprint: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize one declarative kernel blueprint.

    The returned report is JSON-serializable and contains derived resource and
    cache information. Validation is deliberately fail-closed: an unknown
    topology, layout, or pipeline is an error rather than a guessed fallback.
    """

    root = _mapping(blueprint, "blueprint")
    missing = sorted(_REQUIRED - root.keys())
    if missing:
        raise BlueprintValidationError("blueprint", f"missing required fields: {missing}")
    architecture = _choice(root["architecture_target"], "architecture_target", _ARCHITECTURES)
    precision = _choice(root["precision"], "precision", _PRECISIONS)
    template = root["template"]
    _choice(template, "template", {item[1] for item in _TOPOLOGIES})
    if (precision, template) not in _TOPOLOGIES:
        raise BlueprintValidationError(
            "template", f"unsupported topology for precision={precision}: {template}"
        )

    tile = _mapping(root["tile"], "tile")
    dimensions = {
        name: _positive_int(tile.get(name), f"tile.{name}") for name in ("m", "n", "k")
    }
    for name, value in dimensions.items():
        if value % 16:
            raise BlueprintValidationError(f"tile.{name}", "must be divisible by 16")
    if precision == "nvfp4_scaled":
        for name in ("n", "k"):
            if dimensions[name] % 32:
                raise BlueprintValidationError(f"tile.{name}", "NVFP4 must be divisible by 32")

    pipeline = _mapping(root["pipeline"], "pipeline")
    stages = _positive_int(pipeline.get("stages"), "pipeline.stages")
    if stages > 8:
        raise BlueprintValidationError("pipeline.stages", "must be <= 8")
    pipeline_kind = _choice(
        pipeline.get("kind", "single_stage"),
        "pipeline.kind",
        {"single_stage", "multi_stage"},
    )
    asynchronous = bool(pipeline.get("asynchronous", False))
    load = pipeline.get("load", "direct")
    synchronization = pipeline.get("synchronization", "none")
    if asynchronous and load != "tma":
        raise BlueprintValidationError("pipeline.load", "asynchronous pipelines require tma")
    if asynchronous and synchronization != "mbarrier":
        raise BlueprintValidationError(
            "pipeline.synchronization", "asynchronous pipelines require mbarrier"
        )
    if pipeline_kind == "single_stage" and stages != 1:
        raise BlueprintValidationError("pipeline", "single_stage requires stages=1")
    if pipeline_kind == "multi_stage" and stages < 2:
        raise BlueprintValidationError("pipeline", "multi_stage requires stages>=2")
    if template == "fp16_gemm" and (asynchronous or pipeline_kind != "single_stage"):
        raise BlueprintValidationError(
            "pipeline", "fp16_gemm template only permits the direct single-stage control"
        )

    layout = _mapping(root["layout"], "layout")
    for name in ("a", "b", "c"):
        _choice(layout.get(name), f"layout.{name}", _LAYOUTS)
    if asynchronous and layout.get("swizzle") not in {"swizzled", "interleaved"}:
        raise BlueprintValidationError(
            "layout.swizzle", "TMA asynchronous templates require swizzled or interleaved layout"
        )

    epilogue = _mapping(root["epilogue"], "epilogue")
    epilogue_kind = _choice(epilogue.get("kind"), "epilogue.kind", _EPILOGUES)
    scheduling = _mapping(root["scheduling"], "scheduling")
    schedule = _choice(scheduling.get("mode"), "scheduling.mode", _SCHEDULES)
    split_k = _positive_int(scheduling.get("split_k", 1), "scheduling.split_k")
    if schedule != "stream_k" and split_k != 1:
        raise BlueprintValidationError(
            "scheduling.split_k", "split_k>1 requires scheduling.mode=stream_k"
        )

    dynamic = root.get("dynamic_dimensions", [])
    if not isinstance(dynamic, list) or any(item not in {"m", "n", "k"} for item in dynamic):
        raise BlueprintValidationError(
            "dynamic_dimensions", "must be a list containing only m, n, or k"
        )
    budget = _mapping(root.get("resource_budget", {}), "resource_budget")
    shared_memory_limit = _positive_int(
        budget.get("shared_memory_bytes", 227 * 1024), "resource_budget.shared_memory_bytes"
    )
    element_bytes = 1 if precision in {"fp8_scaled", "nvfp4_scaled"} else 2
    tile_bytes = (dimensions["m"] * dimensions["k"] + dimensions["k"] * dimensions["n"]) * element_bytes
    shared_memory_estimate = tile_bytes * stages
    if shared_memory_estimate > shared_memory_limit:
        raise BlueprintValidationError(
            "resource_budget.shared_memory_bytes",
            f"estimated tile storage {shared_memory_estimate} exceeds {shared_memory_limit}",
        )

    normalized = deepcopy(root)
    normalized["architecture_target"] = architecture
    normalized["precision"] = precision
    normalized["template"] = template
    normalized["tile"] = dimensions
    normalized["pipeline"] = {
        **pipeline,
        "stages": stages,
        "kind": pipeline_kind,
        "asynchronous": asynchronous,
    }
    normalized["scheduling"] = {**scheduling, "mode": schedule, "split_k": split_k}
    normalized["dynamic_dimensions"] = list(dynamic)
    return {
        "valid": True,
        "blueprint_hash": _blueprint_hash(normalized),
        "topology": f"{template}:{precision}",
        "normalized": normalized,
        "derived": {
            "shared_memory_estimate_bytes": shared_memory_estimate,
            "shared_memory_limit_bytes": shared_memory_limit,
            "cache_key_dimensions": [
                "blueprint_hash",
                "shape",
                "driver_version",
                "gpu_capability",
                "template_version",
            ],
            "dynamic_dimensions": list(dynamic),
            "fallback": root.get("fallback", "reference"),
            "epilogue": epilogue_kind,
            "schedule": schedule,
        },
    }
