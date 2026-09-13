import unittest

from kairo_lab.blueprint import (
    BlueprintValidationError,
    make_compile_plan,
    validate_blueprint,
)


def valid_blueprint(**overrides):
    blueprint = {
        "architecture_target": "sm120",
        "precision": "fp16",
        "template": "tma_wmma",
        "tile": {"m": 128, "n": 128, "k": 64},
        "pipeline": {
            "kind": "multi_stage",
            "stages": 2,
            "asynchronous": True,
            "load": "tma",
            "synchronization": "mbarrier",
        },
        "layout": {
            "a": "row_major",
            "b": "col_major",
            "c": "row_major",
            "swizzle": "swizzled",
        },
        "epilogue": {"kind": "identity"},
        "scheduling": {"mode": "cta", "split_k": 1},
    }
    blueprint.update(overrides)
    return blueprint


class BlueprintTests(unittest.TestCase):
    def test_valid_blueprint_returns_hash_and_derived_plan(self):
        report = validate_blueprint(valid_blueprint())
        self.assertTrue(report["valid"])
        self.assertEqual(report["topology"], "tma_wmma:fp16")
        self.assertEqual(report["derived"]["shared_memory_estimate_bytes"], 65536)
        self.assertEqual(len(report["blueprint_hash"]), 64)

    def test_unknown_precision_template_pair_fails_closed(self):
        with self.assertRaisesRegex(BlueprintValidationError, "unsupported topology"):
            validate_blueprint(
                valid_blueprint(precision="nvfp4_scaled", template="tma_wmma")
            )

    def test_async_pipeline_requires_tma_and_mbarrier(self):
        pipeline = valid_blueprint()["pipeline"] | {"load": "direct"}
        with self.assertRaisesRegex(BlueprintValidationError, "require tma"):
            validate_blueprint(valid_blueprint(pipeline=pipeline))

    def test_nvfp4_requires_scale_aligned_n_and_k(self):
        blueprint = valid_blueprint(
            precision="nvfp4_scaled",
            template="nvfp4_scaled_gemm",
            tile={"m": 32, "n": 48, "k": 64},
            pipeline={"kind": "multi_stage", "stages": 2},
        )
        with self.assertRaisesRegex(BlueprintValidationError, "NVFP4"):
            validate_blueprint(blueprint)

    def test_shared_memory_budget_is_enforced(self):
        with self.assertRaisesRegex(BlueprintValidationError, "exceeds"):
            validate_blueprint(
                valid_blueprint(resource_budget={"shared_memory_bytes": 1024})
            )

    def test_dynamic_dimensions_are_explicit(self):
        report = validate_blueprint(valid_blueprint(dynamic_dimensions=["m"]))
        self.assertEqual(report["derived"]["dynamic_dimensions"], ["m"])
        with self.assertRaisesRegex(BlueprintValidationError, "dynamic_dimensions"):
            validate_blueprint(valid_blueprint(dynamic_dimensions=["batch"]))

    def test_compile_plan_uses_hardware_and_shape_in_cache_key(self):
        report = validate_blueprint(valid_blueprint())
        plan = make_compile_plan(
            report,
            (1024, 1024, 1024),
            driver_version="596.36",
            gpu_capability="sm120",
        )
        self.assertEqual(plan["steps"][0], "validate_blueprint")
        self.assertTrue(plan["gates"]["correctness_required_before_benchmark"])
        self.assertEqual(plan["cache_key"]["shape"], [1024, 1024, 1024])
        self.assertEqual(plan["cache_key"]["gpu_capability"], "sm120")
        self.assertEqual(len(plan["cache_key"]["digest"]), 64)

    def test_compile_plan_rejects_unvalidated_report(self):
        with self.assertRaisesRegex(BlueprintValidationError, "valid"):
            make_compile_plan(
                {"valid": False},
                (128, 128, 64),
                driver_version="596.36",
                gpu_capability="sm120",
            )
