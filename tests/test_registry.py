import unittest

from mmt.applications import MosiAdapter, Pamap2Adapter
from mmt.models import V1GatedFusion
from mmt.registry import ComponentRegistry, RegistryError, build_default_registries
from mmt.runtime.engine import InferenceEngine


class RegistryTests(unittest.TestCase):
    def test_registry_rejects_duplicate_and_unknown_names(self):
        registry = ComponentRegistry("adapter")
        registry.register("toy", object)

        with self.assertRaises(RegistryError):
            registry.register("toy", object)
        with self.assertRaises(RegistryError):
            registry.create("unknown")

    def test_default_registries_construct_known_components(self):
        registries = build_default_registries()

        self.assertIsInstance(registries.adapters.create("pamap2"), Pamap2Adapter)
        self.assertIsInstance(registries.adapters.create("mosi"), MosiAdapter)
        model = registries.models.create(
            "v1", {"imu": 2}, {"negative": 0, "positive": 1}, {"hidden_size": 8}
        )
        self.assertIsInstance(model, V1GatedFusion)
        self.assertIs(registries.runtime.resolve("predictor"), InferenceEngine)

    def test_each_default_registry_bundle_is_independent(self):
        first = build_default_registries()
        second = build_default_registries()
        first.adapters.register("custom", object)

        self.assertNotIn("custom", second.adapters)


if __name__ == "__main__":
    unittest.main()
