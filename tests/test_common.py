#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# flake8: noqa
# pylint: disable=broad-exception-raised, raise-missing-from, too-many-arguments, redefined-outer-name
# pylint: disable=multiple-statements, logging-fstring-interpolation, trailing-whitespace, line-too-long
# pylint: disable=broad-exception-caught, missing-function-docstring, missing-class-docstring
# pylint: disable=f-string-without-interpolation
# pylance: disable=reportMissingImports, reportMissingModuleSource

import unittest
import sys
import os

# Fix imports by adding parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import (
    weights_to_umf,
    umf_to_weights,
    calculate_umf_from_recipe,
    make_json_safe,
    load_materials,
    resolve_inventory,
    filter_materials_by_inventory,
)

# Relative tolerance for round-trip checks, in percent. Both conversions round
# their output (2 decimals for weights, 3 for UMF) and molar masses themselves
# are rounded in the database, so an exact round-trip is not expected.
TOLERANCE_PERCENT = 1.0

# Absolute floor of the tolerance, for trace oxides whose 1% band is narrower
# than the rounding step of the result. UMF values are rounded to 3 decimals, so
# the floor is exactly one such step: enough to absorb a rounding difference, but
# small enough that dropping even the smallest expected oxide (0.002) fails the
# check. Measured deviations of the checked compositions are 0.000-0.017, and
# every non-zero one of them is already covered by the 1% relative band.
TOLERANCE_ABSOLUTE = 0.001


def assert_umf_close(test_case, expected_umf, actual_umf):
    """Compare UMF dictionaries oxide by oxide within the shared tolerance"""
    for oxide, expected_value in expected_umf.items():
        actual_value = actual_umf.get(oxide, 0)
        allowed = max(TOLERANCE_ABSOLUTE, abs(expected_value) * TOLERANCE_PERCENT / 100)
        test_case.assertLessEqual(
            abs(actual_value - expected_value), allowed,
            f"{oxide}: expected {expected_value}, got {actual_value}")


class TestUmfConversions(unittest.TestCase):

    def assert_umf_close(self, expected_umf, actual_umf):
        assert_umf_close(self, expected_umf, actual_umf)

    def test_round_trip_transparent_glaze(self):
        """UMF -> weights -> UMF keeps the transparent glaze formula"""
        umf = {
            "SiO2": 3.151,
            "Al2O3": 0.379,
            "B2O3": 0.266,
            "Na2O": 0.143,
            "K2O": 0.086,
            "CaO": 0.718,
            "MgO": 0.048,
        }

        self.assert_umf_close(umf, weights_to_umf(umf_to_weights(umf)))

    def test_round_trip_simple_feldspathic(self):
        """UMF -> weights -> UMF keeps a simple feldspathic formula"""
        umf = {
            "SiO2": 4.0,
            "Al2O3": 1.0,
            "Na2O": 0.5,
            "K2O": 0.5,
        }

        self.assert_umf_close(umf, weights_to_umf(umf_to_weights(umf)))

    def test_round_trip_with_zinc(self):
        """UMF -> weights -> UMF keeps a formula with several RO fluxes"""
        umf = {
            "SiO2": 2.5,
            "Al2O3": 0.4,
            "CaO": 0.6,
            "MgO": 0.2,
            "ZnO": 0.2,
        }

        self.assert_umf_close(umf, weights_to_umf(umf_to_weights(umf)))

    def test_weights_sum_to_100(self):
        """umf_to_weights returns weight percentages that add up to 100%"""
        umf = {"SiO2": 3.151, "Al2O3": 0.379, "CaO": 0.718, "Na2O": 0.143}

        weights = umf_to_weights(umf)

        self.assertAlmostEqual(sum(weights.values()), 100.0, delta=0.05)


class TestCalculateUmfFromRecipe(unittest.TestCase):

    def weight_composition_of(self, recipe):
        """Sum up the oxide formulas of a {material: percent} recipe"""
        materials = {m['name']: m for m in load_materials(only_inventory=False, priority=False)}

        composition = {}
        for name, percentage in recipe.items():
            self.assertIn(name, materials, f"material {name} is missing from the database")
            for oxide, content in materials[name].get('formula', {}).items():
                composition[oxide] = composition.get(oxide, 0.0) + content * (percentage / 100.0)
        return composition

    def test_umf_of_reference_transparent_recipe(self):
        """UMF of the reference transparent glaze recipe matches the known values"""
        recipe = {
            "Нефелин-сиенит VR13": 30,
            "Кварцевая мука Кварцверке W12": 20,
            "Волластонит МИВОЛЛ": 20,
            "Улексит (Химпэк)": 15,
            "Каолин КЖФ-1": 15,
        }

        expected_umf = {
            "SiO2": 3.144,
            "Al2O3": 0.378,
            "B2O3": 0.265,
            "Na2O": 0.143,
            "K2O": 0.086,
            "CaO": 0.717,
            "MgO": 0.048,
            "SrO": 0.005,
            "Fe2O3": 0.002,
            "TiO2": 0.003,
        }

        umf, raw_umf = calculate_umf_from_recipe(self.weight_composition_of(recipe))

        assert_umf_close(self, expected_umf, umf)

        # The raw values are the same numbers before rounding
        for oxide, value in umf.items():
            self.assertAlmostEqual(value, raw_umf[oxide], places=3)

    def test_matches_weights_to_umf(self):
        """calculate_umf_from_recipe agrees with weights_to_umf on the same input"""
        weight_composition = umf_to_weights({"SiO2": 3.0, "Al2O3": 0.4, "CaO": 0.7, "Na2O": 0.3})

        umf, _raw_umf = calculate_umf_from_recipe(weight_composition)

        self.assertEqual(umf, weights_to_umf(weight_composition))

    def test_fluxless_composition_normalizes_to_smallest_oxide(self):
        """Without fluxes the smallest molar amount is used as unity"""
        umf, _raw_umf = calculate_umf_from_recipe({"SiO2": 90.0, "Al2O3": 10.0})

        self.assertEqual(umf["Al2O3"], 1.0)
        self.assertGreater(umf["SiO2"], 1.0)


class TestMakeJsonSafe(unittest.TestCase):

    def test_infinities_and_nan_become_strings(self):
        self.assertEqual(make_json_safe(float('inf')), "Infinity")
        self.assertEqual(make_json_safe(float('-inf')), "-Infinity")
        self.assertEqual(make_json_safe(float('nan')), "NaN")

    def test_finite_values_are_untouched(self):
        self.assertEqual(make_json_safe(1.5), 1.5)
        self.assertEqual(make_json_safe(0), 0)
        self.assertEqual(make_json_safe("text"), "text")
        self.assertIsNone(make_json_safe(None))

    def test_nested_structures(self):
        obj = {
            "ratios": {"SiO2:Al2O3": float('inf'), "R2O:RO": 0.25},
            "errors": [float('nan'), 1.0, [float('-inf'), {"x": float('inf')}]],
            "name": "solution",
        }

        expected = {
            "ratios": {"SiO2:Al2O3": "Infinity", "R2O:RO": 0.25},
            "errors": ["NaN", 1.0, ["-Infinity", {"x": "Infinity"}]],
            "name": "solution",
        }

        self.assertEqual(make_json_safe(obj), expected)

    def test_result_is_json_serializable(self):
        import json

        payload = make_json_safe({"a": float('inf'), "b": [float('nan')]})

        self.assertEqual(json.loads(json.dumps(payload)), {"a": "Infinity", "b": ["NaN"]})


class TestInventoryHelpers(unittest.TestCase):

    def test_resolve_inventory_returns_explicit_data(self):
        explicit = ["Каолин КЖФ-1", "Мел, CaCO3"]

        self.assertEqual(resolve_inventory(explicit), explicit)

    def test_resolve_inventory_reads_in_inventory_flag(self):
        inventory = resolve_inventory()
        flagged = [m['name'] for m in load_materials(only_inventory=False, priority=False)
                   if m.get('inInventory') is True]

        self.assertGreater(len(inventory), 0)
        self.assertEqual(sorted(inventory), sorted(flagged))

    def test_resolve_inventory_none_falls_back_to_the_database(self):
        """None means "no explicit inventory": the inInventory materials are used"""
        flagged = [m['name'] for m in load_materials(only_inventory=False, priority=False)
                   if m.get('inInventory') is True]

        self.assertEqual(len(resolve_inventory(None)), len(flagged))

    def test_resolve_inventory_keeps_an_empty_list_empty(self):
        """An explicitly empty inventory must not fall back to the database.

        The distinction is load-bearing: the callers turn an empty inventory into
        a "no materials available" error, so resolving [] to the full database
        would silently solve against materials the caller does not have.
        """
        self.assertEqual(resolve_inventory([]), [])

    def test_filter_materials_by_inventory(self):
        materials = load_materials(only_inventory=False, priority=False)

        filtered = filter_materials_by_inventory(materials, ["Каолин КЖФ-1", "нет такого материала"])

        self.assertEqual([m['name'] for m in filtered], ["Каолин КЖФ-1"])


class TestMaterialPriorities(unittest.TestCase):

    def test_base_materials_outrank_unlisted_ones(self):
        """Materials missing from priorities.json must not outrank the base ones"""
        materials = {m['name']: m['priority']
                     for m in load_materials(only_inventory=True, priority=True)}

        self.assertEqual(materials["Нефелин-сиенит VR13"], 2)
        self.assertGreater(materials["Бентонит"], materials["Нефелин-сиенит VR13"])
        self.assertGreater(materials["Карбонат цинка, ZnCO3"], materials["Нефелин-сиенит VR13"])


if __name__ == "__main__":
    unittest.main()
