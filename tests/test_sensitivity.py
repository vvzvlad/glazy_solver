#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# flake8: noqa
# pylint: disable=broad-exception-raised, raise-missing-from, too-many-arguments, redefined-outer-name
# pylint: disable=multiple-statements, logging-fstring-interpolation, trailing-whitespace, line-too-long
# pylint: disable=broad-exception-caught, missing-function-docstring, missing-class-docstring
# pylint: disable=f-string-without-interpolation
# pylance: disable=reportMissingImports, reportMissingModuleSource

import copy
import unittest
import sys
import os

# Fix imports by adding parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import load_materials, weights_to_umf
from solver_classic import calculate_recipe_composition
from sensitivity import load_tolerances, material_sigma, recipe_sensitivity

# The reference "Прозрачная глазурь △6" of database/recipes.json, the same one
# tests/test_common.py checks the UMF of
TRANSPARENT_RECIPE = {
    "Нефелин-сиенит VR13": 30,
    "Кварцевая мука Кварцверке W12": 20,
    "Волластонит МИВОЛЛ": 20,
    "Улексит (Химпэк)": 15,
    "Каолин КЖФ-1": 15,
}

# A material with an empty formula is a legal record here, not a broken one:
# 37 of the 216 entries are pigments, SiC, CMC, water or gypsum
EMPTY_FORMULA_MATERIAL = "Гипс"


def all_materials():
    return load_materials(only_inventory=False, priority=False)


def share_of(result, material_name):
    for row in result["by_material"]:
        if row["material"] == material_name:
            return row["share"]
    raise AssertionError(f"{material_name} is missing from by_material")


class TestMaterialSigma(unittest.TestCase):

    def setUp(self):
        self.tolerances = load_tolerances()
        self.materials = {m["name"]: m for m in all_materials()}

    def test_shipped_file_has_the_expected_shape(self):
        self.assertIn("default_relative", self.tolerances)
        self.assertIn("classes", self.tolerances)
        self.assertIn("materials", self.tolerances)
        self.assertGreater(self.tolerances["default_relative"], 0)

    def test_every_inventory_material_has_a_class(self):
        """The 19 materials of the default inventory are the ones actually used"""
        inventory = load_materials(only_inventory=True, priority=False)
        self.assertEqual(len(inventory), 19)

        for material in inventory:
            entry = self.tolerances["materials"].get(material["name"])
            self.assertIsNotNone(entry, f"{material['name']} has no tolerance entry")
            self.assertIn(entry.get("class"), self.tolerances["classes"],
                          f"{material['name']} has an unknown class {entry.get('class')}")

    def test_every_named_material_exists_in_the_database(self):
        """A typo in a name would silently fall back to the default tolerance"""
        known = set(self.materials)
        for name in self.tolerances["materials"]:
            self.assertIn(name, known, f"{name} is not a material of the database")

    def test_class_tolerance_is_applied(self):
        sigmas = material_sigma(self.materials["Кварцевая мука Кварцверке W12"], self.tolerances)
        self.assertEqual(sigmas, {"SiO2": self.tolerances["classes"]["silica"]})

    def test_per_oxide_override_wins_over_the_class(self):
        sigmas = material_sigma(self.materials["Улексит (Химпэк)"], self.tolerances)
        self.assertEqual(sigmas["B2O3"], 0.10)
        self.assertEqual(sigmas["CaO"], self.tolerances["classes"]["hydrate_borate"])

    def test_unlisted_material_gets_the_default(self):
        sigmas = material_sigma(self.materials["Карбонат бария, BaCO3"], self.tolerances)
        self.assertTrue(sigmas)
        for sigma in sigmas.values():
            self.assertEqual(sigma, self.tolerances["default_relative"])

    def test_empty_formula_gives_no_sigmas(self):
        self.assertEqual(material_sigma(self.materials[EMPTY_FORMULA_MATERIAL], self.tolerances), {})


class TestRecipeSensitivity(unittest.TestCase):

    def setUp(self):
        self.materials = all_materials()

    def test_ulexite_outranks_quartz_on_the_reference_recipe(self):
        """
        The whole point of the metric: lever TIMES uncertainty

        Quartz has by far the biggest lever on SiO2 of this recipe, but its
        analysis (99-100% SiO2) never lies. Ulexite has a smaller lever and a
        much wider spread. A metric ranking by lever alone would put quartz on
        top, which is exactly the answer that helps nobody.
        """
        result = recipe_sensitivity(TRANSPARENT_RECIPE, self.materials)

        self.assertIsNone(result["error"])
        self.assertGreater(share_of(result, "Улексит (Химпэк)"),
                           share_of(result, "Кварцевая мука Кварцверке W12"))
        self.assertEqual(result["by_material"][0]["material"], "Улексит (Химпэк)")
        self.assertEqual(result["by_material"][0]["via_oxide"], "B2O3")

    def test_shares_sum_to_one(self):
        result = recipe_sensitivity(TRANSPARENT_RECIPE, self.materials)

        total = sum(row["share"] for row in result["by_material"])
        self.assertAlmostEqual(total, 1.0, delta=1e-6)

    def test_shares_are_sorted_descending(self):
        result = recipe_sensitivity(TRANSPARENT_RECIPE, self.materials)

        shares = [row["share"] for row in result["by_material"]]
        self.assertEqual(shares, sorted(shares, reverse=True))

    def test_base_umf_matches_the_direct_calculation(self):
        """The reported base formula is the plain UMF of the recipe, nothing else"""
        result = recipe_sensitivity(TRANSPARENT_RECIPE, self.materials)
        expected = weights_to_umf(calculate_recipe_composition(self.materials, TRANSPARENT_RECIPE))

        self.assertEqual(result["umf"], expected)

    def test_per_oxide_covers_every_oxide_and_is_sorted_by_relative(self):
        result = recipe_sensitivity(TRANSPARENT_RECIPE, self.materials)

        self.assertEqual({item["oxide"] for item in result["per_oxide"]}, set(result["umf"]))

        relatives = [item["relative"] for item in result["per_oxide"]]
        self.assertEqual(relatives, sorted(relatives, reverse=True))

        for item in result["per_oxide"]:
            self.assertGreater(item["sigma"], 0, f"{item['oxide']} has no spread at all")

    def test_boron_is_the_least_trustworthy_oxide_of_the_reference_recipe(self):
        """Ulexite carries the boron and ulexite is the loosest analysis here"""
        result = recipe_sensitivity(TRANSPARENT_RECIPE, self.materials)

        self.assertEqual(result["per_oxide"][0]["oxide"], "B2O3")

    def test_affects_names_the_oxides_the_material_moves(self):
        result = recipe_sensitivity(TRANSPARENT_RECIPE, self.materials)

        ulexite = next(row for row in result["by_material"] if row["material"] == "Улексит (Химпэк)")
        self.assertIn("B2O3", ulexite["affects"])
        self.assertLessEqual(len(ulexite["affects"]), 3)
        self.assertEqual(ulexite["sigma_used"], 0.10)

        quartz = next(row for row in result["by_material"]
                      if row["material"] == "Кварцевая мука Кварцверке W12")
        self.assertEqual(quartz["affects"], ["SiO2"])

    def test_empty_formula_material_contributes_nothing(self):
        """A pigment or a binder cannot move the formula, and that is an answer"""
        recipe = dict(TRANSPARENT_RECIPE)
        recipe[EMPTY_FORMULA_MATERIAL] = 5

        result = recipe_sensitivity(recipe, self.materials)

        self.assertIsNone(result["error"])

        row = next(row for row in result["by_material"] if row["material"] == EMPTY_FORMULA_MATERIAL)
        self.assertEqual(row["share"], 0.0)
        self.assertIsNone(row["via_oxide"])
        self.assertIsNone(row["sigma_used"])
        self.assertEqual(row["affects"], [])

        self.assertAlmostEqual(sum(r["share"] for r in result["by_material"]), 1.0, delta=1e-6)

        # And it changes nothing for the rest of the recipe
        baseline = recipe_sensitivity(TRANSPARENT_RECIPE, self.materials)
        self.assertEqual(result["umf"], baseline["umf"])
        self.assertAlmostEqual(share_of(result, "Улексит (Химпэк)"),
                               share_of(baseline, "Улексит (Химпэк)"), places=6)

    def test_single_material_recipe(self):
        result = recipe_sensitivity({"Нефелин-сиенит VR13": 100}, self.materials)

        self.assertIsNone(result["error"])
        self.assertEqual(len(result["by_material"]), 1)
        self.assertAlmostEqual(result["by_material"][0]["share"], 1.0, delta=1e-6)
        self.assertTrue(result["per_oxide"])

    def test_unknown_material_is_skipped_with_a_warning(self):
        result = recipe_sensitivity({"Нефелин-сиенит VR13": 70, "Философский камень": 30},
                                    self.materials)

        self.assertIsNone(result["error"])
        self.assertEqual([row["material"] for row in result["by_material"]], ["Нефелин-сиенит VR13"])
        self.assertTrue(any("Философский камень" in warning for warning in result["warnings"]))

    def test_recipe_without_fluxes_is_refused_instead_of_answered(self):
        """
        weights_to_umf falls back to "the smallest oxide is unity" without
        fluxes. That basis is arbitrary, so a sensitivity computed on it would
        be a number without a meaning.
        """
        result = recipe_sensitivity({"Кварцевая мука Кварцверке W12": 60, "Глинозем, Al203": 40},
                                    self.materials)

        self.assertEqual(result["error"], "no_fluxes")
        self.assertEqual(result["by_material"], [])
        self.assertEqual(result["per_oxide"], [])
        self.assertEqual(result["umf"], {})
        self.assertTrue(result["message"])

    def test_empty_recipe_is_refused(self):
        self.assertEqual(recipe_sensitivity({}, self.materials)["error"], "empty_recipe")

    def test_no_known_material_is_refused(self):
        result = recipe_sensitivity({"Философский камень": 100}, self.materials)
        self.assertEqual(result["error"], "no_known_materials")

    def test_manganese_recipe_warns_instead_of_crashing(self):
        """
        MnO2 belongs to no group of oxides_classification() (DATA_NOTES.md,
        section 2), so the flux sum of this recipe collapses to traces and its
        UMF is inflated ~117x. Fixing the classification is a separate job; this
        module only has to survive it and say that the numbers are suspect.
        """
        recipe = {"Оксид марганца": 50, "Каолин КЖФ-1": 30, "Кварцевая мука Кварцверке W12": 20}

        result = recipe_sensitivity(recipe, self.materials)

        self.assertIsNone(result["error"])
        self.assertTrue(result["by_material"])
        self.assertTrue(any("флюс" in warning for warning in result["warnings"]),
                        f"expected a low flux warning, got {result['warnings']}")


class TestSigmaMonotonicity(unittest.TestCase):
    """A looser analysis of one material can only raise its own share"""

    def setUp(self):
        self.materials = all_materials()
        self.tolerances = load_tolerances()

    def with_sigma(self, material_name, sigma):
        tolerances = copy.deepcopy(self.tolerances)
        entry = tolerances["materials"].setdefault(material_name, {})
        entry.pop("oxides", None)
        entry["class"] = f"_test_{material_name}"
        tolerances["classes"][entry["class"]] = sigma
        return tolerances

    def test_raising_the_sigma_of_a_material_raises_its_share(self):
        shares = []
        for sigma in (0.01, 0.05, 0.15):
            result = recipe_sensitivity(TRANSPARENT_RECIPE, self.materials,
                                        self.with_sigma("Каолин КЖФ-1", sigma))
            shares.append(share_of(result, "Каолин КЖФ-1"))

        self.assertEqual(shares, sorted(shares), f"share did not grow with sigma: {shares}")
        self.assertLess(shares[0], shares[-1])

    def test_quartz_takes_the_lead_once_its_analysis_is_declared_unreliable(self):
        """
        The complement of the ulexite test: the ranking follows the tolerance
        data and is not a property of the materials hardcoded anywhere.
        """
        tolerances = self.with_sigma("Кварцевая мука Кварцверке W12", 0.5)

        result = recipe_sensitivity(TRANSPARENT_RECIPE, self.materials, tolerances)

        self.assertEqual(result["by_material"][0]["material"], "Кварцевая мука Кварцверке W12")


class TestWeightsToUmfSignature(unittest.TestCase):
    """
    Regression: weights_to_umf grew a round_digits parameter for this module

    Every existing caller passes no second argument and must keep getting the
    values rounded to 3 decimals it got before.
    """

    # Recomputed from the reference transparent recipe, same numbers as the
    # rest of the suite checks
    REFERENCE_COMPOSITION = {
        "SiO2": 46.2145, "Al2O3": 12.2415, "K2O": 2.5285, "Na2O": 2.7645,
        "CaO": 12.98, "MgO": 0.435, "B2O3": 5.55, "SrO": 0.15,
        "Fe2O3": 0.1136, "TiO2": 0.06,
    }

    def test_default_call_still_rounds_to_three_decimals(self):
        umf = weights_to_umf({"SiO2": 65.2, "Al2O3": 18.1, "Na2O": 8.4, "K2O": 8.3})

        self.assertEqual(umf, {"SiO2": 4.853, "Al2O3": 0.794, "Na2O": 0.606, "K2O": 0.394})

    def test_explicit_default_matches_the_implicit_one(self):
        self.assertEqual(weights_to_umf(self.REFERENCE_COMPOSITION),
                         weights_to_umf(self.REFERENCE_COMPOSITION, 3))

    def test_rounded_result_is_the_raw_one_rounded(self):
        raw = weights_to_umf(self.REFERENCE_COMPOSITION, round_digits=None)
        rounded = weights_to_umf(self.REFERENCE_COMPOSITION)

        self.assertEqual(rounded, {oxide: round(value, 3) for oxide, value in raw.items()})

    def test_raw_result_keeps_the_digits_rounding_would_destroy(self):
        """The reason the parameter exists: a trace oxide survives"""
        raw = weights_to_umf(self.REFERENCE_COMPOSITION, round_digits=None)

        self.assertNotEqual(raw["Fe2O3"], round(raw["Fe2O3"], 3))
        self.assertGreater(raw["Fe2O3"], 0)


if __name__ == '__main__':
    unittest.main()
