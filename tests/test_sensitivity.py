#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# flake8: noqa
# pylint: disable=broad-exception-raised, raise-missing-from, too-many-arguments, redefined-outer-name
# pylint: disable=multiple-statements, logging-fstring-interpolation, trailing-whitespace, line-too-long
# pylint: disable=broad-exception-caught, missing-function-docstring, missing-class-docstring
# pylint: disable=f-string-without-interpolation
# pylance: disable=reportMissingImports, reportMissingModuleSource

import copy
import json
import math
import tempfile
import unittest
import sys
import os

# Fix imports by adding parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import load_materials, weights_to_umf
from solver_classic import calculate_recipe_composition
from sensitivity import (DEGRADED_TOLERANCES_WARNING, FALLBACK_RELATIVE, MAX_PERCENTAGE,
                         NONFINITE_CONTRIBUTION_WARNING, ZERO_CONTRIBUTION_WARNING,
                         _all_finite, _material_shares, _top_affected,
                         load_tolerances, material_sigma, recipe_sensitivity)

# A path that cannot exist, to stand in for an unreadable tolerance database
MISSING_TOLERANCES = os.path.join(tempfile.gettempdir(), 'no_such_material_tolerance.json')

SHIPPED_TOLERANCES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'database', 'material_tolerance.json')

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


def shares(result):
    return {row["material"]: row["share"] for row in result["by_material"]}


def shipped_tolerances_file():
    with open(SHIPPED_TOLERANCES_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


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


class TestToleranceLoading(unittest.TestCase):
    """How the file is read: no cache, and an unusable file is reported"""

    def write_tolerances(self, payload):
        handle = tempfile.NamedTemporaryFile('w', suffix='.json', encoding='utf-8', delete=False)
        self.addCleanup(os.unlink, handle.name)
        with handle:
            handle.write(payload if isinstance(payload, str) else json.dumps(payload))
        return handle.name

    def test_the_shipped_file_is_not_degraded(self):
        self.assertFalse(load_tolerances()["degraded"])

    def test_an_edit_of_the_file_takes_effect_without_a_restart(self):
        """
        material_tolerance.md tells the user to edit the file by hand ("set a
        small sigma and the material drops down the ranking"). A process wide
        cache would silently postpone every such edit to the next restart.
        """
        assigned = {"Каолин КЖФ-1": {"class": "clay"}}
        path = self.write_tolerances({"default_relative": 0.05, "classes": {"clay": 0.05},
                                      "materials": assigned})
        self.assertEqual(load_tolerances(path)["classes"]["clay"], 0.05)

        with open(path, 'w', encoding='utf-8') as f:
            json.dump({"default_relative": 0.05, "classes": {"clay": 0.30},
                       "materials": assigned}, f)

        fresh = load_tolerances(path)
        self.assertEqual(fresh["classes"]["clay"], 0.30)
        self.assertFalse(fresh["degraded"])

    def test_a_caller_mutating_the_result_cannot_corrupt_the_next_read(self):
        tolerances = load_tolerances()
        tolerances["classes"]["clay"] = 999
        tolerances["default_relative"] = 999

        fresh = load_tolerances()
        self.assertNotEqual(fresh["classes"].get("clay"), 999)
        self.assertNotEqual(fresh["default_relative"], 999)

    def test_a_missing_file_degrades_to_one_flat_sigma_and_says_so(self):
        with self.assertLogs('sensitivity', level='WARNING'):
            tolerances = load_tolerances(MISSING_TOLERANCES)

        self.assertTrue(tolerances["degraded"])
        self.assertEqual(tolerances["classes"], {})
        self.assertEqual(tolerances["materials"], {})
        self.assertEqual(tolerances["default_relative"], FALLBACK_RELATIVE)

    def test_a_file_that_is_not_json_degrades_the_same_way(self):
        with self.assertLogs('sensitivity', level='WARNING'):
            self.assertTrue(load_tolerances(self.write_tolerances("{not json at all"))["degraded"])

    def test_a_json_file_that_is_not_an_object_degrades_the_same_way(self):
        with self.assertLogs('sensitivity', level='WARNING'):
            self.assertTrue(load_tolerances(self.write_tolerances([0.05, 0.10]))["degraded"])


class TestUnusableToleranceFiles(unittest.TestCase):
    """
    A file that parses is not a file that works

    The file is documented as one to edit by hand, so the way it breaks in
    practice is a typo in a key or a section deleted along with what it served -
    not an unreadable disk. Every payload below is valid JSON and an object, and
    every one of them used to come back with degraded: False and no warning at
    all, while the ranking it produced was the flat one of a missing file.
    """

    def setUp(self):
        self.materials = all_materials()
        self.shipped = shipped_tolerances_file()

        with self.assertLogs('sensitivity', level='WARNING'):
            self.flat = shares(recipe_sensitivity(TRANSPARENT_RECIPE, self.materials,
                                                  load_tolerances(MISSING_TOLERANCES)))

    def write_tolerances(self, payload):
        handle = tempfile.NamedTemporaryFile('w', suffix='.json', encoding='utf-8', delete=False)
        self.addCleanup(os.unlink, handle.name)
        with handle:
            json.dump(payload, handle)
        return handle.name

    def unusable_files(self):
        """(name, payload, degraded) - the table of the review, as a fixture"""
        return [
            ("an empty object", {}, True),
            ("materials written as a list",
             {"default_relative": 0.05, "classes": self.shipped["classes"], "materials": []},
             True),
            ("a junk default_relative and nothing else",
             {"default_relative": "abc"}, True),
            ("a typo in both section keys",
             {"default_relative": 0.05, "class": self.shipped["classes"],
              "material": self.shipped["materials"]}, True),
            ("classes defined but assigned to nobody",
             {"default_relative": 0.05, "classes": self.shipped["classes"], "materials": {}},
             True),
            ("classes written as a list, materials intact",
             {"default_relative": 0.05, "classes": [], "materials": self.shipped["materials"]},
             False),
            ("a junk default_relative with both sections intact",
             {"default_relative": "abc", "classes": self.shipped["classes"],
              "materials": self.shipped["materials"]}, False),
        ]

    def test_the_flag_follows_what_the_file_can_actually_do(self):
        for name, payload, degraded in self.unusable_files():
            with self.subTest(name):
                with self.assertLogs('sensitivity', level='WARNING'):
                    tolerances = load_tolerances(self.write_tolerances(payload))

                self.assertEqual(tolerances["degraded"], degraded)

    def test_none_of_them_is_answered_in_silence(self):
        """
        The invariant, and not a list of specific checks: a warning is owed
        whenever the file did not fully work, whatever the shape of the damage
        """
        for name, payload, _degraded in self.unusable_files():
            with self.subTest(name):
                with self.assertLogs('sensitivity', level='WARNING'):
                    tolerances = load_tolerances(self.write_tolerances(payload))

                result = recipe_sensitivity(TRANSPARENT_RECIPE, self.materials, tolerances)

                self.assertIsNone(result["error"])
                self.assertTrue(result["warnings"],
                                f"{name}: answered with no warning at all")

    def test_an_answer_equal_to_the_flat_one_always_carries_the_degradation_warning(self):
        """
        Where the line is drawn: not "the file was readable" but "the numbers
        differ from the ones a missing file gives". Those two answers are
        indistinguishable to a caller, so they must not differ in what they say.
        """
        for name, payload, _degraded in self.unusable_files():
            with self.subTest(name):
                with self.assertLogs('sensitivity', level='WARNING'):
                    tolerances = load_tolerances(self.write_tolerances(payload))

                result = recipe_sensitivity(TRANSPARENT_RECIPE, self.materials, tolerances)

                if shares(result) == self.flat:
                    self.assertIn(DEGRADED_TOLERANCES_WARNING, result["warnings"],
                                  f"{name}: the flat ranking without the flat warning")
                else:
                    self.assertNotIn(DEGRADED_TOLERANCES_WARNING, result["warnings"],
                                     f"{name}: a working file reported as degraded")

    def test_a_partly_usable_file_keeps_working_and_still_reports_the_damage(self):
        """
        classes gone, materials intact: the per oxide overrides still resolve, so
        ulexite stays the leader - with 0.56 instead of 0.700. Neither a flat
        answer nor the right one, and the least visible of the three.
        """
        with self.assertLogs('sensitivity', level='WARNING'):
            tolerances = load_tolerances(self.write_tolerances(
                {"default_relative": 0.05, "classes": [],
                 "materials": self.shipped["materials"]}))

        result = recipe_sensitivity(TRANSPARENT_RECIPE, self.materials, tolerances)

        self.assertFalse(tolerances["degraded"])
        self.assertEqual(result["by_material"][0]["material"], "Улексит (Химпэк)")
        self.assertNotEqual(shares(result), self.flat)
        self.assertNotEqual(share_of(result, "Улексит (Химпэк)"),
                            share_of(recipe_sensitivity(TRANSPARENT_RECIPE, self.materials),
                                     "Улексит (Химпэк)"))
        self.assertTrue(any("classes" in warning for warning in result["warnings"]),
                        f"nothing said about the dropped section: {result['warnings']}")

    def test_a_substituted_section_is_logged_and_not_only_warned_about(self):
        with self.assertLogs('sensitivity', level='WARNING') as logs:
            load_tolerances(self.write_tolerances(
                {"default_relative": 0.05, "classes": self.shipped["classes"],
                 "materials": []}))

        self.assertTrue(any('material_tolerance_section_ignored' in line for line in logs.output),
                        f"the type substitution was not logged: {logs.output}")

    def test_a_substituted_default_relative_is_logged_too(self):
        with self.assertLogs('sensitivity', level='WARNING') as logs:
            tolerances = load_tolerances(self.write_tolerances(
                {"default_relative": "abc", "classes": self.shipped["classes"],
                 "materials": self.shipped["materials"]}))

        self.assertEqual(tolerances["default_relative"], FALLBACK_RELATIVE)
        self.assertTrue(any('material_tolerance_bad_default' in line for line in logs.output),
                        f"the substituted default was not logged: {logs.output}")

    def test_a_broken_material_entry_does_not_take_the_rest_with_it(self):
        materials_section = dict(self.shipped["materials"])
        materials_section["Каолин КЖФ-1"] = "clay"

        with self.assertLogs('sensitivity', level='WARNING') as logs:
            tolerances = load_tolerances(self.write_tolerances(
                {"default_relative": 0.05, "classes": self.shipped["classes"],
                 "materials": materials_section}))

        result = recipe_sensitivity(TRANSPARENT_RECIPE, self.materials, tolerances)

        self.assertFalse(tolerances["degraded"])
        self.assertEqual(result["by_material"][0]["material"], "Улексит (Химпэк)")
        self.assertTrue(any('material_tolerance_entry_ignored' in line for line in logs.output))
        self.assertTrue(result["warnings"], "the dropped entry was not reported")

    def test_the_shipped_file_reports_nothing_at_all(self):
        """The complement: the fixture above must not fire on a healthy file"""
        tolerances = load_tolerances()

        self.assertFalse(tolerances["degraded"])
        self.assertEqual(tolerances["issues"], [])


class TestDegradedTolerances(unittest.TestCase):
    """An unavailable tolerance database changes the answer, so it must be visible in it"""

    def setUp(self):
        self.materials = all_materials()
        with self.assertLogs('sensitivity', level='WARNING'):
            self.degraded = load_tolerances(MISSING_TOLERANCES)

    def test_the_degradation_reaches_the_warnings_and_not_only_the_log(self):
        result = recipe_sensitivity(TRANSPARENT_RECIPE, self.materials, self.degraded)

        self.assertIsNone(result["error"])
        self.assertTrue(any("база допусков недоступна" in warning for warning in result["warnings"]),
                        f"expected a degradation warning, got {result['warnings']}")

    def test_the_ranking_really_does_change_without_the_file(self):
        """
        Why the warning is not cosmetic: with flat sigmas the answer is the
        ranking by lever alone, the one the module exists to avoid. Ulexite
        stops being the leader and nothing in the response would show it.
        """
        with_file = recipe_sensitivity(TRANSPARENT_RECIPE, self.materials)
        flat = recipe_sensitivity(TRANSPARENT_RECIPE, self.materials, self.degraded)

        self.assertEqual(with_file["by_material"][0]["material"], "Улексит (Химпэк)")
        self.assertNotEqual(flat["by_material"][0]["material"], "Улексит (Химпэк)")

    def test_a_readable_database_adds_no_such_warning(self):
        self.assertEqual(recipe_sensitivity(TRANSPARENT_RECIPE, self.materials)["warnings"], [])


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

    def assert_nonfinite_share_is_skipped(self, amount):
        recipe = dict(TRANSPARENT_RECIPE)
        recipe["Мел, CaCO3"] = amount

        result = recipe_sensitivity(recipe, self.materials)

        self.assertIsNone(result["error"])
        self.assertNotIn("Мел, CaCO3", {row["material"] for row in result["by_material"]})
        self.assertTrue(any("Мел, CaCO3" in warning for warning in result["warnings"]),
                        f"expected a skip warning, got {result['warnings']}")

        for item in result["per_oxide"]:
            self.assertTrue(math.isfinite(item["value"]), f"{item['oxide']} value is not finite")
            self.assertTrue(math.isfinite(item["sigma"]), f"{item['oxide']} sigma is not finite")
        for row in result["by_material"]:
            self.assertTrue(math.isfinite(row["share"]), f"{row['material']} share is not finite")

        # The rest of the recipe is answered exactly as if the bad row were absent
        baseline = recipe_sensitivity(TRANSPARENT_RECIPE, self.materials)
        self.assertEqual(result["umf"], baseline["umf"])
        self.assertEqual([row["material"] for row in result["by_material"]],
                         [row["material"] for row in baseline["by_material"]])

    def test_infinite_share_is_skipped_instead_of_poisoning_the_answer(self):
        """1e400 is valid JSON and Python parses it into inf"""
        self.assert_nonfinite_share_is_skipped(float('inf'))

    def test_nan_share_is_skipped_too(self):
        """The "amount <= 0" filter cannot catch this one: nan <= 0 is False"""
        self.assert_nonfinite_share_is_skipped(float('nan'))

    def test_a_umf_of_a_single_oxide_warns_that_the_shares_do_not_sum_to_one(self):
        """
        Chalk alone is exactly CaO 1.0 whatever its analysis says: the only
        oxide of the recipe IS the unity basis. Every share is then honestly
        zero, and a consumer normalizing by their sum divides by zero.
        """
        result = recipe_sensitivity({"Мел, CaCO3": 100}, self.materials)

        self.assertIsNone(result["error"])
        self.assertEqual([row["share"] for row in result["by_material"]], [0.0])
        self.assertTrue(any("не сдвигает ни один материал" in warning
                            for warning in result["warnings"]),
                        f"expected a zero contribution warning, got {result['warnings']}")

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

    def test_the_refusal_does_not_claim_a_material_is_missing_when_it_is_not(self):
        """
        Chalk IS in the database; what was rejected is its share. The old message
        ("ни один материал рецепта не найден в базе") sent the reader looking for
        a typo in a name that is spelled perfectly, and the real reason - which
        is in "warnings" - was never pointed at.
        """
        result = recipe_sensitivity({"Мел, CaCO3": float('inf')}, self.materials)

        self.assertEqual(result["error"], "no_known_materials")
        self.assertNotIn("не найден в базе", result["message"])
        self.assertIn("warnings", result["message"])
        self.assertTrue(any("Мел, CaCO3" in warning for warning in result["warnings"]),
                        f"the reason is not in the warnings either: {result['warnings']}")

    def test_recipe_standing_on_trace_fluxes_warns_instead_of_crashing(self):
        """
        A kaolin and quartz body carries no flux of its own: the only R2O/RO in
        it are the traces the kaolin analysis lists (0.6% of all the moles of
        the recipe, against the 19-21% of a real glaze). The unity basis is
        therefore defined - so the answer is computed and not refused - but it
        rests on those traces and inflates the whole formula by two orders of
        magnitude, and the module has to say that the numbers are suspect.
        """
        recipe = {"Каолин КЖФ-1": 60, "Кварцевая мука Кварцверке W12": 40}

        result = recipe_sensitivity(recipe, self.materials)

        self.assertIsNone(result["error"])
        self.assertTrue(result["by_material"])
        self.assertTrue(any("флюс" in warning for warning in result["warnings"]),
                        f"expected a low flux warning, got {result['warnings']}")
        # This is what the warning is about: a real glaze sits near SiO2 3
        self.assertGreater(result["umf"]["SiO2"], 100)

    def test_manganese_recipe_is_ranked_like_any_other(self):
        """
        MnO2 used to belong to no group of oxides_classification() at all
        (DATA_NOTES.md, section 2.2), so the flux sum of this recipe collapsed
        to the traces of the kaolin (~0.0086) and the UMF came out ~117x too
        large - the case the low flux warning above was first written for. MnO2
        is in "ro" now, it carries the unity basis of this recipe almost by
        itself, and the recipe must be ranked as the ordinary one it is: no
        complaint about the fluxes and a formula on the scale of a real glaze.
        """
        recipe = {"Оксид марганца": 50, "Каолин КЖФ-1": 30, "Кварцевая мука Кварцверке W12": 20}

        result = recipe_sensitivity(recipe, self.materials)

        self.assertIsNone(result["error"])
        self.assertEqual(len(result["by_material"]), 3)
        self.assertEqual([warning for warning in result["warnings"] if "флюс" in warning], [])
        # The published numbers of the reference recipe: MnO2 0.989, SiO2 1.123
        self.assertAlmostEqual(result["umf"]["MnO2"], 0.989, delta=0.01)
        self.assertAlmostEqual(result["umf"]["SiO2"], 1.123, delta=0.01)


class TestFiniteResult(unittest.TestCase):
    """
    Being finite on the way in does not make the answer finite on the way out

    The variance accumulates the SQUARE of the response, the shares divide by
    their own sum and the per oxide spread takes a square root, so a number that
    passed every guard on the input can still leave the finite range three steps
    later. make_json_safe then turns it into the string "Infinity" or "NaN"
    sitting in a field documented as a number, next to error: null.
    """

    def setUp(self):
        self.materials = all_materials()

    def write_tolerances(self, payload):
        handle = tempfile.NamedTemporaryFile('w', suffix='.json', encoding='utf-8', delete=False)
        self.addCleanup(os.unlink, handle.name)
        with handle:
            json.dump(payload, handle)
        return handle.name

    def assert_every_number_is_finite(self, result):
        self.assertTrue(_all_finite(result), f"a number of the answer is not finite: {result}")

    def test_a_share_beyond_the_percent_scale_is_skipped(self):
        """1e156 still squares into a number, 1e160 does not - and both are shares in percent"""
        result = recipe_sensitivity({"Кварцевая мука Кварцверке W12": 1e160, "Мел, CaCO3": 10},
                                    self.materials)

        self.assertIsNone(result["error"])
        self.assert_every_number_is_finite(result)
        self.assertNotIn("Кварцевая мука Кварцверке W12",
                         {row["material"] for row in result["by_material"]})
        self.assertTrue(any("Кварцевая мука" in warning for warning in result["warnings"]),
                        f"expected a skip warning, got {result['warnings']}")

    def test_the_bound_is_on_the_share_and_not_on_the_recipe_summing_to_100(self):
        """A recipe of 200 g of chalk per 100 g of dry mix is a normal thing to ask about"""
        result = recipe_sensitivity({"Нефелин-сиенит VR13": 300, "Мел, CaCO3": 120},
                                    self.materials)

        self.assertIsNone(result["error"])
        self.assertEqual(result["warnings"], [])
        self.assert_every_number_is_finite(result)

    def test_the_bound_admits_a_share_just_under_it(self):
        result = recipe_sensitivity({"Нефелин-сиенит VR13": MAX_PERCENTAGE, "Мел, CaCO3": 10},
                                    self.materials)

        self.assertIsNone(result["error"])
        self.assertEqual(result["warnings"], [])
        self.assert_every_number_is_finite(result)

    def test_a_wall_of_huge_shares_cannot_overflow_the_composition_itself(self):
        result = recipe_sensitivity({"Кварцевая мука Кварцверке W12": 1.5e308,
                                     "Мел, CaCO3": 1.5e308,
                                     "Нефелин-сиенит VR13": 1.5e308}, self.materials)

        self.assertEqual(result["error"], "no_known_materials")
        self.assertEqual(len(result["warnings"]), 3)

    def test_a_hand_written_sigma_that_overflows_is_refused_instead_of_returned(self):
        """
        The tolerance file is edited by hand and _positive_float() only asks the
        sigma to be a positive finite number. A sigma of 1e308 passes that and
        comes back as "sigma": Infinity for SiO2 and "share": NaN for quartz -
        under error: null, because inf/inf is nan and nan is neither <= 0 nor > 0.
        """
        tolerances = load_tolerances(self.write_tolerances({
            "default_relative": 0.05,
            "classes": {"silica": 1e308, "clay": 0.05},
            "materials": {"Кварцевая мука Кварцверке W12": {"class": "silica"},
                          "Каолин КЖФ-1": {"class": "clay"}},
        }))

        with self.assertLogs('sensitivity', level='ERROR'):
            result = recipe_sensitivity(TRANSPARENT_RECIPE, self.materials, tolerances)

        self.assertFalse(tolerances["degraded"])
        self.assertEqual(result["error"], "nonfinite_result")
        self.assertEqual(result["per_oxide"], [])
        self.assertEqual(result["by_material"], [])
        self.assertTrue(result["message"])

    def test_a_nonfinite_cell_of_a_material_formula_is_refused_and_named(self):
        """
        materials.json is written by an importer and edited by hand as well. Such
        a cell poisons the base composition before any perturbation happens, so
        the answer cannot be saved - but the log has to say which cell it was,
        instead of leaving a bare "the result is not finite".
        """
        materials = copy.deepcopy(self.materials)
        next(m for m in materials
             if m["name"] == "Каолин КЖФ-1")["formula"]["Al2O3"] = float('inf')

        with self.assertLogs('sensitivity', level='WARNING') as logs:
            result = recipe_sensitivity({"Каолин КЖФ-1": 50, "Мел, CaCO3": 50}, materials)

        self.assertEqual(result["error"], "nonfinite_result")
        self.assertTrue(any('sensitivity_nonfinite_formula_cell' in line and 'Al2O3' in line
                            for line in logs.output),
                        f"the broken cell was not named in the log: {logs.output}")

    def test_the_answer_of_the_reference_recipe_is_finite_everywhere(self):
        self.assert_every_number_is_finite(recipe_sensitivity(TRANSPARENT_RECIPE, self.materials))


class TestTotalContributionIsNotANumber(unittest.TestCase):
    """
    "not a number" must take the same branch as "zero", never the healthy one

    Every comparison against nan is False, so a "total <= 0" test calls a
    poisoned total healthy while the "total > 0" test right next to it zeroes
    every share: the caller gets exactly the degenerate answer API.md promises to
    always explain, with nothing explaining it.
    """

    def rows(self, *contributions):
        return [{"material": f"m{index}", "contribution": value, "via_oxide": "SiO2",
                 "sigma_used": 0.05, "affects": ["SiO2"]}
                for index, value in enumerate(contributions)]

    def test_a_zero_total_zeroes_the_shares_and_says_why(self):
        rows, warning = _material_shares(self.rows(0.0, 0.0))

        self.assertEqual([row["share"] for row in rows], [0.0, 0.0])
        self.assertEqual(warning, ZERO_CONTRIBUTION_WARNING)

    def test_a_nan_total_does_not_pass_for_a_healthy_one(self):
        with self.assertLogs('sensitivity', level='ERROR'):
            rows, warning = _material_shares(self.rows(1.0, float('nan')))

        self.assertEqual([row["share"] for row in rows], [0.0, 0.0])
        self.assertEqual(warning, NONFINITE_CONTRIBUTION_WARNING)

    def test_an_infinite_total_is_reported_the_same_way(self):
        with self.assertLogs('sensitivity', level='ERROR'):
            rows, warning = _material_shares(self.rows(1.0, float('inf')))

        self.assertEqual([row["share"] for row in rows], [0.0, 0.0])
        self.assertEqual(warning, NONFINITE_CONTRIBUTION_WARNING)

    def test_a_healthy_total_normalizes_and_warns_about_nothing(self):
        rows, warning = _material_shares(self.rows(3.0, 1.0))

        self.assertIsNone(warning)
        self.assertEqual([row["share"] for row in rows], [0.75, 0.25])

    def test_a_material_with_a_nan_contribution_claims_to_affect_nothing(self):
        """
        The row it used to produce contradicted itself: affects ["MgO", "CaO",
        "SiO2"] next to share 0.0 and via_oxide null, because "nan < 0.05" is
        False and the cut on the share of the leader never fired
        """
        self.assertEqual(_top_affected({"SiO2": float('nan'), "CaO": 1.0}, float('nan')), [])
        self.assertEqual(_top_affected({"SiO2": 1.0, "CaO": 1.0}, float('inf')), [])
        self.assertEqual(_top_affected({"SiO2": 1.0, "CaO": 1.0}, 0.0), [])

    def test_all_finite_looks_inside_lists_and_dictionaries(self):
        self.assertTrue(_all_finite({"a": [1, 2.0, {"b": None}], "c": "text", "d": True}))
        self.assertFalse(_all_finite({"per_oxide": [{"sigma": float('inf')}]}))
        self.assertFalse(_all_finite([{"share": float('nan')}]))


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


class TestNoSideEffects(unittest.TestCase):
    """
    The module perturbs analyses for a living and the material records it gets
    are the shared ones a caller may keep using afterwards
    """

    def test_the_materials_list_comes_back_untouched(self):
        materials = all_materials()
        before = copy.deepcopy(materials)

        recipe_sensitivity(TRANSPARENT_RECIPE, materials)

        self.assertEqual(materials, before)

    def test_the_recipe_comes_back_untouched(self):
        recipe = dict(TRANSPARENT_RECIPE)
        before = copy.deepcopy(recipe)

        recipe_sensitivity(recipe, all_materials())

        self.assertEqual(recipe, before)


class TestIncrementalComposition(unittest.TestCase):
    """
    The perturbed weight composition is built by moving one cell of the base one
    instead of recomputing the whole recipe - the hot loop of the module, run
    once per (material, oxide) pair. It is equivalent because the contribution
    of a material is linear in its formula, and nothing else checks that.
    """

    def test_matches_a_full_recalculation_on_the_reference_recipe(self):
        materials = all_materials()
        tolerances = load_tolerances()
        by_name = {material["name"]: material for material in materials}
        used = [by_name[name] for name in TRANSPARENT_RECIPE]

        base = calculate_recipe_composition(used, TRANSPARENT_RECIPE)

        checked = 0
        for name, amount in TRANSPARENT_RECIPE.items():
            material = by_name[name]
            for oxide, sigma in material_sigma(material, tolerances).items():
                content = float(material["formula"][oxide])

                # What sensitivity.py does: one cell of the composition moves
                incremental = dict(base)
                incremental[oxide] = incremental.get(oxide, 0.0) + content * sigma * (amount / 100.0)

                # What it means: that one analysis is A[i][j] * (1 + sigma)
                perturbed_materials = copy.deepcopy(used)
                next(m for m in perturbed_materials
                     if m["name"] == name)["formula"][oxide] = content * (1 + sigma)
                full = calculate_recipe_composition(perturbed_materials, TRANSPARENT_RECIPE)

                self.assertEqual(set(incremental), set(full))
                for result_oxide, value in full.items():
                    self.assertAlmostEqual(incremental[result_oxide], value, delta=1e-12,
                                           msg=f"{name} / {oxide} -> {result_oxide}")
                checked += 1

        self.assertGreater(checked, 10, "the reference recipe should cover many pairs")


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
                         weights_to_umf(self.REFERENCE_COMPOSITION, round_digits=3))

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
