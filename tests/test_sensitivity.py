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
from common import load_materials, load_molar_masses, weights_to_umf
from solver_classic import calculate_recipe_composition
from sensitivity import (FALLBACK_RELATIVE, FLAT_SIGMA_WARNING, MAX_PERCENTAGE, MAX_SIGMA,
                         NONFINITE_CONTRIBUTION_WARNING, UNREADABLE_TOLERANCES_ISSUE,
                         ZERO_CONTRIBUTION_WARNING, ZERO_FLUX_MOLES, _all_finite,
                         _first_nonfinite, _material_shares, _top_affected,
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


def flat_warning(result):
    """
    The flat ranking warning of an answer, None when it carries none

    FLAT_SIGMA_WARNING is the stem of that message and not the whole of it: the
    fact it states is the same every time, and what follows is what was observed
    here - the one number every applied sigma came out as, and a cause only where
    this run of the calculation could check one. The stem is all this helper
    matches; the text that follows it is pinned by
    test_the_message_claims_only_what_the_set_of_sigmas_holds, because a prefix
    match let two refuted wordings through in a row.
    """
    return next((warning for warning in result["warnings"]
                 if warning.startswith(FLAT_SIGMA_WARNING)), None)


def numbers_in(payload):
    """Every number written anywhere in a tolerance file, whatever section it sits in"""
    if isinstance(payload, dict):
        return {number for value in payload.values() for number in numbers_in(value)}

    if isinstance(payload, list):
        return {number for value in payload for number in numbers_in(value)}

    if isinstance(payload, bool) or not isinstance(payload, (int, float)):
        return set()

    return {float(payload)}


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

    def test_a_cell_that_cannot_be_perturbed_gets_no_sigma_of_its_own(self):
        """
        A direct call, because it is the only way in

        Through recipe_sensitivity() a non-numeric cell never reaches this code:
        calculate_recipe_composition() multiplies it by the share several steps
        earlier and raises there. A line trace of the whole suite (240 tests)
        found the `except (TypeError, ValueError)` of material_sigma() and the
        skip of a zero cell never executed once, while API.md described both as
        the module's own guard - so the guard was documented and unrun.

        The material is built here rather than looked up: no record of
        database/materials.json carries a cell like these (checked over all 216).
        """
        material = {"name": "материал не из базы",
                    "formula": {"SiO2": 50, "Al2O3": "много", "CaO": None, "K2O": 0}}

        self.assertEqual(material_sigma(material, self.tolerances),
                         {"SiO2": self.tolerances["default_relative"]})


class TestToleranceLoading(unittest.TestCase):
    """How the file is read: no cache, and an unusable file is reported"""

    def write_tolerances(self, payload):
        handle = tempfile.NamedTemporaryFile('w', suffix='.json', encoding='utf-8', delete=False)
        self.addCleanup(os.unlink, handle.name)
        with handle:
            handle.write(payload if isinstance(payload, str) else json.dumps(payload))
        return handle.name

    def test_the_shipped_file_loses_nothing_on_the_way_in(self):
        self.assertEqual(load_tolerances()["issues"], [])

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
        self.assertEqual(fresh["issues"], [])

    def test_a_caller_mutating_the_result_cannot_corrupt_the_next_read(self):
        tolerances = load_tolerances()
        tolerances["classes"]["clay"] = 999
        tolerances["default_relative"] = 999

        fresh = load_tolerances()
        self.assertNotEqual(fresh["classes"].get("clay"), 999)
        self.assertNotEqual(fresh["default_relative"], 999)

    def test_a_missing_file_falls_back_to_one_flat_sigma_and_says_so(self):
        with self.assertLogs('sensitivity', level='WARNING'):
            tolerances = load_tolerances(MISSING_TOLERANCES)

        self.assertEqual(tolerances["issues"], [UNREADABLE_TOLERANCES_ISSUE])
        self.assertEqual(tolerances["classes"], {})
        self.assertEqual(tolerances["materials"], {})
        self.assertEqual(tolerances["default_relative"], FALLBACK_RELATIVE)

    def test_a_file_that_is_not_json_is_reported_the_same_way(self):
        with self.assertLogs('sensitivity', level='WARNING'):
            issues = load_tolerances(self.write_tolerances("{not json at all"))["issues"]
        self.assertEqual(issues, [UNREADABLE_TOLERANCES_ISSUE])

    def test_a_json_file_that_is_not_an_object_is_reported_the_same_way(self):
        with self.assertLogs('sensitivity', level='WARNING'):
            issues = load_tolerances(self.write_tolerances([0.05, 0.10]))["issues"]
        self.assertEqual(issues, [UNREADABLE_TOLERANCES_ISSUE])

    def test_the_loader_never_claims_to_know_what_the_answer_will_look_like(self):
        """
        The rule this module now follows, as a test: load_tolerances() reports
        what the FILE lost and nothing about the ranking. Three rounds of fixes
        put a "degraded" verdict here instead, each one a cleverer count of the
        file's contents, and each one was walked past by a file that counted as
        healthy and never reached the recipe. Whether the answer came out flat is
        decided in recipe_sensitivity(), by looking at the answer.
        """
        with self.assertLogs('sensitivity', level='WARNING'):
            self.assertNotIn("degraded", load_tolerances(MISSING_TOLERANCES))

        self.assertNotIn("degraded", load_tolerances())


class TestToleranceFilesThatDoNotWork(unittest.TestCase):
    """
    A file that parses is not a file that works

    The file is documented as one to edit by hand, so the way it breaks in
    practice is a typo in a key or a section deleted along with what it served -
    not an unreadable disk. Every payload below is valid JSON and an object, and
    every one of them used to come back with no warning at all while the ranking
    it produced was the flat one of a missing file.
    """

    def setUp(self):
        self.materials = all_materials()
        self.shipped = shipped_tolerances_file()

    def write_tolerances(self, payload):
        handle = tempfile.NamedTemporaryFile('w', suffix='.json', encoding='utf-8', delete=False)
        self.addCleanup(os.unlink, handle.name)
        with handle:
            json.dump(payload, handle)
        return handle.name

    def flat_shares(self, recipe, sigma):
        """
        The answer of a file that says nothing but "every material is this sigma"

        Taken at a given sigma and not at a fixed 0.05, because the shares are
        NOT invariant to scaling every sigma by the same factor: the UMF
        renormalizes onto its unity basis, so the response is not linear in
        sigma. Wollastonite moves 0.327 -> 0.204 between a flat 0.05 and a flat
        0.9, which is what made the fixed comparison a false statement rather
        than a strict one.
        """
        return shares(recipe_sensitivity(
            recipe, self.materials,
            {"default_relative": sigma, "classes": {}, "materials": {}}))

    def matching_flat_sigma(self, recipe, result, payload):
        """
        The sigma of the flat file this answer is the answer of, None when none is

        The candidates are every number written anywhere in the file under test
        plus the fallback it can drop to, so the search never has to know HOW a
        sigma resolves - only that a resolved one came from somewhere in the
        file. That independence is the point: the previous form of this test
        compared against one fixed answer, the one at "default_relative", and a
        file whose materials all landed on 0.02 while its default said 0.05 was
        called a working file by it.
        """
        for sigma in sorted(numbers_in(payload) | {FALLBACK_RELATIVE}):
            if not 0 < sigma <= MAX_SIGMA:
                continue
            if shares(result) == self.flat_shares(recipe, sigma):
                return sigma

        return None

    def unusable_files(self):
        """(name, payload) - the table of the review, as a fixture"""
        return [
            ("an empty object", {}),
            ("materials written as a list",
             {"default_relative": 0.05, "classes": self.shipped["classes"], "materials": []}),
            ("a junk default_relative and nothing else", {"default_relative": "abc"}),
            ("a default_relative and nothing else", {"default_relative": 0.02}),
            ("a typo in both section keys",
             {"default_relative": 0.05, "class": self.shipped["classes"],
              "material": self.shipped["materials"]}),
            ("classes defined but assigned to nobody",
             {"default_relative": 0.05, "classes": self.shipped["classes"], "materials": {}}),
            ("classes written as a list, materials intact",
             {"default_relative": 0.05, "classes": [], "materials": self.shipped["materials"]}),
            ("a junk default_relative with both sections intact",
             {"default_relative": "abc", "classes": self.shipped["classes"],
              "materials": self.shipped["materials"]}),
        ]

    def answers_that_may_come_out_flat(self):
        """
        (name, recipe, payload) - every input the invariant below is checked on

        The four at the end are answered by the SHIPPED file, in perfect order,
        and they are the half no previous form of the test could see: their
        materials resolve to a class each and the classes land on one and the
        same number. That is not an exotic file, that is what grouping materials
        into classes is FOR - feldspar and silica are both 0.02, both ashes are
        0.20, both carbonates are 0.01. Over the 2-4 material combinations of the
        19 inventory materials the shipped file answers 103 of them flat, and
        equality to "default_relative" saw 11 of those.
        """
        by_the_file = [(name, TRANSPARENT_RECIPE, payload)
                       for name, payload in self.unusable_files()]

        return by_the_file + [
            ("the shipped file on a feldspar recipe, every sigma 0.02",
             {"Нефелин-сиенит VR13": 40, "Полевой шпат FFF": 30,
              "Кварцевая мука Кварцверке W12": 30}, self.shipped),
            ("the shipped file on two ashes, both 0.20",
             {"Древесная зола": 60, "Костная зола": 40}, self.shipped),
            ("the shipped file on two carbonates, both 0.01",
             {"Мел, CaCO3": 50, "Доломит МИДОЛ": 50}, self.shipped),
            ("the shipped file on a recipe of a single material",
             {"Нефелин-сиенит VR13": 100}, self.shipped),
        ]

    def test_none_of_them_is_answered_in_silence(self):
        """
        The invariant, and not a list of specific checks: a warning is owed
        whenever the file did not fully work, whatever the shape of the damage
        """
        for name, payload in self.unusable_files():
            with self.subTest(name):
                with self.assertLogs('sensitivity', level='WARNING'):
                    tolerances = load_tolerances(self.write_tolerances(payload))
                    result = recipe_sensitivity(TRANSPARENT_RECIPE, self.materials, tolerances)

                self.assertIsNone(result["error"])
                self.assertTrue(result["warnings"],
                                f"{name}: answered with no warning at all")

    def test_an_answer_equal_to_a_flat_one_always_carries_the_flat_warning(self):
        """
        Where the line is drawn: not "the file was readable", not "some material
        got a sigma of its own", but "these numbers are the numbers of a file
        that distinguishes nothing". Those two answers are indistinguishable to a
        caller, so they must not differ in what they say - in both directions.

        The comparison is against the flat answer at the sigma that was actually
        applied and not at the "default_relative" of the file. The two are not
        the same question: a file whose materials all resolve to 0.02 while its
        default says 0.05 ranks nothing at all, and its answer is bit for bit the
        answer of {"default_relative": 0.02} - umf, per_oxide and by_material
        alike. The previous form of the test compared against the default and
        called that file a working one.
        """
        for name, recipe, payload in self.answers_that_may_come_out_flat():
            with self.subTest(name):
                path = self.write_tolerances(payload)
                with self.assertLogs('sensitivity', level='WARNING'):
                    tolerances = load_tolerances(path)
                    result = recipe_sensitivity(recipe, self.materials, tolerances)

                # Outside the assertLogs, like write_tolerances above it: this
                # helper calls recipe_sensitivity itself, once per candidate
                # sigma, on a file that is flat by construction - so it always
                # logs sensitivity_flat_sigmas and there is always at least one
                # candidate. Inside the block it satisfied the assertion on its
                # own and the call under test could have gone silent unnoticed.
                flat_sigma = self.matching_flat_sigma(recipe, result, payload)

                if flat_sigma is None:
                    self.assertIsNone(flat_warning(result), f"{name}: a working file called flat")
                else:
                    self.assertIsNotNone(
                        flat_warning(result),
                        f"{name}: the ranking of a flat {flat_sigma} without the flat warning")

    def test_a_default_relative_of_its_own_is_still_a_flat_answer(self):
        """
        The entry that broke the previous version of the test above

        {"default_relative": 0.02} ranks nothing: every material gets the same
        sigma. Its shares are nevertheless NOT the shares of a missing file
        (wollastonite 0.3328 against 0.3266), so a comparison against one fixed
        flat answer called this file a working one.
        """
        with self.assertLogs('sensitivity', level='WARNING'):
            tolerances = load_tolerances(self.write_tolerances({"default_relative": 0.02}))
            result = recipe_sensitivity(TRANSPARENT_RECIPE, self.materials, tolerances)

        # Outside the block for the same reason as in the test above: flat_shares
        # calls recipe_sensitivity on a file that is flat by construction, so it
        # logs sensitivity_flat_sigmas on its own and satisfied the assertion by
        # itself. Silencing ONLY the call under test - the two are told apart by
        # the "issues" key load_tolerances always adds - left this test passing
        # while its neighbour above failed 8 of its 12 subtests.
        at_the_fallback = self.flat_shares(TRANSPARENT_RECIPE, FALLBACK_RELATIVE)
        at_its_own = self.flat_shares(TRANSPARENT_RECIPE, 0.02)

        self.assertIsNotNone(flat_warning(result))
        self.assertAlmostEqual(share_of(result, "Волластонит МИВОЛЛ"), 0.332833969, places=6)
        self.assertNotEqual(shares(result), at_the_fallback)
        self.assertEqual(shares(result), at_its_own)

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
            flat = self.flat_shares(TRANSPARENT_RECIPE, 0.05)

        result = recipe_sensitivity(TRANSPARENT_RECIPE, self.materials, tolerances)

        self.assertIsNone(flat_warning(result))
        self.assertEqual(result["by_material"][0]["material"], "Улексит (Химпэк)")
        self.assertNotEqual(shares(result), flat)
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

        self.assertIsNone(flat_warning(result))
        self.assertEqual(result["by_material"][0]["material"], "Улексит (Химпэк)")
        self.assertTrue(any('material_tolerance_entry_ignored' in line for line in logs.output))
        self.assertTrue(result["warnings"], "the dropped entry was not reported")

    def test_the_shipped_file_reports_nothing_at_all(self):
        """The complement: the fixture above must not fire on a healthy file"""
        self.assertEqual(load_tolerances()["issues"], [])
        self.assertEqual(recipe_sensitivity(TRANSPARENT_RECIPE, self.materials)["warnings"], [])


class TestAHealthyFileThatNeverReachesTheRecipe(unittest.TestCase):
    """
    The hole three rounds of counting the file's contents could not close

    Nothing is wrong with these files. Every sigma in them is a good number, in a
    section of the right type, under a key spelled correctly - a count of what
    the file contains sees a working tolerance database and says so. They simply
    never meet this recipe: material_tolerance.json is documented as a file of
    its own, edited by hand, with names that must match database/materials.json
    exactly, and a supplier renaming a material is enough to part the two.

    The answer is then the answer of a missing file, bit for bit, and that is
    what has to be reported - which is only visible from the answer.
    """

    def setUp(self):
        self.materials = all_materials()
        self.shipped = shipped_tolerances_file()

        with self.assertLogs('sensitivity', level='WARNING'):
            self.missing_file = shares(recipe_sensitivity(
                TRANSPARENT_RECIPE, self.materials, load_tolerances(MISSING_TOLERANCES)))

    def write_tolerances(self, payload):
        handle = tempfile.NamedTemporaryFile('w', suffix='.json', encoding='utf-8', delete=False)
        self.addCleanup(os.unlink, handle.name)
        with handle:
            json.dump(payload, handle)
        return handle.name

    def test_names_out_of_sync_with_materials_json_are_not_answered_in_silence(self):
        """Every entry renamed: a healthy file about materials nobody asked about"""
        renamed = {f"{name} (партия 2024)": entry
                   for name, entry in self.shipped["materials"].items()}

        tolerances = load_tolerances(self.write_tolerances(
            {"default_relative": 0.05, "classes": self.shipped["classes"],
             "materials": renamed}))

        self.assertEqual(tolerances["issues"], [], "the file itself is in perfect order")

        with self.assertLogs('sensitivity', level='WARNING') as logs:
            result = recipe_sensitivity(TRANSPARENT_RECIPE, self.materials, tolerances)

        warning = flat_warning(result)
        self.assertIsNotNone(warning)
        self.assertEqual(shares(result), self.missing_file,
                         "the same numbers a missing file gives, so the same warning")
        self.assertEqual(result["by_material"][0]["material"], "Волластонит МИВОЛЛ")
        self.assertTrue(any('sensitivity_flat_sigmas' in line for line in logs.output))

        # And here the cause IS observed - the file names 19 materials and not
        # one of them is in this recipe - so here it is named
        self.assertIn("Ни одно имя", warning,
                      f"the names did not match and nothing said so: {warning}")

    def test_an_override_on_an_oxide_the_material_does_not_carry_is_not_a_sigma(self):
        """
        A per oxide override is only usable if the material has that oxide:
        ulexite carries no ZrO2, so this sigma resolves for nothing at all
        """
        tolerances = load_tolerances(self.write_tolerances(
            {"default_relative": 0.05, "classes": {},
             "materials": {"Улексит (Химпэк)": {"oxides": {"ZrO2": 0.5}}}}))

        self.assertEqual(tolerances["issues"], [])

        with self.assertLogs('sensitivity', level='WARNING'):
            result = recipe_sensitivity(TRANSPARENT_RECIPE, self.materials, tolerances)

        self.assertIsNotNone(flat_warning(result))
        self.assertEqual(shares(result), self.missing_file)

    def test_a_recipe_of_materials_the_file_does_not_describe_is_flat_too(self):
        """
        The shipped file, in perfect order, and a recipe of materials none of
        which it mentions: the ranking is by lever alone all the same
        """
        recipe = {"Карбонат бария, BaCO3": 40, "Литий углекислый, Li2CO3": 20,
                  "Кварцевая мука Кварцверке W12": 40}
        unlisted = {name: entry for name, entry in self.shipped["materials"].items()
                    if name != "Кварцевая мука Кварцверке W12"}

        tolerances = load_tolerances(self.write_tolerances(
            {"default_relative": 0.05, "classes": self.shipped["classes"],
             "materials": unlisted}))

        with self.assertLogs('sensitivity', level='WARNING'):
            result = recipe_sensitivity(recipe, self.materials, tolerances)

        self.assertIsNone(result["error"])
        self.assertIsNotNone(flat_warning(result))

    def test_one_material_on_a_class_number_of_its_own_answers_without_it(self):
        """
        The complement: the warning is about the recipe, not about the file

        Named after what is set up here and not after a rule. "A sigma of its
        own is enough" would be the rule, and it is false three ways, each of
        them measured: a class number equal to default_relative is not a sigma
        of its own (the test below), an override equal to the class number is
        not either, and one written onto an oxide that never reaches the UMF -
        Loi, or one the material does not carry - changes nothing at all.
        """
        tolerances = load_tolerances(self.write_tolerances(
            {"default_relative": 0.05, "classes": self.shipped["classes"],
             "materials": {"Улексит (Химпэк)": {"class": "hydrate_borate"}}}))

        result = recipe_sensitivity(TRANSPARENT_RECIPE, self.materials, tolerances)

        self.assertEqual(result["warnings"], [])
        self.assertEqual(result["by_material"][0]["material"], "Улексит (Химпэк)")

    def test_a_sigma_equal_to_the_default_does_not_count_as_one_of_its_own(self):
        """
        clay is 0.05 and so is default_relative: a file assigning that class to
        everything produces the flat answer and has to be reported as one, even
        though every material of the recipe did resolve to a class
        """
        tolerances = load_tolerances(self.write_tolerances(
            {"default_relative": 0.05, "classes": {"clay": 0.05},
             "materials": {name: {"class": "clay"} for name in TRANSPARENT_RECIPE}}))

        with self.assertLogs('sensitivity', level='WARNING'):
            result = recipe_sensitivity(TRANSPARENT_RECIPE, self.materials, tolerances)

        self.assertIsNotNone(flat_warning(result))
        self.assertEqual(shares(result), self.missing_file)

    def test_classes_that_land_on_one_number_do_not_distinguish_anything_either(self):
        """
        And the default has nothing to do with it

        Every material here resolves to a class of its own, none of them to
        default_relative, the file is the shipped one and it is in perfect order.
        feldspar and silica are simply both 0.02, so the ranking is by lever
        alone and the answer is bit for bit the answer of a file that says
        nothing but {"default_relative": 0.02}.
        """
        recipe = {"Нефелин-сиенит VR13": 40, "Полевой шпат FFF": 30,
                  "Кварцевая мука Кварцверке W12": 30}

        with self.assertLogs('sensitivity', level='WARNING'):
            result = recipe_sensitivity(recipe, self.materials)
            flat = recipe_sensitivity(recipe, self.materials,
                                      {"default_relative": 0.02, "classes": {}, "materials": {}})

        self.assertIsNotNone(flat_warning(result))
        self.assertEqual(result["by_material"], flat["by_material"])
        self.assertEqual(result["per_oxide"], flat["per_oxide"])
        self.assertEqual(result["umf"], flat["umf"])
        self.assertEqual([round(row["share"], 9) for row in result["by_material"]],
                         [0.606560622, 0.31383887, 0.079600508])

    def test_the_warning_names_the_sigma_that_was_applied(self):
        """
        The checkable half of the message

        "все получили 0.2" is a sentence its reader can hold against
        material_tolerance.json and see both ashes sitting in one class. "все
        получили одинаковую" leaves them to work out which one, and the answer
        carries no other place to look it up: sigma_used is per material and
        only names the leading oxide of each.
        """
        cases = [("0.02", {"Нефелин-сиенит VR13": 40, "Полевой шпат FFF": 30,
                           "Кварцевая мука Кварцверке W12": 30}),
                 ("0.2", {"Древесная зола": 60, "Костная зола": 40}),
                 ("0.01", {"Мел, CaCO3": 50, "Доломит МИДОЛ": 50}),
                 ("0.02", {"Нефелин-сиенит VR13": 100})]

        for sigma, recipe in cases:
            with self.subTest(sigma):
                with self.assertLogs('sensitivity', level='WARNING'):
                    result = recipe_sensitivity(recipe, self.materials)

                warning = flat_warning(result)
                self.assertIsNotNone(warning)
                self.assertIn(sigma, warning, f"the sigma is not in the message: {warning}")

    def test_the_flat_warning_does_not_diagnose_a_file_that_is_in_order(self):
        """
        The half of the message that was a guess, and a wrong one

        A recipe of clays gets the flat answer out of the SHIPPED file: clay is
        0.05 for all four of them. The file is present, complete, and every name
        in it matches database/materials.json exactly - a separate test asserts
        that. The message used to finish with "the tolerance database is
        unavailable, does not describe these materials or has drifted from the
        names in database/materials.json", all three false at once, and sent
        whoever read it to fix a file with nothing wrong in it.
        """
        recipe = {"Каолин КЖФ-1": 40, "Бентонит": 20, "Тальк Онотский": 20,
                  "Волластонит МИВОЛЛ": 20}

        with self.assertLogs('sensitivity', level='WARNING'):
            result = recipe_sensitivity(recipe, self.materials)

        warning = flat_warning(result)
        self.assertIsNotNone(warning)
        self.assertEqual(result["warnings"], [warning],
                         "the shipped file lost nothing, so nothing else is owed")
        self.assertIn("0.05", warning)

        for claim in ("недоступн", "не описывает", "разошл", "не прочитан"):
            self.assertNotIn(claim, warning,
                             f"a healthy file diagnosed as broken: {warning}")


class TestDroppedSigmasAreReported(unittest.TestCase):
    """
    Every value the resolution refuses is named, and not quietly replaced

    The failure mode of all of these is the same and is the quietest one there
    is: the material falls back a level, its number in the answer shrinks, and
    nothing anywhere says that a line of the file was thrown away.
    """

    def setUp(self):
        self.materials = all_materials()
        self.shipped = shipped_tolerances_file()
        self.baseline = recipe_sensitivity(TRANSPARENT_RECIPE, self.materials)

    def write_tolerances(self, payload):
        handle = tempfile.NamedTemporaryFile('w', suffix='.json', encoding='utf-8', delete=False)
        self.addCleanup(os.unlink, handle.name)
        with handle:
            json.dump(payload, handle)
        return handle.name

    def with_ulexite_entry(self, entry):
        materials_section = dict(self.shipped["materials"])
        materials_section["Улексит (Химпэк)"] = entry
        return self.write_tolerances({"default_relative": 0.05,
                                      "classes": self.shipped["classes"],
                                      "materials": materials_section})

    def test_true_is_not_a_sigma_of_one_hundred_percent(self):
        """
        float(True) is 1.0, so "clay": true used to be accepted as the widest
        spread in the file and handed the clays the top of the ranking
        """
        with self.assertLogs('sensitivity', level='WARNING'):
            tolerances = load_tolerances(self.write_tolerances(
                {"default_relative": 0.05,
                 "classes": dict(self.shipped["classes"], clay=True),
                 "materials": self.shipped["materials"]}))

        result = recipe_sensitivity(TRANSPARENT_RECIPE, self.materials, tolerances)

        sigmas = material_sigma(
            next(m for m in self.materials if m["name"] == "Каолин КЖФ-1"), tolerances)
        self.assertTrue(sigmas)
        self.assertEqual(set(sigmas.values()), {tolerances["default_relative"]},
                         f"true was taken for a sigma of 100%: {sigmas}")
        self.assertEqual(result["by_material"][0]["material"], "Улексит (Химпэк)")
        self.assertTrue(any("класс" in warning for warning in result["warnings"]),
                        f"the dropped class was not reported: {result['warnings']}")

    def test_an_oxides_section_of_the_wrong_type_is_reported(self):
        """"oxides": [0.10] moved ulexite from 0.700 to 0.618 under warnings: []"""
        with self.assertLogs('sensitivity', level='WARNING') as logs:
            tolerances = load_tolerances(self.with_ulexite_entry(
                {"class": "hydrate_borate", "oxides": [0.10]}))

        result = recipe_sensitivity(TRANSPARENT_RECIPE, self.materials, tolerances)

        self.assertTrue(any('material_tolerance_oxides_ignored' in line for line in logs.output),
                        f"the substituted section was not logged: {logs.output}")
        self.assertTrue(any("oxides" in warning for warning in result["warnings"]),
                        f"the substituted section was not reported: {result['warnings']}")
        self.assertAlmostEqual(share_of(result, "Улексит (Химпэк)"), 0.618371896, places=6)
        self.assertNotEqual(share_of(result, "Улексит (Химпэк)"),
                            share_of(self.baseline, "Улексит (Химпэк)"))

    def test_an_unusable_value_inside_oxides_is_reported(self):
        """
        A junk sigma of a CLASS was reported through the materials pointing at
        it; a junk sigma of an OXIDE was the one kind of damage that had no
        channel at all - it was simply not counted and disappeared
        """
        for value in ("десять", 0, -0.1, None, True):
            with self.subTest(repr(value)):
                with self.assertLogs('sensitivity', level='WARNING') as logs:
                    tolerances = load_tolerances(self.with_ulexite_entry(
                        {"class": "hydrate_borate", "oxides": {"B2O3": value}}))

                result = recipe_sensitivity(TRANSPARENT_RECIPE, self.materials, tolerances)

                self.assertTrue(
                    any('material_tolerance_override_ignored' in line and 'B2O3' in line
                        for line in logs.output),
                    f"the dropped override was not logged: {logs.output}")
                self.assertTrue(any("оксид" in warning for warning in result["warnings"]),
                                f"the dropped override was not reported: {result['warnings']}")
                self.assertAlmostEqual(share_of(result, "Улексит (Химпэк)"), 0.618371896,
                                       places=6)

    def test_a_sigma_above_the_sane_bound_is_dropped_and_the_answer_survives(self):
        """
        A relative sigma of 5.0 says the passport can be wrong by 500%. It used
        to be taken at face value and turn the ranking over - kaolin first with
        0.589 - which is a worse outcome than the mistyped line being ignored.
        """
        with self.assertLogs('sensitivity', level='WARNING') as logs:
            tolerances = load_tolerances(self.write_tolerances(
                {"default_relative": 0.05,
                 "classes": dict(self.shipped["classes"], clay=5.0),
                 "materials": self.shipped["materials"]}))

        result = recipe_sensitivity(TRANSPARENT_RECIPE, self.materials, tolerances)

        self.assertIsNone(result["error"])
        self.assertEqual(result["by_material"][0]["material"], "Улексит (Химпэк)")
        self.assertTrue(any('material_tolerance_sigma_out_of_range' in line
                            for line in logs.output), f"not logged: {logs.output}")
        self.assertTrue(any("100%" in warning for warning in result["warnings"]),
                        f"the oversized sigma was not reported: {result['warnings']}")

    def test_the_bound_admits_the_widest_sigma_that_still_means_something(self):
        tolerances = load_tolerances(self.write_tolerances(
            {"default_relative": 0.05,
             "classes": dict(self.shipped["classes"], ash=MAX_SIGMA),
             "materials": self.shipped["materials"]}))

        self.assertEqual(tolerances["issues"], [])
        self.assertEqual(tolerances["classes"]["ash"], MAX_SIGMA)

    def test_an_absurd_sigma_no_longer_takes_the_whole_answer_down(self):
        """
        1e308 used to reach the variance, square into inf and come back as
        nonfinite_result (422) - the entire ranking refused because one class of
        the file was mistyped. It is now one dropped value among the rest.
        """
        with self.assertLogs('sensitivity', level='WARNING'):
            tolerances = load_tolerances(self.write_tolerances({
                "default_relative": 0.05,
                "classes": {"silica": 1e308, "clay": 0.05},
                "materials": {"Кварцевая мука Кварцверке W12": {"class": "silica"},
                              "Каолин КЖФ-1": {"class": "clay"}},
            }))

            result = recipe_sensitivity(TRANSPARENT_RECIPE, self.materials, tolerances)

        self.assertIsNone(result["error"])
        self.assertTrue(_all_finite(result))
        self.assertTrue(any("100%" in warning for warning in result["warnings"]))
        self.assertIsNotNone(flat_warning(result),
                             "with silica dropped nothing is left but the default, and that is said")
        self.assertEqual(material_sigma(
            next(m for m in self.materials if m["name"] == "Кварцевая мука Кварцверке W12"),
            tolerances), {"SiO2": 0.05})


class TestFlatSigmaAnswer(unittest.TestCase):
    """An unavailable tolerance database changes the answer, so it must be visible in it"""

    def setUp(self):
        self.materials = all_materials()
        with self.assertLogs('sensitivity', level='WARNING'):
            self.no_file = load_tolerances(MISSING_TOLERANCES)

    def test_the_flat_answer_reaches_the_warnings_and_not_only_the_log(self):
        with self.assertLogs('sensitivity', level='WARNING'):
            result = recipe_sensitivity(TRANSPARENT_RECIPE, self.materials, self.no_file)

        self.assertIsNone(result["error"])
        self.assertIsNotNone(flat_warning(result))
        self.assertIn(UNREADABLE_TOLERANCES_ISSUE, result["warnings"],
                      "the unreadable file is a fact of its own and is said as well")

    def test_the_ranking_really_does_change_without_the_file(self):
        """
        Why the warning is not cosmetic: with flat sigmas the answer is the
        ranking by lever alone, the one the module exists to avoid. Ulexite
        stops being the leader and nothing in the response would show it.

        What that ranking does NOT do is promote quartz - it is last in both,
        and the header of sensitivity.py used to say otherwise. Pinned here so
        the sentence that replaced it stays measured rather than argued.

        Pinned with it: "the ranking by lever alone" is not one ranking. The
        response is not linear in the sigma, so which material leads the flat
        answer depends on the flat number - wollastonite at 0.05, nepheline at
        0.2. The replacement sentence said a common factor cannot reorder
        anything, and this is what refuted it before it was committed.
        """
        with_file = recipe_sensitivity(TRANSPARENT_RECIPE, self.materials)
        with self.assertLogs('sensitivity', level='WARNING'):
            flat = recipe_sensitivity(TRANSPARENT_RECIPE, self.materials, self.no_file)

        self.assertEqual(with_file["by_material"][0]["material"], "Улексит (Химпэк)")
        self.assertNotEqual(flat["by_material"][0]["material"], "Улексит (Химпэк)")
        self.assertEqual(flat["by_material"][-1]["material"], "Кварцевая мука Кварцверке W12")
        self.assertEqual(with_file["by_material"][-1]["material"], "Кварцевая мука Кварцверке W12")
        self.assertEqual([row["material"] for row in flat["by_material"]].index("Улексит (Химпэк)"), 2)

        leaders = {}
        for sigma in (0.05, 0.2):
            with self.assertLogs('sensitivity', level='WARNING'):
                answer = recipe_sensitivity(
                    TRANSPARENT_RECIPE, self.materials,
                    {"default_relative": sigma, "classes": {}, "materials": {}})
            leaders[sigma] = answer["by_material"][0]["material"]

        self.assertEqual(leaders, {0.05: "Волластонит МИВОЛЛ", 0.2: "Нефелин-сиенит VR13"})

    def test_a_readable_database_adds_no_such_warning(self):
        self.assertEqual(recipe_sensitivity(TRANSPARENT_RECIPE, self.materials)["warnings"], [])

    def test_the_message_claims_only_what_the_set_of_sigmas_holds(self):
        """
        The text itself, pinned, on the two answers that refuted its predecessors

        Nothing checked this string before - flat_warning() matches the stem and
        stops - so both earlier wordings travelled in shipped answers next to a
        row that contradicted them. Both spoke about MATERIALS; the set behind
        the message holds sigmas.

        Chalk is the degenerate case of API.md. Its 0.01 IS applied and the
        perturbation does run, and the formula still does not move by a digit: a
        UMF whose only oxide is a flux is CaO 1.0 whatever the analysis says. So
        the same answer carries ZERO_CONTRIBUTION_WARNING, which "все материалы,
        способные сдвинуть формулу, получили одну и ту же сигму" contradicted in
        the line above it.

        Gypsum is the other side: an empty formula contributes no sigma at all
        and its row says so with sigma_used: null, which is what refuted the
        wording before that one, "все материалы рецепта".

        Spelled out and not composed from FLAT_SIGMA_OBSERVED: a test that builds
        its expectation out of the constant under test agrees with whatever the
        constant says. Written that way first, it passed with the refuted wording
        restored - the same shape of hole as the message this test exists for.
        """
        with self.assertLogs('sensitivity', level='WARNING'):
            chalk = recipe_sensitivity({"Мел, CaCO3": 100}, self.materials)

        self.assertEqual(flat_warning(chalk),
                         "ранжирование идёт только по плечу: разброс паспортов не различает "
                         "материалы этого рецепта — все применённые сигмы оказались одним "
                         "числом 0.01")
        self.assertIn(ZERO_CONTRIBUTION_WARNING, chalk["warnings"])
        self.assertEqual(chalk["by_material"],
                         [{"material": "Мел, CaCO3", "share": 0.0, "via_oxide": "CaO",
                           "sigma_used": 0.01, "affects": []}])

        with self.assertLogs('sensitivity', level='WARNING'):
            with_gypsum = recipe_sensitivity(
                {EMPTY_FORMULA_MATERIAL: 50, "Нефелин-сиенит VR13": 50}, self.materials)

        self.assertEqual(flat_warning(with_gypsum),
                         "ранжирование идёт только по плечу: разброс паспортов не различает "
                         "материалы этого рецепта — все применённые сигмы оказались одним "
                         "числом 0.02")
        self.assertEqual(
            next(row for row in with_gypsum["by_material"]
                 if row["material"] == EMPTY_FORMULA_MATERIAL),
            {"material": EMPTY_FORMULA_MATERIAL, "share": 0.0, "via_oxide": None,
             "sigma_used": None, "affects": []})

    def test_the_other_half_of_the_message_when_no_sigma_was_applied_at_all(self):
        """
        The empty set says so instead of naming a number it does not have

        The comment at the flat check names an inf cell of a formula as the way
        in, and that is the way taken here - a hand edit of materials.json, which
        is how that file is maintained. The answer is refused with
        nonfinite_result, but the warnings travel with the 422 body, so the text
        is what a caller reads. Untested until now, and a line trace of the suite
        confirmed it: the branch never executed once.
        """
        poisoned = copy.deepcopy(
            next(m for m in self.materials if m["name"] == "Мел, CaCO3"))
        poisoned["formula"] = {"CaO": float('inf')}

        with self.assertLogs('sensitivity', level='WARNING'):
            result = recipe_sensitivity({"Мел, CaCO3": 100}, [poisoned])

        self.assertEqual(result["error"], "nonfinite_result")
        self.assertEqual(flat_warning(result),
                         "ранжирование идёт только по плечу: разброс паспортов не различает "
                         "материалы этого рецепта — ни одна сигма не вошла в расчёт")


class TestRecipeSensitivity(unittest.TestCase):

    def setUp(self):
        self.materials = all_materials()

    def test_an_oxide_with_no_molar_mass_costs_the_material_a_sigma(self):
        """
        Why API.md says "no more than the oxides of formula" and not "as many as"

        Three records of database/materials.json carry Loi, and Loi is not in
        database/molar_masses.json, so it never reaches the UMF and perturbing it
        provably changes nothing. Their applied sigmas come out one short of
        their formula: Нефелин-сиенит А-270 8 and 7, Метакаолин BMK-45 4 and 3,
        Дисульфид Молибдена 2 and 1.

        The second half is the consequence a reader of the file would want: a
        sigma written onto such an oxide is a sigma written into the void. A
        0.9 on Loi - the loosest value the file accepts - leaves the answer equal
        to the shipped one field by field, warning for warning.
        """
        molar_masses = load_molar_masses()
        self.assertNotIn("Loi", molar_masses)

        tolerances = load_tolerances()
        for name, oxides, applied in [("Нефелин-сиенит А-270", 8, 7),
                                      ("Метакаолин BMK-45", 4, 3),
                                      ("Дисульфид Молибдена", 2, 1)]:
            material = next(m for m in self.materials if m["name"] == name)
            sigmas = material_sigma(material, tolerances)
            self.assertEqual(len(material["formula"]), oxides, name)
            self.assertEqual(len([o for o in sigmas if o in molar_masses]), applied, name)

        recipe = {"Метакаолин BMK-45": 50, "Волластонит МИВОЛЛ": 50}
        into_the_void = copy.deepcopy(tolerances)
        into_the_void["materials"]["Метакаолин BMK-45"] = {"oxides": {"Loi": 0.9}}

        with self.assertLogs('sensitivity', level='WARNING'):
            shipped = recipe_sensitivity(recipe, self.materials, tolerances)
        with self.assertLogs('sensitivity', level='WARNING'):
            overridden = recipe_sensitivity(recipe, self.materials, into_the_void)

        self.assertEqual(shipped, overridden)

    def test_ulexite_outranks_quartz_on_the_reference_recipe(self):
        """
        The whole point of the metric: lever TIMES uncertainty

        Quartz has the biggest lever on the SiO2 of this recipe in raw UMF units
        (+0.011 per 1% of its analysis against +0.003 for the B2O3 of ulexite),
        but its analysis (99-100% SiO2) never lies, while the hydration of
        ulexite moves the B2O3 it brings.

        What the uncertainty buys is measured by
        test_the_ranking_really_does_change_without_the_file, and it is
        NOT "quartz would be on top" - quartz is last either way, 0.003 here
        and 0.027 flat, because the contribution is scored against the scale of
        the oxide moved and SiO2 is 3.15. It is that with one flat sigma ulexite
        falls from 0.700 to 0.275 and third place, behind wollastonite and
        nepheline.
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
        # Nepheline syenite alone comes out flat because every oxide of it takes
        # the "feldspar" 0.02 and the shipped file overrides none of them - NOT
        # because a one-material recipe is flat by definition. It is not: an
        # override applies per material-oxide pair, so ulexite alone has two
        # applied sigmas (0.08 by class, 0.10 on B2O3) and answers with no
        # warning at all. See test_a_single_material_is_not_flat_by_itself.
        with self.assertLogs('sensitivity', level='WARNING'):
            result = recipe_sensitivity({"Нефелин-сиенит VR13": 100}, self.materials)

        self.assertIsNone(result["error"])
        self.assertEqual(len(result["by_material"]), 1)
        self.assertAlmostEqual(result["by_material"][0]["share"], 1.0, delta=1e-6)
        self.assertTrue(result["per_oxide"])
        self.assertIsNotNone(flat_warning(result))

    def test_a_single_material_is_not_flat_by_itself(self):
        """
        The generalisation the documentation used to make, pinned as false

        "A recipe of one material" and "every material in one class" were listed
        as cases that always carry the flat warning. They do not: a sigma is
        applied per material-oxide pair, and the shipped file overrides B2O3 of
        ulexite and borax to 0.10 over the 0.08 of their "hydrate_borate" class.
        So one ulexite has two applied sigmas and is ranked by them, and a reader
        who trusted the old text would read a one-row by_material with share 1.0
        as a meaningful ranking on the strength of a warning that never came.
        """
        for recipe in ({"Улексит (Химпэк)": 100},
                       {"Бура, Na2O 2 B2O3 10 H2O": 100},
                       {"Улексит (Химпэк)": 50, "Бура, Na2O 2 B2O3 10 H2O": 50}):
            with self.subTest(str(recipe)):
                result = recipe_sensitivity(recipe, self.materials)

                self.assertIsNone(result["error"])
                self.assertIsNone(flat_warning(result),
                                  f"{recipe}: called flat although B2O3 is overridden")

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
        with self.assertLogs('sensitivity', level='WARNING'):
            result = recipe_sensitivity({"Мел, CaCO3": 100}, self.materials)

        self.assertIsNone(result["error"])
        self.assertEqual([row["share"] for row in result["by_material"]], [0.0])
        self.assertTrue(any("не сдвигает ни один материал" in warning
                            for warning in result["warnings"]),
                        f"expected a zero contribution warning, got {result['warnings']}")

    def test_unknown_material_is_skipped_with_a_warning(self):
        with self.assertLogs('sensitivity', level='WARNING'):
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

    def with_flux_moles(self, moles):
        """
        A one material recipe carrying exactly this many moles of flux per 100 g

        Built rather than looked up, because both no_fluxes tests above hand the
        module a recipe with no fluxes AT ALL - quartz and alumina - and a recipe
        of zero passes any threshold whatsoever. Nothing was checking where the
        line actually sits, so moving ZERO_FLUX_MOLES by ten orders of magnitude
        left the suite green.
        """
        molar_masses = load_molar_masses()
        content = moles * molar_masses["CaO"]

        # The share is 100, so the weight composition carries `content` of CaO
        # exactly and _flux_moles divides it by the same molar mass it was built
        # from. The round trip is checked and not assumed.
        self.assertEqual(content / molar_masses["CaO"], moles, "the fixture is not exact")

        return {"Проба": 100}, [{"name": "Проба", "formula": {"SiO2": 60.0, "CaO": content}}]

    def test_a_flux_sum_at_the_threshold_is_refused(self):
        """The comparison is "<=", so the threshold itself is on the refused side"""
        recipe, materials = self.with_flux_moles(ZERO_FLUX_MOLES)

        with self.assertLogs('sensitivity', level='WARNING'):
            result = recipe_sensitivity(recipe, materials)

        self.assertEqual(result["error"], "no_fluxes")

    def test_a_flux_sum_of_one_ulp_above_the_threshold_is_answered(self):
        """
        And the other side of it, as close as a float can get: the recipe is
        absurd and the answer says so - the UMF stands on traces and is inflated
        by nine orders of magnitude - but it IS an answer, on a unity basis that
        exists. That is the whole difference the threshold draws.
        """
        recipe, materials = self.with_flux_moles(math.nextafter(ZERO_FLUX_MOLES, math.inf))

        with self.assertLogs('sensitivity', level='WARNING'):
            result = recipe_sensitivity(recipe, materials)

        self.assertIsNone(result["error"])
        self.assertTrue(_all_finite(result))
        self.assertTrue(any("флюс" in warning for warning in result["warnings"]),
                        f"expected a low flux warning, got {result['warnings']}")

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

    def test_manganese_recipe_warns_instead_of_crashing(self):
        """
        MnO2 belongs to no group of oxides_classification() (DATA_NOTES.md,
        section 2), so the flux sum of this recipe collapses to traces and its
        UMF is inflated ~117x. Fixing the classification is a separate job; this
        module only has to survive it and say that the numbers are suspect.
        """
        recipe = {"Оксид марганца": 50, "Каолин КЖФ-1": 30, "Кварцевая мука Кварцверке W12": 20}

        with self.assertLogs('sensitivity', level='WARNING'):
            result = recipe_sensitivity(recipe, self.materials)

        self.assertIsNone(result["error"])
        self.assertTrue(result["by_material"])
        self.assertTrue(any("флюс" in warning for warning in result["warnings"]),
                        f"expected a low flux warning, got {result['warnings']}")


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
        with self.assertLogs('sensitivity', level='WARNING'):
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

    def test_the_refusal_names_the_row_that_is_not_finite(self):
        """
        The log used to say only "the computed answer carries inf or nan" while
        every number needed to point at the guilty row was already computed one
        line above it, leaving whoever reads it to reproduce the recipe by hand.
        """
        materials = copy.deepcopy(self.materials)
        next(m for m in materials
             if m["name"] == "Каолин КЖФ-1")["formula"]["Al2O3"] = float('inf')

        with self.assertLogs('sensitivity', level='ERROR') as logs:
            recipe_sensitivity({"Каолин КЖФ-1": 50, "Мел, CaCO3": 50}, materials)

        refusal = next(line for line in logs.output if 'sensitivity_nonfinite_result' in line)
        self.assertTrue(any(token in refusal for token in ('umf', 'per_oxide', 'by_material')),
                        f"the refusal does not say where: {refusal}")
        self.assertTrue(any(token in refusal for token in ('nan', 'inf')),
                        f"the refusal does not say what: {refusal}")

    def test_the_path_of_a_nonfinite_number_names_the_row_and_not_its_index(self):
        result = {"umf": {"SiO2": 3.0},
                  "per_oxide": [{"oxide": "SiO2", "sigma": 0.1},
                                {"oxide": "B2O3", "sigma": float('inf')}],
                  "by_material": []}

        self.assertEqual(_first_nonfinite(result), "result.per_oxide[B2O3].sigma=inf")
        self.assertIsNone(_first_nonfinite({"umf": {"SiO2": 3.0}, "error": None}))

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
