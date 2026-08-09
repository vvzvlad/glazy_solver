#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# flake8: noqa
# pylint: disable=broad-exception-raised, raise-missing-from, too-many-arguments, redefined-outer-name
# pylint: disable=multiple-statements, logging-fstring-interpolation, trailing-whitespace, line-too-long
# pylint: disable=broad-exception-caught, missing-function-docstring, missing-class-docstring
# pylint: disable=f-string-without-interpolation
# pylance: disable=reportMissingImports, reportMissingModuleSource

import os
import sys
import unittest

# Fix imports by adding parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from validate_database import (CATEGORY_BAD_VALUE, CATEGORY_DUPLICATE_NAME,
                               CATEGORY_FORMULA_SUM, CATEGORY_LOI_KEY,
                               CATEGORY_NON_OXIDE_MATERIAL, CATEGORY_UNKNOWN_OXIDE,
                               CATEGORY_UNKNOWN_PRIORITY, LEVEL_ERROR, LEVEL_NOTE,
                               LEVEL_WARNING, by_level, count_by_category,
                               format_report, validate_database)

MASSES = {"SiO2": 60.08, "Al2O3": 101.96, "CaO": 56.08, "Fe2O3": 159.69}


def material(name, formula):
    return {"name": name, "formula": formula}


def categories(issues, level):
    return {item['category'] for item in by_level(issues, level)}


def materials_in(issues, category):
    return {item['material'] for item in issues if item['category'] == category}


class TestShippedDatabase(unittest.TestCase):
    """
    The checks against the real database/materials.json

    Fails on errors only. Warnings and notes are printed, because both describe
    records that are legal and known - 37 pigments and the like with no oxide in
    them, two analyses that legitimately sum above 100 - and a red test on those
    would only teach everyone to ignore this file.
    """

    @classmethod
    def setUpClass(cls):
        cls.issues = validate_database()

    def test_no_errors(self):
        errors = by_level(self.issues, LEVEL_ERROR)
        self.assertEqual(errors, [], f"database errors: {format_report(errors)}")

    def test_warnings_are_printed_not_asserted_away(self):
        warnings = by_level(self.issues, LEVEL_WARNING)
        notes = by_level(self.issues, LEVEL_NOTE)

        print(f"\ndatabase validation: {len(warnings)} warnings, {len(notes)} notes")
        for item in warnings:
            print(f"  warning [{item['category']}] {item['material']}: {item['message']}")
        for category, count in sorted(count_by_category(notes).items()):
            print(f"  note    [{category}] x{count}")

        self.assertIsInstance(warnings, list)

    def test_the_two_known_wide_analyses_are_warnings(self):
        # Both are correct analyses: the LOI model allows a sum above 100
        wide = materials_in(self.issues, CATEGORY_FORMULA_SUM)

        self.assertIn('Криолит', wide)
        self.assertIn('Циркон микронный, ZrSiO4', wide)
        self.assertTrue(all(item['level'] == LEVEL_WARNING
                            for item in self.issues
                            if item['category'] == CATEGORY_FORMULA_SUM))

    def test_the_non_oxide_materials_are_notes(self):
        # 37 of them at the time of writing - pigments, SiC, CMC, water, gypsum.
        # The count is asserted as a floor, not an equality: adding a pigment to
        # the database must not turn this test red
        non_oxide = by_level(self.issues, LEVEL_NOTE)
        names = materials_in(self.issues, CATEGORY_NON_OXIDE_MATERIAL)

        self.assertGreaterEqual(len(names), 37)
        self.assertTrue(all(item['level'] == LEVEL_NOTE for item in non_oxide))

    def test_the_loi_key_is_a_note_and_never_an_unknown_oxide(self):
        loi = materials_in(self.issues, CATEGORY_LOI_KEY)
        unknown = materials_in(self.issues, CATEGORY_UNKNOWN_OXIDE)

        self.assertIn('Метакаолин BMK-45', loi)
        self.assertEqual(unknown, set())


class TestRules(unittest.TestCase):
    """Each rule on injected data, so that a green database proves nothing"""

    def test_zero_formula_is_a_note_and_not_an_error(self):
        issues = validate_database([material("Вода", {})], {}, MASSES)

        self.assertEqual(by_level(issues, LEVEL_ERROR), [])
        self.assertEqual(categories(issues, LEVEL_NOTE), {CATEGORY_NON_OXIDE_MATERIAL})

    def test_formula_sum_out_of_band_is_a_warning(self):
        low = validate_database([material("Огрызок", {"SiO2": 10.0})], {}, MASSES)
        high = validate_database([material("Криолит-подобный", {"SiO2": 120.0})], {}, MASSES)
        fine = validate_database([material("Кварц", {"SiO2": 100.0})], {}, MASSES)

        self.assertEqual(categories(low, LEVEL_WARNING), {CATEGORY_FORMULA_SUM})
        self.assertEqual(categories(high, LEVEL_WARNING), {CATEGORY_FORMULA_SUM})
        self.assertEqual(by_level(fine, LEVEL_WARNING), [])
        self.assertEqual(by_level(low, LEVEL_ERROR), [])

    def test_an_oxide_without_a_molar_mass_is_an_error(self):
        issues = validate_database([material("Странный", {"SiO2": 50.0, "Xy2O3": 40.0})],
                                   {}, MASSES)
        errors = by_level(issues, LEVEL_ERROR)

        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]['category'], CATEGORY_UNKNOWN_OXIDE)
        self.assertIn('Xy2O3', errors[0]['message'])

    def test_loi_is_whitelisted_and_left_out_of_the_sum(self):
        issues = validate_database([material("Метакаолин", {"SiO2": 52.0, "Al2O3": 45.0,
                                                            "Loi": 0.8})], {}, MASSES)

        self.assertEqual(by_level(issues, LEVEL_ERROR), [])
        self.assertEqual(categories(issues, LEVEL_NOTE), {CATEGORY_LOI_KEY})
        # 97.0 without the LOI, 97.8 with it - both inside the band, so the
        # exclusion is checked where it changes the verdict instead
        edge = validate_database([material("На грани", {"SiO2": 104.0, "Loi": 5.0})],
                                 {}, MASSES)
        self.assertEqual(by_level(edge, LEVEL_WARNING), [])

    def test_duplicate_names_are_an_error(self):
        issues = validate_database([material("Мел", {"CaO": 56.0}),
                                    material("Мел", {"CaO": 55.0})], {}, MASSES)
        errors = by_level(issues, LEVEL_ERROR)

        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]['category'], CATEGORY_DUPLICATE_NAME)
        self.assertEqual(errors[0]['material'], "Мел")

    def test_a_priority_for_a_missing_material_is_a_warning(self):
        issues = validate_database([material("Кварц", {"SiO2": 100.0})],
                                   {"Кварц": 1, "Ушедший поставщик": 5}, MASSES)
        warnings = by_level(issues, LEVEL_WARNING)

        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0]['category'], CATEGORY_UNKNOWN_PRIORITY)
        self.assertEqual(warnings[0]['material'], "Ушедший поставщик")
        self.assertEqual(by_level(issues, LEVEL_ERROR), [])

    def test_a_malformed_record_is_an_error_and_not_a_crash(self):
        issues = validate_database([material("Строка вместо числа", {"SiO2": "много"}),
                                    {"formula": {"SiO2": 100.0}},
                                    "не объект вовсе"], {}, MASSES)

        self.assertEqual(len(by_level(issues, LEVEL_ERROR)), 3)

    def test_every_value_the_code_cannot_use_is_an_error(self):
        """
        The four the validator used to pass while the code choked on them

        Each one was checked downstream before being listed here: Infinity makes
        check_feasibility refuse the whole inventory, NaN makes the material
        vanish silently because NaN > 0 is False, a negative content has the
        feasibility "why" announce that no material contains the oxide, and a
        quoted number makes filter_materials_with_formula raise TypeError.
        """
        cases = {
            'infinity': float('inf'),
            'nan': float('nan'),
            'negative': -5.0,
            'string': "68.0",
            'boolean': True,
            'null': None,
        }

        for label, value in cases.items():
            issues = validate_database([material("Битый", {"SiO2": value})], {}, MASSES)
            errors = by_level(issues, LEVEL_ERROR)

            self.assertEqual(len(errors), 1, f"{label} produced {len(errors)} errors")
            self.assertEqual(errors[0]['category'], CATEGORY_BAD_VALUE)
            self.assertEqual(errors[0]['material'], "Битый")

    def test_a_legal_zero_is_not_an_unusable_value(self):
        # 0.0 is a perfectly good analysis cell ("this batch has none of it"),
        # and an int is as good as a float
        issues = validate_database([material("Кварц", {"SiO2": 100, "Fe2O3": 0.0})],
                                   {}, MASSES)

        self.assertEqual(by_level(issues, LEVEL_ERROR), [])

    def test_the_report_names_every_level(self):
        issues = validate_database([material("Вода", {}),
                                    material("Огрызок", {"SiO2": 10.0}),
                                    material("Странный", {"Xy2O3": 40.0})], {}, MASSES)
        report = format_report(issues)

        self.assertIn('ERRORS', report)
        self.assertIn('WARNINGS', report)
        self.assertIn('NOTES', report)
        self.assertIn('summary', report)


if __name__ == '__main__':
    unittest.main()
