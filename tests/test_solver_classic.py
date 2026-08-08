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
from solver_classic import solve_glaze_recipe, find_multiple_solutions

# Target UMF of the reference transparent glaze
TEST_UMF = {
    "Na2O": 0.143,
    "K2O": 0.086,
    "MgO": 0.048,
    "CaO": 0.717,
    "SrO": 0.005,
    "Fe2O3": 0.002,
    "Al2O3": 0.378,
    "B2O3": 0.265,
    "TiO2": 0.003,
    "SiO2": 3.144,
}

# Materials of the original recipe this UMF was calculated from
EXPECTED_MATERIALS = [
    "Нефелин-сиенит VR13",
    "Кварцевая мука Кварцверке W12",
    "Волластонит МИВОЛЛ",
    "Улексит (Химпэк)",
    "Каолин КЖФ-1",
]

# Recipe percentages are rounded to 2 decimals and materials below 0.1% are
# dropped, so the total is only approximately 100%
TOTAL_DELTA = 0.5


class TestSolveGlazeRecipe(unittest.TestCase):
    """Integration test of the classic solver against the real material database"""

    def setUp(self):
        self.solution = solve_glaze_recipe(TEST_UMF)

    def test_solution_structure(self):
        for key in ('recipe', 'error', 'target_composition', 'actual_composition',
                    'weight_composition', 'materials_count'):
            self.assertIn(key, self.solution)

        self.assertEqual(self.solution['target_composition'], TEST_UMF)
        self.assertEqual(self.solution['materials_count'], len(self.solution['recipe']))

    def test_recipe_is_not_empty(self):
        self.assertGreater(len(self.solution['recipe']), 0)

    def test_recipe_percentages_sum_to_100(self):
        total = sum(self.solution['recipe'].values())

        self.assertAlmostEqual(total, 100.0, delta=TOTAL_DELTA)

    def test_expected_materials_are_found(self):
        found = [name for name in EXPECTED_MATERIALS if name in self.solution['recipe']]

        self.assertGreaterEqual(
            len(found), 3,
            f"only {len(found)} of the expected materials found: {found}")

    def test_actual_umf_is_close_to_target(self):
        actual = self.solution['actual_composition']

        for oxide, target_value in TEST_UMF.items():
            self.assertLessEqual(
                abs(actual.get(oxide, 0) - target_value), max(0.05, target_value * 0.1),
                f"{oxide}: target {target_value}, actual {actual.get(oxide, 0)}")

        self.assertLess(self.solution['error'], 0.1)

    def test_custom_inventory_reproduces_original_recipe(self):
        """With only the original five materials available the recipe is recovered"""
        solution = solve_glaze_recipe(TEST_UMF, inventory_data=EXPECTED_MATERIALS)

        original_recipe = {
            "Нефелин-сиенит VR13": 30,
            "Кварцевая мука Кварцверке W12": 20,
            "Волластонит МИВОЛЛ": 20,
            "Улексит (Химпэк)": 15,
            "Каолин КЖФ-1": 15,
        }

        self.assertEqual(sorted(solution['recipe'].keys()), sorted(original_recipe.keys()))
        for name, percentage in original_recipe.items():
            self.assertAlmostEqual(solution['recipe'][name], percentage, delta=1.0)

    def test_empty_inventory_reports_error(self):
        solution = solve_glaze_recipe(TEST_UMF, inventory_data=["материал которого нет в базе"])

        self.assertIn('error', solution)
        self.assertNotIn('recipe', solution)


class TestFindMultipleSolutions(unittest.TestCase):
    """The search used by the API. It is randomized, so only structure is checked"""

    def test_returns_usable_solutions(self):
        solutions = find_multiple_solutions(
            TEST_UMF, max_solutions=3, error_tolerance=0.01, logging=False)

        self.assertIsInstance(solutions, list)
        self.assertGreater(len(solutions), 0)
        self.assertLessEqual(len(solutions), 3)

        for solution in solutions:
            self.assertGreater(len(solution['recipe']), 0)
            self.assertAlmostEqual(sum(solution['recipe'].values()), 100.0, delta=TOTAL_DELTA)
            self.assertLess(solution['error'], 0.1)
            self.assertEqual(solution['materials_count'], len(solution['recipe']))

    def test_empty_inventory_reports_error(self):
        result = find_multiple_solutions(
            TEST_UMF, inventory_data=["материал которого нет в базе"], logging=False)

        self.assertIsInstance(result, dict)
        self.assertIn('error', result)


if __name__ == "__main__":
    unittest.main()
