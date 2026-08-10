#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# flake8: noqa
# pylint: disable=broad-exception-raised, raise-missing-from, too-many-arguments, redefined-outer-name
# pylint: disable=multiple-statements, logging-fstring-interpolation, trailing-whitespace, line-too-long
# pylint: disable=broad-exception-caught, missing-function-docstring, missing-class-docstring
# pylint: disable=f-string-without-interpolation
# pylance: disable=reportMissingImports, reportMissingModuleSource

import contextlib
import io
import unittest
import sys
import os
from unittest import mock

from scipy.optimize import nnls

# Fix imports by adding parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common
from common import load_materials, weights_to_umf
import numpy as np

from solver_classic import (
    calculate_recipe_composition,
    create_oxide_matrix,
    find_multiple_solutions,
    main as solver_classic_main,
    solve_glaze_recipe,
    solve_recipe,
)
from solver_iterative import find_best_recipe

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

    @classmethod
    def setUpClass(cls):
        # Deterministic for a fixed target UMF and never mutated by the tests
        # below, so it is computed once for the whole class
        cls.solution = solve_glaze_recipe(TEST_UMF)

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

    def test_reported_umf_is_the_umf_of_the_recipe(self):
        """actual_composition — это UMF самого рецепта, без подгонки под цель.

        Решатель когда-то домасштабировал результат по наименьшему оксиду цели,
        из-за чего репортуемая ошибка была меньше настоящей, а сумма флюсов
        переставала быть единицей. Тест держит эту перенормировку удалённой.
        """
        materials = load_materials(only_inventory=False, priority=True)
        recomputed = weights_to_umf(
            calculate_recipe_composition(materials, self.solution['recipe']))

        self.assertEqual(sorted(recomputed.keys()),
                         sorted(self.solution['actual_composition'].keys()))
        for oxide, value in recomputed.items():
            self.assertAlmostEqual(
                self.solution['actual_composition'][oxide], value, places=4,
                msg=f"{oxide}: reported {self.solution['actual_composition'][oxide]}, "
                    f"recipe gives {value}")

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

    def test_unknown_materials_report_error(self):
        """An inventory of names absent from the database leaves nothing to solve with"""
        solution = solve_glaze_recipe(TEST_UMF, inventory_data=["материал которого нет в базе"])

        self.assertIn('error', solution)
        self.assertNotIn('recipe', solution)

    def test_empty_inventory_reports_error(self):
        """An empty inventory must report an error, not fall back to the database"""
        solution = solve_glaze_recipe(TEST_UMF, inventory_data=[])

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

    def test_unknown_materials_report_error(self):
        """An inventory of names absent from the database leaves nothing to solve with"""
        result = find_multiple_solutions(
            TEST_UMF, inventory_data=["материал которого нет в базе"], logging=False)

        self.assertIsInstance(result, dict)
        self.assertIn('error', result)

    def test_empty_inventory_reports_error(self):
        """An empty inventory must report an error, not fall back to the database"""
        result = find_multiple_solutions(TEST_UMF, inventory_data=[], logging=False)

        self.assertIsInstance(result, dict)
        self.assertIn('error', result)

    def test_same_seed_gives_the_same_solutions(self):
        """Один и тот же seed — побитово тот же список решений"""
        first = find_multiple_solutions(
            TEST_UMF, max_solutions=3, error_tolerance=0.01, logging=False, seed=1234)
        second = find_multiple_solutions(
            TEST_UMF, max_solutions=3, error_tolerance=0.01, logging=False, seed=1234)

        self.assertEqual([solution['recipe'] for solution in first],
                         [solution['recipe'] for solution in second])
        self.assertEqual([solution['error'] for solution in first],
                         [solution['error'] for solution in second])

    def test_another_seed_also_gives_valid_solutions(self):
        """Другой seed вправе дать другие решения, но они обязаны быть валидными.

        Совпадение списков не запрещено — поиск случайный, и разные seed'ы
        вполне могут сойтись к одному набору, — поэтому проверяется только
        качество, а не различие.
        """
        solutions = find_multiple_solutions(
            TEST_UMF, max_solutions=3, error_tolerance=0.01, logging=False, seed=99)

        self.assertIsInstance(solutions, list)
        self.assertGreater(len(solutions), 0)

        for solution in solutions:
            self.assertGreater(len(solution['recipe']), 0)
            self.assertAlmostEqual(sum(solution['recipe'].values()), 100.0, delta=TOTAL_DELTA)
            self.assertLess(solution['error'], 0.1)


class TestCorruptClassificationIsNotAnEmptyAnswer(unittest.TestCase):
    """Битая классификация обязана долетать до вызывающего, а не превращаться в «решений нет».

    Оба движка считают состав через solver_classic.solve_recipe, у которого
    широкий `except Exception`. Без отдельной ветки `except ClassificationError:
    raise` перед ним классический решатель вернул бы {'error': ...} и упал бы
    дальше на KeyError('materials_count'), а итеративный — пустой список,
    неотличимый от честного «решений не найдено». Тесты держат эту ветку на
    месте.
    """

    def setUp(self):
        # The helper below resets the cache, so the real one is put back
        # afterwards whatever happens
        self.saved_cache = common._OXIDE_CLASSIFICATION_CACHE
        self.addCleanup(self.restore_cache)

    def restore_cache(self):
        common._OXIDE_CLASSIFICATION_CACHE = self.saved_cache

    def broken_classification(self):
        """Подменяет загрузчик классификации на падающий.

        Патчится именно загрузчик, а не кэш: кэш заполняется уже проверенными
        данными и повторно не валидируется, поэтому «битый кэш» ошибку бы не
        поднял. _oxide_classification() — та самая точка, где реальный битый
        файл и кидает ClassificationError.
        """
        common._OXIDE_CLASSIFICATION_CACHE = None
        return mock.patch(
            'common._oxide_classification',
            side_effect=common.ClassificationError('broken.json: "unity" must be a non-empty list'))

    def test_classic_single_solution_raises(self):
        with self.broken_classification():
            with self.assertRaises(common.ClassificationError):
                solve_glaze_recipe(TEST_UMF)

    def test_classic_search_raises_instead_of_reporting_an_error_dict(self):
        with self.broken_classification():
            with self.assertRaises(common.ClassificationError):
                find_multiple_solutions(TEST_UMF, max_solutions=1, logging=False)

    def test_iterative_raises_instead_of_returning_an_empty_list(self):
        with self.broken_classification():
            with self.assertRaises(common.ClassificationError):
                find_best_recipe(EXPECTED_MATERIALS, TEST_UMF)

    def test_an_ordinary_failure_is_still_reported_as_a_result(self):
        """Широкий except не должен пострадать: обычная численная беда — по-прежнему словарь"""
        with mock.patch('solver_classic.weights_to_umf', side_effect=ValueError('degenerate')):
            solution = solve_glaze_recipe(TEST_UMF)

        self.assertIn('error', solution)
        self.assertEqual(solution['recipe'], {})


class TestRowWeights(unittest.TestCase):
    """
    The weighting seam of solve_recipe: it has to scale BOTH sides

    Weighting row i means minimizing (w_i * (A_i x - b_i))^2. Scaling the matrix
    alone is the same problem only while b_i == 0, which is what the caller
    outside this function used to rely on and what stopped being true when the
    oxides the target names started carrying weights (TZ_SOLVER_V2.md 10.5,
    10.19).
    """

    # Three materials against four oxides, and the only source of Al2O3 is also
    # the only source of K2O, so their ratio is fixed and the fit has to be a
    # compromise. An exactly determined system would hit the target whatever the
    # weights are and could not show that the weights do anything.
    MATERIALS = [
        {'name': 'Silica', 'formula': {'SiO2': 100.0}},
        {'name': 'Whiting', 'formula': {'CaO': 56.1}},
        {'name': 'Feldspar', 'formula': {'Al2O3': 19.0, 'SiO2': 68.0, 'K2O': 12.0}},
    ]
    TARGET = {'SiO2': 3.0, 'Al2O3': 0.4, 'CaO': 0.7, 'K2O': 0.1}

    # Al2O3 is the second oxide of TARGET, and it is the one this vector forces
    WEIGHTS = np.array([1.0, 50.0, 1.0, 1.0])

    def setUp(self):
        self.matrix, self.names = create_oxide_matrix(self.MATERIALS, list(self.TARGET))

    def solve(self, row_weights=None, materials=None):
        return solve_recipe(self.matrix, self.TARGET, self.names, materials,
                            row_weights=row_weights)

    def test_uniform_weights_change_nothing(self):
        """Scaling every row by the same number cannot move the argmin"""
        plain = self.solve()
        scaled = self.solve(row_weights=np.full(len(self.TARGET), 7.0))

        self.assertEqual(plain['recipe'], scaled['recipe'])

    def test_weighting_a_row_really_moves_the_fit(self):
        """...and a non-uniform one does, or the parameter would be decoration"""
        plain = self.solve()
        weighted = self.solve(row_weights=self.WEIGHTS)

        self.assertNotEqual(plain['recipe'], weighted['recipe'])

    def test_the_right_hand_side_travels_with_the_row(self):
        """
        The invariant, as an equality against the scaled system it has to equal
        and an inequality against the mistake it has to avoid

        Scaling A alone is a DIFFERENT problem here, and visibly so: the same
        weights give 67% feldspar when both sides travel and 2% when only the
        matrix does. That is what would have happened silently had the caller
        gone on scaling the matrix outside this function.
        """
        weighted = self.solve(row_weights=self.WEIGHTS)

        target_weights = common.umf_to_weights(self.TARGET)
        scaled_rhs = np.array([target_weights[oxide] * weight
                               for oxide, weight in zip(self.TARGET, self.WEIGHTS)])
        expected, _residual = nnls(self.matrix * self.WEIGHTS[:, None], scaled_rhs)
        expected = 100.0 * expected / expected.sum()

        for name, share in zip(self.names, expected):
            self.assertAlmostEqual(weighted['recipe'][name], share, places=2)

        matrix_only = solve_recipe(self.matrix * self.WEIGHTS[:, None], self.TARGET, self.names)
        self.assertNotEqual(weighted['recipe'], matrix_only['recipe'])

    def test_the_matrix_of_the_caller_is_not_touched(self):
        before = self.matrix.copy()
        self.solve(row_weights=self.WEIGHTS)

        np.testing.assert_array_equal(self.matrix, before)

    def test_the_matrix_only_composition_estimate_stays_unscaled(self):
        """
        The trap: without available_materials the composition is read straight
        off oxide_matrix, and reading a SCALED matrix there would silently
        report a formula nobody mixed. solver_iterative always passes materials,
        so this branch would never have complained.
        """
        estimated = self.solve(row_weights=self.WEIGHTS)
        exact = self.solve(row_weights=self.WEIGHTS, materials=self.MATERIALS)

        for oxide, value in exact['weight_composition'].items():
            self.assertAlmostEqual(estimated['weight_composition'][oxide], value, delta=0.05,
                                   msg=f"{oxide} of the matrix-only estimate carries the weights")

    def test_a_wrong_length_is_an_error_and_not_an_empty_answer(self):
        with self.assertRaises(ValueError):
            self.solve(row_weights=np.ones(len(self.TARGET) + 1))


class TestCommandLineCleansItsTarget(unittest.TestCase):
    """
    find_multiple_solutions does not validate its target, and main() must

    The library contract is "the caller cleans" - /api/solve does it above the
    choice of engine - and a command line IS a caller, not a second library.
    Before it cleaned, `--umf '{"SiO2": -4, "Al2O3": -1}'` printed three
    "solutions" fitted to negative oxides with nothing said about it.
    """

    def run_main(self, umf):
        argv = ['solver_classic.py', '--umf', umf, '--solutions', '1']
        printed = io.StringIO()
        with mock.patch.object(sys, 'argv', argv), contextlib.redirect_stdout(printed):
            solver_classic_main()
        return printed.getvalue()

    def test_a_refused_oxide_is_named_before_any_recipe_is_printed(self):
        output = self.run_main('{"SiO2": 4, "Al2O3": 1, "Na2O": 0.5, "K2O": 0.5, "Xx": 1}')

        self.assertIn('Xx', output)
        self.assertIn('не распознаны', output)
        self.assertIn('решени', output, "the rest of the target is still solved")

    def test_a_target_with_nothing_left_is_refused_instead_of_solved(self):
        output = self.run_main('{"SiO2": -4, "Al2O3": -1}')

        self.assertIn('Al2O3, SiO2', output)
        self.assertNotIn('состав рецепта', output)

    def test_a_target_of_nothing_but_zeros_is_refused_instead_of_crashing(self):
        """It used to die of ZeroDivisionError inside umf_to_weights"""
        output = self.run_main('{"SiO2": 0, "Al2O3": 0}')

        self.assertIn('нулев', output)
        self.assertNotIn('состав рецепта', output)

    def test_a_zero_next_to_a_real_value_is_still_solved(self):
        output = self.run_main('{"SiO2": 4, "Al2O3": 1, "Na2O": 0.5, "K2O": 0.5, "Fe2O3": 0}')

        self.assertIn('состав рецепта', output)

    def test_a_clean_target_is_not_commented_on(self):
        output = self.run_main('{"SiO2": 4, "Al2O3": 1, "Na2O": 0.5, "K2O": 0.5}')

        self.assertNotIn('не распознаны', output)
        self.assertIn('состав рецепта', output)


if __name__ == "__main__":
    unittest.main()
