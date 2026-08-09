#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# flake8: noqa
# pylint: disable=broad-exception-raised, raise-missing-from, too-many-arguments, redefined-outer-name
# pylint: disable=multiple-statements, logging-fstring-interpolation, trailing-whitespace, line-too-long
# pylint: disable=broad-exception-caught, missing-function-docstring, missing-class-docstring
# pylint: disable=f-string-without-interpolation
# pylance: disable=reportMissingImports, reportMissingModuleSource

"""
Offline oracles of the inverse problem (TZ_SOLVER_V2.md 7.4 and 7.5)

The inverse problem has no unique solution in general, so recipe weights may
only be asserted where the system is determined - as many independent materials
as oxides. That is what the two cases of tests/fixtures/inverse_exact.json are:
textbook problems with a published answer, run through both engines and through
our own forward calculation. Everything they need is in the fixture, as plain
numbers, so nothing here depends on database/materials.json.

tests/fixtures/collinear_feldspars.json is the opposite case on purpose: an
underdetermined system where the answer is not unique and only the chemistry and
the conditioning of the material set can be judged.
"""

import json
import math
import os
import sys
import unittest

# Fix imports by adding parent directory to path
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_DIR)
from common import load_materials, resolve_inventory, resolve_material_pool, weights_to_umf
from quality_metrics import MAX_CONDITION_NUMBER, solution_quality
from solver_classic import (
    calculate_recipe_composition,
    calculate_umf_error,
    find_multiple_solutions,
    solve_glaze_recipe,
)
from solver_iterative import find_best_recipe

FIXTURES_DIR = os.path.join(PROJECT_DIR, 'tests', 'fixtures')

# Per oxide sanity check of a solved recipe: even where the weights are allowed
# to differ from the published ones, the chemistry may not.
MAX_OXIDE_DEVIATION = 0.02

# Chemistry gate of the degeneracy stress (TZ_SOLVER_V2.md 7.5)
MAX_DEGENERACY_UMF_ERROR = 0.05


def load_fixture(name):
    """Read a JSON fixture of tests/fixtures by file name"""
    with open(os.path.join(FIXTURES_DIR, name), 'r', encoding='utf-8') as f:
        return json.load(f)


def as_floats(recipe):
    """Plain floats out of a recipe, whose values may be numpy scalars"""
    return {name: float(weight) for name, weight in recipe.items()}


def forward_umf(materials, recipe):
    """Our own forward calculation: recipe -> weight composition -> UMF"""
    return weights_to_umf(calculate_recipe_composition(materials, recipe))


def assert_classic_solved(test_case, solution):
    """
    Check that the classic engine actually solved something

    Its "error" key is overloaded: normally it holds the numeric UMF distance,
    but a failure puts a message string there instead. Testing for the presence
    of the key would therefore reject every good answer, so what is checked is
    the type of the value and the presence of a non-empty recipe.
    """
    test_case.assertNotIsInstance(solution.get('error'), str, msg=str(solution))
    test_case.assertTrue(solution.get('recipe'), msg=str(solution))
    return solution


class TestMaterialInjectionSeam(unittest.TestCase):
    """
    The materials= seam has to replace the database, not be filtered by it

    Every oracle below stands on this: the idealised materials of the fixtures
    are not in database/materials.json and are not in anybody's inventory, so if
    the injected records were still run through resolve_inventory() the solvers
    would see an empty material set and the oracles would degrade into "no
    solution found" instead of failing.
    """

    def setUp(self):
        self.materials = [
            {'name': 'Pure Silica', 'formula': {'SiO2': 100.0}},
            {'name': 'Pure Whiting', 'formula': {'CaO': 56.10}},
            {'name': 'Pure Kaolin', 'formula': {'Al2O3': 40.21, 'SiO2': 47.29}},
        ]
        self.target = forward_umf(
            self.materials,
            {'Pure Silica': 30.0, 'Pure Whiting': 40.0, 'Pure Kaolin': 30.0})

    def test_injected_materials_need_no_inventory(self):
        """materials= without an inventory means "all of this is available\""""
        solution = assert_classic_solved(
            self, solve_glaze_recipe(self.target, materials=self.materials))

        self.assertEqual(set(solution['recipe']), {'Pure Silica', 'Pure Whiting', 'Pure Kaolin'})

    def test_injected_materials_are_the_whole_database(self):
        """No record of database/materials.json may leak into the answer"""
        for engine_recipe in (self._classic_recipe(), self._iterative_recipe()):
            self.assertTrue(set(engine_recipe) <= {material['name'] for material in self.materials},
                            msg=f"unexpected materials in {engine_recipe}")

    def test_an_inventory_still_filters_injected_materials(self):
        """Given both, the inventory narrows the injected catalogue as usual"""
        solution = solve_glaze_recipe(self.target, inventory_data=['Pure Silica', 'Pure Whiting'],
                                      materials=self.materials)

        self.assertNotIn('Pure Kaolin', solution['recipe'])

    def test_an_inventory_matching_nothing_is_an_error_not_a_silent_answer(self):
        solution = solve_glaze_recipe(self.target, inventory_data=['Кварцевая мука Кварцверке W12'],
                                      materials=self.materials)

        self.assertIsInstance(solution.get('error'), str)
        self.assertFalse(solution.get('recipe'))

    def test_an_injected_material_with_an_empty_formula_is_still_dropped(self):
        """The zero formula filter applies to an injected catalogue too"""
        with_dead_column = self.materials + [{'name': 'Water', 'formula': {}}]

        classic = solve_glaze_recipe(self.target, materials=with_dead_column)
        iterative = find_best_recipe(None, self.target, materials=with_dead_column)

        self.assertNotIn('Water', assert_classic_solved(self, classic)['recipe'])
        self.assertTrue(iterative)
        self.assertNotIn('Water', iterative[0]['recipe'])

    def test_injected_materials_need_no_priority_key(self):
        """priority is optional; the iterative solver falls back to DEFAULT_PRIORITY"""
        for material in self.materials:
            self.assertNotIn('priority', material)

        self.assertTrue(self._iterative_recipe())

    def test_the_default_still_reads_the_database(self):
        """materials=None must keep behaving exactly as before"""
        records, inventory = resolve_material_pool(None, None)

        self.assertEqual(records, load_materials(only_inventory=False, priority=True))
        self.assertEqual(inventory, resolve_inventory(None))

    def _classic_recipe(self):
        solution = assert_classic_solved(
            self, solve_glaze_recipe(self.target, materials=self.materials))
        return as_floats(solution['recipe'])

    def _iterative_recipe(self):
        solutions = find_best_recipe(None, self.target, materials=self.materials)
        self.assertTrue(solutions, msg="the iterative solver returned nothing")
        return as_floats(solutions[0]['recipe'])


class TestExactOracles(unittest.TestCase):
    """
    Determined systems with a published answer (TZ_SOLVER_V2.md 7.4)

    Case A is Leach 4321, case B is problem 3 of Linda Arbuckle's glaze
    chemistry handout. Both have as many materials as oxides and a unique
    solution, which is the only situation where asserting recipe weights is
    legitimate at all.
    """

    @classmethod
    def setUpClass(cls):
        cls.cases = load_fixture('inverse_exact.json')['cases']

    def assert_umf_matches_target(self, target_umf, actual_umf, label):
        """Per oxide sanity check on top of the weight comparison"""
        for oxide, expected_value in target_umf.items():
            deviation = abs(actual_umf.get(oxide, 0.0) - expected_value)
            self.assertLessEqual(
                deviation, MAX_OXIDE_DEVIATION,
                f"{label}: {oxide} is {actual_umf.get(oxide, 0.0)}, target {expected_value}")

    def assert_recipe_matches_expectation(self, case, recipe, label):
        """Every material within the tolerance the fixture states, and no other"""
        expected = case['expected_recipe']
        tolerance = case['weight_tolerance']

        self.assertEqual(set(recipe), set(expected),
                         f"{label}: expected exactly {sorted(expected)}, got {sorted(recipe)}")

        for material, expected_weight in expected.items():
            self.assertAlmostEqual(
                recipe[material], expected_weight, delta=tolerance,
                msg=f"{label}: {material} is {recipe[material]}, expected {expected_weight}"
                    f" +-{tolerance}")

        self.assertAlmostEqual(sum(recipe.values()), 100.0, delta=0.5, msg=label)

    def test_the_fixture_is_self_consistent(self):
        """
        The published recipe reproduces the published target through OUR forward math

        This one holds no solver at all: it proves the fixture is internally
        consistent, so that a failure of the cases below is a solver failure and
        not a typo in an analysis. The published percentages of case B are
        rounded to one decimal, which is where its residual 0.0017 comes from -
        case A is exact.
        """
        for case in self.cases:
            with self.subTest(case=case['id']):
                actual_umf = forward_umf(case['materials'], case['expected_recipe'])
                error = calculate_umf_error(case['target_umf'], actual_umf)

                self.assertLessEqual(error, 0.01, f"{case['id']}: forward error {error}")
                self.assert_umf_matches_target(case['target_umf'], actual_umf,
                                               f"{case['id']} forward")

    def test_classic_solver_recovers_the_published_recipe(self):
        for case in self.cases:
            with self.subTest(case=case['id']):
                solution = assert_classic_solved(
                    self, solve_glaze_recipe(case['target_umf'], materials=case['materials']))

                self.assert_recipe_matches_expectation(case, as_floats(solution['recipe']),
                                                       f"{case['id']} classic")
                self.assert_umf_matches_target(case['target_umf'],
                                               as_floats(solution['actual_composition']),
                                               f"{case['id']} classic")

    def test_classic_subset_search_recovers_the_published_recipe(self):
        """The seeded entry point of the classic engine, on the same two cases"""
        for case in self.cases:
            with self.subTest(case=case['id']):
                solutions = find_multiple_solutions(case['target_umf'], max_solutions=3,
                                                    min_materials=True, logging=False, seed=0,
                                                    materials=case['materials'])

                self.assertTrue(solutions, msg=f"{case['id']}: no solution")
                self.assert_recipe_matches_expectation(case, as_floats(solutions[0]['recipe']),
                                                       f"{case['id']} classic search")

    def test_iterative_solver_recovers_the_published_recipe(self):
        for case in self.cases:
            with self.subTest(case=case['id']):
                solutions = find_best_recipe(None, case['target_umf'], materials=case['materials'])

                self.assertTrue(solutions, msg=f"{case['id']}: no solution")
                self.assert_recipe_matches_expectation(case, as_floats(solutions[0]['recipe']),
                                                       f"{case['id']} iterative")
                self.assert_umf_matches_target(case['target_umf'],
                                               as_floats(solutions[0]['result_umf']),
                                               f"{case['id']} iterative")


class CollinearFeldsparsMixin:
    """
    Shared setup of the degeneracy stress (TZ_SOLVER_V2.md 7.5, amended by 10.9)

    The fixture holds Custer Feldspar and two near-collinear synthetic twins of
    it, plus silica, whiting and kaolin. The target is our own forward
    calculation of 40 / 30 / 20 / 10 on the FIRST feldspar, so an honest answer
    exists and uses four materials with cond = 15.1.
    """

    @classmethod
    def setUpClass(cls):
        fixture = load_fixture('collinear_feldspars.json')
        cls.materials = fixture['materials']
        cls.target_umf = fixture['target_umf']
        cls.honest_recipe = fixture['honest_recipe']

    def solve_classic(self):
        solution = assert_classic_solved(
            self, solve_glaze_recipe(self.target_umf, materials=self.materials))
        return as_floats(solution['recipe']), as_floats(solution['actual_composition'])

    def solve_iterative(self):
        solutions = find_best_recipe(None, self.target_umf, materials=self.materials)
        self.assertTrue(solutions, msg="the iterative solver returned nothing")
        return as_floats(solutions[0]['recipe']), as_floats(solutions[0]['result_umf'])

    def condition_number(self, recipe):
        """cond of the "oxides x used materials" matrix behind the recipe"""
        quality = solution_quality(recipe, self.honest_recipe, self.materials)
        return quality['conditioning']['cond']


class TestCollinearFeldsparsChemistry(CollinearFeldsparsMixin, unittest.TestCase):
    """
    What holds today: a solution comes back and its chemistry is right

    The weights are deliberately NOT asserted. Three near-identical feldspars
    make the system underdetermined, so the answer is not unique by
    construction and any particular split of the feldspar mass is as valid as
    another - as chemistry. Whether it is a good recipe is what the conditioning
    test below is for.
    """

    def test_the_target_is_the_forward_calculation_of_the_honest_recipe(self):
        """Guards the fixture itself: the target must be reachable exactly"""
        self.assertEqual(forward_umf(self.materials, self.honest_recipe), self.target_umf)

    def test_the_honest_answer_is_well_conditioned(self):
        """The existence proof: a four material answer with cond about 15"""
        self.assertLessEqual(self.condition_number(self.honest_recipe), MAX_CONDITION_NUMBER)

    def test_classic_returns_finite_chemistry_within_tolerance(self):
        self.assert_solution_is_sane(*self.solve_classic())

    def test_iterative_returns_finite_chemistry_within_tolerance(self):
        self.assert_solution_is_sane(*self.solve_iterative())

    def assert_solution_is_sane(self, recipe, result_umf):
        self.assertTrue(recipe, "no recipe came back")

        for material, weight in recipe.items():
            self.assertTrue(math.isfinite(weight), f"{material} weighs {weight}")
            self.assertGreater(weight, 0.0, f"{material} weighs {weight}")

        for oxide, value in result_umf.items():
            self.assertTrue(math.isfinite(value), f"{oxide} is {value}")

        error = calculate_umf_error(self.target_umf, result_umf)
        self.assertTrue(math.isfinite(error), f"the UMF error is {error}")
        self.assertLessEqual(error, MAX_DEGENERACY_UMF_ERROR)


class TestCollinearFeldsparsConditioning(CollinearFeldsparsMixin, unittest.TestCase):
    """
    What should hold and does not: the answer must not stand on a compensating pair

    Both tests below are expected failures. See their docstrings; the short
    version is that this is a solver defect, not a threshold that wants
    loosening, and the assertion must stay exactly as it is.
    """

    @unittest.expectedFailure
    def test_classic_does_not_stand_on_a_compensating_pair(self):
        """
        EXPECTED FAILURE - measured cond = 1849.6 against a threshold of 1e3.

        The classic solver answers Variant A 11.78 + Variant B 27.92 + Silica
        30.28 + Whiting 20.01 + Kaolin 10.02 and never touches Custer Feldspar
        at all, splitting the feldspar mass across the two near-identical twins.
        The chemistry is perfect (UMF error 0.0) and that is exactly the problem:
        NNLS over the full material set has no reason to prefer one feldspar to a
        combination of two almost identical ones, so it returns a five material
        recipe whose weights are barely determined - cond 1849.6 against the 15.1
        of the honest four material answer that exists in the same catalogue.

        This is a real defect of the solver, not a threshold to relax.
        MAX_CONDITION_NUMBER = 1e3 sits between the 15.1 of an honest recipe and
        the 1849.6 of a degenerate one with two orders of magnitude of room on
        either side, and weakening it would remove the only measure that tells
        the two apart (TZ_SOLVER_V2.md 10.9 measured that rounding_drift does
        not: it maxes out at 0.0153 on the whole degenerate family).

        When the solver learns to drop a redundant material this test will report
        an UNEXPECTED SUCCESS. That is the signal to delete this decorator, not
        to delete the test.
        """
        recipe, _ = self.solve_classic()

        self.assertLessEqual(self.condition_number(recipe), MAX_CONDITION_NUMBER)

    @unittest.expectedFailure
    def test_iterative_does_not_stand_on_a_compensating_pair(self):
        """
        EXPECTED FAILURE - measured cond = 1849.6 against a threshold of 1e3.

        The iterative solver lands on the same degenerate answer as the classic
        one (Variant A 11.78 + Variant B 27.92 + Silica 30.27 + Whiting 20.01 +
        Kaolin 10.02, UMF error 0.0), by a different route: all six materials of
        the fixture share one priority, so _priority_start_set takes the whole
        group at once and the search starts from every material there is. From
        there the beam only ever ADDS materials - nothing in it tries dropping
        one - and _shrink_to_limit only fires above max_materials, which six
        materials never reach. So the starting set is the answer, twins included.

        This is a real defect of the solver, not a threshold to relax; see the
        classic counterpart above for why MAX_CONDITION_NUMBER = 1e3 stays.

        When the solver learns to drop a redundant material this test will report
        an UNEXPECTED SUCCESS. That is the signal to delete this decorator, not
        to delete the test.
        """
        recipe, _ = self.solve_iterative()

        self.assertLessEqual(self.condition_number(recipe), MAX_CONDITION_NUMBER)


if __name__ == '__main__':
    unittest.main()
