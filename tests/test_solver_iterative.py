#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# flake8: noqa
# pylint: disable=broad-exception-raised, raise-missing-from, too-many-arguments, redefined-outer-name
# pylint: disable=multiple-statements, logging-fstring-interpolation, trailing-whitespace, line-too-long
# pylint: disable=broad-exception-caught, missing-function-docstring, missing-class-docstring
# pylint: disable=f-string-without-interpolation
# pylance: disable=reportMissingImports, reportMissingModuleSource
# type: ignore

"""
Contract and edge case tests for the iterative solver.

The reference recipes themselves are covered by tests/test_individual_recipes.py
and are not repeated here; this module tests what find_best_recipe promises
about its arguments and about the shape of what it returns.
"""

import json
import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import weights_to_umf
from solver_classic import calculate_recipe_composition, calculate_umf_error
from solver_iterative import (
    SEARCH_EXHAUSTIVE,
    SEARCH_HEURISTIC,
    find_best_recipe,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# A complete target: every oxide the reference recipe brings is listed
FULL_TARGET = {
    "Al2O3": 0.379,
    "B2O3": 0.266,
    "CaO": 0.718,
    "Fe2O3": 0.002,
    "K2O": 0.086,
    "MgO": 0.048,
    "Na2O": 0.143,
    "SiO2": 3.151,
    "SrO": 0.005,
    "TiO2": 0.003,
}

# A partial target: what a human types into the UI, only the oxides they care about
PARTIAL_TARGET = {"SiO2": 3.0, "Al2O3": 0.4, "CaO": 0.7, "Na2O": 0.3}


def material_records():
    """Every record of database/materials.json, inventory flag ignored"""
    with open(os.path.join(ROOT, "database", "materials.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def inventory_from_database():
    """Names of the materials flagged inInventory in database/materials.json"""
    return [m["name"] for m in material_records() if m.get("inInventory")]


class SolverTestCase(unittest.TestCase):
    """Shares the inventory between the test cases so the file is read once"""

    _inventory = None

    @classmethod
    def setUpClass(cls):
        if SolverTestCase._inventory is None:
            SolverTestCase._inventory = inventory_from_database()
        cls.inventory = list(SolverTestCase._inventory)


class TestMaterialLimits(SolverTestCase):

    def test_max_materials_below_the_start_set_is_honoured(self):
        """A two component recipe is a legitimate request, not a request for three"""
        for limit in (1, 2, 3):
            with self.subTest(max_materials=limit):
                solutions = find_best_recipe(self.inventory, FULL_TARGET, max_materials=limit)
                self.assertTrue(solutions, f"no solution for max_materials={limit}")
                for solution in solutions:
                    self.assertLessEqual(solution['materials_count'], limit)
                    self.assertLessEqual(len(solution['recipe']), limit)

    def test_max_materials_is_never_exceeded(self):
        for limit in (4, 6, 10):
            with self.subTest(max_materials=limit):
                for solution in find_best_recipe(self.inventory, FULL_TARGET, max_materials=limit):
                    self.assertLessEqual(solution['materials_count'], limit)

    def test_min_materials_is_never_violated(self):
        """An unreachable min_materials returns nothing instead of a smaller recipe"""
        for floor in (8, 20):
            with self.subTest(min_materials=floor):
                solutions = find_best_recipe(self.inventory, FULL_TARGET, min_materials=floor)
                for solution in solutions:
                    self.assertGreaterEqual(solution['materials_count'], floor)

    def test_min_materials_above_max_materials_returns_nothing(self):
        solutions = find_best_recipe(self.inventory, FULL_TARGET, min_materials=6, max_materials=4)
        self.assertEqual(solutions, [])

    def test_max_materials_below_one_returns_nothing(self):
        for limit in (0, -3):
            with self.subTest(max_materials=limit):
                self.assertEqual(find_best_recipe(self.inventory, FULL_TARGET, max_materials=limit), [])

    def test_reachable_min_materials_is_respected(self):
        solutions = find_best_recipe(self.inventory, FULL_TARGET, min_materials=4, max_materials=8)
        self.assertTrue(solutions)
        for solution in solutions:
            self.assertGreaterEqual(solution['materials_count'], 4)
            self.assertLessEqual(solution['materials_count'], 8)


class TestSolutionCount(SolverTestCase):

    def test_zero_solutions_returns_empty_list(self):
        self.assertEqual(find_best_recipe(self.inventory, FULL_TARGET, max_solutions=0), [])

    def test_negative_solutions_returns_empty_list(self):
        self.assertEqual(find_best_recipe(self.inventory, FULL_TARGET, max_solutions=-1), [])

    def test_one_solution_returns_exactly_one(self):
        self.assertEqual(len(find_best_recipe(self.inventory, FULL_TARGET, max_solutions=1)), 1)

    def test_never_more_than_requested(self):
        for wanted in (1, 2, 3, 5):
            with self.subTest(max_solutions=wanted):
                self.assertLessEqual(len(find_best_recipe(self.inventory, FULL_TARGET, max_solutions=wanted)), wanted)

    def test_solutions_are_unique_recipes(self):
        solutions = find_best_recipe(self.inventory, FULL_TARGET, max_solutions=5)
        keys = [tuple(sorted((name, round(weight, 1)) for name, weight in s['recipe'].items()))
                for s in solutions]
        self.assertEqual(len(keys), len(set(keys)))


class TestDegenerateTargets(SolverTestCase):
    """None of these may raise: through the API they would all become a 500"""

    def test_empty_target(self):
        self.assertEqual(find_best_recipe(self.inventory, {}), [])

    def test_none_target(self):
        self.assertEqual(find_best_recipe(self.inventory, None), [])

    def test_unknown_oxide_only(self):
        self.assertEqual(find_best_recipe(self.inventory, {"Xx2O7": 1.0}), [])

    def test_all_zero_target(self):
        self.assertEqual(find_best_recipe(self.inventory, {"SiO2": 0.0, "CaO": 0.0}), [])

    def test_negative_target(self):
        self.assertEqual(find_best_recipe(self.inventory, {"SiO2": -2.0}), [])

    def test_non_numeric_target(self):
        self.assertEqual(find_best_recipe(self.inventory, {"SiO2": "плохо"}), [])

    def test_unknown_oxides_are_dropped_and_the_rest_still_solves(self):
        target = dict(FULL_TARGET)
        target["Xx2O7"] = 5.0
        solutions = find_best_recipe(self.inventory, target, max_solutions=1)
        self.assertTrue(solutions)
        self.assertNotIn("Xx2O7", solutions[0]['target_umf'])


class TestDegenerateInventories(SolverTestCase):

    def test_empty_inventory(self):
        self.assertEqual(find_best_recipe([], FULL_TARGET), [])

    def test_names_absent_from_the_database(self):
        self.assertEqual(find_best_recipe(["нет такого материала", "и такого тоже"], FULL_TARGET), [])

    def test_single_material_inventory_does_not_raise(self):
        solutions = find_best_recipe(["Кварцевая мука Кварцверке W12"], FULL_TARGET, max_solutions=1)
        for solution in solutions:
            self.assertEqual(solution['materials_count'], 1)


class TestArgumentValidation(SolverTestCase):

    def test_unknown_candidate_search_is_rejected(self):
        with self.assertRaises(ValueError):
            find_best_recipe(self.inventory, FULL_TARGET, candidate_search='magic')

    def test_both_candidate_search_modes_solve(self):
        for mode in (SEARCH_EXHAUSTIVE, SEARCH_HEURISTIC):
            with self.subTest(candidate_search=mode):
                solutions = find_best_recipe(self.inventory, FULL_TARGET, candidate_search=mode)
                self.assertTrue(solutions)


class TestSolutionShape(SolverTestCase):

    REQUIRED_KEYS = ('recipe', 'error', 'objective_error', 'result_umf', 'target_umf',
                     'effective_target_umf', 'unlisted_weight', 'materials_count', 'iterations')

    def test_every_documented_key_is_present(self):
        for solution in find_best_recipe(self.inventory, FULL_TARGET, max_solutions=3):
            for key in self.REQUIRED_KEYS:
                self.assertIn(key, solution)

    def test_recipe_weights_sum_to_exactly_100(self):
        targets = (FULL_TARGET, PARTIAL_TARGET)
        for target in targets:
            for wanted in (1, 3, 5):
                for solution in find_best_recipe(self.inventory, target, max_solutions=wanted):
                    with self.subTest(target=sorted(target), max_solutions=wanted):
                        self.assertAlmostEqual(sum(solution['recipe'].values()), 100.0, places=6)

    def test_materials_count_matches_the_recipe(self):
        for solution in find_best_recipe(self.inventory, FULL_TARGET, max_solutions=5):
            self.assertEqual(solution['materials_count'], len(solution['recipe']))

    def test_no_material_below_the_noise_floor(self):
        for solution in find_best_recipe(self.inventory, FULL_TARGET, max_solutions=5):
            for weight in solution['recipe'].values():
                self.assertGreater(weight, 0.0)

    def test_solutions_are_json_serializable(self):
        solutions = find_best_recipe(self.inventory, FULL_TARGET, max_solutions=5)
        restored = json.loads(json.dumps(solutions, ensure_ascii=False))
        self.assertEqual(len(restored), len(solutions))
        self.assertEqual(restored[0]['recipe'], solutions[0]['recipe'])

    def test_solutions_are_ordered_best_first(self):
        solutions = find_best_recipe(self.inventory, FULL_TARGET, max_solutions=5)
        objectives = [s['objective_error'] for s in solutions]
        self.assertEqual(objectives[0], min(objectives))

    def test_iterations_stay_within_the_human_budget(self):
        for solution in find_best_recipe(self.inventory, FULL_TARGET, max_solutions=5):
            self.assertGreaterEqual(solution['iterations'], 1)
            self.assertLessEqual(solution['iterations'], 8)


class TestErrorSelfConsistency(SolverTestCase):
    """A consumer must be able to recompute the reported error from the result"""

    def test_error_matches_the_returned_target_on_a_full_target(self):
        for solution in find_best_recipe(self.inventory, FULL_TARGET, max_solutions=5):
            recomputed = float(calculate_umf_error(solution['target_umf'], solution['result_umf']))
            self.assertAlmostEqual(solution['error'], recomputed, places=9)

    def test_error_matches_the_returned_target_on_a_partial_target(self):
        for weight in (0.0, 0.5, 1.0):
            with self.subTest(penalize_unlisted=weight):
                for solution in find_best_recipe(self.inventory, PARTIAL_TARGET, max_solutions=5,
                                                 penalize_unlisted=weight):
                    recomputed = float(calculate_umf_error(solution['target_umf'], solution['result_umf']))
                    self.assertAlmostEqual(solution['error'], recomputed, places=9)

    def test_objective_equals_error_when_nothing_is_penalized(self):
        for solution in find_best_recipe(self.inventory, PARTIAL_TARGET, max_solutions=5,
                                         penalize_unlisted=0.0):
            self.assertAlmostEqual(solution['error'], solution['objective_error'], places=9)

    def test_objective_is_never_below_the_reported_error(self):
        for solution in find_best_recipe(self.inventory, PARTIAL_TARGET, max_solutions=5,
                                         penalize_unlisted=1.0):
            self.assertGreaterEqual(solution['objective_error'] + 1e-12, solution['error'])

    def test_effective_target_extends_the_requested_one(self):
        solution = find_best_recipe(self.inventory, PARTIAL_TARGET, max_solutions=1)[0]
        for oxide, value in solution['target_umf'].items():
            self.assertEqual(solution['effective_target_umf'][oxide], value)
        self.assertGreater(len(solution['effective_target_umf']), len(solution['target_umf']))


class TestPartialTarget(SolverTestCase):
    """
    A target typed by hand lists only the oxides the user wants. Zeroing out
    everything else is an assumption, not a fact, and it costs the requested
    oxides - so it has to be switchable.
    """

    def best_error(self, **kwargs):
        solutions = find_best_recipe(self.inventory, PARTIAL_TARGET, max_solutions=5, **kwargs)
        self.assertTrue(solutions)
        return solutions[0]

    def test_not_penalizing_unlisted_oxides_hits_the_requested_ones(self):
        solution = self.best_error(penalize_unlisted=0.0)
        for oxide, expected in PARTIAL_TARGET.items():
            self.assertAlmostEqual(solution['result_umf'].get(oxide, 0.0), expected, delta=0.02,
                                   msg=f"{oxide} missed the target")

    def test_penalizing_unlisted_oxides_is_worse_on_the_requested_ones(self):
        """The documented trade-off: a hard zero on the unlisted oxides costs accuracy"""
        soft = self.best_error(penalize_unlisted=0.0)
        hard = self.best_error(penalize_unlisted=1.0)
        self.assertLess(soft['error'], hard['error'])

    def test_penalizing_unlisted_oxides_keeps_them_out(self):
        """...and buys a cleaner formula in return, which is why it is an option"""
        soft = self.best_error(penalize_unlisted=0.0)
        hard = self.best_error(penalize_unlisted=1.0)

        def contamination(solution):
            return sum(value for oxide, value in solution['result_umf'].items()
                       if oxide not in PARTIAL_TARGET)

        self.assertLess(contamination(hard), contamination(soft))

    def test_boolean_and_float_forms_agree(self):
        for flag, weight in ((True, 1.0), (False, 0.0)):
            with self.subTest(penalize_unlisted=flag):
                by_flag = find_best_recipe(self.inventory, PARTIAL_TARGET, max_solutions=3,
                                           penalize_unlisted=flag)
                by_weight = find_best_recipe(self.inventory, PARTIAL_TARGET, max_solutions=3,
                                             penalize_unlisted=weight)
                self.assertEqual([s['recipe'] for s in by_flag], [s['recipe'] for s in by_weight])

    def test_full_target_is_unaffected_by_the_flag(self):
        """Nothing is unlisted in a complete target, so the weight cannot matter"""
        hard = find_best_recipe(self.inventory, FULL_TARGET, max_solutions=3, penalize_unlisted=1.0)
        self.assertTrue(hard)


class TestTraceIngredient(SolverTestCase):
    """
    Half a percent of a colourant is an ingredient, not noise

    This is the regression of the defect that MIN_MATERIAL_WEIGHT = 1.0 caused
    and that _prune_solution replaced. A weight threshold high enough to keep
    trace junk out of a recipe is also high enough to throw a colourant out of
    it, and the answer that comes back after it is thrown out looks fine: the
    UMF error stays well under the default error_threshold, so nothing in the
    response says that the glaze just went from blue to clear.

    The target is built by the forward calculation from a recipe that really
    needs the colourant, so the test states its own premise instead of trusting
    a number pasted from somewhere.

    Everything below runs at the DEFAULT max_solutions, and that is not an
    accident. The same target at max_solutions=1 still comes back without the
    colourant, for a completely different reason that the pruning pass does not
    touch: the colourant-free answer scores 0.0575, the default error_threshold
    is 0.1, so find_best_recipe declares the branch converged and stops the
    search before the colourant is ever tried. Passing error_threshold=0.01
    recovers it at max_solutions=1 as well. That is a second, independent way
    for a trace ingredient to go missing and it lives in the acceptance rule,
    not in the weight floor.
    """

    COBALT = "Карбонат кобальта, CoCO3"

    COLOURED_RECIPE = {
        "Полевой шпат FFF": 39.5,
        "Кварцевая мука Кварцверке W12": 30.0,
        "Мел, CaCO3": 20.0,
        "Каолин КЖФ-1": 10.0,
        COBALT: 0.5,
    }

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        records = {m["name"]: m for m in material_records()}
        cls.records = [records[name] for name in cls.COLOURED_RECIPE]
        cls.target = weights_to_umf(
            calculate_recipe_composition(cls.records, cls.COLOURED_RECIPE))
        # The inventory of the database has no colourant in it; this is the
        # potter who owns one and wants it used
        cls.coloured_inventory = list(cls.inventory) + [cls.COBALT]

    def test_the_target_really_needs_the_colourant(self):
        """The premise: CoO is in the target and nothing else can bring it"""
        self.assertGreater(self.target.get("CoO", 0.0), 0.0)

        carriers = [m["name"] for m in material_records()
                    if m["name"] in self.coloured_inventory and "CoO" in m.get("formula", {})]
        self.assertEqual(carriers, [self.COBALT])

    def test_a_half_percent_colourant_is_not_dropped(self):
        solutions = find_best_recipe(self.coloured_inventory, self.target)

        self.assertTrue(solutions, "no solution for the coloured target")
        recipe = solutions[0]['recipe']
        self.assertIn(self.COBALT, recipe,
                      f"the colourant was silently dropped: {recipe}")
        self.assertAlmostEqual(recipe[self.COBALT], self.COLOURED_RECIPE[self.COBALT], delta=0.2)
        self.assertAlmostEqual(solutions[0]['result_umf'].get('CoO', 0.0),
                               self.target['CoO'], delta=0.002)

    def test_dropping_it_would_have_passed_for_an_acceptable_answer(self):
        """
        Why this needs a test at all: the failure was invisible

        Solved without the colourant in the inventory, the very same target
        comes back with an error of about 0.057 - under the default
        error_threshold of 0.1, so the caller is handed a clear glaze and told
        it is a good answer. Nothing but the material list gives it away.
        """
        without = find_best_recipe(self.inventory, self.target)

        self.assertTrue(without)
        self.assertNotIn(self.COBALT, without[0]['recipe'])
        self.assertLess(without[0]['error'], 0.1)

        with_colourant = find_best_recipe(self.coloured_inventory, self.target)
        self.assertLess(with_colourant[0]['error'], without[0]['error'])


class TestDeterminism(SolverTestCase):

    def test_two_runs_are_identical(self):
        first = find_best_recipe(self.inventory, FULL_TARGET, max_solutions=5)
        second = find_best_recipe(self.inventory, FULL_TARGET, max_solutions=5)
        self.assertEqual(json.dumps(first, sort_keys=True, ensure_ascii=False),
                         json.dumps(second, sort_keys=True, ensure_ascii=False))

    def test_inventory_order_does_not_matter(self):
        straight = find_best_recipe(self.inventory, FULL_TARGET, max_solutions=5)
        reversed_inventory = find_best_recipe(list(reversed(self.inventory)), FULL_TARGET, max_solutions=5)
        self.assertEqual([s['recipe'] for s in straight], [s['recipe'] for s in reversed_inventory])

    def test_heuristic_mode_is_deterministic_too(self):
        first = find_best_recipe(self.inventory, FULL_TARGET, max_solutions=5,
                                 candidate_search=SEARCH_HEURISTIC)
        second = find_best_recipe(self.inventory, FULL_TARGET, max_solutions=5,
                                  candidate_search=SEARCH_HEURISTIC)
        self.assertEqual([s['recipe'] for s in first], [s['recipe'] for s in second])

    def test_the_solver_does_not_mutate_its_arguments(self):
        target = dict(FULL_TARGET)
        inventory = list(self.inventory)
        find_best_recipe(inventory, target, max_materials=2, max_solutions=3)
        self.assertEqual(target, FULL_TARGET)
        self.assertEqual(inventory, list(self.inventory))


class TestCandidateHeuristic(SolverTestCase):
    """
    The heuristic has to actually drive the search, not decorate it. These tests
    pin the two properties the review asked for: the focus oxide is the one the
    recipe is really furthest from, and the score is not a plain dot product of
    the residual with the material formula.
    """

    def test_focus_oxide_is_the_largest_deviation_and_is_deterministic(self):
        from solver_iterative import _focus_oxide

        residual = {'SiO2': -1.0, 'CaO': 4.0, 'MgO': -7.5, 'Na2O': 0.2}
        self.assertEqual(_focus_oxide(residual), 'MgO')
        self.assertIsNone(_focus_oxide({}))
        self.assertIsNone(_focus_oxide({'SiO2': 0.0}))

        # An exact tie must always resolve the same way, whatever the hash seed
        tie = {'ZnO': 3.0, 'Al2O3': 3.0, 'SiO2': 3.0}
        self.assertEqual({_focus_oxide(dict(tie)) for _ in range(10)}, {'Al2O3'})

    def test_score_really_depends_on_the_focus_oxide(self):
        from solver_iterative import _score_candidate

        material = {'name': 'test', 'formula': {'SiO2': 50.0, 'MgO': 50.0}}
        residual = {'SiO2': 4.0, 'MgO': 4.0}

        on_silica = _score_candidate(material, residual, 'SiO2')
        on_magnesia = _score_candidate(material, residual, 'MgO')
        on_nothing = _score_candidate(material, residual, None)

        # Symmetric material and symmetric residual: the two focus choices agree
        self.assertAlmostEqual(on_silica, on_magnesia, places=9)
        # ...but focusing on something changes the score compared to focusing on
        # nothing, which is exactly what the previous version failed to do
        self.assertNotAlmostEqual(on_silica, on_nothing, places=6)

    def test_deficit_and_contamination_are_separate_terms(self):
        """A plain dot product would score these two materials identically"""
        from solver_iterative import _score_candidate

        residual = {'SiO2': 10.0, 'ZnO': -10.0}
        filler = {'name': 'filler', 'formula': {'SiO2': 100.0}}
        polluter = {'name': 'polluter', 'formula': {'ZnO': 100.0}}
        mixed = {'name': 'mixed', 'formula': {'SiO2': 50.0, 'ZnO': 50.0}}

        # The dot product of `mixed` with the residual is zero, and so is the
        # average of the two pure materials; an asymmetric penalty must push the
        # polluting half below that
        self.assertLess(_score_candidate(mixed, residual, 'SiO2'),
                        0.5 * (_score_candidate(filler, residual, 'SiO2')
                               + _score_candidate(polluter, residual, 'SiO2')) + 1e-9)
        self.assertLess(_score_candidate(polluter, residual, 'SiO2'), 0.0)
        self.assertGreater(_score_candidate(filler, residual, 'SiO2'), 0.0)

    def test_priority_outranks_a_slightly_better_candidate(self):
        from solver_iterative import _rank_candidates

        residual = {'SiO2': 10.0}
        # The low priority material is chemically a little better, the high
        # priority one must still come first
        better = {'name': 'low priority', 'priority': 50, 'formula': {'SiO2': 100.0}}
        preferred = {'name': 'high priority', 'priority': 1, 'formula': {'SiO2': 90.0, 'Al2O3': 10.0}}

        ranked = _rank_candidates([better, preferred], residual, 'SiO2')
        self.assertEqual(ranked[0][1]['name'], 'high priority')

        # ...but a candidate that is much better wins despite its low priority
        far_better = {'name': 'low priority', 'priority': 50, 'formula': {'SiO2': 100.0}}
        weak = {'name': 'high priority', 'priority': 1, 'formula': {'SiO2': 5.0, 'ZnO': 95.0}}
        ranked = _rank_candidates([far_better, weak], {'SiO2': 10.0, 'ZnO': -10.0}, 'SiO2')
        self.assertEqual(ranked[0][1]['name'], 'low priority')

    def test_heuristic_mode_costs_far_fewer_solver_runs(self):
        """The point of the heuristic is the cost, and it has to be visible"""
        import solver_classic
        from scipy.optimize import nnls as real_nnls

        calls = {'n': 0}

        def counting_nnls(a, b, *args, **kwargs):
            calls['n'] += 1
            return real_nnls(a, b, *args, **kwargs)

        original = solver_classic.nnls
        solver_classic.nnls = counting_nnls
        try:
            calls['n'] = 0
            find_best_recipe(self.inventory, FULL_TARGET, max_solutions=5,
                             candidate_search=SEARCH_EXHAUSTIVE)
            exhaustive = calls['n']

            calls['n'] = 0
            find_best_recipe(self.inventory, FULL_TARGET, max_solutions=5,
                             candidate_search=SEARCH_HEURISTIC)
            heuristic = calls['n']
        finally:
            solver_classic.nnls = original

        self.assertGreater(exhaustive, 0)
        self.assertLess(heuristic, exhaustive)


if __name__ == "__main__":
    unittest.main()
