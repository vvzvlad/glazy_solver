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
import math
import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import (
    OXIDE_SCALE_FLOOR,
    load_materials,
    load_molar_masses,
    umf_deviation,
    weights_to_umf,
)
from feasibility import DEFAULT_FEASIBILITY_TOL as CHEMISTRY_TOLERANCE
from solver_classic import calculate_recipe_composition, calculate_umf_error
from solver_iterative import (
    OBJECTIVE_DEADBAND,
    SEARCH_EXHAUSTIVE,
    SEARCH_HEURISTIC,
    _build_problem,
    _focus_oxide,
    _known_oxide,
    _score_candidate,
    _solve_material_set,
    _weight_residual,
    find_best_recipe,
    usable_target,
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
                     'effective_target_umf', 'unlisted_weight', 'materials_count',
                     'merged_variants', 'iterations')

    def test_every_documented_key_is_present(self):
        for solution in find_best_recipe(self.inventory, FULL_TARGET, max_solutions=3):
            for key in self.REQUIRED_KEYS:
                self.assertIn(key, solution)

    def test_merged_variants_counts_the_recipes_that_collapsed_onto_this_one(self):
        """
        Fewer solutions than asked for is not silence any more

        Several recipes of the search can prune onto the same answer, and they
        are merged instead of being listed as near-duplicates. The count is what
        tells a caller that happened, so it has to be an honest integer even
        when nothing collapsed.
        """
        solutions = find_best_recipe(self.inventory, FULL_TARGET, max_solutions=5)

        for solution in solutions:
            self.assertIsInstance(solution['merged_variants'], int)
            self.assertGreaterEqual(solution['merged_variants'], 0)

        # A target with many near-equivalent answers has to show at least one
        # merge somewhere, otherwise this field is never exercised at all
        crowded = find_best_recipe(self.inventory, PARTIAL_TARGET, max_solutions=5,
                                   penalize_unlisted=0.0)
        self.assertTrue(any(s['merged_variants'] > 0 for s in solutions + crowded),
                        "no recipe collapsed onto another on either target")

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

    def test_objective_is_recomputable_when_nothing_is_penalized(self):
        """
        The objective is no longer the reported error, and a consumer has to be
        able to rebuild it anyway

        It used to be exactly calculate_umf_error over the requested oxides when
        penalize_unlisted was 0, and a test pinned that equality. That equality
        is gone by design: the objective is now the L2 of the per-oxide RELATIVE
        deviations, each one given OBJECTIVE_DEADBAND for free. What replaces it
        is the same promise on the new definition - the number can be rebuilt
        from target_umf and result_umf, and nothing else goes into it while the
        unlisted oxides carry no weight.
        """
        for solution in find_best_recipe(self.inventory, PARTIAL_TARGET, max_solutions=5,
                                         penalize_unlisted=0.0):
            squared = 0.0
            for oxide, expected in solution['target_umf'].items():
                scale = max(expected, OXIDE_SCALE_FLOOR)
                residual = (abs(expected - solution['result_umf'].get(oxide, 0.0)) / scale
                            - OBJECTIVE_DEADBAND)
                if residual > 0.0:
                    squared += residual ** 2
            self.assertAlmostEqual(solution['objective_error'], math.sqrt(squared), places=9)

    def test_the_objective_is_not_the_reported_error(self):
        """
        The two numbers measure different things and must not be read as one

        A consumer comparing objective_error against an absolute tolerance, or
        error against the 0.05 the feasibility LP and the benchmark speak, is
        making a units mistake. The partial target here is off by about 2% on
        every requested oxide, which is a hair over the deadband and worth
        0.002 of objective, while the same answer carries 0.066 of absolute L2 -
        the two are not within an order of magnitude of each other.
        """
        solution = find_best_recipe(self.inventory, PARTIAL_TARGET, max_solutions=1,
                                    penalize_unlisted=0.0)[0]
        self.assertNotAlmostEqual(solution['error'], solution['objective_error'], places=3)

    def test_the_contamination_term_lives_in_the_objective_only(self):
        """
        What the contamination term does to the objective, on a target that has
        one

        This replaces an assertion that the objective is NEVER below the
        reported error, which was true while the two were the same absolute norm
        plus a non-negative term and is false now: the relative objective
        divides a deviation of a big oxide by that oxide, so a well fitted
        target of large numbers comes out below its own absolute L2. Measured
        over the 300 case corpus, 340 of the returned solutions have an
        objective below their error, the widest by 13.8 - it is not an ordering
        that exists, and a test claiming it would only be pinning a fixture.

        What IS true is the reason the two differ at all: penalize_unlisted
        loads the objective with contamination that `error` does not see, so on
        a partial target the hard setting must raise the objective while leaving
        the reported error to its own trade-off.
        """
        soft = find_best_recipe(self.inventory, PARTIAL_TARGET, max_solutions=1,
                                penalize_unlisted=0.0)[0]
        hard = find_best_recipe(self.inventory, PARTIAL_TARGET, max_solutions=1,
                                penalize_unlisted=1.0)[0]

        self.assertGreater(hard['objective_error'], hard['error'])
        self.assertAlmostEqual(soft['objective_error'], 0.0, delta=0.05)

    def test_effective_target_extends_the_requested_one(self):
        solution = find_best_recipe(self.inventory, PARTIAL_TARGET, max_solutions=1)[0]
        for oxide, value in solution['target_umf'].items():
            self.assertEqual(solution['effective_target_umf'][oxide], value)
        self.assertGreater(len(solution['effective_target_umf']), len(solution['target_umf']))


class TestKeysWithoutAMolarMassGetNoRow(unittest.TestCase):
    """
    A key the system cannot express in UMF takes no part in the fit at all

    The rule is the molar mass, not a list of names, and it has to hold in every
    place the search reads a formula - the NNLS rows, the objective, the
    residual and the candidate score. A key with no molar mass has no relative
    deviation to minimize, so any weight it gets in weight percent is arbitrary;
    and under the relative weights an arbitrary weight is decisive rather than
    small. A chemical row is 1 / (molar mass * scale), which over this database
    runs from 0.005548 (SiO2 at a target of 3.0) to at most 0.5264 (fluorine at
    the scale floor), so a row left on the bare penalty of 1.0 outweighs every
    chemical row there can be - by 1.9x at worst and 180x in the common case
    (1 / 0.005548; the 182 that stood here was 1 / 0.0055, the reciprocal of the
    rounded row rather than of the row).

    Loss on ignition is the instance that occurs in the shipped data, and it is
    rarer than the first version of this docstring claimed: 3 of the 216
    materials of database/materials.json carry a Loi key and all three are
    inInventory: false. Neither measurement rig can see it - the default 19
    material inventory excludes all three and bench/corpus strips LOI from every
    Glazy formula - so the fixtures below carry their own.

    "Carbon" stands for the general case: an unanalysed key nobody has ever put
    on a list.
    """

    MATERIALS = [
        {'name': 'Whiting', 'formula': {'CaO': 56.1, 'Loi': 43.9}},
        {'name': 'Silica', 'formula': {'SiO2': 100.0}},
        {'name': 'Kaolin', 'formula': {'Al2O3': 40.0, 'SiO2': 47.0, 'LOI': 13.0}},
    ]
    TARGET = {'SiO2': 3.0, 'Al2O3': 0.4, 'CaO': 0.7}

    # The general case: the same shelf with and without an unanalysed key on the
    # one material that carries the requested P2O5
    ASH_BASE = [
        {'name': 'Silica', 'formula': {'SiO2': 100.0}},
        {'name': 'Whiting', 'formula': {'CaO': 56.1}},
        {'name': 'Kaolin', 'formula': {'Al2O3': 40.0, 'SiO2': 47.0}},
    ]
    ASH_TARGET = {'SiO2': 3.0, 'Al2O3': 0.4, 'CaO': 0.7, 'P2O5': 0.05}
    CLEAN_ASH = {'CaO': 30.0, 'SiO2': 30.0, 'P2O5': 5.0}
    DIRTY_ASH = {'CaO': 30.0, 'SiO2': 30.0, 'P2O5': 5.0, 'Carbon': 35.0}

    def setUp(self):
        self.problem = _build_problem(usable_target(self.TARGET)[0], self.MATERIALS, 1.0)

    def solve_ash(self, ash_formula):
        materials = self.ASH_BASE + [{'name': 'Ash', 'formula': ash_formula}]
        problem = _build_problem(usable_target(self.ASH_TARGET)[0], materials, 1.0)
        return problem, _solve_material_set(materials, problem)

    def test_a_key_without_a_molar_mass_is_not_an_oxide_of_the_problem(self):
        for key in ('Loi', 'LOI'):
            self.assertNotIn(key, self.problem['oxides'])
            self.assertNotIn(key, self.problem['full_target'])
            self.assertNotIn(key, self.problem['unlisted'])

    def test_every_row_weight_is_priced_by_a_molar_mass(self):
        """
        The symptom the bug would have shown: a row at 1.0 among rows at 0.006

        Stated as the invariant rather than as a ceiling, because a ceiling is
        the wrong test here - fluorine at the scale floor legitimately reaches
        0.5264, so "below 0.5" would fail on a fluorine-bearing shelf for the
        right chemistry and the wrong reason. Every row has to BE
        penalty / (molar mass * scale); a row that is not is a key that got
        through without a mass.
        """
        molar_masses = load_molar_masses()
        problem = self.problem

        self.assertTrue(len(problem['row_weights']) > 0)
        for oxide, weight in zip(problem['oxides'], problem['row_weights']):
            penalty = 1.0 if oxide in problem['target_umf'] else problem['unlisted_weight']
            scale = max(problem['full_target'][oxide], OXIDE_SCALE_FLOOR)
            self.assertAlmostEqual(weight, penalty / (molar_masses[oxide] * scale), places=12,
                                   msg=f"{oxide} is not priced by its molar mass")

    def test_an_unanalysed_key_does_not_change_the_answer(self):
        """
        The general case, and the one that showed the damage

        The two shelves differ by one unanalysed key on one material. Before the
        rule was the molar mass, that key got a row at 1.0 against chemical rows
        of 0.0055 to 0.0705, the fit went to minimizing it, and the only carrier
        of the requested P2O5 was thrown out whole: the answer became
        {Kaolin, Silica, Whiting} at an objective of 0.4800 instead of
        {Ash, Kaolin, Silica} at 0.0117, a 41-fold regression announced by one
        log line.
        """
        clean_problem, clean = self.solve_ash(self.CLEAN_ASH)
        dirty_problem, dirty = self.solve_ash(self.DIRTY_ASH)

        self.assertEqual(clean_problem['oxides'], dirty_problem['oxides'])
        self.assertEqual(clean['recipe'], dirty['recipe'])
        self.assertAlmostEqual(clean['objective_error'], dirty['objective_error'], places=12)
        self.assertIn('Ash', dirty['recipe'], "the only carrier of the requested P2O5 was dropped")

    def test_the_residual_and_the_focus_oxide_never_see_it(self):
        """
        The second route into the fit, which the problem-level tests miss

        calculate_recipe_composition keeps every key of every formula, so the
        actual side of the residual arrives carrying Loi whatever _expand_target
        did. Taking the union of the two sides put it back: measured on this
        very fixture, the residual held Loi -10.1 and LOI -4.4 against a worst
        real oxide of +10.0, and _focus_oxide picked "Loi" as the oxide of the
        step - so the whole candidate ranking of that step was about which
        material burns off least.
        """
        state = _solve_material_set(self.MATERIALS, self.problem)
        residual = _weight_residual(self.problem, state)

        self.assertEqual(sorted(residual), sorted(self.problem['oxides']))
        self.assertIn(_focus_oxide(residual), set(self.problem['oxides']))

    def test_the_residual_is_normalized_over_the_oxides_alone(self):
        """
        Order matters: filter first, normalize second

        Scaling the actual composition to 100 with the loss on ignition still in
        it divides every real oxide by a total 10-15% too large, which pushes
        EVERY residual up at once - a systematic deficit the search then chases.
        This fixture is solved exactly, so every residual has to be ~0; with the
        old order SiO2 alone read +10.0.
        """
        state = _solve_material_set(self.MATERIALS, self.problem)
        residual = _weight_residual(self.problem, state)

        for oxide, gap in residual.items():
            self.assertAlmostEqual(gap, 0.0, delta=0.05, msg=f"{oxide} residual is {gap}")

    def test_writing_the_loss_out_or_not_does_not_change_the_score(self):
        """
        The key is invisible to the score in BOTH directions

        It contributes to none of the four terms - it used to land in
        "disturbance", because residual.get(key, 0.0) is 0.0 and 0.0 is inside
        MATCHED_OXIDE_TOLERANCE, so every material was penalized in proportion
        to how much of it burns off - and it does not move the denominator
        either, because the denominator is 100, the basis the analysis is stated
        on, not the sum of the analysis.

        Those two are the same requirement seen twice: whiting with its loss
        written out and whiting without it are THE SAME MATERIAL described two
        ways, and they have to score the same. Under the old denominator they
        did not - the sum was 56.1 in one case and 100 in the other, so the
        CaO fraction came out 1.0 against 0.561 and the analysis that omits its
        loss looked like pure calcium oxide.

        Pure lime IS a different material and must score differently, which is
        what keeps this test from passing for the trivial reason.
        """
        state = _solve_material_set(self.MATERIALS, self.problem)
        residual = _weight_residual(self.problem, state)
        focus = _focus_oxide(residual)

        spelled_out = _score_candidate({'name': 'Whiting', 'formula': {'CaO': 56.1, 'Loi': 43.9}},
                                       residual, focus)
        implied = _score_candidate({'name': 'Whiting', 'formula': {'CaO': 56.1}},
                                   residual, focus)
        quicklime = _score_candidate({'name': 'Quicklime', 'formula': {'CaO': 100.0}},
                                     residual, focus)

        self.assertAlmostEqual(spelled_out, implied, places=12,
                               msg="the same material scores differently depending on whether "
                                   "its loss on ignition is written out")
        self.assertNotAlmostEqual(implied, quicklime, places=6,
                                  msg="whiting and quicklime are not the same material")

    def test_the_denominator_is_the_weighed_basis_and_not_the_analysis_sum(self):
        """
        Stated as the arithmetic, because the bias it removes is a ratio

        Only 65 of the 179 analysed materials of the shipped database have an
        analysis that adds up to 100, so for the other 114 the old denominator
        inflated every fraction by 1 / (oxide sum): whiting by 1.783, dolomite
        by 1.926, borax by 1.894, quartz by 1.000. The score therefore preferred
        exactly the materials that lose the most in the kiln - a bias in the
        ranking, and NOT a measured loss of quality: A/B'd over the 300 case
        corpus it moves one answer of 300 under 'heuristic' and none at all
        under 'exhaustive' (the numbers are in _score_candidate). What the test
        pins is the arithmetic, which is wrong or right regardless.
        """
        residual = {'CaO': 10.0}
        score = _score_candidate({'name': 'Whiting', 'formula': {'CaO': 56.1}}, residual, 'CaO')

        # focus gain = residual * fraction, and the fraction has to be 0.561
        self.assertAlmostEqual(score, 10.0 * 0.561, places=12)

    def test_the_carbonate_is_not_penalized_for_burning_off(self):
        """Whiting is the only CaO source, so a Loi row could only hurt it"""
        state = _solve_material_set(self.MATERIALS, self.problem)

        self.assertIsNotNone(state)
        self.assertIn('Whiting', state['recipe'])

    def test_the_whole_search_is_unmoved_by_the_key(self):
        """
        End to end through the documented materials= seam, so that every place
        that reads a formula is on the path: _expand_state and _rank_candidates
        as well as _build_problem and _solve_material_set.
        """
        clean = find_best_recipe(None, self.ASH_TARGET, max_solutions=5,
                                 materials=self.ASH_BASE + [{'name': 'Ash',
                                                             'formula': self.CLEAN_ASH}])
        dirty = find_best_recipe(None, self.ASH_TARGET, max_solutions=5,
                                 materials=self.ASH_BASE + [{'name': 'Ash',
                                                             'formula': self.DIRTY_ASH}])

        self.assertTrue(clean)
        self.assertEqual([s['recipe'] for s in clean], [s['recipe'] for s in dirty])
        self.assertEqual([round(s['objective_error'], 12) for s in clean],
                         [round(s['objective_error'], 12) for s in dirty])


class TestTheRuleOnAShippedCarrier(unittest.TestCase):
    """
    The same rule, on a material nobody invented for the occasion

    Every fixture above carries its own synthetic Loi, and so does every other
    automated run in this repository - which means that until this class existed
    the rule had no witness on real data at all. The reason is worth writing
    down, because it is a property of the shipped database rather than an
    oversight of the tests: three of the 216 materials of database/materials.json
    carry a Loi key and all three are inInventory: false, while
    tests/test_individual_recipes.py and corpus scenario B build their inventory
    from that very flag and bench/corpus strips LOI out of every Glazy formula
    before it makes a case. Nothing that runs reaches a real carrier by accident.

    So this class reaches one on purpose, through the documented materials= seam
    and end to end, and asserts the two halves of the rule that the shipped data
    can show: the key gets no row of the fit, and a material scores the same
    whether or not its analysis writes the loss out.
    """

    # A real material with a real loss on ignition, chosen for being an ordinary
    # glaze ingredient rather than for being convenient: metakaolin brings the
    # Al2O3 of the target below and is the only carrier of it on this shelf.
    CARRIER = 'Метакаолин BMK-45'
    SHELF = (CARRIER, 'Кварцевая мука Кварцверке W12', 'Мел, CaCO3')
    TARGET = {'SiO2': 3.0, 'Al2O3': 0.4, 'CaO': 0.7}

    @classmethod
    def setUpClass(cls):
        cls.shipped = {material['name']: material
                       for material in load_materials(only_inventory=False, priority=False)}

    def shelf(self, strip_loss=False):
        materials = []
        for name in self.SHELF:
            material = dict(self.shipped[name])
            formula = dict(material['formula'])
            if strip_loss:
                formula.pop('Loi', None)
            material['formula'] = formula
            materials.append(material)
        return materials

    def test_the_premise_the_rest_of_this_class_rests_on(self):
        """
        The witness is real and it is unreachable any other way

        Stated as assertions rather than as prose because every part of it is a
        property of a data file this repository keeps editing. If the carrier
        loses its Loi key the class below has stopped testing anything; if ANY
        shipped carrier enters the default inventory, the comments in
        solver_iterative and the docstrings here - which say "all three are
        inInventory: false", and conclude from that that no measurement rig can
        see this rule - have gone stale. The last assertion covers what those
        comments actually claim, not just the one material this class uses,
        because the claim is about the database and not about the fixture.

        The refusal is spelled with _known_oxide and not with `key in
        molar_masses`, because the second is the WEAKER of the two writings the
        predicate was extracted to unify - it misses a mass of zero or NaN. On
        the shipped table the two agree; a test that reproduces the weak writing
        is a test that would keep passing through exactly the divergence
        _known_oxide exists to prevent.
        """
        molar_masses = load_molar_masses()
        material = self.shipped.get(self.CARRIER)

        def refused(entry):
            return [key for key in (entry.get('formula') or {})
                    if not _known_oxide(key, molar_masses)]

        self.assertIsNotNone(material, f"{self.CARRIER} is gone from the shipped database")
        self.assertTrue(refused(material),
                        f"{self.CARRIER} no longer carries a key without a molar mass, so this "
                        f"class has lost its witness - point it at another shipped carrier")

        stocked = sorted(name for name, entry in self.shipped.items()
                         if refused(entry) and entry.get('inInventory'))

        self.assertFalse(stocked,
                         f"{stocked} carry a key without a molar mass AND are in the default "
                         f"inventory, so the rule is now reached by the ordinary runs - the "
                         f"comments in solver_iterative and above that say otherwise are stale")

    def test_the_loss_of_a_shipped_material_gets_no_row(self):
        solutions = find_best_recipe(None, self.TARGET, max_solutions=3,
                                     materials=self.shelf())

        self.assertTrue(solutions)
        best = solutions[0]
        self.assertIn(self.CARRIER, best['recipe'],
                      "the carrier is not in the answer, so nothing here was exercised")
        # effective_target_umf IS the oxide set of the fit, so a key missing
        # from it is a key with no NNLS row and no term of the objective
        self.assertNotIn('Loi', best['effective_target_umf'])
        self.assertNotIn('Loi', best['result_umf'])

    def test_a_shipped_answer_does_not_depend_on_the_loss_being_written_out(self):
        """
        The consistency statement, on the database as shipped: metakaolin with
        its 1% loss listed and metakaolin without it are the same material
        described two ways, and the search may not tell them apart.
        """
        with_loss = find_best_recipe(None, self.TARGET, max_solutions=3,
                                     materials=self.shelf())
        without_loss = find_best_recipe(None, self.TARGET, max_solutions=3,
                                        materials=self.shelf(strip_loss=True))

        self.assertTrue(with_loss)
        self.assertEqual([s['recipe'] for s in with_loss],
                         [s['recipe'] for s in without_loss])
        self.assertEqual([round(s['objective_error'], 12) for s in with_loss],
                         [round(s['objective_error'], 12) for s in without_loss])


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
        """
        Measured RELATIVELY, which is what "hits" means here

        The tolerance used to be an absolute 0.02 of UMF, which on the SiO2 of
        3.0 in this target is 0.67% and on the Na2O of 0.3 is 6.7% - one numeral
        asking for two very different things. The solver does not promise
        either: it optimizes the relative deviation and stops caring inside
        OBJECTIVE_DEADBAND, so every requested oxide comes back about 2% low
        here (the recipe brings unrequested fluxes, which is exactly what
        penalize_unlisted=0.0 permits, and they inflate the unity denominator).
        The gate is the tolerance the whole system judges by.
        """
        solution = self.best_error(penalize_unlisted=0.0)
        deviation = umf_deviation(PARTIAL_TARGET,
                                  {oxide: solution['result_umf'].get(oxide, 0.0)
                                   for oxide in PARTIAL_TARGET})

        self.assertLessEqual(deviation['max_relative'], CHEMISTRY_TOLERANCE,
                             msg=f"{deviation['worst_oxide']} missed the target")

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

    What keeps the colourant here is the sole carrier rule, not the size of the
    error. Cobalt happens to be a flux, so its removal drags the unity
    denominator and costs ten tolerances, which once looked like proof that the
    tolerance protects colourants - it is not, and
    tests/test_solver_inverse.py TestSoleCarrierRule holds the counterexample
    with a colourant that is not a flux.

    The target is built by the forward calculation from a recipe that really
    needs the colourant, so the test states its own premise instead of trusting
    a number pasted from somewhere.

    THE SECOND WAY A TRACE INGREDIENT USED TO GO MISSING IS GONE, and the
    paragraph that used to be here described it: at max_solutions=1 the same
    target came back without the colourant, because the colourant-free answer
    scored 0.0575 against a default error_threshold of 0.1, so the branch was
    declared converged and the search stopped before the colourant was ever
    tried. Only error_threshold=0.01 recovered it.

    That was the absolute objective failing to see a trace oxide, which is the
    defect the relative objective exists to fix. Re-measured: the same
    colourant-free answer now scores 0.1300, because CoO is missed by 0.15 of
    its own scale and only the deadband comes off that. 0.1300 is above the
    default threshold, the branch is not declared converged, and the colourant
    comes back at max_solutions=1 with every argument at its default -
    test_the_colourant_is_found_at_max_solutions_one below pins exactly that.
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

    def test_the_colourant_is_found_at_max_solutions_one(self):
        """
        The regression of the acceptance-rule failure, not of the weight floor

        Everything at its default, including max_solutions=1, where the beam
        width and the number of children are both 1 and the early-convergence
        rule has the most room to fire. It cannot fire here any more: the
        colourant-free branch is at 0.13 on the relative objective, above the
        0.1 threshold, so the search keeps going and finds the cobalt.
        """
        solutions = find_best_recipe(self.coloured_inventory, self.target, max_solutions=1)

        self.assertTrue(solutions)
        self.assertIn(self.COBALT, solutions[0]['recipe'],
                      f"the colourant was dropped at max_solutions=1: {solutions[0]['recipe']}")

    def test_the_colourant_free_branch_is_no_longer_acceptable(self):
        """The premise of the test above, measured rather than asserted"""
        without = find_best_recipe(self.inventory, self.target, max_solutions=1)

        self.assertTrue(without)
        self.assertNotIn(self.COBALT, without[0]['recipe'])
        # the absolute norm called this answer good; the relative one does not
        self.assertLess(without[0]['error'], 0.1)
        self.assertGreater(without[0]['objective_error'], 0.1)

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
