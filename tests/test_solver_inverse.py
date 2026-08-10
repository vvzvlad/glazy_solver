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
from solver_iterative import (
    PRUNE_ERROR_TOLERANCE,
    PRUNE_OBJECTIVE_TOLERANCE,
    _build_problem,
    _prune_solution,
    _sole_carriers,
    _solve_material_set,
    find_best_recipe,
    usable_target,
)

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
    The answer must not stand on a compensating pair

    One engine holds this and one does not, and the difference is the whole
    point of the pair: the iterative solver gained a backward elimination pass
    (solver_iterative._prune_solution) and stopped splitting the feldspar mass
    across the twins; the classic solver never learned to drop a material and
    still does. Its test therefore stays an expected failure - the threshold is
    right and the solver is wrong, not the other way round.
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

        The iterative counterpart below used to carry the same decorator and no
        longer does, so the fix is known and it is not a threshold change: what
        solve_recipe() lacks is a pass that asks whether a material can be
        removed without making the fit worse. Its own x[i] > 0.1 filter cannot
        answer that question - both twins weigh far more than 0.1%. When it grows
        one this test will report an UNEXPECTED SUCCESS. That is the signal to
        delete this decorator, not to delete the test.
        """
        recipe, _ = self.solve_classic()

        self.assertLessEqual(self.condition_number(recipe), MAX_CONDITION_NUMBER)

    def test_iterative_does_not_stand_on_a_compensating_pair(self):
        """
        Held by the pruning pass of the iterative solver - measured cond = 14.8.

        The search still starts from the degenerate answer: all six materials of
        the fixture share one priority, so _priority_start_set takes the whole
        group at once, the beam only ever ADDS materials and _shrink_to_limit
        only fires above max_materials, which six materials never reach. What
        changed is what happens to a solved state on the way out.
        solver_iterative._prune_solution() re-solves the recipe once per
        material with that material left out and accepts the removal whenever
        the objective error grows by no more than PRUNE_OBJECTIVE_TOLERANCE
        (and the reported error by no more than PRUNE_ERROR_TOLERANCE - two
        different gates in two different units, see _prune_solution). One of
        the two twins is exactly that kind of material - the other one can carry
        its share almost unchanged - so it goes, and the answer collapses onto
        the well conditioned four material recipe that was always available:

            before  Variant A 11.78, Variant B 27.92, Silica 30.27,
                    Whiting 20.01, Kaolin 10.02   err 0.0     cond 1849.6
            after   Variant B 39.29, Silica 30.68,
                    Whiting 20.01, Kaolin 10.02   err 0.0033  cond 14.8

        Note what the trade actually is: the fit gets very slightly WORSE (0.0
        to 0.0033, three thousandths of a UMF unit, invisible in a fired glaze)
        and the recipe gets two orders of magnitude better conditioned. A
        threshold on the weight of a material could not have made that trade -
        both twins weigh double digits.

        MAX_CONDITION_NUMBER stays at 1e3 untouched, with 14.8 on one side of it
        and the classic solver's 1849.6 on the other.
        """
        recipe, _ = self.solve_iterative()

        self.assertLessEqual(self.condition_number(recipe), MAX_CONDITION_NUMBER)


class TestPruningPass(CollinearFeldsparsMixin, unittest.TestCase):
    """
    solver_iterative._prune_solution() on the fixture that motivated it

    The conditioning test above pins the OUTCOME of pruning through the public
    entry point. These tests pin the rule itself: what may be removed, how much
    that is allowed to cost, and what stops it.
    """

    def setUp(self):
        self.problem = _build_problem(usable_target(self.target_umf)[0], self.materials, 1.0)
        self.unpruned = _solve_material_set(self.materials, self.problem)
        self.assertIsNotNone(self.unpruned, "the fixture produced no recipe at all")

    def test_the_unpruned_answer_really_does_carry_a_redundant_material(self):
        """
        Guards the premise: without this the two tests below prove nothing

        NNLS over the whole catalogue splits the feldspar mass across the two
        near-identical twins, and neither of them is small - the redundancy is
        structural, not a rounding artefact, which is exactly why a threshold on
        the weight of a material cannot see it.
        """
        recipe = self.unpruned['recipe']

        self.assertIn('Custer Feldspar Variant A', recipe)
        self.assertIn('Custer Feldspar Variant B', recipe)
        for twin in ('Custer Feldspar Variant A', 'Custer Feldspar Variant B'):
            self.assertGreater(recipe[twin], 1.0, f"{twin} is not a trace component")

    def test_a_redundant_material_is_dropped_and_the_fit_barely_moves(self):
        pruned = _prune_solution(self.unpruned, self.problem, 1)

        self.assertLess(pruned['materials_count'], self.unpruned['materials_count'])
        self.assertEqual(len(pruned['recipe']), pruned['materials_count'])

        removed = set(self.unpruned['recipe']) - set(pruned['recipe'])
        self.assertEqual(len(removed), 1, f"expected exactly one removal, got {sorted(removed)}")
        self.assertTrue(removed <= {'Custer Feldspar Variant A', 'Custer Feldspar Variant B'},
                        f"pruned something other than a feldspar twin: {sorted(removed)}")

        # The rule is per removal, so one removal may cost one tolerance. The
        # OBJECTIVE tolerance: this is the objective, and the two halves of the
        # old single constant are not interchangeable just because they happen
        # to carry the same numeral today
        growth = pruned['objective_error'] - self.unpruned['objective_error']
        self.assertLessEqual(growth, PRUNE_OBJECTIVE_TOLERANCE * len(removed) + 1e-12,
                             f"the objective grew by {growth}")
        self.assertAlmostEqual(sum(pruned['recipe'].values()), 100.0, places=6)

    def test_a_needed_material_survives_the_pass(self):
        """Silica, whiting and kaolin are the only source of their oxides"""
        pruned = _prune_solution(self.unpruned, self.problem, 1)

        for material in ('Silica', 'Whiting', 'Kaolin'):
            self.assertIn(material, pruned['recipe'])

    def test_pruning_never_goes_below_min_materials(self):
        """
        The floor is a hard bound, and on this fixture it really binds

        min_materials=5 is the count of the unpruned answer, so the one removal
        that would otherwise happen is not allowed and the degenerate recipe is
        returned untouched - a worse recipe, but the one the caller asked for.
        """
        for floor in (1, 2, 3, 4, 5):
            with self.subTest(min_materials=floor):
                pruned = _prune_solution(self.unpruned, self.problem, floor)
                self.assertGreaterEqual(pruned['materials_count'], min(floor, self.unpruned['materials_count']))

        blocked = _prune_solution(self.unpruned, self.problem, self.unpruned['materials_count'])
        self.assertEqual(blocked['recipe'], self.unpruned['recipe'])

    def test_a_min_materials_of_zero_still_leaves_one_material(self):
        """An empty recipe is not a recipe, whatever the caller asked for"""
        self.assertGreaterEqual(_prune_solution(self.unpruned, self.problem, 0)['materials_count'], 1)

    def test_the_floor_reaches_the_public_entry_point(self):
        """The same two answers, through find_best_recipe rather than by hand"""
        free = find_best_recipe(None, self.target_umf, materials=self.materials,
                                min_materials=1, max_solutions=1)
        floored = find_best_recipe(None, self.target_umf, materials=self.materials,
                                   min_materials=5, max_solutions=1)

        self.assertEqual(free[0]['materials_count'], 4)
        self.assertEqual(floored[0]['materials_count'], 5)


class TestSoleCarrierRule(unittest.TestCase):
    """
    The structural half of _prune_solution: what may never be removed

    _sole_carriers answers one question - "is this material the only thing in
    the recipe supplying an oxide the target asked for" - and the answer is what
    protects colourants. No quantity takes part in it, which is the whole point:
    a colourant works optically and contributes almost nothing to the chemistry,
    so a threshold on chemical error cannot recognise one at any calibration.
    """

    SILICA = {'name': 'Silica', 'formula': {'SiO2': 100.0}}
    WHITING = {'name': 'Whiting', 'formula': {'CaO': 56.1}}
    KAOLIN = {'name': 'Kaolin', 'formula': {'Al2O3': 40.21, 'SiO2': 47.29}}
    COBALT = {'name': 'Cobalt Carbonate', 'formula': {'CoO': 63.0}}
    CHROME = {'name': 'Chrome Oxide', 'formula': {'Cr2O3': 100.0}}

    def test_the_only_source_of_a_requested_oxide_is_protected(self):
        used = [self.SILICA, self.WHITING, self.KAOLIN, self.COBALT]
        target = {'SiO2': 3.0, 'Al2O3': 0.4, 'CaO': 1.0, 'CoO': 0.015}

        # Cobalt for CoO, whiting for CaO, kaolin for Al2O3; SiO2 has three
        # sources, so nothing is protected on its account
        self.assertEqual(_sole_carriers(used, target),
                         {'Cobalt Carbonate', 'Whiting', 'Kaolin'})

    def test_a_second_source_removes_the_protection(self):
        both = [self.SILICA, self.WHITING, self.KAOLIN, self.COBALT,
                {'name': 'Cobalt Oxide', 'formula': {'CoO': 93.35}}]
        target = {'SiO2': 3.0, 'Al2O3': 0.4, 'CaO': 1.0, 'CoO': 0.015}

        carriers = _sole_carriers(both, target)
        self.assertNotIn('Cobalt Carbonate', carriers)
        self.assertNotIn('Cobalt Oxide', carriers)

    def test_an_oxide_requested_as_zero_protects_nobody(self):
        """"none of this" is a reason to drop the carrier, not to keep it"""
        used = [self.SILICA, self.WHITING, self.CHROME]

        self.assertNotIn('Chrome Oxide', _sole_carriers(used, {'SiO2': 3.0, 'Cr2O3': 0.0}))

    def test_an_oxide_the_target_never_mentions_protects_nobody(self):
        """An unlisted oxide is contamination by definition"""
        used = [self.SILICA, self.WHITING, self.CHROME]

        self.assertNotIn('Chrome Oxide', _sole_carriers(used, {'SiO2': 3.0}))

    def test_a_non_flux_colourant_survives_a_removal_the_tolerance_would_allow(self):
        """
        The case that the error tolerance alone gets wrong

        Chrome oxide is not a flux, so losing it does not drag the unity
        denominator the way losing cobalt does, and its whole contribution to
        the search objective is smaller than PRUNE_OBJECTIVE_TOLERANCE. On the
        numbers alone the removal is free. On the question the rule actually
        asks - is anything else in this recipe carrying the Cr2O3 the target
        asked for - it is not free at all.
        """
        materials = [self.SILICA, self.WHITING, self.KAOLIN, self.CHROME]
        recipe = {'Silica': 30.0, 'Whiting': 25.0, 'Kaolin': 44.85, 'Chrome Oxide': 0.15}
        target = forward_umf(materials, recipe)

        problem = _build_problem(usable_target(target)[0], materials, 1.0)
        state = _solve_material_set(materials, problem)
        self.assertIn('Chrome Oxide', state['recipe'], "the fixture lost the colourant before pruning")

        # The premise, and it takes BOTH gates: a removal passes only when it
        # costs at most PRUNE_OBJECTIVE_TOLERANCE on the objective AND at most
        # PRUNE_ERROR_TOLERANCE on the reported error, so asserting one of them
        # leaves the test able to pass for the wrong reason - drop the other
        # gate below what this removal costs and the colourant would survive
        # because the removal was refused on price, not because it carries the
        # Cr2O3. Measured on this fixture: 0.0200 of objective against a gate of
        # 0.03 and 0.0030 of error against a gate of 0.03, both open.
        without = _solve_material_set(
            [m for m in materials if m['name'] != 'Chrome Oxide'], problem)
        self.assertLessEqual(without['objective_error'],
                             state['objective_error'] + PRUNE_OBJECTIVE_TOLERANCE,
                             "this colourant is no longer a counterexample to the tolerance")
        self.assertLessEqual(without['error'],
                             state['error'] + PRUNE_ERROR_TOLERANCE,
                             "the error gate now refuses the removal on its own, so this test "
                             "no longer shows that the sole carrier rule is what keeps the "
                             "colourant")

        # ...and it stays anyway
        self.assertIn('Chrome Oxide', _prune_solution(state, problem, 1)['recipe'])


class TestPruningChecksBothErrors(unittest.TestCase):
    """
    The removal test runs on `error` as well as on the search objective

    The two numbers are not the same quantity and no longer even the same units:
    the objective is the L2 of the RELATIVE per-oxide deviations with a
    deadband, bounded by PRUNE_OBJECTIVE_TOLERANCE, while `error` is the
    absolute L2 the caller receives, bounded by PRUNE_ERROR_TOLERANCE. A removal
    can therefore be cheap on one and dear on the other, and only the second
    gate bounds what the caller is handed.

    The mechanism that used to produce the divergence is still named in
    _prune_solution - with penalize_unlisted > 0 the objective folds in the
    contamination of the unlisted oxides, which a removal can shrink to pay for
    a rise in `error` - but it is weaker than it was, because that contamination
    term is on the relative scale now too. What is easy to build instead, and
    what the fixture below builds, is the plain difference of scale: a removal
    that walks a big oxide by a percent of ITSELF is nearly free on the relative
    objective and expensive on the absolute norm.
    """

    def setUp(self):
        self.materials = [
            {'name': 'Silica', 'formula': {'SiO2': 100.0}},
            {'name': 'Whiting', 'formula': {'CaO': 56.1}},
            {'name': 'Kaolin', 'formula': {'Al2O3': 40.0, 'SiO2': 47.0}},
            # The trap: it carries the requested K2O and a lot of unrequested BaO
            {'name': 'Dirty Feldspar', 'formula': {'K2O': 20.0, 'SiO2': 30.0, 'BaO': 50.0}},
            # ...and this one keeps it from being the SOLE carrier of K2O, so the
            # structural rule stays out of the way and the tolerance has to decide
            {'name': 'Clean Feldspar', 'formula': {'Al2O3': 19.0, 'SiO2': 80.8, 'K2O': 0.2}},
        ]
        self.target = {'SiO2': 3.0, 'Al2O3': 0.5, 'CaO': 0.7, 'K2O': 0.1}
        self.problem = _build_problem(usable_target(self.target)[0], self.materials, 1.0)
        self.state = _solve_material_set(self.materials, self.problem)
        self.assertIsNotNone(self.state)

    def test_the_two_error_numbers_really_diverge_here(self):
        """Without this the test below would pass for the wrong reason"""
        self.assertGreater(self.state['objective_error'], self.state['error'])
        self.assertNotIn('Dirty Feldspar',
                         _sole_carriers([m for m in self.materials
                                         if m['name'] in self.state['recipe']],
                                        usable_target(self.target)[0]))

    def test_a_removal_that_only_looks_free_on_the_objective_is_rejected(self):
        used = [m for m in self.materials if m['name'] in self.state['recipe']]
        tempting = []

        for dropped in used:
            reduced = [m for m in used if m['name'] != dropped['name']]
            candidate = _solve_material_set(reduced, self.problem)
            if candidate is None:
                continue
            cheap_on_objective = (candidate['objective_error']
                                  <= self.state['objective_error'] + PRUNE_OBJECTIVE_TOLERANCE)
            dear_on_error = candidate['error'] > self.state['error'] + PRUNE_ERROR_TOLERANCE
            if cheap_on_objective and dear_on_error:
                tempting.append(dropped['name'])

        self.assertTrue(tempting, "the fixture no longer offers the trade this test is about")

        pruned = _prune_solution(self.state, self.problem, 1)
        for name in tempting:
            self.assertIn(name, pruned['recipe'],
                          f"{name} was removed on the objective while raising the reported error")

    def test_the_reported_error_never_runs_away(self):
        """The invariant the second gate buys: one tolerance per removal, at most"""
        pruned = _prune_solution(self.state, self.problem, 1)
        removals = self.state['materials_count'] - pruned['materials_count']

        self.assertLessEqual(pruned['error'],
                             self.state['error'] + PRUNE_ERROR_TOLERANCE * removals + 1e-12)


if __name__ == '__main__':
    unittest.main()
