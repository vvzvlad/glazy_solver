#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# flake8: noqa
# pylint: disable=broad-exception-raised, raise-missing-from, too-many-arguments, redefined-outer-name
# pylint: disable=multiple-statements, logging-fstring-interpolation, trailing-whitespace, line-too-long
# pylint: disable=broad-exception-caught, missing-function-docstring, missing-class-docstring
# pylint: disable=f-string-without-interpolation
# pylance: disable=reportMissingImports, reportMissingModuleSource

import json
import math
import os
import sys
import unittest
import warnings

import numpy as np

# Fix imports by adding parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import flux_oxides, load_materials, load_molar_masses
from feasibility import (DEFAULT_FEASIBILITY_TOL, MAX_CONDITION_NUMBER, MIN_FLUX_SHARE,
                         OXIDE_SCALE_FLOOR, achievable_ranges, build_molar_matrix,
                         check_feasibility, flux_row, matrix_diagnostics,
                         projected_range_lps, usable_oxides)

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fixtures')

# A synthetic material set, used wherever the point of the test is an exact
# number rather than the real database. Three materials, one flux, and exactly
# one source of alumina - which also carries iron, so that "the iron comes with
# the materials whether anyone likes it or not" can be shown in figures.
SILICA = {"name": "Кремнезём (тест)", "formula": {"SiO2": 100.0}}
CHALK = {"name": "Кальцит (тест)", "formula": {"CaO": 100.0}}
IRONY_ALUMINA = {"name": "Глинозём с железом (тест)", "formula": {"Al2O3": 90.0, "Fe2O3": 10.0}}
SYNTHETIC = [SILICA, CHALK, IRONY_ALUMINA]

# A legal record with no oxide in it: 37 of the 216 real materials look like this
WATER = {"name": "Вода (тест)", "formula": {}}


def reference_recipes():
    with open(os.path.join(FIXTURES, 'reference_recipes.json'), 'r', encoding='utf-8') as f:
        return json.load(f)


def recipe_01():
    """The reference "Прозрачная глазурь △6", the first fixture recipe"""
    for recipe in reference_recipes():
        if recipe['id'] == 'recipe_01_transparent_glaze':
            return recipe
    raise AssertionError("recipe_01_transparent_glaze is missing from the fixtures")


def inventory_materials():
    return load_materials(only_inventory=True, priority=False)


def row_for(result, oxide):
    for row in result['per_oxide']:
        if row['oxide'] == oxide:
            return row
    raise AssertionError(f"{oxide} is missing from per_oxide: "
                         f"{[row['oxide'] for row in result['per_oxide']]}")


def all_finite(obj):
    """True when no float anywhere in the structure is NaN or infinite"""
    if isinstance(obj, dict):
        return all(all_finite(value) for value in obj.values())
    if isinstance(obj, (list, tuple)):
        return all(all_finite(value) for value in obj)
    if isinstance(obj, float):
        return math.isfinite(obj)
    return True


class TestMatrix(unittest.TestCase):

    def test_molar_matrix_is_moles_per_100g(self):
        A, names = build_molar_matrix([SILICA, CHALK], ['SiO2', 'CaO'])
        masses = load_molar_masses()

        self.assertEqual(names, [SILICA['name'], CHALK['name']])
        self.assertAlmostEqual(A[0, 0], 100.0 / masses['SiO2'])
        self.assertAlmostEqual(A[1, 1], 100.0 / masses['CaO'])
        self.assertAlmostEqual(A[0, 1], 0.0)
        self.assertAlmostEqual(A[1, 0], 0.0)

    def test_material_without_a_formula_gets_no_column(self):
        A, names = build_molar_matrix([SILICA, WATER, CHALK], ['SiO2', 'CaO'])

        self.assertEqual(names, [SILICA['name'], CHALK['name']])
        self.assertEqual(A.shape, (2, 2))

    def test_oxide_without_a_molar_mass_gets_no_row(self):
        # Not a crash and not a silent zero row: the name is dropped, and
        # usable_oxides() is the function that says which names survived
        self.assertEqual(usable_oxides(['SiO2', 'Unobtainium', 'CaO']), ['SiO2', 'CaO'])
        A, _ = build_molar_matrix([SILICA, CHALK], ['SiO2', 'Unobtainium', 'CaO'])
        self.assertEqual(A.shape, (2, 2))

    def test_loi_is_not_an_oxide_row(self):
        self.assertEqual(usable_oxides(['SiO2', 'Loi', 'LOI']), ['SiO2'])

    def test_flux_row_follows_the_classification(self):
        oxides = ['SiO2', 'CaO', 'K2O', 'Al2O3', 'MgO', 'TiO2']
        fluxes = set(flux_oxides())
        expected = np.array([1.0 if oxide in fluxes else 0.0 for oxide in oxides])

        np.testing.assert_array_equal(flux_row(oxides), expected)
        # And the classification is what decides it, not a list written here
        self.assertEqual(flux_row(['CaO'])[0], 1.0)
        self.assertEqual(flux_row(['SiO2'])[0], 0.0)

    def test_diagnostics_on_the_real_inventory(self):
        materials = inventory_materials()
        oxides = sorted({oxide for material in materials
                         for oxide in (material.get('formula') or {})})
        diagnostics = matrix_diagnostics(materials, oxides)

        self.assertEqual(diagnostics['n_oxides'], len(oxides))
        self.assertIsNotNone(diagnostics['cond'])
        self.assertEqual(diagnostics['rank'], min(len(oxides), len(materials)))
        self.assertEqual(diagnostics['ill_conditioned'],
                         diagnostics['cond'] > MAX_CONDITION_NUMBER)

    def test_diagnostics_report_an_unspanned_oxide_as_degenerate(self):
        # Two materials that are the same material, asked about two oxides: the
        # rank is 1 where it could have been 2, so nothing here can move CaO
        twin = {"name": "Кремнезём-двойник (тест)", "formula": {"SiO2": 100.0}}
        diagnostics = matrix_diagnostics([SILICA, twin], ['SiO2', 'CaO'])

        self.assertIsNone(diagnostics['cond'])
        self.assertTrue(diagnostics['ill_conditioned'])
        self.assertEqual(diagnostics['rank'], 1)

    def test_more_materials_than_oxides_is_not_a_defect(self):
        # quality_metrics judges the materials of ONE RECIPE and calls a column
        # beyond the oxide count degenerate; an inventory is over-complete by
        # construction (19 materials, 12 oxides) and must not be condemned for it
        twin = {"name": "Кремнезём-двойник (тест)", "formula": {"SiO2": 100.0}}
        diagnostics = matrix_diagnostics([SILICA, twin], ['SiO2'])

        self.assertIsNotNone(diagnostics['cond'])
        self.assertFalse(diagnostics['ill_conditioned'])


class TestCheckFeasibility(unittest.TestCase):

    def test_reference_recipe_is_reachable_from_the_full_inventory(self):
        target = recipe_01()['umf']
        result = check_feasibility(target, inventory_materials())

        self.assertTrue(result['feasible'])
        self.assertLess(result['max_relative_deviation'], DEFAULT_FEASIBILITY_TOL)
        self.assertEqual(result['unreachable_oxides'], [])
        self.assertEqual(result['why'], {})
        self.assertTrue(all_finite(result))

        # Every oxide of the target is reported, and reported as reached
        for oxide in target:
            row = row_for(result, oxide)
            self.assertTrue(row['reachable'], f"{oxide} unexpectedly out of reach")
            self.assertAlmostEqual(row['delta'], row['closest'] - row['target'])

    def test_closest_recipe_sums_to_100(self):
        result = check_feasibility(recipe_01()['umf'], inventory_materials())
        total = sum(result['closest_recipe'].values())

        self.assertAlmostEqual(total, 100.0, delta=0.1)
        self.assertTrue(all(value > 0 for value in result['closest_recipe'].values()))

    def test_lithium_without_a_lithium_material_is_unreachable_and_says_why(self):
        # Verified against the shipped stock: not one inInventory material
        # carries Li2O, so this is the real "no carrier" case and not a contrived one
        materials = inventory_materials()
        self.assertFalse(any('Li2O' in (material.get('formula') or {})
                             for material in materials))

        target = dict(recipe_01()['umf'])
        target['Li2O'] = 0.5
        result = check_feasibility(target, materials)

        self.assertFalse(result['feasible'])
        self.assertIn('Li2O', result['unreachable_oxides'])
        self.assertIn('Li2O', result['why'])
        self.assertIn('Li2O', result['why']['Li2O'])
        self.assertIn('не содержит', result['why']['Li2O'])

        # And nothing else is named. The min-sum point misses CaO here as well -
        # lithium is a flux, so the fluxes that remain have to cover a unity
        # budget half of which was asked of a material that does not exist - but
        # CaO CAN be held inside the tolerance at the same t*, so it is an
        # artefact of the point and not part of the answer. The LP that decides
        # this is the whole reason unreachable_oxides is not read off the point
        self.assertEqual(result['unreachable_oxides'], ['Li2O'])

    def test_the_verdict_list_is_checked_against_the_range_machinery(self):
        """
        A named oxide is one no point can hold, cross-checked another way

        achievable_ranges(tol=t*) explores exactly the optimal face of the
        verdict - every oxide inside t* * s, contamination inside t* * 0.1 - so
        the interval it reports for an oxide is the full set of values that
        oxide can take there. An oxide whose interval reaches into its ordinary
        tolerance band could have been held and must not be named; one whose
        interval misses the band entirely is genuinely forced. Two different
        code paths, one answer.
        """
        target = dict(recipe_01()['umf'])
        target['Li2O'] = 0.5
        result = check_feasibility(target, inventory_materials())
        t_star = result['max_relative_deviation']
        on_the_face = achievable_ranges(target, inventory_materials(), tol=t_star)

        for row in result['per_oxide']:
            oxide = row['oxide']
            if oxide not in on_the_face['oxide_ranges']:
                continue
            low, high = on_the_face['oxide_ranges'][oxide]
            if high is None:
                high = float('inf')
            wanted = row['target']
            scale = max(wanted, OXIDE_SCALE_FLOOR)
            band_low = wanted - DEFAULT_FEASIBILITY_TOL * scale
            band_high = wanted + DEFAULT_FEASIBILITY_TOL * scale
            holdable = low <= band_high + 1e-9 and high >= band_low - 1e-9

            self.assertEqual(holdable, oxide not in result['unreachable_oxides'],
                             f"{oxide}: reachable on the t* face = {holdable}, "
                             f"named unreachable = {oxide in result['unreachable_oxides']}")

    def test_rows_and_the_verdict_list_agree(self):
        target = dict(recipe_01()['umf'])
        target['Li2O'] = 0.5
        result = check_feasibility(target, inventory_materials())

        named = set(result['unreachable_oxides'])
        for row in result['per_oxide']:
            self.assertEqual(row['reachable'], row['oxide'] not in named)

    def test_per_oxide_is_sorted_by_relative_deviation(self):
        target = dict(recipe_01()['umf'])
        target['Li2O'] = 0.5
        rows = check_feasibility(target, inventory_materials())['per_oxide']

        self.assertEqual([row['relative'] for row in rows],
                         sorted((row['relative'] for row in rows), reverse=True))

    def test_only_the_guilty_oxide_is_named(self):
        # The minimax alone would report every oxide as unreachable: once t* is
        # paid for by the one oxide nobody can make, the others may drift inside
        # it for free. The polish LP is what keeps this list honest. ZrO2 is the
        # clean case - no carrier in the stock AND not a flux, so it does not
        # disturb the unity budget of the others
        target = dict(recipe_01()['umf'])
        target['ZrO2'] = 0.2
        result = check_feasibility(target, inventory_materials())

        self.assertEqual(result['unreachable_oxides'], ['ZrO2'])

    def test_carried_but_not_far_enough_states_the_extreme(self):
        # Strontium is in the stock, but only as the 1% that comes with the
        # ulexite, and that ulexite brings 20 times more flux along with it: the
        # most SrO this inventory can reach is about 0.021
        target = {"SiO2": 3.0, "CaO": 0.8, "SrO": 0.2}
        result = check_feasibility(target, inventory_materials())

        self.assertFalse(result['feasible'])
        self.assertIn('SrO', result['unreachable_oxides'])
        self.assertIn('максимум', result['why']['SrO'])
        self.assertIn('0.02', result['why']['SrO'])

    def test_an_empty_formula_material_changes_nothing(self):
        target = {"SiO2": 3.0, "Al2O3": 0.3, "CaO": 1.0}
        without = check_feasibility(target, SYNTHETIC)
        with_water = check_feasibility(target, SYNTHETIC + [WATER])

        self.assertEqual(with_water['feasible'], without['feasible'])
        self.assertAlmostEqual(with_water['max_relative_deviation'],
                               without['max_relative_deviation'])
        self.assertEqual(with_water['closest_recipe'], without['closest_recipe'])
        self.assertNotIn(WATER['name'], with_water['closest_recipe'])
        self.assertTrue(all_finite(with_water))

    def test_a_target_off_the_unity_basis_is_rescaled_not_condemned(self):
        # The same glaze written with the fluxes summing to 2 is the same glaze
        target = {"SiO2": 3.0, "Al2O3": 0.3, "CaO": 1.0}
        doubled = {oxide: value * 2 for oxide, value in target.items()}

        plain = check_feasibility(target, SYNTHETIC)
        scaled = check_feasibility(doubled, SYNTHETIC)

        self.assertEqual(scaled['feasible'], plain['feasible'])
        self.assertAlmostEqual(scaled['max_relative_deviation'],
                               plain['max_relative_deviation'], places=6)
        self.assertTrue(any('плавн' in warning for warning in scaled['warnings']))

    def test_a_target_without_fluxes_is_refused(self):
        result = check_feasibility({"SiO2": 3.0, "Al2O3": 0.3}, SYNTHETIC)

        self.assertIsNone(result['feasible'])
        self.assertEqual(result['error'], 'no_target_fluxes')

    def test_nothing_raises_out_of_the_module(self):
        # Feasibility is diagnostics: a solver above must keep working whatever
        # is thrown at this
        for target, materials in (({"SiO2": "много"}, SYNTHETIC),
                                  ({}, SYNTHETIC),
                                  ({"SiO2": 3.0, "CaO": 1.0}, []),
                                  ({"SiO2": float('nan'), "CaO": 1.0}, SYNTHETIC),
                                  (None, None)):
            result = check_feasibility(target, materials)
            self.assertIsInstance(result, dict)
            self.assertIn('feasible', result)

    def test_a_nonfinite_analysis_is_reported_not_computed_with(self):
        broken = [{"name": "Битый анализ (тест)", "formula": {"SiO2": float('inf'), "CaO": 10.0}},
                  CHALK]
        result = check_feasibility({"SiO2": 3.0, "CaO": 1.0}, broken)

        self.assertIsNone(result['feasible'])
        self.assertEqual(result['error'], 'nonfinite_analysis')

    def test_the_whole_database_does_not_raise_numpy_warnings(self):
        # numpy 2.2 reports the floating point flags left by the vectorized BLAS
        # kernel as ours: on the 50 x 179 matrix of the whole database "flux @ A"
        # claimed divide-by-zero, overflow and invalid value at once, on operands
        # that are finite and a product correct to 6e-17
        materials = load_materials(only_inventory=False, priority=False)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result = check_feasibility(recipe_01()['umf'], materials)

        self.assertTrue(result['feasible'])
        self.assertEqual([str(item.message) for item in caught
                          if issubclass(item.category, RuntimeWarning)], [])

    def test_impossible_ceilings_are_reported_not_raised(self):
        # Every flux capped at zero contradicts the unity normalization
        ceilings = {oxide: 0.0 for oxide in flux_oxides()}
        result = check_feasibility({"SiO2": 3.0, "CaO": 1.0}, SYNTHETIC, passengers=ceilings)

        self.assertFalse(result['feasible'])
        self.assertEqual(result['closest_recipe'], {})
        self.assertTrue(any('пассажир' in warning for warning in result['warnings']))

    def test_absurd_targets_are_named_and_never_blamed_on_passengers(self):
        # Both of these used to reach the infeasible branch with no passenger in
        # the request and be answered "the passenger ceilings are incompatible
        # with the unity normalization" - a confident sentence about a
        # constraint the caller never wrote. They are now refused by name,
        # before any LP, and the passenger sentence is tied to having any
        for absurd in ({"CaO": 1.0, "SiO2": 1e20}, {"CaO": 1e-300, "SiO2": 3.0}):
            result = check_feasibility(absurd, inventory_materials())

            self.assertEqual(result['error'], 'degenerate_target')
            self.assertFalse(any('пассажир' in warning for warning in result['warnings']))

    def test_a_large_but_workable_target_is_still_answered(self):
        # The guard is at 1e6 and must not swallow a target that is merely
        # extreme: a silica of 1e5 per unity is reachable, absurd as it is
        result = check_feasibility({"CaO": 1.0, "SiO2": 1e5}, inventory_materials())

        self.assertTrue(result['feasible'])

    def test_the_verdict_does_not_hang_on_the_polish_slack(self):
        # A caller passing tol exactly equal to t* used to be told "unreachable"
        # by the verdict and "feasible" by the ranges, because the polish spent
        # its whole absolute slack and pushed the reported worst to t* + 1e-6
        target = dict(recipe_01()['umf'])
        target['Li2O'] = 0.5
        t_star = check_feasibility(target, inventory_materials())['max_relative_deviation']

        verdict = check_feasibility(target, inventory_materials(), tol=t_star)
        ranges = achievable_ranges(target, inventory_materials(), tol=t_star)

        self.assertTrue(verdict['feasible'])
        self.assertTrue(ranges['feasible'])
        self.assertEqual(verdict['max_relative_deviation'], t_star)

    def test_a_negative_content_is_not_reported_as_a_missing_oxide(self):
        # A hand edited analysis with Li2O: -5 is a carrier as far as the
        # arithmetic goes, so "no material of the set contains it" is a lie
        materials = [{"name": "Отрицательный (тест)",
                      "formula": {"Li2O": -5.0, "SiO2": 50.0, "CaO": 20.0}}, CHALK]
        result = check_feasibility({"SiO2": 3.0, "CaO": 1.0, "Li2O": 0.2}, materials)

        self.assertIn('Li2O', result['unreachable_oxides'])
        self.assertNotIn('не содержит', result['why']['Li2O'])


class TestPassengers(unittest.TestCase):
    """
    The one point of the parameter, in numbers

    The set has a single alumina source and it carries 10% iron, so asking for
    Al2O3 0.3 brings in Fe2O3 0.0213 whether anyone wants it or not.
    """

    TARGET = {"SiO2": 3.0, "Al2O3": 0.3, "CaO": 1.0}

    def test_an_unlisted_oxide_is_driven_to_zero_and_costs_the_target(self):
        result = check_feasibility(self.TARGET, SYNTHETIC)

        # Iron nobody asked for is measured against zero, so the LP gives up
        # alumina to avoid it: 0.247 instead of the 0.3 that was asked for
        self.assertFalse(result['feasible'])
        self.assertAlmostEqual(result['max_relative_deviation'], 0.1755, places=3)
        self.assertAlmostEqual(row_for(result, 'Al2O3')['closest'], 0.2474, places=3)
        self.assertIn('Fe2O3', result['unreachable_oxides'])

    def test_the_same_oxide_as_a_passenger_costs_nothing(self):
        result = check_feasibility(self.TARGET, SYNTHETIC, passengers={"Fe2O3": 0.05})

        # The target is hit exactly, and the iron is not in the deviation at all
        self.assertTrue(result['feasible'])
        self.assertAlmostEqual(result['max_relative_deviation'], 0.0, places=9)
        self.assertAlmostEqual(row_for(result, 'Al2O3')['closest'], 0.3, places=6)
        self.assertNotIn('Fe2O3', [row['oxide'] for row in result['per_oxide']])

    def test_a_passenger_is_not_driven_to_its_ceiling(self):
        result = check_feasibility(self.TARGET, SYNTHETIC, passengers={"Fe2O3": 0.05})
        passenger = result['passengers'][0]

        self.assertEqual(passenger['oxide'], 'Fe2O3')
        self.assertEqual(passenger['limit'], 0.05)
        self.assertTrue(passenger['within_limit'])
        # It lands where the chemistry puts it, well below the ceiling
        self.assertAlmostEqual(passenger['closest'], 0.02128, places=4)
        self.assertLess(passenger['closest'], 0.05)

    def test_the_same_number_as_an_ordinary_target_makes_it_worse(self):
        # Chasing a value nobody chose: the LP now overshoots the alumina to
        # reach the iron, which is exactly how a junk component is born
        as_target = check_feasibility(dict(self.TARGET, Fe2O3=0.05), SYNTHETIC)
        as_passenger = check_feasibility(self.TARGET, SYNTHETIC, passengers={"Fe2O3": 0.05})

        self.assertFalse(as_target['feasible'])
        self.assertTrue(as_passenger['feasible'])
        self.assertGreater(as_target['max_relative_deviation'],
                           as_passenger['max_relative_deviation'])
        self.assertGreater(row_for(as_target, 'Al2O3')['closest'], 0.35)

    def test_a_passenger_wins_over_the_same_oxide_in_the_target(self):
        result = check_feasibility(dict(self.TARGET, Fe2O3=0.05), SYNTHETIC,
                                   passengers={"Fe2O3": 0.05})

        self.assertTrue(result['feasible'])
        self.assertEqual([row['oxide'] for row in result['passengers']], ['Fe2O3'])
        self.assertTrue(any('пассажир' in warning for warning in result['warnings']))


class TestAchievableRanges(unittest.TestCase):

    def test_ranges_around_the_reference_recipe(self):
        target = recipe_01()['umf']
        materials = inventory_materials()
        result = achievable_ranges(target, materials)

        self.assertTrue(result['feasible'])
        # 1 probe + 2 per oxide + 2 per material
        self.assertEqual(result['lp_count'],
                         1 + 2 * len(result['oxide_ranges']) + 2 * len(result['material_ranges']))
        self.assertEqual(len(result['material_ranges']), len(materials))

        for oxide, value in target.items():
            low, high = result['oxide_ranges'][oxide]
            scale = max(value, OXIDE_SCALE_FLOOR)
            # The interval is non empty, contains the target, and cannot leave
            # the box the verdict draws around it
            self.assertLessEqual(low, high)
            self.assertLessEqual(low, value + 1e-9)
            self.assertGreaterEqual(high, value - 1e-9)
            self.assertGreaterEqual(low, value - DEFAULT_FEASIBILITY_TOL * scale - 1e-9)
            self.assertLessEqual(high, value + DEFAULT_FEASIBILITY_TOL * scale + 1e-9)

        for low, high in result['material_ranges'].values():
            self.assertGreaterEqual(low, -1e-9)
            self.assertLessEqual(high, 100.0 + 1e-9)
            self.assertLessEqual(low, high)

        self.assertAlmostEqual(sum(result['example_recipe'].values()), 100.0, delta=0.1)

    def test_the_two_functions_answer_the_same_question(self):
        """
        With no material constraints the ranges are exactly the verdict's region

        This is what point 1 of the review was about, stated as a property
        rather than as a number: an unreachable target has an empty region, so
        the ranges must say so instead of describing recipes the verdict
        rejects. Four targets, two of each kind.
        """
        materials = inventory_materials()
        cases = {
            'reference': recipe_01()['umf'],
            'reference + Li2O': dict(recipe_01()['umf'], Li2O=0.5),
            'round numbers': {"SiO2": 3.0, "Al2O3": 0.3, "CaO": 0.7, "K2O": 0.3},
            'five oxides': {"SiO2": 3.0, "Al2O3": 0.35, "CaO": 0.6, "Na2O": 0.25, "K2O": 0.15},
        }

        for label, target in cases.items():
            verdict = check_feasibility(target, materials)
            ranges = achievable_ranges(target, materials)
            self.assertEqual(verdict['feasible'], ranges['feasible'],
                             f"{label}: verdict {verdict['feasible']}, "
                             f"ranges {ranges['feasible']}")

    def test_a_material_range_is_a_real_interval(self):
        result = achievable_ranges(recipe_01()['umf'], inventory_materials())
        low, high = result['material_ranges']['Улексит (Химпэк)']

        # The reference recipe uses 15% of it, so an interval that does not
        # contain 15 would mean the ranges and the verdict disagree
        self.assertLessEqual(low, 15.0)
        self.assertGreaterEqual(high, 15.0)

    def test_the_region_is_the_one_the_verdict_accepts(self):
        """
        Point 1 of the review, in the numbers that showed it

        An oxide the target does not name is contamination, and the verdict
        scores it against zero at the scale floor. Leaving it unbounded here
        answered "up to 19.9% bone ash" for a mixture carrying P2O5 0.198 -
        forty times the tolerance, rejected by the same module.
        """
        result = achievable_ranges(recipe_01()['umf'], inventory_materials())
        allowance = DEFAULT_FEASIBILITY_TOL * OXIDE_SCALE_FLOOR

        for oxide, (low, high) in result['oxide_ranges'].items():
            if oxide in recipe_01()['umf']:
                continue
            self.assertIsNotNone(high, f"{oxide} is contamination and must be capped")
            self.assertLessEqual(high, allowance + 1e-9,
                                 f"{oxide} may reach {high}, the verdict allows {allowance}")

        # Bone ash was the headline case: 19.915% before, under 1% now
        self.assertLess(result['material_ranges']['Костная зола'][1], 1.0)

    def test_an_unbounded_maximum_comes_back_as_none(self):
        # Pure quartz in the set and nothing capping SiO2: there is no largest
        # SiO2. Since every oxide is now bounded by default, "nothing capping
        # it" has to be said out loud - and saying it is the only way to get an
        # unbounded end at all
        result = achievable_ranges({"CaO": 1.0}, [SILICA, CHALK],
                                   oxide_constraints={"SiO2": [None, None]})

        self.assertTrue(result['feasible'])
        self.assertEqual(result['oxide_ranges']['SiO2'][0], 0.0)
        self.assertIsNone(result['oxide_ranges']['SiO2'][1])

    def test_the_flux_floor_keeps_undefined_mixtures_out(self):
        # A flux free batch satisfies every UMF constraint as 0 <= 0 and has no
        # UMF at all. With SiO2 explicitly unbounded, 100% quartz is such a
        # batch, and without the floor it is reported as an achievable share
        constraints = {"SiO2": [None, None]}
        result = achievable_ranges({"CaO": 1.0}, inventory_materials(),
                                   oxide_constraints=constraints)
        quartz = result['material_ranges']['Кварцевая мука Кварцверке W12']

        self.assertLess(quartz[1], 100.0)
        self.assertAlmostEqual(quartz[1], 100.0 * (1.0 - MIN_FLUX_SHARE), delta=1.0)

    def test_a_flux_only_target_says_that_it_constrains_nothing(self):
        result = achievable_ranges({"CaO": 1.0}, inventory_materials())

        self.assertTrue(any('одних плавней' in warning for warning in result['warnings']))
        # And the example is a mixture with a UMF, not 100% aluminium powder
        self.assertNotIn('Алюминиевая пудра', result['example_recipe'])

    def test_an_empty_target_is_refused_like_the_verdict_refuses_it(self):
        result = achievable_ranges({}, SYNTHETIC)

        self.assertIsNone(result['feasible'])
        self.assertEqual(result['error'], 'empty_target')

    def test_a_ceiling_makes_the_maximum_finite_again(self):
        result = achievable_ranges({"CaO": 1.0}, [SILICA, CHALK],
                                   oxide_constraints={"SiO2": [None, 2.5]})

        self.assertIsNotNone(result['oxide_ranges']['SiO2'][1])
        self.assertAlmostEqual(result['oxide_ranges']['SiO2'][1], 2.5, places=6)

    def test_contradictory_constraints_stop_after_the_first_lp(self):
        result = achievable_ranges(
            {"CaO": 1.0}, inventory_materials(),
            material_constraints={"Кварцевая мука Кварцверке W12": [60, 100],
                                  "Мел, CaCO3": [60, 100]})

        self.assertFalse(result['feasible'])
        self.assertEqual(result['lp_count'], 1)
        self.assertEqual(result['oxide_ranges'], {})
        self.assertEqual(result['material_ranges'], {})

    def test_contradictory_oxide_constraints_are_infeasible_too(self):
        # A range given the wrong way round is a contradiction like any other,
        # and it must not quietly pass as an empty interval
        result = achievable_ranges({"CaO": 1.0}, inventory_materials(),
                                   oxide_constraints={"SiO2": [3.0, 2.0]})

        self.assertFalse(result['feasible'])
        # Two rather than one: the probe runs on sum(x) = 100, where a flux free
        # point (100% alumina powder: no SiO2, no flux) satisfies both bounds as
        # 0 <= 0, and the contradiction only bites once S_flux is pinned to 1.
        # Still a bail out, not a grind through all 63
        self.assertLessEqual(result['lp_count'], 2)

    def test_unknown_names_in_the_constraints_are_warnings_not_failures(self):
        result = achievable_ranges({"CaO": 1.0, "SiO2": 3.0}, SYNTHETIC,
                                   oxide_constraints={"Unobtainium": [0, 1]},
                                   material_constraints={"Нет такого материала": [0, 10]})

        self.assertTrue(result['feasible'])
        self.assertEqual(len(result['warnings']), 2)

    def test_nothing_raises_out_of_the_range_machinery(self):
        for target, materials in (({"CaO": 1.0}, []),
                                  ({}, SYNTHETIC),
                                  (None, None)):
            result = achievable_ranges(target, materials)
            self.assertIsInstance(result, dict)
            self.assertIn('feasible', result)


class TestFeasibilityEndpoint(unittest.TestCase):
    """
    POST /api/feasibility

    Kept here rather than in tests/test_api_server.py: the endpoint is a thin
    wrapper over this module, and the two belong in one place while the contract
    is new.
    """

    @classmethod
    def setUpClass(cls):
        import api_server
        cls.client = api_server.app.test_client()

    def test_the_reference_target_is_reachable_through_the_api(self):
        response = self.client.post('/api/feasibility', json={"umf": recipe_01()['umf']})
        body = response.get_json()

        self.assertEqual(response.status_code, 200)
        self.assertTrue(body['feasible'])
        self.assertIn('diagnostics', body)
        self.assertIn('achievable_ranges', body)
        self.assertEqual(body['achievable_ranges']['lp_count'], 63)

    def test_a_null_inventory_means_the_default_stock(self):
        # The same reading /api/solve gives the word, and deliberately not the
        # one /api/sensitivity gives it (there null means the whole database)
        explicit = self.client.post('/api/feasibility', json={
            "umf": recipe_01()['umf'],
            "inventory": [material['name'] for material in inventory_materials()]})
        implicit = self.client.post('/api/feasibility', json={
            "umf": recipe_01()['umf'], "inventory": None})

        self.assertEqual(explicit.get_json()['closest_recipe'],
                         implicit.get_json()['closest_recipe'])
        self.assertEqual(len(implicit.get_json()['achievable_ranges']['material_ranges']),
                         len(inventory_materials()))

    def test_every_range_the_endpoint_reports_is_finite(self):
        # Through the API every oxide is bounded - by the target box, by a
        # passenger ceiling or by the contamination allowance - so the null end
        # that the library can return cannot arise here. Worth pinning: the
        # opposite would mean an oxide escaped the region the verdict accepts
        response = self.client.post('/api/feasibility', json={
            "umf": recipe_01()['umf'], "passengers": {"Fe2O3": 0.03}})
        ranges = response.get_json()['achievable_ranges']['oxide_ranges']

        self.assertTrue(all(bound is not None for pair in ranges.values() for bound in pair))
        self.assertLessEqual(ranges['Fe2O3'][1], 0.03 + 1e-9)

    def test_a_rejected_passenger_is_rejected_in_both_halves(self):
        # The ranges used to be built from the RAW request while the verdict
        # cleaned it, so this answered feasible: true next to
        # achievable_ranges.feasible: false, with one warning between them
        response = self.client.post('/api/feasibility', json={
            "umf": recipe_01()['umf'], "passengers": {"Fe2O3": -1.0}})
        body = response.get_json()

        self.assertTrue(body['feasible'])
        self.assertTrue(body['achievable_ranges']['feasible'])
        self.assertEqual(body['passengers'], [])
        self.assertTrue(any('пассажир' in warning for warning in body['warnings']))

    def test_the_verdict_can_be_asked_for_on_its_own(self):
        response = self.client.post('/api/feasibility', json={
            "umf": recipe_01()['umf'], "ranges": False})
        body = response.get_json()

        self.assertEqual(response.status_code, 200)
        self.assertNotIn('achievable_ranges', body)
        self.assertIn('diagnostics', body)
        self.assertTrue(body['feasible'])

    def test_an_inventory_too_big_for_the_ranges_is_refused(self):
        everything = [material['name'] for material
                      in load_materials(only_inventory=False, priority=False)]

        refused = self.client.post('/api/feasibility',
                                   json={"umf": recipe_01()['umf'], "inventory": everything})
        self.assertEqual(refused.status_code, 413)
        self.assertEqual(refused.get_json()['error'], 'inventory_too_large')

        # ... and the escape hatch named in the message works
        verdict = self.client.post('/api/feasibility',
                                   json={"umf": recipe_01()['umf'], "inventory": everything,
                                         "ranges": False})
        self.assertEqual(verdict.status_code, 200)
        self.assertTrue(verdict.get_json()['feasible'])

    def test_the_projection_matches_what_the_ranges_actually_solve(self):
        materials = inventory_materials()
        target = recipe_01()['umf']

        self.assertEqual(projected_range_lps(target, materials),
                         achievable_ranges(target, materials)['lp_count'])

    def test_passengers_and_material_constraints_are_accepted(self):
        response = self.client.post('/api/feasibility', json={
            "umf": recipe_01()['umf'],
            "passengers": {"Fe2O3": 0.03},
            "material_constraints": {"Мел, CaCO3": [0, 10]}})
        body = response.get_json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual([row['oxide'] for row in body['passengers']], ['Fe2O3'])
        low, high = body['achievable_ranges']['material_ranges']['Мел, CaCO3']
        self.assertLessEqual(high, 10.0 + 1e-6)

    def test_bad_requests_are_400_and_unusable_targets_are_422(self):
        self.assertEqual(self.client.post('/api/feasibility', json={}).status_code, 400)
        self.assertEqual(self.client.post('/api/feasibility',
                                          json={"umf": recipe_01()['umf'],
                                                "passengers": [1, 2]}).status_code, 400)
        self.assertEqual(self.client.post('/api/feasibility',
                                          json={"umf": recipe_01()['umf'],
                                                "tol": "скоро"}).status_code, 400)
        # A formula with no flux is not a UMF: there is no unity to normalize by
        no_flux = self.client.post('/api/feasibility', json={"umf": {"SiO2": 3.0}})
        self.assertEqual(no_flux.status_code, 422)
        self.assertEqual(no_flux.get_json()['error'], 'no_target_fluxes')


if __name__ == '__main__':
    unittest.main()
