#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# flake8: noqa
# pylint: disable=broad-exception-raised, raise-missing-from, too-many-arguments, redefined-outer-name
# pylint: disable=multiple-statements, logging-fstring-interpolation, trailing-whitespace, line-too-long
# pylint: disable=broad-exception-caught, missing-function-docstring, missing-class-docstring
# pylint: disable=f-string-without-interpolation
# pylance: disable=reportMissingImports, reportMissingModuleSource

"""
Unit tests of the quality metrics

Everything here is synthetic: no solver is run and no test depends on the
current contents of database/materials.json. The material records are built
inline so that a change of the real database cannot silently change what these
tests measure.
"""

import json
import os
import shutil
import sys
import tempfile
import unittest

# Fix imports by adding parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import quality_metrics as qm
from common import DEFAULT_PRIORITY

# Synthetic materials. The two feldspars are near collinear on purpose: half a
# percent of SiO2 is traded against Al2O3 and K2O with the sum preserved, and
# everything else is identical, which makes any mix of them chemically almost
# the same material. These are the analyses behind the measurements recorded in
# TZ_SOLVER_V2.md 10.9 (cond 15.1 honest against 3668.8 degenerate). The rest
# are the classic well separated members of a glaze.
FELDSPAR = {
    'name': 'Potash Feldspar',
    'formula': {'SiO2': 68.50, 'Al2O3': 17.00, 'Na2O': 3.00, 'K2O': 10.00, 'CaO': 0.30, 'Fe2O3': 0.10},
}
FELDSPAR_TWIN = {
    'name': 'Feldspar Twin',
    'formula': {'SiO2': 69.00, 'Al2O3': 16.70, 'Na2O': 3.00, 'K2O': 9.80, 'CaO': 0.30, 'Fe2O3': 0.10},
}
SILICA = {'name': 'Silica', 'formula': {'SiO2': 100.0}}
WHITING = {'name': 'Whiting', 'formula': {'CaO': 56.10}}
KAOLIN = {'name': 'Kaolin', 'formula': {'Al2O3': 40.21, 'SiO2': 47.29}}
IRON_OXIDE = {'name': 'Red Iron Oxide', 'formula': {'Fe2O3': 100.0}}
ALUMINA = {'name': 'Глинозём Г-00', 'formula': {'Al2O3': 65.40}}
KAOLIN_RU = {'name': 'КАОЛИН КЖФ-1', 'formula': {'Al2O3': 40.21, 'SiO2': 47.29}}
# A real edge case of the project database: two analyses sum to more than 100
# (Cryolite 122.90, Zircon 135.22), which makes the batch LOI negative.
ZIRCON = {'name': 'Zircon', 'formula': {'ZrO2': 66.00, 'SiO2': 69.22}}
# 37 of the 216 real materials carry no oxide at all and all of them store an
# empty formula: every pigment, the silicon carbide fractions, water, CMC,
# charcoal, gypsum, alum. They are legal recipe entries with no chemistry.
PIGMENT = {'name': 'Кобальт голубой пигмент 6226', 'formula': {}}

MATERIALS = [FELDSPAR, FELDSPAR_TWIN, SILICA, WHITING, KAOLIN, IRON_OXIDE, ALUMINA,
             KAOLIN_RU, ZIRCON, PIGMENT]

# A plain, well behaved recipe used as the original in most of the tests
BASE_RECIPE = {'Potash Feldspar': 40.0, 'Silica': 30.0, 'Whiting': 20.0, 'Kaolin': 10.0}


class TestJunkAndMinPortion(unittest.TestCase):
    """A component too light to weigh has to be visible in the report"""

    def test_light_component_is_junk_and_fails_min_portion(self):
        original = dict(BASE_RECIPE, Silica=27.0, **{'Red Iron Oxide': 3.0})
        recipe = dict(BASE_RECIPE, Silica=29.3, **{'Red Iron Oxide': 0.7})

        report = qm.solution_quality(recipe, original, MATERIALS)

        self.assertEqual(report['junk']['solution'], 1)
        self.assertEqual(report['junk']['original'], 0)
        self.assertFalse(report['junk']['ok'])
        self.assertAlmostEqual(report['min_portion']['solution'], 0.7)
        self.assertAlmostEqual(report['min_portion']['original'], 3.0)
        self.assertTrue(report['min_portion']['required'])
        self.assertFalse(report['min_portion']['ok'])
        self.assertIn('junk', report['failures'])
        self.assertIn('min_portion', report['failures'])

    def test_same_recipe_passes_when_the_component_is_heavy_enough(self):
        original = dict(BASE_RECIPE, Silica=27.0, **{'Red Iron Oxide': 3.0})
        recipe = dict(original)

        report = qm.solution_quality(recipe, original, MATERIALS)

        self.assertEqual(report['junk']['solution'], 0)
        self.assertTrue(report['junk']['ok'])
        self.assertAlmostEqual(report['min_portion']['solution'], 3.0)
        self.assertTrue(report['min_portion']['ok'])
        self.assertEqual(report['failures'], [])

    def test_the_rule_is_waived_when_the_original_is_itself_below_the_limit(self):
        # The original proves that this chemistry needs a sub-percent colorant,
        # so the solution may not be blamed for needing one too
        original = dict(BASE_RECIPE, Silica=29.5, **{'Red Iron Oxide': 0.5})
        recipe = dict(BASE_RECIPE, Silica=29.4, **{'Red Iron Oxide': 0.6})

        report = qm.solution_quality(recipe, original, MATERIALS)

        self.assertFalse(report['min_portion']['required'])
        self.assertTrue(report['min_portion']['ok'])
        self.assertNotIn('min_portion', report['failures'])


class TestCount(unittest.TestCase):
    """The solution may use one material more than the original, not two"""

    def test_one_extra_material_passes(self):
        recipe = dict(BASE_RECIPE, Silica=25.0, **{'Red Iron Oxide': 5.0})

        report = qm.solution_quality(recipe, BASE_RECIPE, MATERIALS)

        self.assertEqual(report['count']['solution'], 5)
        self.assertEqual(report['count']['original'], 4)
        self.assertEqual(report['count']['delta'], qm.MAX_COUNT_DELTA)
        self.assertTrue(report['count']['ok'])
        self.assertNotIn('count', report['failures'])

    def test_two_extra_materials_fail(self):
        recipe = dict(BASE_RECIPE, Silica=20.0, **{'Red Iron Oxide': 5.0, 'Глинозём Г-00': 5.0})

        report = qm.solution_quality(recipe, BASE_RECIPE, MATERIALS)

        self.assertEqual(report['count']['delta'], qm.MAX_COUNT_DELTA + 1)
        self.assertFalse(report['count']['ok'])
        self.assertIn('count', report['failures'])


class TestRoundingDrift(unittest.TestCase):
    """
    The weighing-tolerance diagnostic

    The drift is a true statement about whether a recipe survives being weighed
    on a 0.5% grid, and nothing more. It is NOT the degeneracy detector - see
    TestConditioning for that, and TZ_SOLVER_V2.md 10.9 for why the two are not
    the same thing.
    """

    def test_a_collinear_split_is_harder_to_weigh_than_a_well_separated_recipe(self):
        # Same chemistry both ways: the 41.1% of feldspar is simply split across
        # the twin. Splitting one mass over two grid-rounded knobs doubles the
        # rounding error the pair can accumulate, so the split does cost
        # something on the scale - but only about a factor of three, and it
        # stays well inside the band that honest recipes also occupy, which is
        # precisely why the drift cannot be used as a degeneracy gate.
        honest = {'Potash Feldspar': 41.1, 'Silica': 28.9, 'Whiting': 19.5, 'Kaolin': 10.5}
        collinear = {'Potash Feldspar': 12.3, 'Feldspar Twin': 28.8,
                     'Silica': 28.9, 'Whiting': 19.5, 'Kaolin': 10.5}

        report = qm.solution_quality(collinear, honest, MATERIALS)
        honest_drift = report['rounding_drift']['original']
        collinear_drift = report['rounding_drift']['value']

        # Measured: honest 0.0052, collinear 0.0145 - a factor of 2.79
        self.assertLess(honest_drift, qm.MAX_ROUNDING_DRIFT)
        self.assertGreater(collinear_drift, 2.0 * honest_drift)

    def test_the_drift_never_fails_a_solution(self):
        # An unweighable recipe is worth reporting, but the gate lives on the
        # conditioning now and the drift may not send anything to "failures"
        vanishing = {'Silica': 0.2, 'Whiting': 0.1}

        report = qm.solution_quality(vanishing, BASE_RECIPE, MATERIALS)

        self.assertFalse(report['rounding_drift']['ok'])
        self.assertNotIn('rounding_drift', report['failures'])
        self.assertNotIn('rounding_drift', qm.GATED_METRICS)

    def test_a_recipe_that_rounds_away_entirely_does_not_divide_by_zero(self):
        # Both components round down to nothing, so the rounded batch has no
        # mass at all; the metric must report the damage, not crash
        vanishing = {'Silica': 0.2, 'Whiting': 0.1}

        report = qm.solution_quality(vanishing, BASE_RECIPE, MATERIALS)

        self.assertGreater(report['rounding_drift']['value'], qm.MAX_ROUNDING_DRIFT)
        self.assertFalse(report['rounding_drift']['ok'])

    def test_a_recipe_of_unknown_materials_has_no_drift_to_measure(self):
        report = qm.solution_quality({'Mystery Powder': 100.0}, BASE_RECIPE, MATERIALS)

        self.assertIsNone(report['rounding_drift']['value'])
        self.assertIsNone(report['rounding_drift']['ok'])
        self.assertNotIn('rounding_drift', report['failures'])


class TestConditioning(unittest.TestCase):
    """
    The real anti-degeneracy metric

    A recipe standing on a compensating pair of near-identical materials has an
    almost singular material matrix. Unlike the rounding drift, the condition
    number separates the two populations by orders of magnitude and does not
    care about the decimals of the shares.
    """

    HONEST = {'Potash Feldspar': 41.1, 'Silica': 28.9, 'Whiting': 19.5, 'Kaolin': 10.5}
    COLLINEAR = {'Potash Feldspar': 12.3, 'Feldspar Twin': 28.8,
                 'Silica': 28.9, 'Whiting': 19.5, 'Kaolin': 10.5}

    def test_a_well_separated_recipe_passes(self):
        report = qm.solution_quality(self.HONEST, self.HONEST, MATERIALS)

        # Measured: 15.1, two orders of magnitude below the threshold
        self.assertAlmostEqual(report['conditioning']['cond'], 15.11, places=2)
        self.assertLess(report['conditioning']['cond'], qm.MAX_CONDITION_NUMBER)
        self.assertEqual(report['conditioning']['rank'], 4)
        self.assertEqual(report['conditioning']['redundancy'], 0)
        self.assertFalse(report['conditioning']['rank_deficient'])
        self.assertTrue(report['conditioning']['ok'])
        self.assertEqual(report['failures'], [])

    def test_a_collinear_pair_fails(self):
        report = qm.solution_quality(self.COLLINEAR, self.HONEST, MATERIALS)

        # Measured: 3668.8 against the original's 15.1 - a factor of 243
        self.assertAlmostEqual(report['conditioning']['cond'], 3668.8, places=1)
        self.assertGreater(report['conditioning']['cond'], qm.MAX_CONDITION_NUMBER)
        self.assertAlmostEqual(report['conditioning']['original'], 15.11, places=2)
        self.assertFalse(report['conditioning']['ok'])
        self.assertIn('conditioning', report['failures'])

    def test_the_condition_number_does_not_depend_on_the_decimals(self):
        # The point of the whole metric: the same material set scores the same
        # however the shares are printed, while the drift of these two identical
        # recipes differs by a factor of five for no chemical reason at all
        unlucky = {'Potash Feldspar': 41.3, 'Silica': 28.7, 'Whiting': 19.4, 'Kaolin': 10.6}

        lucky_report = qm.solution_quality(self.HONEST, self.HONEST, MATERIALS)
        unlucky_report = qm.solution_quality(unlucky, unlucky, MATERIALS)

        self.assertEqual(lucky_report['conditioning']['cond'],
                         unlucky_report['conditioning']['cond'])
        self.assertTrue(unlucky_report['conditioning']['ok'])
        # ... and this is what the drift makes of the very same pair
        self.assertGreater(unlucky_report['rounding_drift']['value'],
                           3.0 * lucky_report['rounding_drift']['value'])
        self.assertGreater(unlucky_report['rounding_drift']['value'], qm.MAX_ROUNDING_DRIFT)

    def test_two_materials_of_the_same_analysis_are_rank_deficient(self):
        # Kaolin twice under two names: no chemistry can tell the pair apart, so
        # the material set is linearly dependent and infinitely ill-conditioned
        recipe = {'Potash Feldspar': 41.1, 'Silica': 28.9, 'Whiting': 19.5,
                  'Kaolin': 5.5, 'КАОЛИН КЖФ-1': 5.0}

        report = qm.solution_quality(recipe, self.HONEST, MATERIALS)

        self.assertIsNone(report['conditioning']['cond'])
        self.assertTrue(report['conditioning']['rank_deficient'])
        self.assertEqual(report['conditioning']['rank'], 4)
        self.assertEqual(report['conditioning']['redundancy'], 1)
        self.assertFalse(report['conditioning']['ok'])
        self.assertIn('conditioning', report['failures'])

    def test_a_pigment_does_not_make_a_recipe_degenerate(self):
        # A material with no oxides is an empty column: no rows, one column, so
        # a naive matrix loses rank and condemns every pigmented recipe. There
        # are 37 such materials in the database, and a glaze really does carry
        # 1% of cobalt pigment. Compared with itself, a recipe cannot possibly
        # be worse than the original, so this must come back clean.
        pigmented = {'Potash Feldspar': 39.0, 'Silica': 30.0, 'Whiting': 20.0,
                     'Kaolin': 10.0, 'Кобальт голубой пигмент 6226': 1.0}

        report = qm.solution_quality(pigmented, pigmented, MATERIALS)

        self.assertEqual(report['failures'], [])
        self.assertTrue(report['conditioning']['ok'])
        self.assertFalse(report['conditioning']['rank_deficient'])
        self.assertIsNotNone(report['conditioning']['cond'])
        self.assertLess(report['conditioning']['cond'], qm.MAX_CONDITION_NUMBER)
        # The pigment is reported rather than silently ignored, and it is not
        # confused with a material we have never heard of
        self.assertEqual(report['unanalysed_materials'], ['Кобальт голубой пигмент 6226'])
        self.assertEqual(report['unknown_materials'], [])
        # It has no column, so it cannot raise the rank - but it is still a bag
        # on the shelf whose contribution we cannot justify
        self.assertEqual(report['conditioning']['rank'], 4)
        self.assertEqual(report['conditioning']['redundancy'], 1)

    def test_a_pigment_does_not_hide_a_real_degeneracy(self):
        # The exclusion must not become a way to smuggle a dependent set past
        # the gate: the duplicated kaolin still sinks this recipe
        pigmented_and_degenerate = {'Potash Feldspar': 39.0, 'Silica': 30.0, 'Whiting': 20.0,
                                    'Kaolin': 5.0, 'КАОЛИН КЖФ-1': 5.0,
                                    'Кобальт голубой пигмент 6226': 1.0}

        report = qm.solution_quality(pigmented_and_degenerate, self.HONEST, MATERIALS)

        self.assertIsNone(report['conditioning']['cond'])
        self.assertTrue(report['conditioning']['rank_deficient'])
        self.assertFalse(report['conditioning']['ok'])
        self.assertIn('conditioning', report['failures'])
        self.assertEqual(report['unanalysed_materials'], ['Кобальт голубой пигмент 6226'])

    def test_a_single_material_recipe_does_not_crash(self):
        report = qm.solution_quality({'Silica': 100.0}, {'Silica': 100.0}, MATERIALS)

        self.assertAlmostEqual(report['conditioning']['cond'], 1.0)
        self.assertEqual(report['conditioning']['rank'], 1)
        self.assertEqual(report['conditioning']['redundancy'], 0)
        self.assertTrue(report['conditioning']['ok'])

    def test_a_recipe_of_unknown_materials_is_the_worst_case_not_an_abstention(self):
        report = qm.solution_quality({'Mystery Powder': 100.0}, self.HONEST, MATERIALS)

        self.assertIsNone(report['conditioning']['cond'])
        self.assertEqual(report['conditioning']['rank'], 0)
        self.assertEqual(report['conditioning']['redundancy'], 1)
        self.assertFalse(report['conditioning']['ok'])
        self.assertIn('conditioning', report['failures'])


class TestCost(unittest.TestCase):
    """Prices are region dependent, so an incomplete price list buys no verdict"""

    PRICES = {'Potash Feldspar': 100.0, 'Silica': 50.0, 'Whiting': 40.0, 'Kaolin': 90.0}

    def test_missing_price_leaves_the_ratio_undefined(self):
        recipe = dict(BASE_RECIPE, Silica=25.0, **{'Red Iron Oxide': 5.0})
        prices = dict(self.PRICES)  # no price for the iron oxide

        report = qm.solution_quality(recipe, BASE_RECIPE, MATERIALS, prices=prices)

        self.assertIsNone(report['cost']['ratio'])
        self.assertIsNone(report['cost']['cost_abs'])
        self.assertIsNone(report['cost']['ok'])
        self.assertLess(report['cost']['coverage'], 1.0)
        self.assertAlmostEqual(report['cost']['coverage'], 4.0 / 5.0)
        self.assertNotIn('cost', report['failures'])

    def test_full_coverage_yields_a_ratio(self):
        recipe = {'Potash Feldspar': 44.0, 'Silica': 26.0, 'Whiting': 20.0, 'Kaolin': 10.0}

        report = qm.solution_quality(recipe, BASE_RECIPE, MATERIALS, prices=self.PRICES)

        self.assertAlmostEqual(report['cost']['coverage'], 1.0)
        self.assertAlmostEqual(report['cost']['original'], 72.0)
        self.assertAlmostEqual(report['cost']['solution'], 74.0)
        self.assertAlmostEqual(report['cost']['cost_abs'], 74.0)
        self.assertAlmostEqual(report['cost']['ratio'], 74.0 / 72.0)
        self.assertTrue(report['cost']['ok'])
        # 74.00 roubles per kg of batch across 4 bags, 72.00 across 4
        self.assertAlmostEqual(report['assembly_score']['value'], 296.0)
        self.assertAlmostEqual(report['assembly_score']['original'], 288.0)

    def test_a_solution_more_expensive_than_the_band_fails(self):
        prices = {'Potash Feldspar': 200.0, 'Silica': 10.0, 'Whiting': 10.0, 'Kaolin': 10.0}
        recipe = {'Potash Feldspar': 60.0, 'Silica': 10.0, 'Whiting': 20.0, 'Kaolin': 10.0}

        report = qm.solution_quality(recipe, BASE_RECIPE, MATERIALS, prices=prices)

        self.assertGreater(report['cost']['ratio'], qm.MAX_COST_RATIO)
        self.assertFalse(report['cost']['ok'])
        self.assertIn('cost', report['failures'])

    def test_cost_abs_survives_an_unpriced_original(self):
        # The scenario the absolute cost exists for: our own solution is fully
        # priced, the foreign original is not, so no ratio - but our own price
        # per kg of batch is still a fact worth reporting
        original = dict(BASE_RECIPE, Silica=25.0, **{'Frit 3134': 5.0})
        recipe = {'Potash Feldspar': 44.0, 'Silica': 26.0, 'Whiting': 20.0, 'Kaolin': 10.0}

        report = qm.solution_quality(recipe, original, MATERIALS, prices=self.PRICES)

        self.assertAlmostEqual(report['cost']['cost_abs'], 74.0)
        self.assertIsNone(report['cost']['original'])
        self.assertIsNone(report['cost']['ratio'])
        self.assertIsNone(report['cost']['ok'])
        self.assertAlmostEqual(report['cost']['coverage'], 4.0 / 5.0)
        self.assertAlmostEqual(report['assembly_score']['value'], 296.0)
        # Only the side we cannot price loses its score
        self.assertIsNone(report['assembly_score']['original'])

    def test_no_prices_at_all(self):
        report = qm.solution_quality(BASE_RECIPE, BASE_RECIPE, MATERIALS)

        self.assertIsNone(report['cost']['ratio'])
        self.assertAlmostEqual(report['cost']['coverage'], 0.0)
        self.assertIsNone(report['assembly_score']['cost_abs'])
        self.assertIsNone(report['assembly_score']['value'])


class TestAssemblyScore(unittest.TestCase):
    """
    Roubles times pieces, or nothing at all

    The score must never silently switch units: a baseline diff comparing a
    component count against a roubles-times-pieces score reads the change as an
    86-fold improvement (TZ_SOLVER_V2.md 10.10).
    """

    PRICES = {'Potash Feldspar': 100.0, 'Silica': 50.0, 'Whiting': 40.0, 'Kaolin': 90.0}

    def test_an_unpriced_recipe_has_no_score_rather_than_a_component_count(self):
        recipe = dict(BASE_RECIPE, Silica=25.0, **{'Red Iron Oxide': 5.0})

        report = qm.solution_quality(recipe, BASE_RECIPE, MATERIALS, prices=self.PRICES)

        self.assertIsNone(report['assembly_score']['value'])
        self.assertIsNone(report['assembly_score']['cost_abs'])
        # Specifically not the number of components, in either notation
        self.assertNotEqual(report['assembly_score']['value'], len(recipe))
        self.assertNotEqual(report['assembly_score']['value'], float(len(recipe)))
        # ... and nothing is lost, because the count is its own metric
        self.assertEqual(report['count']['solution'], len(recipe))

    def test_a_fully_priced_pair_scores_in_roubles_times_pieces(self):
        # 40*100 + 30*50 + 20*40 + 10*90, all per 100 g of batch, is 72.00 per
        # kg; the solution swaps 4 points of silica for feldspar and costs 74.00
        recipe = {'Potash Feldspar': 44.0, 'Silica': 26.0, 'Whiting': 20.0, 'Kaolin': 10.0}

        report = qm.solution_quality(recipe, BASE_RECIPE, MATERIALS, prices=self.PRICES)

        self.assertAlmostEqual(report['assembly_score']['cost_abs'], 74.0)
        self.assertAlmostEqual(report['assembly_score']['value'], 74.0 * 4)
        self.assertAlmostEqual(report['assembly_score']['original'], 72.0 * 4)

    def test_only_the_unpriced_side_loses_its_score(self):
        priced = {'Potash Feldspar': 44.0, 'Silica': 26.0, 'Whiting': 20.0, 'Kaolin': 10.0}
        unpriced = dict(BASE_RECIPE, Silica=25.0, **{'Frit 3134': 5.0})

        solution_priced = qm.solution_quality(priced, unpriced, MATERIALS, prices=self.PRICES)
        original_priced = qm.solution_quality(unpriced, priced, MATERIALS, prices=self.PRICES)

        self.assertAlmostEqual(solution_priced['assembly_score']['value'], 296.0)
        self.assertIsNone(solution_priced['assembly_score']['original'])

        self.assertIsNone(original_priced['assembly_score']['value'])
        self.assertAlmostEqual(original_priced['assembly_score']['original'], 296.0)

    def test_the_score_can_never_fail_a_solution(self):
        report = qm.solution_quality(BASE_RECIPE, BASE_RECIPE, MATERIALS)

        self.assertNotIn('ok', report['assembly_score'])
        self.assertNotIn('assembly_score', qm.GATED_METRICS)
        self.assertNotIn('assembly_score', qm.WARNING_METRICS)
        self.assertNotIn('assembly_score', report['failures'])
        self.assertNotIn('assembly_score', report['warnings'])


class TestPriority(unittest.TestCase):
    """Priorities are our own supply reality and are never assumed"""

    def test_no_priorities_means_no_verdict(self):
        report = qm.solution_quality(BASE_RECIPE, BASE_RECIPE, MATERIALS)

        self.assertIsNone(report['priority']['solution'])
        self.assertIsNone(report['priority']['original'])
        self.assertIsNone(report['priority']['ratio'])
        self.assertIsNone(report['priority']['ok'])
        self.assertNotIn('priority', report['failures'])

    def test_an_empty_mapping_abstains_instead_of_inventing_a_verdict(self):
        # Every material would fall to DEFAULT_PRIORITY, which cancels in the
        # ratio and returns a confident 1.0 built out of no data at all
        recipe = dict(BASE_RECIPE, Silica=25.0, **{'Red Iron Oxide': 5.0})

        report = qm.solution_quality(recipe, BASE_RECIPE, MATERIALS, priorities={})

        self.assertIsNone(report['priority']['ratio'])
        self.assertIsNone(report['priority']['solution'])
        self.assertIsNone(report['priority']['ok'])
        self.assertNotIn('priority', report['failures'])

    def test_unlisted_material_falls_back_to_the_default_priority(self):
        priorities = {'Potash Feldspar': 1, 'Silica': 1, 'Whiting': 2, 'Kaolin': 3}
        recipe = dict(BASE_RECIPE, Silica=25.0, **{'Red Iron Oxide': 5.0})

        report = qm.solution_quality(recipe, BASE_RECIPE, MATERIALS, priorities=priorities)

        expected_original = (40 * 1 + 30 * 1 + 20 * 2 + 10 * 3) / 100.0
        expected_solution = (40 * 1 + 25 * 1 + 20 * 2 + 10 * 3 + 5 * DEFAULT_PRIORITY) / 100.0
        self.assertAlmostEqual(report['priority']['original'], expected_original)
        self.assertAlmostEqual(report['priority']['solution'], expected_solution)
        self.assertAlmostEqual(report['priority']['ratio'], expected_solution / expected_original)
        self.assertFalse(report['priority']['ok'])
        self.assertIn('priority', report['failures'])


class TestClayContent(unittest.TestCase):
    """A glaze without clay does not stay in suspension, however good its chemistry"""

    def test_clay_free_solution_warns(self):
        recipe = {'Potash Feldspar': 45.0, 'Silica': 30.0, 'Whiting': 20.0, 'Глинозём Г-00': 5.0}

        report = qm.solution_quality(recipe, BASE_RECIPE, MATERIALS)

        self.assertAlmostEqual(report['clay_content']['solution'], 0.0)
        self.assertAlmostEqual(report['clay_content']['original'], 10.0)
        self.assertAlmostEqual(report['clay_content']['threshold'], 5.0)
        self.assertFalse(report['clay_content']['ok'])
        self.assertEqual(report['warnings'], ['clay_content'])
        # A warning is not a failure of the "no worse than the original" rule
        self.assertNotIn('clay_content', report['failures'])

    def test_clay_is_detected_case_insensitively_in_cyrillic(self):
        recipe = {'Potash Feldspar': 45.0, 'Silica': 30.0, 'Whiting': 10.0, 'КАОЛИН КЖФ-1': 15.0}

        report = qm.solution_quality(recipe, BASE_RECIPE, MATERIALS)

        self.assertAlmostEqual(report['clay_content']['solution'], 15.0)
        self.assertTrue(report['clay_content']['ok'])
        self.assertEqual(report['warnings'], [])

    def test_a_clay_poor_original_does_not_warn_forever(self):
        # The original has 2% of clay, so half of it - 1% - is all we may demand
        original = {'Potash Feldspar': 48.0, 'Silica': 30.0, 'Whiting': 20.0, 'Kaolin': 2.0}
        recipe = {'Potash Feldspar': 48.5, 'Silica': 30.0, 'Whiting': 20.0, 'Kaolin': 1.5}

        report = qm.solution_quality(recipe, original, MATERIALS)

        self.assertAlmostEqual(report['clay_content']['threshold'], 1.0)
        self.assertTrue(report['clay_content']['ok'])


class TestUnknownMaterials(unittest.TestCase):
    """A material we cannot analyse must be reported, not silently priced at zero"""

    def test_unknown_material_is_reported_and_does_not_raise(self):
        recipe = dict(BASE_RECIPE, Silica=25.0, **{'Mystery Powder': 5.0})

        report = qm.solution_quality(recipe, BASE_RECIPE, MATERIALS)

        self.assertEqual(report['unknown_materials'], ['Mystery Powder'])
        self.assertIsNotNone(report['rounding_drift']['value'])
        self.assertEqual(report['count']['solution'], 5)
        self.assertAlmostEqual(report['set_jaccard'], 4.0 / 5.0)
        self.assertAlmostEqual(report['share_delta'], 5.0)

    def test_a_fully_known_pair_reports_nothing(self):
        report = qm.solution_quality(BASE_RECIPE, BASE_RECIPE, MATERIALS)

        self.assertEqual(report['unknown_materials'], [])
        self.assertEqual(report['unanalysed_materials'], [])
        self.assertAlmostEqual(report['set_jaccard'], 1.0)
        self.assertAlmostEqual(report['share_delta'], 0.0)

    def test_unknown_and_unanalysed_are_two_different_states(self):
        # "we have never heard of it" and "we have it on file with no analysis"
        # need separate fields: the first is a data gap, the second is a normal
        # non-oxide ingredient, and only the first is worth chasing
        recipe = dict(BASE_RECIPE, Silica=24.0,
                      **{'Mystery Powder': 5.0, 'Кобальт голубой пигмент 6226': 1.0})

        report = qm.solution_quality(recipe, BASE_RECIPE, MATERIALS)

        self.assertEqual(report['unknown_materials'], ['Mystery Powder'])
        self.assertEqual(report['unanalysed_materials'], ['Кобальт голубой пигмент 6226'])


class TestLossOnIgnition(unittest.TestCase):
    """LOI is a diagnostic and is never clamped"""

    def test_loi_delta_of_two_different_batches(self):
        recipe = {'Silica': 50.0, 'Whiting': 50.0}
        original = {'Silica': 80.0, 'Whiting': 20.0}

        report = qm.solution_quality(recipe, original, MATERIALS)

        self.assertAlmostEqual(report['loi']['solution'], 100.0 - (50.0 + 28.05))
        self.assertAlmostEqual(report['loi']['original'], 100.0 - (80.0 + 11.22))
        self.assertAlmostEqual(report['loi_delta'], abs(21.95 - 8.78))

    def test_analysis_over_100_gives_a_negative_loi(self):
        recipe = {'Zircon': 100.0}
        original = {'Silica': 100.0}

        report = qm.solution_quality(recipe, original, MATERIALS)

        self.assertAlmostEqual(report['loi']['solution'], 100.0 - 135.22)
        self.assertLess(report['loi']['solution'], 0.0)
        self.assertAlmostEqual(report['loi']['original'], 0.0)
        self.assertAlmostEqual(report['loi_delta'], 35.22)


class TestLoadPrices(unittest.TestCase):
    """The price list is optional data and its absence must break nothing"""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix='glazy_prices_')
        self.addCleanup(shutil.rmtree, self.tmp_dir)

    def test_missing_file(self):
        self.assertEqual(qm.load_prices(os.path.join(self.tmp_dir, 'nope.json')), {})

    def test_empty_file(self):
        path = os.path.join(self.tmp_dir, 'empty.json')
        with open(path, 'w', encoding='utf-8') as f:
            f.write('')

        self.assertEqual(qm.load_prices(path), {})

    def test_file_with_prices(self):
        path = os.path.join(self.tmp_dir, 'prices.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'Каолин КЖФ-1': 90}, f, ensure_ascii=False)

        self.assertEqual(qm.load_prices(path), {'Каолин КЖФ-1': 90})

    def test_malformed_json_does_not_escape(self):
        # A price list is optional data: a typo in it must not take down a
        # metric run that does not even need prices
        path = os.path.join(self.tmp_dir, 'broken.json')
        with open(path, 'w', encoding='utf-8') as f:
            f.write('{"Каолин КЖФ-1": 90,}')

        # Soft, but not silent: an unusable file that exists is logged, because
        # pricing nothing looks exactly like having no prices
        with self.assertLogs('quality_metrics', level='WARNING'):
            self.assertEqual(qm.load_prices(path), {})

    def test_a_json_list_is_not_a_price_list(self):
        # Returning the list would blow up later on prices.get(), far away from
        # the file that caused it
        path = os.path.join(self.tmp_dir, 'list.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(['Каолин КЖФ-1', 90], f, ensure_ascii=False)

        with self.assertLogs('quality_metrics', level='WARNING'):
            self.assertEqual(qm.load_prices(path), {})

    def test_a_bare_null_is_an_empty_price_list(self):
        path = os.path.join(self.tmp_dir, 'null.json')
        with open(path, 'w', encoding='utf-8') as f:
            f.write('null')

        self.assertEqual(qm.load_prices(path), {})

    def test_the_committed_file_parses(self):
        # The price list is filled by hand over time, so this asserts the shape
        # and never the contents: the metrics must be correct at any coverage,
        # from an empty file to a complete one
        prices = qm.load_prices()

        self.assertIsInstance(prices, dict)
        for name, price in prices.items():
            self.assertIsInstance(name, str)
            self.assertIsInstance(price, (int, float))
            self.assertGreater(price, 0)


if __name__ == '__main__':
    unittest.main()
