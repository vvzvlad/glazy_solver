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
# A colorant dosed in fractions of a percent that is nevertheless the entire
# reason its recipe exists - the Glazy dump has cobalt carbonate at 0.19% and
# copper carbonate at 0.02%
COBALT_CARBONATE = {'name': 'Углекислый кобальт', 'formula': {'CoO': 63.00}}
COPPER_CARBONATE = {'name': 'Углекислая медь', 'formula': {'CuO': 71.90}}
# A kaolin whose passport carries a trace of titania, after glazy 397 where EP
# Kaolin is the only source of P2O5 in its recipe at 1.45%
TITANIUM_KAOLIN = {'name': 'Каолин с титаном', 'formula': {'Al2O3': 40.00, 'SiO2': 47.00, 'TiO2': 0.05}}
WATER = {'name': 'Вода', 'formula': {}}
CMC = {'name': 'КМЦ', 'formula': {}}
CARBIDE = {'name': 'Карбид кремния', 'formula': {}}

MATERIALS = [FELDSPAR, FELDSPAR_TWIN, SILICA, WHITING, KAOLIN, IRON_OXIDE, ALUMINA,
             KAOLIN_RU, ZIRCON, PIGMENT, WATER, CMC, CARBIDE,
             COBALT_CARBONATE, COPPER_CARBONATE, TITANIUM_KAOLIN]

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


class TestLoadBearingSmallComponents(unittest.TestCase):
    """
    A colourant is not junk for being light

    Weight says nothing about importance: over the Glazy dump, 52.2% of the
    components under 2% are the only source of an oxide in their own recipe.
    Junk is a small component that carries nothing the chemistry could not get
    from the rest of the batch.
    """

    # 0.5% of cobalt is the whole difference between a blue glaze and a clear one
    BLUE = {'Potash Feldspar': 39.5, 'Silica': 30.0, 'Whiting': 20.0,
            'Kaolin': 10.0, 'Углекислый кобальт': 0.5}

    def test_a_light_sole_carrier_is_not_junk(self):
        report = qm.solution_quality(self.BLUE, self.BLUE, MATERIALS)

        self.assertEqual(report['junk']['solution'], 0)
        self.assertEqual(report['small_components']['solution'], 1)
        self.assertTrue(report['junk']['ok'])
        self.assertEqual(report['failures'], [])

    def test_a_light_component_that_carries_nothing_unique_is_junk(self):
        # The 0.37% of feldspar that started this: every oxide it brings is
        # already on the table from the other four materials
        padded = {'Potash Feldspar': 0.37, 'Feldspar Twin': 39.13, 'Silica': 30.0,
                  'Whiting': 20.0, 'Kaolin': 10.5}

        report = qm.solution_quality(padded, BASE_RECIPE, MATERIALS)

        self.assertEqual(report['junk']['solution'], 1)
        self.assertEqual(report['small_components']['solution'], 1)
        self.assertFalse(report['junk']['ok'])
        self.assertIn('junk', report['failures'])

    def test_both_counts_are_reported(self):
        # One light colourant that stays, one light filler that does not
        mixed = {'Potash Feldspar': 39.13, 'Silica': 30.0, 'Whiting': 20.0,
                 'Kaolin': 10.0, 'Углекислый кобальт': 0.5, 'Red Iron Oxide': 0.37}

        report = qm.solution_quality(mixed, mixed, MATERIALS)

        # Iron oxide is not unique - the feldspar carries Fe2O3 as well
        self.assertEqual(report['small_components']['solution'], 2)
        self.assertEqual(report['junk']['solution'], 1)

    def test_sole_carrier_status_belongs_to_the_recipe_not_the_database(self):
        # The same material, load bearing in one recipe and redundant in the
        # next, purely because of what else is in the bucket
        alone = {'Potash Feldspar': 39.5, 'Silica': 30.0, 'Whiting': 20.0,
                 'Kaolin': 10.0, 'Углекислый кобальт': 0.5}
        accompanied = dict(alone, **{'Potash Feldspar': 29.5, 'Zircon': 10.0})

        self.assertEqual(qm.solution_quality(alone, alone, MATERIALS)['junk']['solution'], 0)
        # Zircon brings SiO2 and ZrO2, not CoO, so cobalt stays the sole carrier
        self.assertEqual(qm.solution_quality(accompanied, accompanied, MATERIALS)['junk']['solution'], 0)

        # ... but a second cobalt source takes the exemption away from both
        shared = {'Potash Feldspar': 39.0, 'Silica': 30.0, 'Whiting': 20.0, 'Kaolin': 10.0,
                  'Углекислый кобальт': 0.5, 'Углекислая медь': 0.5}
        shared_report = qm.solution_quality(shared, shared, MATERIALS)
        self.assertEqual(shared_report['small_components']['solution'], 2)
        self.assertEqual(shared_report['junk']['solution'], 0)

    def test_only_oxides_of_the_originals_chemistry_earn_the_exemption(self):
        # Copper the original never asked for is not load bearing, it is noise
        with_copper = {'Potash Feldspar': 39.5, 'Silica': 30.0, 'Whiting': 20.0,
                       'Kaolin': 10.0, 'Углекислая медь': 0.5}

        report = qm.solution_quality(with_copper, BASE_RECIPE, MATERIALS)

        self.assertEqual(report['small_components']['solution'], 1)
        self.assertEqual(report['junk']['solution'], 1)
        self.assertIn('junk', report['failures'])

    def test_an_unanalysable_small_component_is_never_exempt(self):
        # Being unanalysable is not a reason to keep a component: nothing is
        # known to be carried, so nothing unique is carried
        with_pigment = {'Potash Feldspar': 39.5, 'Silica': 30.0, 'Whiting': 20.0,
                        'Kaolin': 10.0, 'Кобальт голубой пигмент 6226': 0.5}

        report = qm.solution_quality(with_pigment, BASE_RECIPE, MATERIALS)

        self.assertEqual(report['junk']['solution'], 1)
        self.assertEqual(report['small_components']['solution'], 1)

    def test_a_trace_oxide_survives_the_umf_rounding_cliff(self):
        # Keying the exemption on the UMF instead of the weight composition
        # would lose 433 of the 17339 exemptions in the Glazy dump, because
        # weights_to_umf() rounds to three decimals and a trace oxide rounds
        # clean out of it. Modelled on glazy 397, where EP Kaolin at 1.45% is
        # the only source of P2O5 in its recipe.
        trace = {'Potash Feldspar': 38.5, 'Silica': 30.0, 'Whiting': 20.0,
                 'Kaolin': 10.0, 'Каолин с титаном': 1.5}

        umf = qm.weights_to_umf(qm.calculate_recipe_composition(MATERIALS, trace))
        report = qm.solution_quality(trace, trace, MATERIALS)

        # The oxide is really there and really invisible in the UMF
        self.assertGreater(qm.calculate_recipe_composition(MATERIALS, trace)['TiO2'], 0.0)
        self.assertEqual(umf.get('TiO2', 0.0), 0.0)
        self.assertEqual(report['junk']['solution'], 0)
        self.assertEqual(report['small_components']['solution'], 1)


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


class TestUnanalysedShare(unittest.TestCase):
    """
    The bound on how much of the batch may have no chemistry behind it

    Excluding oxide-free materials from the conditioning matrix is right - 1% of
    pigment must not condemn a recipe - but unbounded it is a licence. This is
    the bound. Threshold measured over the Glazy corpus: ordinary stain, grog
    and CMC practice lives under a fifth of the batch (median 4.4% among the
    carriers, 90th percentile 19.7%), and the 0.77% of cases above it are built
    on water, grog or a ready-made commercial glaze.
    """

    def test_an_ordinary_pigment_dose_passes(self):
        pigmented = {'Potash Feldspar': 39.0, 'Silica': 30.0, 'Whiting': 20.0,
                     'Kaolin': 10.0, 'Кобальт голубой пигмент 6226': 1.0}

        # Against itself, so that nothing else can colour the verdict: a 1%
        # colorant is normal practice and must clear every gate
        report = qm.solution_quality(pigmented, pigmented, MATERIALS)

        self.assertAlmostEqual(report['unanalysed_share']['solution'], 1.0)
        self.assertAlmostEqual(report['unanalysed_share']['threshold'],
                               qm.MAX_UNANALYSED_SHARE_PERCENT)
        self.assertTrue(report['unanalysed_share']['ok'])
        self.assertEqual(report['failures'], [])

    def test_a_batch_that_is_mostly_unanalysable_fails(self):
        # The case the conditioning exclusion used to catch and then stopped
        # catching: four fifths of the batch is material we cannot analyse, and
        # the conditioning of the one remaining column is a perfect 1.0
        mostly_unanalysable = {'Potash Feldspar': 20.0, 'Кобальт голубой пигмент 6226': 20.0,
                               'Вода': 20.0, 'КМЦ': 20.0, 'Карбид кремния': 20.0}

        report = qm.solution_quality(mostly_unanalysable, BASE_RECIPE, MATERIALS)

        self.assertAlmostEqual(report['unanalysed_share']['solution'], 80.0)
        self.assertFalse(report['unanalysed_share']['ok'])
        self.assertIn('unanalysed_share', report['failures'])
        # ... and the conditioning really is clean, which is the whole point:
        # without this metric nothing would have objected
        self.assertTrue(report['conditioning']['ok'])
        self.assertAlmostEqual(report['conditioning']['cond'], 1.0)

    def test_an_equally_unanalysable_original_grants_no_waiver(self):
        # Unlike min_portion, this rule has no "no worse than the original"
        # escape: a junk record does not make a copy of it judgeable
        mostly_unanalysable = {'Potash Feldspar': 20.0, 'Вода': 40.0, 'КМЦ': 40.0}

        report = qm.solution_quality(mostly_unanalysable, mostly_unanalysable, MATERIALS)

        self.assertAlmostEqual(report['unanalysed_share']['original'], 80.0)
        self.assertFalse(report['unanalysed_share']['ok'])
        self.assertIn('unanalysed_share', report['failures'])

    def test_unknown_materials_count_towards_the_same_share(self):
        # A material we have never heard of contributes no chemistry either, so
        # it is the same hole; which of the two it was stays visible in the
        # two separate lists
        recipe = {'Potash Feldspar': 40.0, 'Silica': 20.0,
                  'Mystery Powder': 25.0, 'Кобальт голубой пигмент 6226': 15.0}

        report = qm.solution_quality(recipe, BASE_RECIPE, MATERIALS)

        self.assertAlmostEqual(report['unanalysed_share']['solution'], 40.0)
        self.assertFalse(report['unanalysed_share']['ok'])
        self.assertEqual(report['unknown_materials'], ['Mystery Powder'])
        self.assertEqual(report['unanalysed_materials'], ['Кобальт голубой пигмент 6226'])

    def test_a_clean_recipe_scores_zero(self):
        report = qm.solution_quality(BASE_RECIPE, BASE_RECIPE, MATERIALS)

        self.assertAlmostEqual(report['unanalysed_share']['solution'], 0.0)
        self.assertAlmostEqual(report['unanalysed_share']['original'], 0.0)
        self.assertTrue(report['unanalysed_share']['ok'])


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

    def test_a_non_utf8_file_does_not_escape(self):
        # A price list saved from a Windows editor in cp1251, or any binary
        # dropped in by accident, must not take down a metric run
        path = os.path.join(self.tmp_dir, 'cp1251.json')
        with open(path, 'wb') as f:
            f.write('{"Мел, CaCO3": 79}'.encode('cp1251'))

        with self.assertLogs('quality_metrics', level='WARNING'):
            self.assertEqual(qm.load_prices(path), {})

    def test_a_non_numeric_price_is_dropped_not_deferred(self):
        # A string price survives the isinstance(dict) check and then explodes
        # inside the cost metric, far away from the file that caused it
        path = os.path.join(self.tmp_dir, 'typo.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'Мел, CaCO3': 'ninety', 'Каолин КЖФ-1': 90,
                       'Кварц': True, 'Тальк': None}, f, ensure_ascii=False)

        with self.assertLogs('quality_metrics', level='WARNING'):
            prices = qm.load_prices(path)

        self.assertEqual(prices, {'Каолин КЖФ-1': 90})

    def test_a_typo_in_the_price_list_only_costs_coverage(self):
        # The end-to-end version of the case above: the metric run completes
        # with a lower coverage instead of raising
        prices = {'Potash Feldspar': 100.0, 'Silica': 'fifty',
                  'Whiting': 40.0, 'Kaolin': 90.0}
        path = os.path.join(self.tmp_dir, 'partial.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(prices, f, ensure_ascii=False)

        with self.assertLogs('quality_metrics', level='WARNING'):
            loaded = qm.load_prices(path)
        report = qm.solution_quality(BASE_RECIPE, BASE_RECIPE, MATERIALS, prices=loaded)

        self.assertAlmostEqual(report['cost']['coverage'], 3.0 / 4.0)
        self.assertIsNone(report['cost']['ratio'])
        self.assertIsNone(report['cost']['ok'])

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
