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
Our forward UMF calculation against the formulas published by segerlab.ru

WHAT THIS IS AN ORACLE OF. This is an oracle of OUR UMF MATH under THEIR
CONVENTION - nothing else. It is NOT a check that the material analyses in
database/materials.json are correct: both sides of the comparison come from the
same source. database/materials.json is a dump of the public SegerLab library
(216 materials, identical ids and analyses) and the eleven recipes below are
their "Основы OVO △6" templates, so an error in an analysis moves our number and
their number by exactly the same amount and this test stays green.

What it does catch is exactly two classes of error, and they are worth a test:

  1. the unity convention drifting - if an oxide silently enters or leaves the
     flux basis, the unity denominator changes and EVERY oxide of EVERY formula
     moves at once, which no single-recipe assertion would localize;
  2. a typo in database/molar_masses.json - a wrong molar mass moves the oxides
     of the affected element and, through the denominator, everything else.

The independence is in the calculation, not in the data: their formulas are
computed by a .NET module in WebAssembly, ours by common.weights_to_umf.

WHY THE "segerlab" PRESET. They count Fe2O3 in the unity denominator; classical
UMF (our default) keeps it in R2O3. That single decision is the whole constant
factor of 1.0022 between their published numbers and ours on every iron-bearing
recipe - see test_the_default_convention_is_offset_by_the_iron below. Our
default does not change; the preset exists so their numbers can be reproduced
digit for digit when cross-checking.

The expected values live in tests/fixtures/segerlab_published_umf.json,
transcribed from the raw API dump (glazes.json, the "segerFormula_computed"
block of each template) rather than typed in here, so that the data stays data.
Their formulas are published rounded to three decimals, hence a tolerance of
0.005 - half of the last printed digit, doubled for the rounding on our side.
"""

import unittest
import sys
import os
import json

# Fix imports by adding parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import flux_oxides, load_materials, load_molar_masses, weights_to_umf
from solver_classic import calculate_recipe_composition

FIXTURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")

# Their formulas are printed to three decimals; half of the last digit on each
# side of the comparison is the whole budget this test allows
TOLERANCE = 0.005


def load_published_formulas():
    """Reads the published SegerLab formulas and their base recipes"""
    path = os.path.join(FIXTURES_DIR, "segerlab_published_umf.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class TestSegerLabConvention(unittest.TestCase):
    """Forward calculation against the eleven published SegerLab formulas"""

    _materials = None

    @classmethod
    def materials(cls):
        """Full material records of database/materials.json (cached)"""
        if cls._materials is None:
            cls._materials = load_materials(only_inventory=False)
        return cls._materials

    def our_umf(self, recipe, convention="segerlab"):
        """Recipe (weight percent) -> UMF, the way the application computes it"""
        composition = calculate_recipe_composition(self.materials(), recipe)
        return weights_to_umf(composition, convention=convention)

    def test_the_fixture_holds_all_eleven_templates(self):
        """A shrinking fixture must fail loudly, not quietly test less"""
        published = load_published_formulas()

        self.assertEqual(len(published), 11)
        self.assertEqual(len({item["id"] for item in published}), 11)

    def test_every_published_formula_is_reproduced(self):
        """Every oxide of every template, within half of their last printed digit"""
        for item in load_published_formulas():
            with self.subTest(recipe=item["name"]):
                expected = item["umf"]
                actual = self.our_umf(item["recipe"])

                # The union of both oxide sets: an oxide we invent out of
                # nowhere is as much a failure as one we lose
                for oxide in sorted(set(expected) | set(actual)):
                    self.assertAlmostEqual(
                        float(actual.get(oxide, 0.0)), float(expected.get(oxide, 0.0)),
                        delta=TOLERANCE,
                        msg=f"'{item['name']}': {oxide} differs from the formula published "
                            f"by SegerLab by more than {TOLERANCE}"
                    )

    def test_the_default_convention_is_offset_by_the_iron(self):
        """The whole difference between the two conventions is Fe2O3 in the basis.

        Not a duplicate of the test above: it pins the CAUSE. If the default
        ever starts reproducing SegerLab as well, someone has moved Fe2O3 into
        our flux basis - a silent change of the classical UMF convention that
        would shift every published number of the project at once.
        """
        transparent = next(item for item in load_published_formulas()
                           if item["id"] == "segerlab_2184")

        self.assertIn("Fe2O3", flux_oxides("segerlab"))
        self.assertNotIn("Fe2O3", flux_oxides())

        theirs = self.our_umf(transparent["recipe"], convention="segerlab")
        ours = self.our_umf(transparent["recipe"], convention=None)

        # Same chemistry, different denominator: our numbers are uniformly larger
        factor = ours["SiO2"] / theirs["SiO2"]
        self.assertAlmostEqual(factor, 1.0022, delta=0.0005)
        for oxide in sorted(set(theirs) & set(ours)):
            if theirs[oxide] < 0.01:
                # Values printed at three decimals carry no ratio worth checking
                continue
            self.assertAlmostEqual(ours[oxide] / theirs[oxide], factor, delta=0.002,
                                   msg=f"{oxide} is not shifted by the same factor")

        # And the factor is literally the iron term of their flux sum
        composition = calculate_recipe_composition(self.materials(), transparent["recipe"])
        molar_masses = load_molar_masses()

        def flux_sum(convention):
            return sum(composition[oxide] / molar_masses[oxide]
                       for oxide in flux_oxides(convention) if oxide in composition)

        self.assertAlmostEqual(flux_sum("segerlab") - flux_sum(None),
                               composition["Fe2O3"] / molar_masses["Fe2O3"], delta=1e-12)
        self.assertAlmostEqual(flux_sum("segerlab") / flux_sum(None), factor, delta=0.0005)


if __name__ == "__main__":
    unittest.main()
