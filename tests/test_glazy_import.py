#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# flake8: noqa
# pylint: disable=broad-exception-raised, raise-missing-from, too-many-arguments, redefined-outer-name
# pylint: disable=multiple-statements, logging-fstring-interpolation, trailing-whitespace, line-too-long
# pylint: disable=broad-exception-caught, missing-function-docstring, missing-class-docstring
# pylint: disable=f-string-without-interpolation
# pylance: disable=reportMissingImports, reportMissingModuleSource

import copy
import unittest
from unittest import mock
import sys
import os

# Fix imports by adding parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import glazy_import
import api_server
from glazy_import import (
    GlazyImportError,
    parse_recipe_id,
    fetch_recipe,
    build_import_result,
    extract_components,
    _numeric_oxides,
    UMF_BASIS_DIFF_NOTE,
    UMF_BASIS_DIFF_WARNING,
)
from common import weights_to_umf

# A trimmed copy of the real answer for recipe 72382. Trimmed on purpose: the
# full payload is 119 KB of data this module does not read, and the tests must
# never reach glazy.org - CI has no business depending on a third-party site.
SAMPLE_RECIPE = {
    "id": 72382,
    "name": "OVO Perfect Matte (40-25-10)",
    "fromOrtonConeName": "5&#189;",
    "toOrtonConeName": "7",
    "materialComponents": [
        {"percentageAmount": "40.0000", "isAdditional": False, "material": {"id": 20668, "name": "Полевой шпат FFF"}},
        {"percentageAmount": "25.0000", "isAdditional": False, "material": {"id": 20485, "name": "Волластонит (МИВОЛЛ)"}},
        {"percentageAmount": "5.0000", "isAdditional": True, "material": {"id": 15393, "name": "Rutile"}},
    ],
    "analysis": {
        "percentageAnalysis": {
            "SiO2": "48.5952", "Al2O3": "18.4238", "B2O3": "3.5238", "Na2O": "2.1238",
            "K2O": "3.1619", "MgO": "0.4876", "CaO": "13.0286", "SrO": "0.0952",
            "TiO2": "4.2882", "Fe2O3": "0.5770", "loi": "0.4310",
        },
        "umfAnalysis": {
            "SiO2": "2.5824", "Al2O3": "0.5770", "B2O3": "0.1616", "Na2O": "0.1094",
            "K2O": "0.1072", "MgO": "0.0386", "CaO": "0.7418", "SrO": "0.0029",
            "TiO2": "0.1714", "Fe2O3": "0.0115",
            "SiO2Al2O3Ratio": "4.4759", "R2OTotal": "0.2166", "ROTotal": "0.7834",
            "xAl2O3": "0.5770", "SiO2xAl2O3Ratio": "4.4759",
        },
        "thermalExpansion": "7.9340",
    },
}

# A lead glaze: PbO carries the melt and the alkali content is tiny. PbO is a
# flux for Glazy and, since the flux list of common.oxides_classification() was
# completed (DATA_NOTES.md, section 2; TZ_SOLVER_V2.md, section 10.13), for us
# as well - both sides normalize this analysis on PbO + K2O and arrive at the
# same numbers. It used to be the standing example of a divergence: with PbO
# counting for nobody here our unity rested on the 1.5% K2O alone and every
# oxide came out ~18x too large. The umfAnalysis below is what this weight
# analysis reads as on the flux basis of Glazy (the "glazy" preset of
# database/oxide_classification.json).
LEAD_RECIPE = {
    "id": 1234,
    "name": "Lead Base",
    "analysis": {
        "percentageAnalysis": {
            "PbO": "60.0000", "SiO2": "31.0000", "Al2O3": "7.5000", "K2O": "1.5000",
        },
        "umfAnalysis": {
            "SiO2": "1.8121", "Al2O3": "0.2583", "PbO": "0.9441", "K2O": "0.0559",
        },
    },
}

# A copper glaze: what a genuine divergence of the two flux bases looks like now
# that PbO is settled. common.oxides_classification() counts the colorants
# (CuO/CoO/NiO/FeO/MnO2) among the fluxes, Glazy keeps them out of the unity
# denominator - so our denominator is the larger one and every oxide of ours
# comes out below the value on glazy.org. That is expected and is NOT an error:
# the solver evaluates its candidates with the same functions, so the target
# stays self-consistent - the import must keep computing it from the weights and
# only report how far the two bases are apart. The umfAnalysis is again the
# "glazy" preset reading of the same weight analysis.
COPPER_RECIPE = {
    "id": 5678,
    "name": "Copper Green",
    "analysis": {
        "percentageAnalysis": {
            "SiO2": "62.0000", "Al2O3": "11.0000", "CaO": "11.0000", "K2O": "4.0000",
            "Na2O": "2.0000", "MgO": "1.0000", "CuO": "6.0000",
        },
        "umfAnalysis": {
            "SiO2": "3.4900", "Al2O3": "0.3649", "CaO": "0.6634", "K2O": "0.1436",
            "Na2O": "0.1091", "MgO": "0.0839", "CuO": "0.2551",
        },
    },
}

# An alumina-silica kiln wash: no oxide of our r2o/ro lists at all, so
# weights_to_umf() would fall back to "the smallest oxide is unity", a basis the
# solver never reproduces. The flux list of Glazy is a subset of ours (it only
# leaves the colorants out), so Glazy has nothing to normalize on either and the
# umfAnalysis below is a basis of its own - which is precisely why the import
# may pass those numbers on only as umf_source "glazy_umf", for the UI to flag.
FLUXLESS_RECIPE = {
    "id": 4321,
    "name": "Kiln Wash",
    "analysis": {
        "percentageAnalysis": {"Al2O3": "48.0000", "SiO2": "49.5000", "TiO2": "2.5000"},
        "umfAnalysis": {"Al2O3": "1.0000", "SiO2": "1.7499", "TiO2": "0.0665"},
    },
}


class FakeResponse:
    """Minimal stand-in for a requests response"""

    def __init__(self, payload, status_code=200, valid_json=True):
        self.payload = payload
        self.status_code = status_code
        self.valid_json = valid_json

    def json(self):
        if not self.valid_json:
            raise ValueError("no json object could be decoded")
        return self.payload


def fake_get(payload, status_code=200, valid_json=True):
    """Patch requests.get inside glazy_import with a fixed answer"""
    return mock.patch.object(
        glazy_import.requests, 'get',
        return_value=FakeResponse(payload, status_code, valid_json))


class TestParseRecipeId(unittest.TestCase):

    def test_accepts_numbers_and_plain_strings(self):
        self.assertEqual(parse_recipe_id(72382), 72382)
        self.assertEqual(parse_recipe_id("72382"), 72382)
        self.assertEqual(parse_recipe_id("  72382  "), 72382)

    def test_accepts_recipe_urls(self):
        self.assertEqual(parse_recipe_id("https://glazy.org/recipes/72382"), 72382)
        self.assertEqual(parse_recipe_id("https://glazy.org/recipes/72382/some-slug"), 72382)
        self.assertEqual(parse_recipe_id("https://glazy.org/recipes/72382?ref=search"), 72382)
        self.assertEqual(parse_recipe_id("https://glazy.org/recipes/72382#analysis"), 72382)
        self.assertEqual(parse_recipe_id("glazy.org/recipes/72382"), 72382)

    def test_accepts_the_api_url_of_a_recipe(self):
        self.assertEqual(parse_recipe_id("https://api.glazy.org/api/recipes/72382"), 72382)
        self.assertEqual(parse_recipe_id("https://www.glazy.org/recipes/72382"), 72382)

    def test_rejects_urls_that_only_mention_recipes(self):
        """A recipe id is only read from a /recipes/<id> link on glazy.org itself"""
        # Another host: whatever its recipe 1 is, it is not the one of Glazy
        self.assertIsNone(parse_recipe_id("https://example.com/recipes/1"))
        # A search result page: the id belongs to the query, not to the page
        self.assertIsNone(parse_recipe_id("https://glazy.org/search?q=recipes/12"))
        # The segment must be a whole one
        self.assertIsNone(parse_recipe_id("prefixrecipes/42"))
        # A material page listing recipes: importing 999 would be an unrelated recipe
        self.assertIsNone(parse_recipe_id("https://glazy.org/materials/123/recipes/999"))
        self.assertIsNone(parse_recipe_id("https://glazy.org/user/123/recipes/72382"))

    def test_rejects_everything_else(self):
        self.assertIsNone(parse_recipe_id(""))
        self.assertIsNone(parse_recipe_id("   "))
        self.assertIsNone(parse_recipe_id("abc"))
        self.assertIsNone(parse_recipe_id("https://glazy.org/materials/123"))
        self.assertIsNone(parse_recipe_id(0))
        self.assertIsNone(parse_recipe_id(-5))
        self.assertIsNone(parse_recipe_id("-5"))
        self.assertIsNone(parse_recipe_id(None))
        self.assertIsNone(parse_recipe_id(True))
        self.assertIsNone(parse_recipe_id(["72382"]))


class TestNumericOxides(unittest.TestCase):

    def test_derived_keys_are_dropped(self):
        oxides = _numeric_oxides(SAMPLE_RECIPE["analysis"]["umfAnalysis"])

        for derived in ("SiO2Al2O3Ratio", "R2OTotal", "ROTotal", "xAl2O3", "SiO2xAl2O3Ratio"):
            self.assertNotIn(derived, oxides)
        self.assertIn("SiO2", oxides)

    def test_loi_is_dropped(self):
        oxides = _numeric_oxides(SAMPLE_RECIPE["analysis"]["percentageAnalysis"])

        self.assertNotIn("loi", oxides)
        self.assertEqual(oxides["SiO2"], 48.5952)

    def test_string_values_become_floats(self):
        oxides = _numeric_oxides({"SiO2": "48.5952", "CaO": "13.0286"})

        self.assertEqual(oxides, {"SiO2": 48.5952, "CaO": 13.0286})

    def test_unusable_values_are_skipped(self):
        oxides = _numeric_oxides({
            "SiO2": "48.5952",
            "Al2O3": "не число",
            "CaO": None,
            "MgO": "0",
            "K2O": "-1.5",
            "Na2O": "NaN",
            "TiO2": "Infinity",
        })

        self.assertEqual(oxides, {"SiO2": 48.5952})

    def test_missing_analysis_gives_an_empty_dict(self):
        self.assertEqual(_numeric_oxides(None), {})
        self.assertEqual(_numeric_oxides("SiO2: 48"), {})


class TestExtractComponents(unittest.TestCase):

    def test_components_keep_their_order_and_flags(self):
        components = extract_components(SAMPLE_RECIPE)

        self.assertEqual(components, [
            {"name": "Полевой шпат FFF", "percentage": 40.0, "is_additional": False, "glazy_material_id": 20668},
            {"name": "Волластонит (МИВОЛЛ)", "percentage": 25.0, "is_additional": False, "glazy_material_id": 20485},
            {"name": "Rutile", "percentage": 5.0, "is_additional": True, "glazy_material_id": 15393},
        ])

    def test_unparsable_percentages_are_skipped(self):
        recipe = {"materialComponents": [
            {"percentageAmount": "нет", "isAdditional": False, "material": {"id": 1, "name": "Bad"}},
            {"percentageAmount": "10.0000", "isAdditional": False, "material": {"name": "Good"}},
        ]}

        self.assertEqual(extract_components(recipe), [
            {"name": "Good", "percentage": 10.0, "is_additional": False, "glazy_material_id": None},
        ])

    def test_non_positive_percentages_are_skipped(self):
        """The card is a reference of the original, "-5.0%" would be worse than nothing"""
        recipe = {"materialComponents": [
            {"percentageAmount": "-5", "isAdditional": False, "material": {"name": "Negative"}},
            {"percentageAmount": "0", "isAdditional": True, "material": {"name": "Zero"}},
            {"percentageAmount": "0.05", "isAdditional": True, "material": {"name": "Tiny"}},
        ]}

        self.assertEqual(extract_components(recipe), [
            {"name": "Tiny", "percentage": 0.05, "is_additional": True, "glazy_material_id": None},
        ])

    def test_html_entities_in_material_names_are_unescaped(self):
        recipe = {"materialComponents": [
            {"percentageAmount": "100", "isAdditional": False,
             "material": {"id": 7, "name": "Tom&#39;s Clay &amp; Co"}},
        ]}

        self.assertEqual(extract_components(recipe)[0]["name"], "Tom's Clay & Co")

    def test_recipe_without_components(self):
        self.assertEqual(extract_components({}), [])


class TestFetchRecipe(unittest.TestCase):

    def test_returns_the_data_dictionary(self):
        with fake_get({"data": SAMPLE_RECIPE}):
            self.assertEqual(fetch_recipe(72382), SAMPLE_RECIPE)

    def test_null_error_key_is_not_an_error(self):
        """A successful answer may carry "error": null next to its data"""
        with fake_get({"error": None, "data": SAMPLE_RECIPE}):
            self.assertEqual(fetch_recipe(72382), SAMPLE_RECIPE)

    def test_error_body_with_http_200_is_an_error(self):
        """Glazy answers 200 with the real status inside the body"""
        body = {"error": {"message": "Recipe does not exist", "status_code": 404}}

        with fake_get(body, status_code=200):
            with self.assertRaises(GlazyImportError) as caught:
                fetch_recipe(999999999)

        self.assertEqual(caught.exception.code, "glazy_not_found")
        self.assertEqual(caught.exception.http_status, 404)
        self.assertEqual(caught.exception.message, "Recipe does not exist")

    def test_private_recipe_is_forbidden(self):
        body = {"error": {"message": "Unauthorized", "status_code": 403}}

        with fake_get(body, status_code=200):
            with self.assertRaises(GlazyImportError) as caught:
                fetch_recipe(72382)

        self.assertEqual(caught.exception.code, "glazy_forbidden")
        self.assertEqual(caught.exception.http_status, 403)

    def test_unauthenticated_recipe_is_forbidden_too(self):
        body = {"error": {"message": "Unauthenticated", "status_code": 401}}

        with fake_get(body, status_code=200):
            with self.assertRaises(GlazyImportError) as caught:
                fetch_recipe(72382)

        self.assertEqual(caught.exception.code, "glazy_forbidden")
        self.assertEqual(caught.exception.http_status, 403)

    def test_bare_string_error_body(self):
        with fake_get({"error": "something went wrong"}, status_code=200):
            with self.assertRaises(GlazyImportError) as caught:
                fetch_recipe(72382)

        self.assertEqual(caught.exception.code, "glazy_unavailable")
        self.assertEqual(caught.exception.http_status, 502)
        self.assertEqual(caught.exception.message, "something went wrong")

    def test_http_error_without_error_body(self):
        with fake_get({}, status_code=500):
            with self.assertRaises(GlazyImportError) as caught:
                fetch_recipe(72382)

        self.assertEqual(caught.exception.code, "glazy_unavailable")
        self.assertEqual(caught.exception.http_status, 502)

    def test_non_json_answer(self):
        with fake_get(None, valid_json=False):
            with self.assertRaises(GlazyImportError) as caught:
                fetch_recipe(72382)

        self.assertEqual(caught.exception.code, "glazy_unavailable")
        self.assertEqual(caught.exception.http_status, 502)

    def test_answer_without_data(self):
        with fake_get({"data": {}}):
            with self.assertRaises(GlazyImportError) as caught:
                fetch_recipe(72382)

        self.assertEqual(caught.exception.code, "glazy_unavailable")
        self.assertEqual(caught.exception.http_status, 502)

    def test_timeout(self):
        with mock.patch.object(glazy_import.requests, 'get',
                               side_effect=glazy_import.requests.Timeout("timed out")):
            with self.assertRaises(GlazyImportError) as caught:
                fetch_recipe(72382)

        self.assertEqual(caught.exception.code, "glazy_timeout")
        self.assertEqual(caught.exception.http_status, 504)

    def test_connection_failure(self):
        with mock.patch.object(glazy_import.requests, 'get',
                               side_effect=glazy_import.requests.ConnectionError("dns failure")):
            with self.assertRaises(GlazyImportError) as caught:
                fetch_recipe(72382)

        self.assertEqual(caught.exception.code, "glazy_unavailable")
        self.assertEqual(caught.exception.http_status, 502)


class TestBuildImportResult(unittest.TestCase):

    def test_target_umf_is_computed_from_the_weight_analysis(self):
        result = build_import_result(SAMPLE_RECIPE)

        expected_weights = _numeric_oxides(SAMPLE_RECIPE["analysis"]["percentageAnalysis"])

        self.assertEqual(result["umf_source"], "weights")
        self.assertEqual(result["weight_percent"], expected_weights)
        self.assertEqual(result["umf"], weights_to_umf(expected_weights))

    def test_target_umf_stays_close_to_the_one_of_glazy(self):
        """Without colorants both flux bases agree, so the check is meaningful"""
        result = build_import_result(SAMPLE_RECIPE)

        for oxide, glazy_value in result["umf_glazy"].items():
            self.assertAlmostEqual(result["umf"][oxide], glazy_value, delta=0.01,
                                   msg=f"{oxide} diverges too much from the UMF of Glazy")

    def test_umf_basis_diff_is_negligible_on_a_normal_recipe(self):
        result = build_import_result(SAMPLE_RECIPE)

        self.assertLessEqual(result["umf_basis_diff"], UMF_BASIS_DIFF_NOTE)

    def test_umf_basis_diff_is_null_without_the_umf_of_glazy(self):
        """Nothing of Glazy to compare against, so the UI must stay silent"""
        recipe = copy.deepcopy(SAMPLE_RECIPE)
        del recipe["analysis"]["umfAnalysis"]

        result = build_import_result(recipe)

        self.assertIsNone(result["umf_basis_diff"])
        self.assertEqual(result["umf_glazy"], {})

    def test_lead_recipe_target_still_comes_from_the_weights(self):
        """
        A lead glaze is imported like any other, and PbO now carries its unity

        What is pinned here is the SOURCE of the target: it is our own
        conversion of the weight analysis and never the umfAnalysis of Glazy,
        because the solver measures its candidates with the very same function.
        The scale is the regression guard on top: PbO belongs to the flux list
        now, so this recipe normalizes on PbO + K2O and lands on the numbers of
        Glazy instead of the ~18x inflation the missing PbO used to produce.
        """
        result = build_import_result(LEAD_RECIPE)

        expected_weights = _numeric_oxides(LEAD_RECIPE["analysis"]["percentageAnalysis"])

        self.assertEqual(result["umf_source"], "weights")
        self.assertEqual(result["umf"], weights_to_umf(expected_weights))
        # PbO + K2O is the unity basis and the lead is nearly all of it
        self.assertAlmostEqual(result["umf"]["PbO"] + result["umf"]["K2O"], 1.0, places=3)
        self.assertGreater(result["umf"]["PbO"], 0.9)
        # Same basis on both sides now, so the UI has nothing to warn about
        self.assertLessEqual(result["umf_basis_diff"], UMF_BASIS_DIFF_NOTE)

    def test_colorant_recipe_target_still_comes_from_the_weights(self):
        """
        A colorant glaze diverges from Glazy by design and must NOT be refused

        CuO is a flux of common.oxides_classification() and not one of Glazy, so
        our unity denominator is the larger of the two and every oxide of ours
        comes out below the value glazy.org shows - far enough apart for the UI
        to warn prominently. The solver measures its candidates with the same
        functions, so the target is self-consistent and reproducible: the import
        reports the divergence instead of failing, and a target in a basis the
        solver does not use is exactly what it must not fall back to.
        """
        result = build_import_result(COPPER_RECIPE)

        expected_weights = _numeric_oxides(COPPER_RECIPE["analysis"]["percentageAnalysis"])

        self.assertEqual(result["umf_source"], "weights")
        self.assertEqual(result["umf"], weights_to_umf(expected_weights))
        for oxide, glazy_value in result["umf_glazy"].items():
            self.assertLess(result["umf"][oxide], glazy_value,
                            f"{oxide} is not below its Glazy value, the bases do not differ")
        self.assertGreater(result["umf_basis_diff"], UMF_BASIS_DIFF_WARNING)

    def test_recipe_without_our_fluxes_falls_back_to_the_umf_of_glazy(self):
        """
        weights_to_umf() would normalize on "the smallest oxide", an arbitrary
        basis the solver never reproduces, so the weights path is skipped
        """
        result = build_import_result(FLUXLESS_RECIPE)

        self.assertEqual(result["umf_source"], "glazy_umf")
        self.assertEqual(result["umf"], result["umf_glazy"])
        # The weight analysis is still reported, only not used as the target
        self.assertEqual(result["weight_percent"]["Al2O3"], 48.0)
        self.assertEqual(result["umf_basis_diff"], 0)

    def test_recipe_without_our_fluxes_and_without_a_glazy_umf(self):
        recipe = copy.deepcopy(FLUXLESS_RECIPE)
        del recipe["analysis"]["umfAnalysis"]

        with self.assertRaises(GlazyImportError) as caught:
            build_import_result(recipe)

        self.assertEqual(caught.exception.code, "no_analysis")
        self.assertEqual(caught.exception.http_status, 422)

    def test_html_entities_in_the_recipe_name_are_unescaped(self):
        recipe = copy.deepcopy(SAMPLE_RECIPE)
        recipe["name"] = "Tom&#39;s Glaze &amp; Co"

        self.assertEqual(build_import_result(recipe)["name"], "Tom's Glaze & Co")

    def test_glazy_umf_is_passed_through_without_derived_keys(self):
        result = build_import_result(SAMPLE_RECIPE)

        self.assertEqual(result["umf_glazy"]["SiO2"], 2.5824)
        self.assertNotIn("SiO2Al2O3Ratio", result["umf_glazy"])

    def test_recipe_metadata(self):
        result = build_import_result(SAMPLE_RECIPE)

        self.assertEqual(result["id"], 72382)
        self.assertEqual(result["name"], "OVO Perfect Matte (40-25-10)")
        self.assertEqual(result["url"], "https://glazy.org/recipes/72382")
        self.assertEqual(result["thermal_expansion"], 7.934)
        # HTML entities of Glazy are unescaped on the server
        self.assertEqual(result["cone_from"], "5½")
        self.assertEqual(result["cone_to"], "7")

    def test_components_are_included(self):
        result = build_import_result(SAMPLE_RECIPE)

        self.assertEqual(len(result["components"]), 3)
        self.assertEqual(result["components"][0]["name"], "Полевой шпат FFF")
        self.assertFalse(result["components"][0]["is_additional"])
        self.assertTrue(result["components"][2]["is_additional"])

    def test_missing_cones_and_expansion_are_null(self):
        recipe = copy.deepcopy(SAMPLE_RECIPE)
        del recipe["fromOrtonConeName"]
        del recipe["toOrtonConeName"]
        del recipe["analysis"]["thermalExpansion"]

        result = build_import_result(recipe)

        self.assertIsNone(result["cone_from"])
        self.assertIsNone(result["cone_to"])
        self.assertIsNone(result["thermal_expansion"])

    def test_falls_back_to_the_umf_of_glazy(self):
        """Without a weight analysis the target is taken as is, and says so"""
        recipe = copy.deepcopy(SAMPLE_RECIPE)
        recipe["analysis"]["percentageAnalysis"] = {"loi": "0.4310"}

        result = build_import_result(recipe)

        self.assertEqual(result["umf_source"], "glazy_umf")
        self.assertEqual(result["umf"], result["umf_glazy"])
        self.assertEqual(result["weight_percent"], {})

    def test_recipe_without_any_analysis(self):
        recipe = copy.deepcopy(SAMPLE_RECIPE)
        recipe["analysis"] = {}

        with self.assertRaises(GlazyImportError) as caught:
            build_import_result(recipe)

        self.assertEqual(caught.exception.code, "no_analysis")
        self.assertEqual(caught.exception.http_status, 422)

    def test_requested_id_is_used_when_the_payload_has_none(self):
        recipe = copy.deepcopy(SAMPLE_RECIPE)
        del recipe["id"]

        result = build_import_result(recipe, 72382)

        self.assertEqual(result["id"], 72382)
        self.assertEqual(result["url"], "https://glazy.org/recipes/72382")


class TestGlazyImportEndpoint(unittest.TestCase):

    def setUp(self):
        self.client = api_server.app.test_client()

    def post(self, payload):
        return self.client.post('/api/glazy_import', json=payload)

    def test_successful_import(self):
        with mock.patch.object(api_server, 'fetch_recipe', return_value=SAMPLE_RECIPE) as fetch:
            response = self.post({"recipe": "https://glazy.org/recipes/72382"})

        fetch.assert_called_once_with(72382)
        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertEqual(data["id"], 72382)
        self.assertEqual(data["name"], "OVO Perfect Matte (40-25-10)")
        self.assertEqual(data["url"], "https://glazy.org/recipes/72382")
        self.assertEqual(data["umf_source"], "weights")
        self.assertEqual(data["umf"], weights_to_umf(_numeric_oxides(SAMPLE_RECIPE["analysis"]["percentageAnalysis"])))
        self.assertEqual(data["umf_glazy"]["SiO2"], 2.5824)
        self.assertLessEqual(data["umf_basis_diff"], UMF_BASIS_DIFF_NOTE)
        self.assertEqual(data["weight_percent"]["SiO2"], 48.5952)
        self.assertEqual(data["components"][2], {
            "name": "Rutile", "percentage": 5.0, "is_additional": True, "glazy_material_id": 15393,
        })
        self.assertEqual(data["cone_from"], "5½")
        self.assertEqual(data["cone_to"], "7")
        self.assertEqual(data["thermal_expansion"], 7.934)

    def test_recipe_id_parameter_is_accepted(self):
        with mock.patch.object(api_server, 'fetch_recipe', return_value=SAMPLE_RECIPE) as fetch:
            response = self.post({"recipe_id": 72382})

        fetch.assert_called_once_with(72382)
        self.assertEqual(response.status_code, 200)

    def test_explicit_null_recipe_falls_back_to_recipe_id(self):
        with mock.patch.object(api_server, 'fetch_recipe', return_value=SAMPLE_RECIPE) as fetch:
            response = self.post({"recipe": None, "recipe_id": 72382})

        fetch.assert_called_once_with(72382)
        self.assertEqual(response.status_code, 200)

    def test_recipe_on_another_flux_basis_is_imported_and_flagged(self):
        """The divergence is reported as a number, the import itself succeeds"""
        with mock.patch.object(api_server, 'fetch_recipe', return_value=COPPER_RECIPE):
            response = self.post({"recipe": "5678"})

        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertEqual(data["umf_source"], "weights")
        self.assertGreater(data["umf_basis_diff"], UMF_BASIS_DIFF_WARNING)

    def test_missing_recipe(self):
        response = self.post({})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["error"], "missing_recipe")

    def test_empty_recipe_is_missing_too(self):
        response = self.post({"recipe": "   "})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["error"], "missing_recipe")

    def test_invalid_recipe_id(self):
        response = self.post({"recipe": "https://glazy.org/materials/123"})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["error"], "invalid_recipe_id")

    def test_glazy_error_is_rendered_with_its_status(self):
        error = GlazyImportError("glazy_not_found", "Recipe does not exist", 404)

        with mock.patch.object(api_server, 'fetch_recipe', side_effect=error):
            response = self.post({"recipe": "999999999"})

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.get_json(), {
            "error": "glazy_not_found", "message": "Recipe does not exist"})

    def test_unexpected_failure_is_a_server_error(self):
        with mock.patch.object(api_server, 'fetch_recipe', side_effect=RuntimeError("boom")):
            response = self.post({"recipe": "72382"})

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.get_json()["error"], "server_error")


if __name__ == "__main__":
    unittest.main()
