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
from unittest.mock import MagicMock, patch

# Fix imports by adding parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import api_server

# A target both engines solve, taken from the example of API.md
TEST_UMF = {"SiO2": 4, "Al2O3": 1, "Na2O": 0.5, "K2O": 0.5}

# An explicit inventory keeps the test independent of the inInventory flags in
# the database and small enough for the classic engine to stay quick
TEST_INVENTORY = [
    "Нефелин-сиенит VR13",
    "Каолин КЖФ-1",
    "Кварцевая мука Кварцверке W12",
    "Улексит (Химпэк)",
    "Волластонит МИВОЛЛ",
]

# Keys that only iterative_solutions_to_classic_format adds. The classic engine
# has no equivalent of any of them, so their presence in a solution is what
# tells the two engines apart in a response.
ITERATIVE_ONLY_KEYS = ('objective_error', 'unlisted_weight', 'unity_scale')


def solve_payload(**overrides):
    payload = {"umf": TEST_UMF, "inventory": TEST_INVENTORY, "max_solutions": 2}
    payload.update(overrides)
    return payload


class TestSolveEndpointSolverSelection(unittest.TestCase):
    """Which engine POST /api/solve dispatches to, and what it answers"""

    def setUp(self):
        api_server.app.config['TESTING'] = True
        self.client = api_server.app.test_client()

    def post_solve(self, **overrides):
        # The classic engine reports its search on stdout; that is not part of
        # what is being tested and it drowns the test output
        with contextlib.redirect_stdout(io.StringIO()):
            return self.client.post('/api/solve', json=solve_payload(**overrides))

    def dispatch_spies(self):
        """
        Wrap both engine entry points so that the call goes through to the real
        solver and is recorded on the way
        """
        return (
            patch.object(api_server, 'find_best_recipe',
                         MagicMock(wraps=api_server.find_best_recipe)),
            patch.object(api_server, 'find_multiple_solutions',
                         MagicMock(wraps=api_server.find_multiple_solutions)),
        )

    def test_default_dispatches_to_the_iterative_engine(self):
        iterative_patch, classic_patch = self.dispatch_spies()
        with iterative_patch as iterative_spy, classic_patch as classic_spy:
            response = self.post_solve()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(iterative_spy.call_count, 1)
        self.assertEqual(classic_spy.call_count, 0)

    def test_default_response_carries_the_iterative_only_fields(self):
        response = self.post_solve()

        self.assertEqual(response.status_code, 200)
        solutions = response.get_json()
        self.assertIsInstance(solutions, list)
        self.assertGreater(len(solutions), 0)
        for solution in solutions:
            for key in ITERATIVE_ONLY_KEYS:
                self.assertIn(key, solution)

    def test_explicit_iterative_matches_the_default(self):
        # The default is not a separate code path, it is the same engine
        default_solutions = self.post_solve().get_json()
        explicit_solutions = self.post_solve(solver='iterative').get_json()

        self.assertEqual(
            [solution['recipe'] for solution in default_solutions],
            [solution['recipe'] for solution in explicit_solutions]
        )

    def test_explicit_classic_still_works(self):
        iterative_patch, classic_patch = self.dispatch_spies()
        with iterative_patch as iterative_spy, classic_patch as classic_spy:
            response = self.post_solve(solver='classic', min_materials=True,
                                       error_tolerance=0.01)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(classic_spy.call_count, 1)
        self.assertEqual(iterative_spy.call_count, 0)

        solutions = response.get_json()
        self.assertIsInstance(solutions, list)
        for solution in solutions:
            # Every key of the classic format is there ...
            for key in ('recipe', 'error', 'target_composition', 'actual_composition',
                        'weight_composition', 'materials_count', 'recipe_umf'):
                self.assertIn(key, solution)
            # ... and none of the iterative additions is
            for key in ITERATIVE_ONLY_KEYS:
                self.assertNotIn(key, solution)

    def test_unknown_solver_returns_400(self):
        response = self.post_solve(solver='magic')

        self.assertEqual(response.status_code, 400)
        body = response.get_json()
        self.assertEqual(body['error'], 'unknown_solver')
        self.assertIn('magic', body['message'])

    def test_empty_solver_value_is_not_silently_defaulted(self):
        # An empty string is a value, not an absent parameter: falling back to
        # the default here would hide a broken caller
        response = self.post_solve(solver='')

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()['error'], 'unknown_solver')


class TestSensitivityEndpoint(unittest.TestCase):
    """POST /api/sensitivity - the math itself lives in tests/test_sensitivity.py"""

    # The reference "Прозрачная глазурь △6"
    RECIPE = {
        "Нефелин-сиенит VR13": 30,
        "Кварцевая мука Кварцверке W12": 20,
        "Волластонит МИВОЛЛ": 20,
        "Улексит (Химпэк)": 15,
        "Каолин КЖФ-1": 15,
    }

    def setUp(self):
        api_server.app.config['TESTING'] = True
        self.client = api_server.app.test_client()

    def test_returns_the_ranking(self):
        response = self.client.post('/api/sensitivity', json={"recipe": self.RECIPE})

        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        for key in ('umf', 'per_oxide', 'by_material', 'warnings', 'error'):
            self.assertIn(key, body)
        self.assertIsNone(body['error'])
        self.assertEqual(body['by_material'][0]['material'], "Улексит (Химпэк)")

    def test_missing_recipe_returns_400(self):
        response = self.client.post('/api/sensitivity', json={})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()['error'], 'missing_recipe')

    def test_recipe_of_the_wrong_type_returns_400(self):
        response = self.client.post('/api/sensitivity', json={"recipe": ["Улексит (Химпэк)"]})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()['error'], 'missing_recipe')

    def test_fluxless_recipe_returns_422(self):
        response = self.client.post('/api/sensitivity', json={
            "recipe": {"Кварцевая мука Кварцверке W12": 60, "Глинозем, Al203": 40}})

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.get_json()['error'], 'no_fluxes')

    def test_material_outside_an_explicit_inventory_is_reported(self):
        response = self.client.post('/api/sensitivity', json={
            "recipe": self.RECIPE,
            "inventory": ["Нефелин-сиенит VR13", "Кварцевая мука Кварцверке W12"]})

        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual({row['material'] for row in body['by_material']},
                         {"Нефелин-сиенит VR13", "Кварцевая мука Кварцверке W12"})
        self.assertTrue(body['warnings'])

    def test_without_an_inventory_the_whole_database_is_searched(self):
        """A recipe names its own materials; being out of stock is not a reason to drop one"""
        recipe = {"Оксид марганца": 50, "Каолин КЖФ-1": 30, "Кварцевая мука Кварцверке W12": 20}

        response = self.client.post('/api/sensitivity', json={"recipe": recipe})

        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        # "Оксид марганца" is inInventory: false and is still analysed
        self.assertIn("Оксид марганца", {row['material'] for row in body['by_material']})


if __name__ == '__main__':
    unittest.main()
