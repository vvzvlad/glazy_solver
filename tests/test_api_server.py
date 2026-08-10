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

    def test_the_answer_is_an_object_carrying_solutions_and_warnings(self):
        """
        The shape itself, because it changed: the endpoint used to answer the
        bare list, and it cannot, because "we ignored an oxide you asked for"
        has nowhere to live in a list - see solve_recipe.
        """
        response = self.post_solve()

        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertIsInstance(body, dict)
        self.assertIsInstance(body['solutions'], list)
        self.assertIsInstance(body['warnings'], list)

    def test_default_response_carries_the_iterative_only_fields(self):
        response = self.post_solve()

        self.assertEqual(response.status_code, 200)
        solutions = response.get_json()['solutions']
        self.assertIsInstance(solutions, list)
        self.assertGreater(len(solutions), 0)
        for solution in solutions:
            for key in ITERATIVE_ONLY_KEYS:
                self.assertIn(key, solution)

    def test_explicit_iterative_matches_the_default(self):
        # The default is not a separate code path, it is the same engine
        default_solutions = self.post_solve().get_json()['solutions']
        explicit_solutions = self.post_solve(solver='iterative').get_json()['solutions']

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

        solutions = response.get_json()['solutions']
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


class TestSolveWarnsAboutOxidesItRefused(unittest.TestCase):
    """
    An oxide the request asked for and the answer does not fit is SAID

    The target is cleaned ONCE, at the endpoint, above the choice of engine, and
    these tests are mostly about that rather than about the wording. The two
    engines do not refuse the same things - the iterative one drops an unknown
    NAME and a bad VALUE, the classic one drops only the name and feeds a
    negative straight into its NNLS - so a warning drawn from one engine's rule
    and handed to the other described a target nobody fitted.
    """

    UNKNOWN = 'Unobtainium'
    DIRTY_TARGET = dict(TEST_UMF, **{UNKNOWN: 0.2})
    # A name both engines know, carrying a value neither may fit
    NEGATIVE_TARGET = dict(TEST_UMF, Fe2O3=-0.5)
    CLASSIC = dict(solver='classic', min_materials=True, error_tolerance=0.01)

    def setUp(self):
        api_server.app.config['TESTING'] = True
        self.client = api_server.app.test_client()

    def post_solve(self, umf, **overrides):
        payload = solve_payload(**overrides)
        payload['umf'] = umf
        with contextlib.redirect_stdout(io.StringIO()):
            return self.client.post('/api/solve', json=payload)

    def post_raw(self, body):
        """
        POST a body EXACTLY as written

        The json= argument of the test client serializes through the app's JSON
        provider, which sorts keys - so a target typed in one order arrives in
        another, and any test about order passes without testing anything.
        """
        with contextlib.redirect_stdout(io.StringIO()):
            return self.client.post('/api/solve', data=body,
                                    content_type='application/json')

    def test_an_unfittable_oxide_is_named_in_the_warnings(self):
        body = self.post_solve(self.DIRTY_TARGET).get_json()

        self.assertEqual(len(body['warnings']), 1, "one warning per request, not one per step")
        self.assertIn(self.UNKNOWN, body['warnings'][0])
        # ...and the request is still answered, because the rest of it is fine
        self.assertTrue(body['solutions'])
        self.assertNotIn(self.UNKNOWN, body['solutions'][0]['target_composition'])

    def test_both_engines_drop_an_unknown_name_out_of_the_reported_target(self):
        """
        Documented for the ENDPOINT, so it has to hold for both engines

        The classic one echoes back whatever target it was handed, so before the
        cleaning moved above the fork it returned 'Unobtainium': 0.2 in
        target_composition next to a warning saying the key was not taken into
        account. A consumer computing "requested minus target_composition" - the
        deduction the field's own documentation suggests - got an empty list.
        """
        for engine in ({}, self.CLASSIC):
            with self.subTest(engine=engine.get('solver', 'default')):
                body = self.post_solve(self.DIRTY_TARGET, **engine).get_json()

                self.assertTrue(body['solutions'])
                for solution in body['solutions']:
                    self.assertNotIn(self.UNKNOWN, solution['target_composition'])

    def test_the_classic_engine_is_answered_the_same_way(self):
        """The refusal is a property of the target, not of the engine"""
        body = self.post_solve(self.DIRTY_TARGET, **self.CLASSIC).get_json()

        self.assertEqual(len(body['warnings']), 1)
        self.assertIn(self.UNKNOWN, body['warnings'][0])

    def test_a_refused_value_is_refused_by_the_engine_that_would_have_used_it(self):
        """
        The case the name-only fixture above cannot see

        A negative UMF is not an unknown name: umf_to_weights lets it through
        and the classic engine fits it. So the endpoint used to answer "Fe2O3 не
        учтён" over a recipe computed FROM Fe2O3 = -0.5, and every material in
        it differed from the answer for the same target with the key removed.
        The assertion is exactly that: warned about AND actually gone, which is
        checkable without pinning a recipe - the answer must equal the answer to
        the target that never carried the key.
        """
        clean_target = {oxide: value for oxide, value in self.NEGATIVE_TARGET.items()
                        if oxide != 'Fe2O3'}

        for engine in ({}, self.CLASSIC):
            with self.subTest(engine=engine.get('solver', 'default')):
                dirty = self.post_solve(self.NEGATIVE_TARGET, **engine).get_json()
                clean = self.post_solve(clean_target, **engine).get_json()

                self.assertEqual(len(dirty['warnings']), 1)
                self.assertIn('Fe2O3', dirty['warnings'][0])
                self.assertEqual([s['recipe'] for s in dirty['solutions']],
                                 [s['recipe'] for s in clean['solutions']],
                                 "the engine fitted a target the warning says was not fitted")

    def test_feasibility_says_the_very_same_sentence(self):
        """
        The two endpoints answering one input differently is the bug this fixes,
        so the agreement is asserted rather than left to two string literals
        drifting apart.

        TWO refused oxides, in a raw body, in non-alphabetical order: with one
        oxide the order of the names is not observable, and through json= the
        two sides agree only because Flask sorted the keys on the way in. Both
        sides sort the names themselves; this is what says so.
        """
        raw = ('{"umf": {"Zz": 0.2, "SiO2": 4, "Aa": 0.1, "Al2O3": 1, '
               '"Na2O": 0.5, "K2O": 0.5}}')
        solve = self.post_raw(raw).get_json()
        feasibility = self.client.post('/api/feasibility',
                                       data=raw.replace('{"umf"', '{"ranges": false, "umf"'),
                                       content_type='application/json').get_json()

        self.assertEqual(len(solve['warnings']), 1)
        self.assertIn('Aa, Zz', solve['warnings'][0], "the names are not in a stable order")
        self.assertIn(solve['warnings'][0], feasibility['warnings'])

    def test_a_target_of_nothing_but_unknown_oxides_still_says_why(self):
        """
        The case a per-solution field could not have covered: there is no
        solution to hang it on, and that is exactly when the caller needs it

        422 and not 200: nothing is left to solve, the client can fix it, and
        the two engines cannot be left to answer it themselves - one returns an
        empty list and the other dies inside numpy.matrix_rank on a zero-row
        problem, so the same request came back 200 or 500 depending on a
        parameter that has nothing to do with the target.
        """
        for engine in ({}, self.CLASSIC):
            with self.subTest(engine=engine.get('solver', 'default')):
                response = self.post_solve({self.UNKNOWN: 1.0}, **engine)
                body = response.get_json()

                self.assertEqual(response.status_code, 422)
                self.assertEqual(body['error'], 'empty_target')
                self.assertEqual(len(body['warnings']), 1)
                self.assertIn(self.UNKNOWN, body['warnings'][0])

    def test_a_target_of_nothing_but_refused_values_is_answered_the_same_way(self):
        """
        The half that a fixture of unknown NAMES cannot show

        For names the classic engine crashed before this change too. For values
        it did not: it fitted them, so {"SiO2": -4, "Al2O3": -1} used to come
        back 200 with a recipe. Both roads now lead to the same 422 rather than
        to a 500 on one engine and a recipe on the other.
        """
        for engine in ({}, self.CLASSIC):
            with self.subTest(engine=engine.get('solver', 'default')):
                response = self.post_solve({"SiO2": -4, "Al2O3": -1}, **engine)
                body = response.get_json()

                self.assertEqual(response.status_code, 422)
                self.assertEqual(body['error'], 'empty_target')
                self.assertIn('Al2O3, SiO2', body['warnings'][0])

    def test_a_target_that_is_not_a_non_empty_object_is_refused_before_anything_reads_it(self):
        """
        Three bodies, one answer, and none of them reaches an engine

        What each engine used to do with them differed, and neither was "tell
        the caller what is wrong": on a target that is not an object the classic
        engine died on `'list' object has no attribute 'keys'` (500) while the
        iterative one answered 200 with an empty list; on {} the classic engine
        died inside numpy.matrix_rank and the iterative one, again, answered 200
        with an empty list. All three are now the 400 and the sentence
        /api/feasibility answers for the same bodies.
        """
        for body in ([], 0, "SiO2", None, {}):
            for engine in ({}, self.CLASSIC):
                with self.subTest(body=body, engine=engine.get('solver', 'default')):
                    response = self.post_solve(body, **engine)

                    self.assertEqual(response.status_code, 400)
                    self.assertEqual(response.get_json()['error'], 'missing_umf')

    # Every POST endpoint that takes a formula, with the parameter it takes and
    # the code it answers when that parameter is unusable. The codes differ
    # because the parameters do; the RULE behind them does not.
    FORMULA_ENDPOINTS = (('/api/solve', 'umf', 'missing_umf'),
                         ('/api/feasibility', 'umf', 'missing_umf'),
                         ('/api/umf_to_weights', 'umf', 'missing_umf'),
                         ('/api/weights_to_umf', 'weights', 'missing_weights'))

    def test_a_body_that_is_not_json_is_a_client_error_and_not_a_crash(self):
        """
        The guard is only "before everything" if the parse cannot raise first

        request.get_json() RAISES on a malformed body, on a body sent under the
        wrong Content-Type and on no body at all; the raise landed in the
        handler's own `except Exception` and came back as a 500 with a stack
        trace, which is exactly the class of answer the guard was added to
        remove. silent=True turns all three into the same 400 these endpoints
        already give an absent parameter.

        "A body that does not parse is a client mistake" is a property of an
        endpoint and not of what its parameter is called, so all four are here
        and /api/weights_to_umf answers it in its own code.
        """
        cases = (('malformed json', '{"umf": {"SiO2": 4}', 'application/json'),
                 ('wrong content type', '{"umf": {"SiO2": 4}}', 'text/plain'),
                 ('no body at all', '', 'application/json'))

        for label, body, content_type in cases:
            for endpoint, _, code in self.FORMULA_ENDPOINTS:
                with self.subTest(case=label, endpoint=endpoint):
                    with contextlib.redirect_stdout(io.StringIO()):
                        response = self.client.post(endpoint, data=body,
                                                    content_type=content_type)

                    self.assertEqual(response.status_code, 400)
                    self.assertEqual(response.get_json()['error'], code)

    def test_every_formula_endpoint_reads_an_unusable_parameter_the_same_way(self):
        """
        One rule, one meaning, across the server

        API.md defines these codes for the whole API, and the two converters
        were left on the older, narrower test: {"umf": []} reached
        umf_to_weights and came back a 500, {"umf": {}} came back 200 with an
        empty result - an answer that looks like a successful conversion of
        nothing.
        """
        for endpoint, parameter, code in self.FORMULA_ENDPOINTS:
            for value in ([], 0, "SiO2", None, {}):
                with self.subTest(endpoint=endpoint, value=value):
                    with contextlib.redirect_stdout(io.StringIO()):
                        response = self.client.post(endpoint, json={parameter: value})

                    self.assertEqual(response.status_code, 400)
                    self.assertEqual(response.get_json()['error'], code)

    def test_the_converters_still_convert(self):
        """The line the refusal above must not cross"""
        weights = self.client.post('/api/umf_to_weights', json={"umf": TEST_UMF})
        self.assertEqual(weights.status_code, 200)
        self.assertIn('SiO2', weights.get_json()['weights'])

        umf = self.client.post('/api/weights_to_umf',
                               json={"weights": weights.get_json()['weights']})
        self.assertEqual(umf.status_code, 200)
        self.assertIn('SiO2', umf.get_json()['umf'])

    def test_the_two_endpoints_refuse_a_bad_target_identically(self):
        """
        The invariant that makes borrowing feasibility's codes worth anything

        It had exactly one exception before this: {"umf": {}} was 422
        empty_target here and 400 missing_umf there - and that is the very body
        API.md used to file under empty_target.
        """
        for umf in ({}, [], {self.UNKNOWN: 1.0}, {"SiO2": -4, "Al2O3": -1}):
            with self.subTest(umf=umf):
                solve = self.post_solve(umf)
                feasibility = self.client.post('/api/feasibility',
                                               json={"umf": umf, "ranges": False})

                self.assertEqual(solve.status_code, feasibility.status_code)
                self.assertEqual(solve.get_json()['error'], feasibility.get_json()['error'])
                self.assertEqual(solve.get_json()['message'], feasibility.get_json()['message'])

    def test_a_target_of_nothing_but_zeros_is_refused_by_both_engines(self):
        """
        Survives the cleaning - a zero is a constraint - and is still no target

        A formula of nothing but zeros is not a UMF: the unity it would be
        normalized by is the sum of its own oxides, which is zero, which is
        literally why common.umf_to_weights divides by zero on it. The classic
        engine used to answer that with a 500 and the iterative one with a quiet
        200 and an empty list; /api/feasibility has always refused it. Now all
        three agree it is unanswerable.
        """
        for engine in ({}, self.CLASSIC):
            with self.subTest(engine=engine.get('solver', 'default')):
                response = self.post_solve({"SiO2": 0.0, "Al2O3": 0.0}, **engine)
                body = response.get_json()

                self.assertEqual(response.status_code, 422)
                self.assertEqual(body['error'], 'zero_target')
                self.assertEqual(body['warnings'], [], "nothing was dropped, the zeros are legal")

    def test_the_two_refusals_do_not_say_the_same_thing(self):
        """
        Two reasons a caller fixes two different ways: correct the oxide names,
        or ask for some oxide at all. One shared sentence would hide which.
        """
        empty = self.post_solve({self.UNKNOWN: 1.0}).get_json()
        zeros = self.post_solve({"SiO2": 0.0}).get_json()

        self.assertNotEqual(empty['error'], zeros['error'])
        self.assertNotEqual(empty['message'], zeros['message'])

    # An ash glaze without the iron of the ash: the target is what
    # {30% Древесная зола, 45% кварц, 25% каолин} comes to with Fe2O3 struck
    # out, so every oxide of it is honestly reachable and the iron is the ONE
    # thing the shelf cannot give up for free. The inventory is spelled out
    # rather than left to the default, so that the flags in materials.json can
    # be edited without silently deciding this test - and the fixture needs the
    # ash, which the five-material TEST_INVENTORY does not carry.
    ASH_INVENTORY = ["Древесная зола", "Нефелин-сиенит VR13",
                     "Кварцевая мука Кварцверке W12", "Мел, CaCO3",
                     "Каолин КЖФ-1", "Костная зола"]
    ASH_TARGET = {"SiO2": 3.897, "Al2O3": 0.345, "CaO": 0.419, "K2O": 0.272,
                  "MgO": 0.185, "Na2O": 0.124, "P2O5": 0.108, "TiO2": 0.005}

    def test_a_zero_next_to_a_real_value_is_a_constraint_and_still_solved(self):
        """
        The line the refusal above must not cross, and what the zero does

        "Fe2O3": 0.0 next to a target means "none of this", it is why
        usable_target keeps zeros, and a zero is a legal input to this API. Only
        a target of NOTHING but zeros is refused.

        THE ASSERTION IS THE DIRECTION, not the echo: target_composition simply
        repeats the cleaned target on both engines, so a solver that dropped the
        zero row entirely would still print it there and pass. Asking for no
        iron has to come back with LESS IRON than not mentioning iron at all.

        Measured on this fixture, actual Fe2O3 of the answer:

            iterative, penalize_unlisted=0.0   0.0650 -> 0.0620
            classic                            0.0650 -> 0.0440

        Not to zero, and that is the fixture being honest rather than the rule
        being weak: the ash IS the source of the potassium and the phosphorus
        the target wants, so the solver trades one against the other instead of
        obeying an impossible instruction.

        Each engine needs the setting under which the zero can bite:

        * iterative, penalize_unlisted=0.0. At the default 1.0 an unlisted oxide
          is ALREADY pushed to zero, so "not listed" and "listed as 0" ask for
          the same thing and NO difference is possible or wanted. The soft
          weight is the case zeros were kept for - the one where "no iron
          please" used to turn into "iron is fine".
        * classic has no such weight: its error is summed over the oxides of the
          target only, so naming Fe2O3 at all is what puts it into the fit.

        An earlier version of this test used TEST_UMF over the small inventory,
        where the target is out of reach by 0.27 relative - it "passed" on two
        equally hopeless answers ranked by _solution_sort_key's tie band, and
        the iron went UP when asked for none. Difference alone is a weak
        assertion on an unreachable target; direction on a reachable one is not.
        """
        cases = (('iterative', dict(penalize_unlisted=0.0)),
                 ('classic', dict(solver='classic', min_materials=True,
                                  error_tolerance=0.01)))

        for engine, extra in cases:
            with self.subTest(engine=engine):
                with_zero = self.post_solve(dict(self.ASH_TARGET, Fe2O3=0.0),
                                            inventory=self.ASH_INVENTORY, **extra)
                without_key = self.post_solve(self.ASH_TARGET,
                                              inventory=self.ASH_INVENTORY, **extra)

                self.assertEqual(with_zero.status_code, 200)
                asked, unasked = with_zero.get_json(), without_key.get_json()
                self.assertTrue(asked['solutions'])
                self.assertEqual(asked['solutions'][0]['target_composition']['Fe2O3'], 0.0)

                self.assertLess(
                    asked['solutions'][0]['actual_composition'].get('Fe2O3', 0.0),
                    unasked['solutions'][0]['actual_composition'].get('Fe2O3', 0.0),
                    "asking for no iron did not buy less iron than not mentioning it, "
                    "so the explicit zero took no part in the fit")

    def test_a_target_the_server_understands_warns_about_nothing(self):
        body = self.post_solve(TEST_UMF).get_json()

        self.assertEqual(body['warnings'], [])

    def test_loss_on_ignition_in_the_target_is_not_reported_as_an_oxide(self):
        """
        Loi is bookkeeping that leaks in from a material analysis, not an oxide
        anybody asked for. It is dropped like any unknown key and, like in
        /api/feasibility, it is dropped without a word.
        """
        body = self.post_solve(dict(TEST_UMF, Loi=5.0)).get_json()

        self.assertEqual(body['warnings'], [])


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

    def assert_no_nonfinite_numbers(self, response):
        """
        make_json_safe turns both of them into strings sitting in a field
        documented as a number, and "Infinity" is the one an overflow produces:
        checking only for "NaN" is what let the previous fix pass while the
        endpoint was still answering 200 with "sigma": "Infinity"
        """
        body = response.get_data(as_text=True)
        for token in ('NaN', 'Infinity'):
            self.assertNotIn(token, body, f"the response carries a {token}: {body[:400]}")

    def test_an_infinite_share_is_not_answered_with_nan(self):
        """
        1e400 is valid JSON and Python parses it into inf, which used to travel
        through the whole calculation and come back as "NaN" strings sitting in
        fields documented as numbers - under error: null
        """
        response = self.client.post(
            '/api/sensitivity',
            data='{"recipe": {"Нефелин-сиенит VR13": 1e400, "Мел, CaCO3": 20}}',
            content_type='application/json')

        self.assertEqual(response.status_code, 200)
        self.assert_no_nonfinite_numbers(response)

        body = response.get_json()
        self.assertIsNone(body['error'])
        self.assertEqual({row['material'] for row in body['by_material']}, {"Мел, CaCO3"})
        self.assertTrue(body['warnings'])

    def test_a_finite_but_absurd_share_is_not_answered_with_infinity(self):
        """
        The share stays finite, the SQUARE of the response to it does not: 1e156
        still comes back as numbers and 1e160 used to come back as
        {"oxide": "SiO2", "value": 1.66e+159, "sigma": "Infinity"} under
        error: null. The guard on "is the input finite" cannot see this one.
        """
        response = self.client.post('/api/sensitivity', json={
            "recipe": {"Кварцевая мука Кварцверке W12": 1e160, "Мел, CaCO3": 10}})

        self.assertEqual(response.status_code, 200)
        self.assert_no_nonfinite_numbers(response)

        body = response.get_json()
        self.assertNotIn("Кварцевая мука Кварцверке W12",
                         {row['material'] for row in body['by_material']})
        self.assertTrue(any("Кварцевая мука" in warning for warning in body['warnings']))

    def test_a_wall_of_huge_shares_does_not_come_back_as_a_composition_of_infinities(self):
        """Several materials at the float ceiling used to overflow the composition itself"""
        response = self.client.post('/api/sensitivity', json={
            "recipe": {"Кварцевая мука Кварцверке W12": 1.5e308, "Мел, CaCO3": 1.5e308,
                       "Нефелин-сиенит VR13": 1.5e308}})

        self.assert_no_nonfinite_numbers(response)
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.get_json()['error'], 'no_known_materials')

    def test_a_rejected_share_is_not_reported_as_a_material_missing_from_the_database(self):
        """The name is spelled perfectly and is in the database; the share is what was refused"""
        response = self.client.post(
            '/api/sensitivity',
            data='{"recipe": {"Мел, CaCO3": 1e400}}',
            content_type='application/json')

        self.assertEqual(response.status_code, 422)
        body = response.get_json()
        self.assertEqual(body['error'], 'no_known_materials')
        self.assertNotIn('не найден в базе', body['message'])
        self.assertTrue(body['warnings'])

    def test_an_inventory_parameter_is_ignored_but_not_in_silence(self):
        """
        The parameter is gone: all it could do was drop materials of the recipe,
        and then "umf" reported the formula of what was left instead of the
        formula the caller asked about. An old client still sending one gets
        different numbers than it used to, so the response has to say so.
        """
        response = self.client.post('/api/sensitivity', json={
            "recipe": self.RECIPE,
            "inventory": ["Нефелин-сиенит VR13", "Кварцевая мука Кварцверке W12"]})

        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual({row['material'] for row in body['by_material']}, set(self.RECIPE))
        self.assertIn(api_server.IGNORED_INVENTORY_WARNING, body['warnings'])

    def test_a_request_without_an_inventory_gets_no_such_warning(self):
        response = self.client.post('/api/sensitivity', json={"recipe": self.RECIPE})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()['warnings'], [])

    def test_an_inventory_of_the_wrong_type_is_ignored_instead_of_crashing(self):
        response = self.client.post('/api/sensitivity',
                                    json={"recipe": self.RECIPE, "inventory": 5})

        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual({row['material'] for row in body['by_material']}, set(self.RECIPE))
        self.assertIn(api_server.IGNORED_INVENTORY_WARNING, body['warnings'])

    def test_the_ignored_inventory_is_reported_on_a_refused_recipe_too(self):
        response = self.client.post('/api/sensitivity', json={
            "recipe": {"Кварцевая мука Кварцверке W12": 60, "Глинозем, Al203": 40},
            "inventory": ["Кварцевая мука Кварцверке W12"]})

        self.assertEqual(response.status_code, 422)
        self.assertIn(api_server.IGNORED_INVENTORY_WARNING, response.get_json()['warnings'])

    def test_the_whole_database_is_searched(self):
        """A recipe names its own materials; being out of stock is not a reason to drop one"""
        recipe = {"Оксид марганца": 50, "Каолин КЖФ-1": 30, "Кварцевая мука Кварцверке W12": 20}

        response = self.client.post('/api/sensitivity', json={"recipe": recipe})

        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        # "Оксид марганца" is inInventory: false and is still analysed
        self.assertIn("Оксид марганца", {row['material'] for row in body['by_material']})


if __name__ == '__main__':
    unittest.main()
