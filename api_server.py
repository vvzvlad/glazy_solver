#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# flake8: noqa
# pylint: disable=broad-exception-raised, raise-missing-from, too-many-arguments, redefined-outer-name
# pylint: disable=multiple-statements, logging-fstring-interpolation, trailing-whitespace, line-too-long
# pylint: disable=broad-exception-caught, missing-function-docstring, missing-class-docstring
# pylint: disable=f-string-without-interpolation
# pylance: disable=reportMissingImports, reportMissingModuleSource
# type: ignore

import json
import logging
import math
import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from solver_classic import find_multiple_solutions, calculate_recipe_composition
from solver_iterative import DROPPED_TARGET_OXIDES_LOG, find_best_recipe, usable_target
from common import (weights_to_umf, umf_to_weights, load_materials, make_json_safe,
                    resolve_inventory, filter_materials_by_inventory,
                    load_oxide_classification)
from feasibility import (DEFAULT_FEASIBILITY_TOL, achievable_ranges, check_feasibility,
                         matrix_diagnostics, projected_range_lps)
from glazy_import import GlazyImportError, parse_recipe_id, fetch_recipe, build_import_result
from sensitivity import recipe_sensitivity

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('glaze_recipe_api')

app = Flask(__name__, static_folder=None)
CORS(app)  # Allow CORS for every route

# Path to the UI directory
UI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'UI')
# Path to the data directory
DATABASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'database')

# The oxide classification decides what counts as a flux, and therefore every
# UMF number the server produces. It is read and validated here, at import, so
# that a corrupt file stops the process at startup: reaching it lazily from
# inside a request means the solvers swallow the failure and answer "no
# solutions found", which looks like an ordinary result and is not one.
try:
    load_oxide_classification()
except Exception as e:
    # Both the ClassificationError and the FileNotFoundError already name the
    # file, so the prefix only adds the code
    logger.critical(f"invalid_oxide_classification: {str(e)}")
    raise

# Available solver engines. The iterative one is the default: measured on the 11
# reference recipes of compare_solvers.py (see REFACTORING.md, section 8) it is
# more accurate (median sum of per-oxide UMF deviations 0.0020 against 0.0030),
# more than three times faster, and it reproduces the exact original set of
# materials in 10 recipes out of 11 against 2 for the classic one.
# The classic engine stays available through an explicit "solver" parameter.
#
# Two corrections from the merge, neither of which overturns the choice. The
# classic engine is no longer non deterministic: it draws its subsets from a
# private seeded generator, so two identical requests give the same answer. And
# the ordering is inventory dependent - on the Glazy corpus, where the inventory
# is the two to twelve materials of the recipe itself, the classic engine passes
# the chemistry gate on 100% of targets against 94.67% and runs about thirty
# times faster, because there its full matrix solution simply is the answer. The
# iterative engine earns the default on the real 19 material inventory, which is
# what the server serves.
SOLVER_CLASSIC = 'classic'
SOLVER_ITERATIVE = 'iterative'
AVAILABLE_SOLVERS = (SOLVER_CLASSIC, SOLVER_ITERATIVE)
DEFAULT_SOLVER = SOLVER_ITERATIVE

# How hard the iterative solver pushes an oxide the request did not mention
# towards zero. 1.0 ("not listed = must be zero") is both the library default of
# find_best_recipe and the behaviour this endpoint had before the parameter
# existed, and it is what actually answers the request best: on the example of
# API.md, {"SiO2": 4, "Al2O3": 1, "Na2O": 0.5, "K2O": 0.5}, weight 1.0 lands at
# an error of 0.14 on the requested oxides while 0.0 lands at 1.41 - ten times
# worse on the very metric a weight of 0.0 is the one minimizing. Letting the
# unlisted oxides float free lets the solver spend the flux budget on oxides
# nobody asked for, and the requested ones pay for it.
#
# It is a parameter and not a constant because the trade-off does exist: with a
# weight of 0.0 the returned formula carries whatever the materials happen to
# bring, which some callers want. Pass it explicitly for that.
#
# A note on the classic engine, since the comparison invites itself: it has no
# equivalent of this weight. Its error is summed over the oxides of the target
# only (solver_classic.calculate_umf_error), so an oxide nobody asked for costs
# it exactly nothing - the unlisted oxides are neither penalized nor measured,
# which is the behaviour a weight of 0.0 approximates here. The two engines are
# therefore still not answering the same question, and their reported errors are
# not comparable even though both are now measured on the plain UMF of the
# recipe.
DEFAULT_PENALIZE_UNLISTED = 1.0

# An oxide the request asked for that neither engine can fit, because
# database/molar_masses.json does not know the name. Word for word the sentence
# /api/feasibility has always answered on the same input: one target, one
# verdict about it, whichever endpoint is asked.
TARGET_OXIDES_DROPPED_WARNING = "оксиды цели не распознаны и не учтены: {oxides}"


def iterative_solutions_to_classic_format(solutions, inventory_data):
    """
    Convert the solutions of the iterative solver into the response format of
    the classic one, so that the UI does not see any difference

    The response has to be self consistent: 'error' is the distance between
    'target_composition' and 'actual_composition' and nothing else, so the
    target reported here is the CLEANED target the solver actually worked on
    (solution['target_umf']), not the raw dictionary of the request. They differ
    whenever the request carries an oxide the solver cannot use - an unknown
    name, a negative or non numeric value - and a consumer recomputing the error
    from the raw request would get a different number for no visible reason.
    Both dictionaries are rounded to four decimals, so a recomputation agrees to
    about 1e-4. WHICH KEYS WENT MISSING that way is not left to be deduced from
    the difference: the endpoint says it in "warnings" - see
    TARGET_OXIDES_DROPPED_WARNING.

    The CLASSIC engine reaches the same place by a different route and it is
    worth saying which: it echoes back whatever target it was given, so what
    makes ITS 'target_composition' clean is the endpoint cleaning the target
    before dispatch rather than anything in this function. That is why the
    cleaning sits above the fork - "the dropped key is not in
    target_composition" is documented for the endpoint, not for one engine of
    it, and it used to be false on the other one.

    Three fields of the iterative solver are passed through as well, because
    without them the answer cannot be interpreted: 'objective_error' (what the
    search actually minimized, and NOT the same quantity as 'error' - it is the
    L2 of the per-oxide RELATIVE deviations with a deadband, plus the damped
    contamination, so it is comparable with the 0.05 the feasibility endpoint
    speaks and not with 'error'), 'unlisted_weight' (the penalize_unlisted that
    was applied, whether it came
    from the request or from the default) and 'unity_scale' (whether the UMF of
    the recipe had to be rescaled onto the basis of the target, and by how
    much). They are additions - every key the classic format has is still there.

    Args:
        solutions: list of solutions returned by find_best_recipe
        inventory_data: optional list of available material names

    Returns:
        list of solutions with the keys of the classic solver
    """
    materials = load_materials(only_inventory=False, priority=True)
    available_materials = filter_materials_by_inventory(materials, resolve_inventory(inventory_data))

    converted = []
    for solution in solutions:
        composition = calculate_recipe_composition(available_materials, solution['recipe'])
        converted.append({
            'recipe': solution['recipe'],
            'error': round(float(solution['error']), 4),
            'target_composition': {oxide: round(float(value), 4) for oxide, value in solution['target_umf'].items()},
            'actual_composition': {oxide: round(float(value), 4) for oxide, value in solution['result_umf'].items()},
            'weight_composition': {oxide: round(float(value), 2) for oxide, value in composition.items()},
            'materials_count': solution['materials_count'],
            'objective_error': round(float(solution['objective_error']), 4),
            'unlisted_weight': float(solution['unlisted_weight']),
            'unity_scale': round(float(solution['unity_scale']), 6),
        })

    return converted


@app.route('/api/solve', methods=['POST'])
def solve_recipe():
    """
    API endpoint that calculates a glaze recipe from a UMF formula

    POST JSON parameters:
    {
        "umf": {"SiO2": 4, "Al2O3": 1, "Na2O": 0.5, "K2O": 0.5},
        "max_solutions": 3,  // optional, 3 by default
        "min_materials": true,  // optional, true by default, classic solver only:
                                // the iterative solver ignores it
        "error_tolerance": 0.01,  // optional, 0.01 by default, classic solver only:
                                  // the iterative solver ignores it
        "inventory": ["Material1", "Material2", ...],  // optional, list of available materials
        "solver": "iterative",  // optional, "iterative" (default) or "classic"
        "penalize_unlisted": 1.0  // optional, 1.0 by default, iterative solver only:
                                  // how hard an oxide missing from "umf" is pushed to
                                  // zero. 1.0/true = not listed means "must be zero"
                                  // (the default, and the most accurate answer on the
                                  // oxides that WERE listed), 0.0/false = not listed
                                  // means "do not care", in between is a soft weight.
                                  // An oxide listed in "umf" as an explicit 0 is a
                                  // constraint and is never treated as unlisted.
    }

    Returns:
    {
        "solutions": [
            {
                "recipe": {"Material1": 45.2, "Material2": 54.8},
                "error": 0.0123,
                "target_composition": {"SiO2": 4, "Al2O3": 1, ...},
                "actual_composition": {"SiO2": 3.98, "Al2O3": 1.02, ...},
                "weight_composition": {"SiO2": 65.2, "Al2O3": 18.1, ...},
                "materials_count": 2,
                "recipe_umf": {"SiO2": 3.98, "Al2O3": 1.02, ...},  // UMF of this particular recipe
                // iterative solver only (so, by default), see
                // iterative_solutions_to_classic_format. "objective_error" is a
                // DIFFERENT quantity from "error" and is deliberately shown as a
                // different number here: it is relative, so on a target of large
                // oxides it comes out below the absolute norm and on a target of
                // trace oxides far above it
                "objective_error": 0.0047,
                "unlisted_weight": 1.0,
                "unity_scale": 1.0
            },
            ...
        ],
        "warnings": ["оксиды цели не распознаны и не учтены: Unobtainium"]
    }

    THE TARGET IS CLEANED HERE, ONCE, ABOVE THE CHOICE OF ENGINE. That is the
    load-bearing decision of this handler and not an implementation detail:
    solver_iterative.usable_target() refuses an unknown NAME and a bad VALUE
    (not a number, negative, NaN, infinity), while the classic engine refuses
    only the name - umf_to_weights drops it - and feeds a negative value
    straight into its NNLS. Letting each engine clean to its own taste produced
    exactly the disagreement this endpoint exists to end: on
    {"Fe2O3": -0.5, "solver": "classic"} the answer carried "оксид не учтён"
    next to a recipe fitted TO -0.5, and every material in it differed from the
    recipe for the same target without that key. Cleaned above the fork, the
    warning, the returned "target_composition" and the vector actually fitted
    are the same object by construction. A negative or NaN UMF is not a legal
    input to either engine, so nothing legitimate was accepted before and is
    refused now.

    THE RESPONSE IS AN OBJECT AND USED TO BE THE BARE LIST, and the reason is
    the "warnings" next to it rather than a taste for envelopes. Until the field
    existed the only trace of a refusal in the answer was an absence: a recipe
    came back, "target_composition" no longer held the key, and nothing said
    why. /api/feasibility, handed the very same target, has always answered
    "оксиды цели не распознаны и не учтены: ..." - word for word what is
    answered here, down to the order of the names. Two endpoints disagreeing
    about one input is bad enough; the silent one being the one that hands back
    a recipe is worse.

    A list cannot carry a field, and the warning cannot live inside the
    solutions either: the request that most needs it - a target of nothing but
    unrecognised oxides - produces no solutions at all. Hence the object. The
    one live consumer, UI/, was moved onto it in the same change, through a
    single unpack_solve_response() that accepts either shape - so the page keeps
    working against a server of either vintage, and the prototype UI-v2/, which
    reads mock data and no endpoint at all, was never on the old shape.

    EVERY ERROR OF THIS ENDPOINT CARRIES "warnings" TOO, empty list included.
    A field that is present on success and absent on failure is a field every
    consumer has to guard, and the failure that arrives WITH a dropped oxide is
    a plausible pair rather than a curiosity - a request can easily be wrong
    twice. The list is empty on the error raised before the target has been read
    at all.

    A TARGET WITH NOTHING TO SOLVE FOR never reaches an engine, in three
    flavours that are told apart because the reader has to do three different
    things about them:

      * "missing_umf"   - 400. No "umf" at all, or one that is not a non-empty
                          object. Checked before anything reads it, in
                          /api/feasibility's words.
      * "empty_target"  - 422. The cleaning emptied it: every name unknown, or
                          every value negative. /api/feasibility's code and
                          sentence, for the same input.
      * "zero_target"   - 422. It survived the cleaning and asks for zero of
                          everything. Not a UMF at all: the unity such a formula
                          would be normalized by is zero.

    ON THE FIRST TWO this endpoint and /api/feasibility answer the same code and
    the same sentence for the same body, which is the whole point of borrowing
    them; the invariant had one exception ({"umf": {}} was 422 here and 400
    there) until "missing_umf" was widened, and now has none.

    THE THIRD IS DELIBERATELY NOT SHARED. /api/feasibility refuses an all-zero
    target as "no_target_fluxes", which is a broader statement - it also refuses
    {"SiO2": 3.0, "Al2O3": 0.3}, a target this endpoint solves, and both engines
    answer it with a recipe. Same 422, own code, own sentence. Borrowing the
    name would have claimed a check that is not made here.

    None of the three is a decision taken for symmetry's sake. The two engines
    disagree about a target of nothing - one returns an empty list, the other
    dies inside numpy - so somebody above them has to say which it is, and a 200
    carrying an empty list is not an answer to the question either.
    """
    # Before the try: the handler's own except must be able to report it, and an
    # exception can be raised by the very first line below
    request_warnings = []

    try:
        # silent=True, like /api/feasibility: a body that is not JSON at all -
        # malformed, or sent under the wrong Content-Type - makes get_json()
        # RAISE, and the raise lands in the handler's own `except Exception`
        # and comes back as a 500 with a stack trace. That is a client mistake
        # and it gets the same 400 as an absent body
        data = request.get_json(silent=True)

        # The shape of "umf" is checked BEFORE anything reads it, in the one
        # test /api/feasibility already uses, word for word. Three inputs used
        # to get past here - no "umf" key, "umf" that is not an object, and an
        # empty object - and the last two then walked into the engines, which do
        # not refuse them in words: the classic one died on
        # `'list' object has no attribute 'keys'` (500) and the iterative one
        # answered 200 with an empty list. Neither is "the caller was told what
        # was wrong with the request".
        umf = data.get('umf') if isinstance(data, dict) else None
        if not isinstance(umf, dict) or not umf:
            logger.warning("missing_umf parameter in request")
            return jsonify({"error": "missing_umf",
                            "message": "umf parameter is required and must be a non-empty object",
                            "warnings": request_warnings}), 400

        max_solutions = data.get('max_solutions', 3)
        min_materials = data.get('min_materials', True)
        error_tolerance = data.get('error_tolerance', 0.01)
        inventory_data = data.get('inventory', None)
        solver_name = data.get('solver', DEFAULT_SOLVER)
        penalize_unlisted = data.get('penalize_unlisted', DEFAULT_PENALIZE_UNLISTED)

        # ONCE PER REQUEST AND ABOVE THE FORK: both engines are handed the same
        # cleaned target, so neither can be fitting something the answer does
        # not describe. It is a non-empty dictionary by now - the guard above is
        # what makes that true, so nothing here has to probe for the shape.
        umf, dropped_oxides = usable_target(umf)

        if dropped_oxides:
            # The solver's wording, not a second one of ours: one request, one
            # line, the same line whichever engine is about to run
            logger.warning(DROPPED_TARGET_OXIDES_LOG.format(oxides=', '.join(dropped_oxides)))
            request_warnings.append(TARGET_OXIDES_DROPPED_WARNING.format(
                oxides=', '.join(dropped_oxides)))

        # The solver name is validated first, so that which error a bad request
        # gets back does not depend on what its target happened to contain
        if solver_name not in AVAILABLE_SOLVERS:
            logger.warning(f"unknown_solver requested: {solver_name}")
            return jsonify({
                "error": "unknown_solver",
                "message": f"unknown solver '{solver_name}', expected one of: {', '.join(AVAILABLE_SOLVERS)}",
                "warnings": request_warnings
            }), 400

        # THERE IS NOTHING TO SOLVE FOR, in one of two ways, and the endpoint
        # answers both because the engines answer them differently and one of
        # them badly: find_best_recipe returns 200 and an empty list, while
        # find_multiple_solutions dies - on an empty oxide set inside
        # numpy.matrix_rank ("zero-size array to reduction operation maximum"),
        # on an all-zero target inside common.umf_to_weights, which divides by
        # the total molar weight. Either way a client mistake used to arrive as
        # a 500 with a stack trace in the log and a 5xx in the monitoring.
        #
        # TWO REASONS, TWO SENTENCES, because the reader has to do two different
        # things about them: fix the oxide names, or ask for some oxide at all.
        # The codes are separate for the same reason.
        #
        # ORDER IS LOAD BEARING: `not any(...)` below is vacuously true of an
        # empty dictionary, so the emptier reason has to be tested first or it
        # would never be reported. What arrives empty here is only ever a target
        # the CLEANING emptied - a request that sent {} was refused above.
        if not umf:
            logger.warning(f"empty_target: nothing left to solve, dropped {dropped_oxides}")
            return jsonify({"error": "empty_target",
                            # The code and the sentence /api/feasibility answers
                            # for this very target
                            "message": "target UMF has no usable oxide",
                            "warnings": request_warnings}), 422

        # A ZERO IS A CONSTRAINT ("none of this") and is honoured everywhere
        # else, so this rejects the target only when there is nothing BUT
        # zeros - the mixed case is an ordinary request and still solved. A
        # formula of nothing but zeros is not a UMF at all: the unity it is
        # normalized by is the sum of its own oxides, which is zero here, so
        # there is no basis to state the answer on and nothing to fit.
        # /api/feasibility already refuses it (as no_target_fluxes, its own
        # reason), so the verdict layer and the recipe layer now agree that this
        # target is unanswerable instead of one of them returning a quiet 200
        # with an empty list, which does not answer the question either.
        if not any(value > 0.0 for value in umf.values()):
            logger.warning("zero_target: every requested oxide is 0, nothing to solve for")
            return jsonify({"error": "zero_target",
                            "message": "target UMF asks for zero of every oxide, so it has no "
                                       "unity to be normalized by and nothing to fit",
                            "warnings": request_warnings}), 422

        logger.info(f"solving recipe for umf: {umf}, max_solutions: {max_solutions}, min_materials: {min_materials}, solver: {solver_name}, penalize_unlisted: {penalize_unlisted}")

        if solver_name == SOLVER_ITERATIVE:
            try:
                iterative_solutions = find_best_recipe(
                    inventory_data,
                    umf,
                    max_solutions=max_solutions,
                    verbose=False,
                    penalize_unlisted=penalize_unlisted
                )
            except ValueError as exc:
                # The solver validates its own arguments and says what is wrong
                # with them; that is a bad request, not a server failure
                logger.warning(f"invalid_parameter: {exc}")
                return jsonify({"error": "invalid_parameter", "message": str(exc),
                                "warnings": request_warnings}), 400

            solutions = iterative_solutions_to_classic_format(iterative_solutions, inventory_data)
        else:
            solutions = find_multiple_solutions(
                umf,
                max_solutions=max_solutions,
                min_materials=min_materials,
                error_tolerance=error_tolerance,
                inventory_data=inventory_data
            )

        if isinstance(solutions, dict) and 'error' in solutions:
            logger.error(f"calculation_error: {solutions['error']}")
            return jsonify({"error": "calculation_error", "message": solutions['error'],
                            "warnings": request_warnings}), 500

        # Add the UMF information to every recipe
        for solution in solutions:
            # The UMF of a particular recipe is already stored in actual_composition,
            # but it is also exposed as a separate field for convenience on the frontend
            solution['recipe_umf'] = solution['actual_composition']

        # Prepare the results for safe JSON serialization
        safe_solutions = make_json_safe(solutions)

        logger.info(f"found {len(solutions)} solutions, warnings={len(request_warnings)}")
        return jsonify({"solutions": safe_solutions, "warnings": request_warnings})

    except Exception as e:
        logger.exception(f"server_error: {str(e)}")
        return jsonify({"error": "server_error", "message": str(e),
                        "warnings": request_warnings}), 500

# A feasibility request that is syntactically fine but has nothing to compute
# from: no usable oxide in the target, a target with no flux at all (there is no
# unity to normalize by), or an inventory in which no material carries an oxide
# analysis. The client can fix all three, so they are 422 and not 500.
FEASIBILITY_UNPROCESSABLE_ERRORS = ('empty_target', 'no_target_fluxes', 'degenerate_target',
                                    'no_usable_materials', 'no_fluxes')

# The largest range problem this endpoint will solve, counted in LPs -
# 1 + 2 * (oxides + materials), which is the thing that actually costs, rather
# than the number of names in the request. Measured on this machine: the real
# 19 material inventory is 63 LPs and 47 ms, 60 materials is 167 LPs and 317 ms,
# and the whole 216 material database is 459 LPs and 4.3 SECONDS on a single
# threaded Flask - a 90x amplification bought with one extra field in a request.
# The per-LP cost itself grows with the problem (0.75 ms at inventory size, 10.8
# ms at database size), so this cap is set by measured latency and not by
# arithmetic: 180 LPs lands around 350 ms.
# A caller who wants the verdict over a big inventory asks for "ranges": false
# and pays 27 ms for the whole database.
MAX_RANGE_LPS = 180


@app.route('/api/feasibility', methods=['POST'])
def feasibility():
    """
    API endpoint that answers whether a target UMF is reachable from a set of
    materials at all, which oxide is to blame when it is not, and how far each
    oxide and each material can move while the target still holds

    No solver runs here. The verdict is a property of the target and the
    inventory, not of a recipe, so the interface can show it while the user is
    still typing the formula - before any recipe exists.

    This is a separate endpoint and NOT a block added to the answer of
    /api/solve, although section 2.4 of TZ_SOLVER_V2.md asked for the latter,
    and the paragraph above is the whole reason: the verdict does not depend on
    a recipe and is wanted before one exists. It is re-asked when the inventory
    changes, not when the answer does - two questions with two lifetimes, not
    one round trip to save.

    (The original text argued from the shape instead - that /api/solve answered
    a bare list which two interfaces read, so wrapping it would break a working
    one. Both halves of that are false now: the response is an object, carrying
    the "warnings" a list could not, and UI-v2/ runs on mock data and calls no
    endpoint at all. The reason above never depended on either.)

    POST JSON parameters:
    {
        "umf": {"SiO2": 3.15, "Al2O3": 0.38, "CaO": 0.72, ...},  // required
        "inventory": ["Material1", ...],   // optional, null = the default stock
        "passengers": {"Fe2O3": 0.03},     // optional, {oxide: upper bound}
        "material_constraints": {"Улексит (Химпэк)": [0, 20]},  // optional, weight %
        "tol": 0.05,                       // optional, the verdict line
        "ranges": true                     // optional, true by default
    }

    "ranges": false returns the verdict alone. That is the difference between
    about 2 ms and about 47 ms on the real inventory, and it is what an
    interface calling this on every keystroke should send: the ranges are worth
    2 x (oxides + materials) LPs and the verdict is worth two.

    "inventory": null means the DEFAULT INVENTORY (the materials flagged
    inInventory), exactly as in /api/solve, and deliberately not what the same
    word means in /api/sensitivity, where it means the whole database. The two
    endpoints are asking different questions: sensitivity analyses a recipe that
    already names its own materials, while this one asks what can be built from
    the stock at hand - and answering it against 216 materials the user does not
    own would be a verdict about someone else's shelf.

    Returns the result of feasibility.check_feasibility plus:
    {
        "diagnostics": {"cond": 488.9, "ill_conditioned": false, "rank": 12,
                        "n_oxides": 12},
        "achievable_ranges": {"feasible": true,
                              "oxide_ranges": {"SiO2": [2.99, 3.31], ...},
                              "material_ranges": {"Каолин КЖФ-1": [0, 27.96], ...},
                              "example_recipe": {...}, "lp_count": 63}
    }

    An unbounded end of a range is null ("as much as you like"), which is a real
    answer and not a missing one: while pure quartz is in the inventory and
    nothing caps SiO2, there is no largest SiO2.
    """
    try:
        data = request.get_json(silent=True)

        umf = data.get('umf') if isinstance(data, dict) else None
        if not isinstance(umf, dict) or not umf:
            logger.warning("feasibility_missing_umf parameter in request")
            return jsonify({"error": "missing_umf",
                            "message": "umf parameter is required and must be a non-empty object"}), 400

        inventory_data = data.get('inventory', None)
        passengers = data.get('passengers', None)
        material_constraints = data.get('material_constraints', None)
        tol = data.get('tol', DEFAULT_FEASIBILITY_TOL)
        want_ranges = data.get('ranges', True)

        if passengers is not None and not isinstance(passengers, dict):
            return jsonify({"error": "invalid_parameter",
                            "message": "passengers must be an object {oxide: upper_bound}"}), 400
        if material_constraints is not None and not isinstance(material_constraints, dict):
            return jsonify({"error": "invalid_parameter",
                            "message": "material_constraints must be an object {material: [min, max]}"}), 400

        try:
            tol = float(tol)
        except (TypeError, ValueError):
            tol = float('nan')
        if not math.isfinite(tol) or tol < 0:
            return jsonify({"error": "invalid_parameter",
                            "message": "tol must be a finite non-negative number"}), 400

        inventory = resolve_inventory(inventory_data)
        materials = filter_materials_by_inventory(
            load_materials(only_inventory=False, priority=False), inventory)

        if want_ranges:
            projected = projected_range_lps(umf, materials)
            if projected > MAX_RANGE_LPS:
                logger.warning(f"feasibility_inventory_too_large: {len(materials)} materials "
                               f"project to {projected} LPs, cap {MAX_RANGE_LPS}")
                return jsonify({
                    "error": "inventory_too_large",
                    "message": f"the achievable ranges over {len(materials)} materials would "
                               f"take {projected} linear programs, above the cap of "
                               f"{MAX_RANGE_LPS}; send a smaller inventory or "
                               f"\"ranges\": false for the verdict alone"
                }), 413

        logger.info(f"feasibility requested for {len(umf)} oxides over {len(materials)} materials, "
                    f"tol={tol}, passengers={len(passengers or {})}, ranges={bool(want_ranges)}")

        result = check_feasibility(umf, materials, tol=tol, passengers=passengers)

        if result.get('error'):
            status = 422 if result['error'] in FEASIBILITY_UNPROCESSABLE_ERRORS else 500
            logger.warning(f"feasibility_failed: {result['error']}: {result.get('message')}")
            return jsonify({"error": result['error'], "message": result.get('message', ''),
                            "warnings": result.get('warnings', [])}), status

        result['diagnostics'] = matrix_diagnostics(
            materials, sorted({oxide for material in materials
                               for oxide in (material.get('formula') or {})}))

        if want_ranges:
            # A passenger is a one sided ceiling in the range problem too, which
            # is the same statement it makes in the verdict: "keep it under
            # this", not "aim for this".
            #
            # The ceilings come from the ANSWER and not from the request:
            # check_feasibility cleans them (unknown oxide, negative, NaN) and
            # says so in "warnings", and feeding the raw dictionary here meant a
            # passenger of -1.0 was dropped by the verdict with a warning and
            # obeyed by the ranges without one - so the same response carried
            # "feasible": true next to "achievable_ranges": {"feasible": false}.
            oxide_constraints = {row['oxide']: [None, row['limit']]
                                 for row in result.get('passengers', [])}
            result['achievable_ranges'] = achievable_ranges(
                umf, materials, oxide_constraints=oxide_constraints,
                material_constraints=material_constraints, tol=tol)

        logger.info(f"feasibility: {result['feasible']}, deviation "
                    f"{result.get('max_relative_deviation')}, "
                    f"unreachable {result.get('unreachable_oxides')}")
        return jsonify(make_json_safe(result))

    except Exception as e:
        logger.exception(f"feasibility_error: {str(e)}")
        return jsonify({"error": "server_error", "message": str(e)}), 500


# HTTP status for a recipe that is syntactically fine but cannot be analysed -
# the same code /api/glazy_import already uses for a recipe without an analysis.
# 'empty_recipe' and 'empty_umf' are guards of sensitivity.py that no request can
# reach through this endpoint (an empty recipe is answered with missing_recipe
# above, and a recipe that passes the flux check always has a UMF); they are
# listed so that a future path to them lands on 422 and not on the 400 default.
SENSITIVITY_UNPROCESSABLE_ERRORS = ('no_fluxes', 'no_known_materials', 'empty_composition',
                                    'empty_recipe', 'empty_umf', 'nonfinite_result')

# The endpoint used to accept an "inventory" and the parameter was removed, not
# renamed: the request that carries it is now answered with different numbers
# than before. Ignoring it without a word is the one outcome an old client cannot
# notice, so the answer says it out loud.
IGNORED_INVENTORY_WARNING = (
    "параметр «inventory» больше не поддерживается и проигнорирован: "
    "чувствительность всегда считается по всей базе материалов")


@app.route('/api/sensitivity', methods=['POST'])
def sensitivity():
    """
    API endpoint that reports what a recipe rests on: how much the uncertainty
    of the material analyses moves its UMF, and which material moves it most

    This is a separate endpoint and not a block of /api/solve on purpose: the
    computation is optional and costs a full UMF recalculation per (material,
    oxide) pair, so a caller that does not need it should not pay for it.

    POST JSON parameters:
    {
        "recipe": {"Нефелин-сиенит VR13": 30.2, ...}  // required, weight percent
    }

    There is no "inventory" parameter here, unlike /api/solve: the names are
    always resolved against the WHOLE database. A recipe names its own materials
    exactly, so there is nothing to search for, and dropping one of them for
    being out of stock would silently analyse a different formula than the one
    reported in "umf". A request that still sends one is answered anyway, with
    IGNORED_INVENTORY_WARNING in "warnings".

    Returns:
    {
        "umf": {"SiO2": 3.151, ...},          // base formula of the recipe
        "per_oxide": [                        // sorted by relative spread, desc
            {"oxide": "B2O3", "value": 0.266, "sigma": 0.02779, "relative": 0.1044},
            ...
        ],
        "by_material": [                      // sorted by share, desc
            {"material": "Улексит (Химпэк)", "share": 0.7004, "via_oxide": "B2O3",
             "sigma_used": 0.1, "affects": ["B2O3", "MgO"]},
            ...
        ],
        "warnings": ["..."],
        "error": null
    }
    """
    try:
        data = request.get_json(silent=True)

        recipe = data.get('recipe') if isinstance(data, dict) else None

        if not isinstance(recipe, dict) or not recipe:
            logger.warning("sensitivity_missing_recipe parameter in request")
            return jsonify({
                "error": "missing_recipe",
                "message": "recipe parameter is required and must be a non-empty object"
            }), 400

        request_warnings = []
        if 'inventory' in data:
            logger.warning("sensitivity_inventory_ignored: the parameter was removed")
            request_warnings.append(IGNORED_INVENTORY_WARNING)

        materials = load_materials(only_inventory=False, priority=True)

        logger.info(f"sensitivity requested for {len(recipe)} materials")

        result = recipe_sensitivity(recipe, materials)
        result['warnings'] = request_warnings + result.get('warnings', [])

        if result.get('error'):
            logger.warning(f"sensitivity_failed: {result['error']}: {result.get('message')}")
            status = 422 if result['error'] in SENSITIVITY_UNPROCESSABLE_ERRORS else 400
            return jsonify({"error": result['error'], "message": result.get('message', ''),
                            "warnings": result['warnings']}), status

        logger.info(f"sensitivity done: {len(result['by_material'])} materials, {len(result['per_oxide'])} oxides, warnings={len(result['warnings'])}")
        return jsonify(make_json_safe(result))

    except Exception as e:
        logger.exception(f"sensitivity_error: {str(e)}")
        return jsonify({"error": "server_error", "message": str(e)}), 500


@app.route('/api/molar_masses', methods=['GET'])
def get_molar_masses():
    """
    API endpoint that returns the list of oxides with their molar masses

    Returns:
    {
        "SiO2": 60.084,
        "Al2O3": 101.961,
        ...
    }
    """
    try:
        molar_masses_path = os.path.join(DATABASE_DIR, 'molar_masses.json')
        
        if not os.path.exists(molar_masses_path):
            logger.error(f"molar_masses_file_not_found: {molar_masses_path}")
            return jsonify({"error": "file_not_found", "message": "Molar masses file not found"}), 404
        
        with open(molar_masses_path, 'r', encoding='utf-8') as f:
            molar_masses = json.load(f)
        
        logger.info(f"returning {len(molar_masses)} molar masses")
        return jsonify(molar_masses)
    
    except Exception as e:
        logger.exception(f"molar_masses_error: {str(e)}")
        return jsonify({"error": "server_error", "message": str(e)}), 500

@app.route('/api/oxide_groups', methods=['GET'])
def get_oxide_groups():
    """
    API endpoint that returns the oxide classification by structural group

    The data comes from common.load_oxide_classification(), not from a second
    read of the file, so what the UI groups oxides by is the very table the
    solvers normalize with - and a file that failed validation cannot be served
    as if it were fine.

    Returns:
    {
        "r2o": ["K2O", "Na2O", "Li2O"],
        "ro": ["MgO", "CaO", ...],
        "r2o3": ["Al2O3", ...],
        "ro2": ["SiO2", ...],
        "unity": ["r2o", "ro"],
        "unity_presets": {"legacy": [...], "glazy": [...], "segerlab": [...]}
    }
    """
    try:
        classification = load_oxide_classification()

        logger.info(f"returning {len(classification)} oxide groups")
        return jsonify(classification)

    except FileNotFoundError:
        classification_path = os.path.join(DATABASE_DIR, 'oxide_classification.json')
        logger.error(f"oxide_classification_file_not_found: {classification_path}")
        return jsonify({"error": "file_not_found", "message": "Oxide classification file not found"}), 404

    except Exception as e:
        logger.exception(f"oxide_groups_error: {str(e)}")
        return jsonify({"error": "server_error", "message": str(e)}), 500

@app.route('/api/umf_to_weights', methods=['POST'])
def convert_umf_to_weights():
    """
    API endpoint that converts a UMF formula into weight percent

    POST JSON parameters:
    {
        "umf": {"SiO2": 4, "Al2O3": 1, "Na2O": 0.5, "K2O": 0.5}
    }

    Returns:
    {
        "weights": {"SiO2": 65.2, "Al2O3": 18.1, "Na2O": 8.4, "K2O": 8.3}
    }
    """
    try:
        # The shape check of /api/solve and /api/feasibility, in the same words:
        # "missing_umf" is one code across this server, so it cannot mean
        # "absent" in one endpoint and "absent or unusable" in another. Before
        # this, {"umf": []} reached umf_to_weights and came back a 500, and
        # {"umf": {}} came back 200 with an empty result
        data = request.get_json(silent=True)

        umf = data.get('umf') if isinstance(data, dict) else None
        if not isinstance(umf, dict) or not umf:
            logger.warning("missing_umf parameter in umf_to_weights request")
            return jsonify({"error": "missing_umf",
                            "message": "umf parameter is required and must be a non-empty object"}), 400

        logger.info(f"converting umf to weights: {umf}")
        
        weights = umf_to_weights(umf)
        
        return jsonify({"weights": weights})
    
    except Exception as e:
        logger.exception(f"umf_to_weights_error: {str(e)}")
        return jsonify({"error": "server_error", "message": str(e)}), 500

@app.route('/api/weights_to_umf', methods=['POST'])
def convert_weights_to_umf():
    """
    API endpoint that converts weight percent into a UMF formula

    POST JSON parameters:
    {
        "weights": {"SiO2": 65.2, "Al2O3": 18.1, "Na2O": 8.4, "K2O": 8.3}
    }

    Returns:
    {
        "umf": {"SiO2": 4, "Al2O3": 1, "Na2O": 0.5, "K2O": 0.5}
    }
    """
    try:
        # The shape check the other three endpoints make, with this one's own
        # code and its own parameter name: "a body that does not parse as JSON
        # is a client mistake and not a server failure" is a property of an
        # endpoint, not of what the parameter happens to be called
        data = request.get_json(silent=True)

        weights = data.get('weights') if isinstance(data, dict) else None
        if not isinstance(weights, dict) or not weights:
            logger.warning("missing_weights parameter in weights_to_umf request")
            return jsonify({"error": "missing_weights",
                            "message": "weights parameter is required and must be "
                                       "a non-empty object"}), 400

        logger.info(f"converting weights to umf: {weights}")
        
        umf = weights_to_umf(weights)
        
        return jsonify({"umf": umf})
    
    except Exception as e:
        logger.exception(f"weights_to_umf_error: {str(e)}")
        return jsonify({"error": "server_error", "message": str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """
    API endpoint that reports whether the server is alive
    """
    logger.debug("health check requested")
    return jsonify({"status": "ok"})

@app.route('/api/materials', methods=['GET'])
def get_materials():
    """
    API endpoint that returns the list of all available materials

    GET parameters:
        inventory_only (bool, optional): if true, only the materials of the inventory are returned

    Returns:
    [
        {
            "name": "Material 1",
            "formula": {"SiO2": 65.2, "Al2O3": 18.1, ...},
            "description": "material description",
            "id": 123,
            "inInventory": true,
            "priority": 2,
            ...
        },
        ...
    ]
    """
    try:
        inventory_only = request.args.get('inventory_only', 'false').lower() == 'true'
        
        # Load every known material
        materials = load_materials(only_inventory=False, priority=True)
        
        if inventory_only:
            # Keep only the materials flagged as inInventory
            materials = [material for material in materials if material.get('inInventory', False)]
        
        logger.info(f"returning {len(materials)} materials, inventory_only={inventory_only}")
        return jsonify(materials)
    
    except Exception as e:
        logger.exception(f"materials_error: {str(e)}")
        return jsonify({"error": "server_error", "message": str(e)}), 500

@app.route('/api/glazy_import', methods=['POST'])
def glazy_import():
    """
    API endpoint that imports a public recipe from glazy.org

    Only the target formula and the original recipe are imported: the materials
    of Glazy are NOT mapped onto the local database, the solver picks its own.

    POST JSON parameters:
    {
        "recipe": "https://glazy.org/recipes/72382"  // recipe URL or id, as typed
        // "recipe_id": 72382                        // accepted as well
    }

    Returns:
    {
        "id": 72382,
        "name": "OVO Perfect Matte (40-25-10)",
        "url": "https://glazy.org/recipes/72382",
        "umf": {"SiO2": 2.583, "Al2O3": 0.577, ...},  // target for /api/solve
        "umf_source": "weights",  // "weights" = recomputed from the weight
                                  // analysis in our flux basis (the normal case),
                                  // "glazy_umf" = taken from Glazy as is, in its
                                  // own basis, because there was nothing to
                                  // recompute from
        "umf_glazy": {"SiO2": 2.5824, ...},   // the UMF of Glazy, for display
        "umf_basis_diff": 0.0006, // largest per-oxide difference between "umf"
                                  // and "umf_glazy"; null when Glazy gave no
                                  // UMF to compare against. A large value means
                                  // the two flux bases differ (a lead recipe)
        "weight_percent": {"SiO2": 48.5952, ...},
        "components": [
            {"name": "...", "percentage": 40.0, "is_additional": false, "glazy_material_id": 20668}
        ],
        "cone_from": "5½",     // null when absent
        "cone_to": "7",        // null when absent
        "thermal_expansion": 7.934  // null when absent
    }
    """
    try:
        # silent=True: a body that is not JSON at all is a missing parameter for
        # this endpoint, not the HTML 400 page Flask would raise on its own
        data = request.get_json(silent=True)

        raw_recipe = None
        if isinstance(data, dict):
            # An explicit "recipe": null must fall back to recipe_id as well, so
            # the fallback is on the VALUE and not on the presence of the key
            raw_recipe = data.get('recipe')
            if raw_recipe is None:
                raw_recipe = data.get('recipe_id')

        if raw_recipe is None or (isinstance(raw_recipe, str) and not raw_recipe.strip()):
            logger.warning("glazy_import_missing_recipe parameter in request")
            return jsonify({
                "error": "missing_recipe",
                "message": "recipe or recipe_id parameter is required"
            }), 400

        recipe_id = parse_recipe_id(raw_recipe)
        if recipe_id is None:
            # repr + a length cap: the raw value is user input and the log format
            # is one line per record, so a newline in it would forge records
            logger.warning(f"glazy_import_invalid_recipe_id: {repr(raw_recipe)[:200]}")
            return jsonify({
                "error": "invalid_recipe_id",
                "message": "expected a glazy.org recipe url or a numeric recipe id"
            }), 400

        logger.info(f"glazy_import_requested: recipe_id={recipe_id}")

        result = build_import_result(fetch_recipe(recipe_id), recipe_id)

        logger.info(f"glazy_import_done: recipe_id={recipe_id}, umf_source={result['umf_source']}, oxides={len(result['umf'])}, components={len(result['components'])}")
        return jsonify(make_json_safe(result))

    except GlazyImportError as e:
        logger.warning(f"glazy_import_failed: {e.code}: {e.message}")
        return jsonify({"error": e.code, "message": e.message}), e.http_status

    except Exception as e:
        logger.exception(f"glazy_import_error: {str(e)}")
        return jsonify({"error": "server_error", "message": str(e)}), 500

# Serving of the static UI files
@app.route('/', defaults={'path': 'index.html'})
@app.route('/<path:path>')
def serve_ui(path):
    """
    Serve the static UI files
    """
    if path.startswith('api/'):
        return jsonify({"error": "not_found", "message": "API endpoint not found"}), 404
    
    logger.debug(f"serving ui file: {path}")
    return send_from_directory(UI_DIR, path)

if __name__ == '__main__':
    logger.info("starting glaze recipe api server on 0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False) 