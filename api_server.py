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
import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from solver_classic import find_multiple_solutions, calculate_recipe_composition
from solver_iterative import find_best_recipe
from common import (weights_to_umf, umf_to_weights, load_materials, make_json_safe,
                    resolve_inventory, filter_materials_by_inventory,
                    load_oxide_classification)
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
    about 1e-4.

    Three fields of the iterative solver are passed through as well, because
    without them the answer cannot be interpreted: 'objective_error' (what the
    search actually minimized, error plus the damped contamination),
    'unlisted_weight' (the penalize_unlisted that was applied, whether it came
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
    [
        {
            "recipe": {"Material1": 45.2, "Material2": 54.8},
            "error": 0.0123,
            "target_composition": {"SiO2": 4, "Al2O3": 1, ...},
            "actual_composition": {"SiO2": 3.98, "Al2O3": 1.02, ...},
            "weight_composition": {"SiO2": 65.2, "Al2O3": 18.1, ...},
            "materials_count": 2,
            "recipe_umf": {"SiO2": 3.98, "Al2O3": 1.02, ...},  // UMF of this particular recipe
            // iterative solver only (so, by default), see
            // iterative_solutions_to_classic_format:
            "objective_error": 0.0123,
            "unlisted_weight": 1.0,
            "unity_scale": 1.0
        },
        ...
    ]
    """
    try:
        data = request.get_json()
        
        if not data or 'umf' not in data:
            logger.warning("missing_umf parameter in request")
            return jsonify({"error": "missing_umf", "message": "umf parameter is required"}), 400
        
        umf = data['umf']
        max_solutions = data.get('max_solutions', 3)
        min_materials = data.get('min_materials', True)
        error_tolerance = data.get('error_tolerance', 0.01)
        inventory_data = data.get('inventory', None)
        solver_name = data.get('solver', DEFAULT_SOLVER)
        penalize_unlisted = data.get('penalize_unlisted', DEFAULT_PENALIZE_UNLISTED)

        if solver_name not in AVAILABLE_SOLVERS:
            logger.warning(f"unknown_solver requested: {solver_name}")
            return jsonify({
                "error": "unknown_solver",
                "message": f"unknown solver '{solver_name}', expected one of: {', '.join(AVAILABLE_SOLVERS)}"
            }), 400

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
                return jsonify({"error": "invalid_parameter", "message": str(exc)}), 400

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
            return jsonify({"error": "calculation_error", "message": solutions['error']}), 500
        
        # Add the UMF information to every recipe
        for solution in solutions:
            # The UMF of a particular recipe is already stored in actual_composition,
            # but it is also exposed as a separate field for convenience on the frontend
            solution['recipe_umf'] = solution['actual_composition']
        
        # Prepare the results for safe JSON serialization
        safe_solutions = make_json_safe(solutions)
        
        logger.info(f"found {len(solutions)} solutions")
        return jsonify(safe_solutions)
    
    except Exception as e:
        logger.exception(f"server_error: {str(e)}")
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
        data = request.get_json()
        
        if not data or 'umf' not in data:
            logger.warning("missing_umf parameter in umf_to_weights request")
            return jsonify({"error": "missing_umf", "message": "umf parameter is required"}), 400
        
        umf = data['umf']
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
        data = request.get_json()
        
        if not data or 'weights' not in data:
            logger.warning("missing_weights parameter in weights_to_umf request")
            return jsonify({"error": "missing_weights", "message": "weights parameter is required"}), 400
        
        weights = data['weights']
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