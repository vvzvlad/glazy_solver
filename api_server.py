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

# Available solver engines; the classic one stays the default so that the
# behaviour without the "solver" parameter is unchanged
SOLVER_CLASSIC = 'classic'
SOLVER_ITERATIVE = 'iterative'
AVAILABLE_SOLVERS = (SOLVER_CLASSIC, SOLVER_ITERATIVE)

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
        "min_materials": true,  // optional, true by default, classic solver only
        "error_tolerance": 0.01,  // optional, 0.01 by default, classic solver only
        "inventory": ["Material1", "Material2", ...],  // optional, list of available materials
        "solver": "classic",  // optional, "classic" (default) or "iterative"
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
            // iterative solver only, see iterative_solutions_to_classic_format:
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
        solver_name = data.get('solver', SOLVER_CLASSIC)
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