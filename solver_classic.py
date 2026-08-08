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
import argparse
import logging
import numpy as np
from scipy.optimize import nnls

from common import (
    ClassificationError,
    umf_to_weights,
    weights_to_umf,
    resolve_inventory,
    filter_materials_by_inventory,
    filter_materials_with_formula,
    load_materials,
)

# find_multiple_solutions() takes a parameter called "logging", which shadows
# the module inside it; every log call in this file therefore goes through this
# object and never through the module name.
logger = logging.getLogger(__name__)


# Build the oxide matrix for all available materials
def create_oxide_matrix(materials, target_oxides):
    n_materials = len(materials)
    n_oxides = len(target_oxides)
    
    oxide_matrix = np.zeros((n_oxides, n_materials))
    material_names = []
    
    for j, material in enumerate(materials):
        material_names.append(material['name'])
        formula = material.get('formula', {})
        
        for i, oxide in enumerate(target_oxides):
            oxide_matrix[i, j] = formula.get(oxide, 0.0)
    
    return oxide_matrix, material_names

# Error between the target and the actual UMF
def calculate_umf_error(target_umf, actual_umf):
    # Only oxides present in the target UMF are taken into account
    squared_error = 0.0
    
    for oxide in target_umf.keys():
        target_value = target_umf.get(oxide, 0.0)
        actual_value = actual_umf.get(oxide, 0.0)
        squared_error += (target_value - actual_value) ** 2
    
    return np.sqrt(squared_error)

# Actual composition in weight percent, derived from the recipe
def calculate_recipe_composition(materials, recipe):
    composition = {}

    for material_name, percentage in recipe.items():
        # Look the material up by name
        material = None
        for m in materials:
            if m['name'] == material_name:
                material = m
                break
        
        if material is None:
            continue
        
        # Add the contribution of every oxide of the material
        for oxide, content in material.get('formula', {}).items():
            if oxide not in composition:
                composition[oxide] = 0.0
            composition[oxide] += content * (percentage / 100.0)
    
    return composition

# Non-negative least squares solution
def solve_recipe(oxide_matrix, target_umf, material_names, available_materials=None, target_weights=None):
    """
    Solve one NNLS problem for a target UMF over the given oxide matrix

    Args:
        oxide_matrix: oxides x materials matrix of the material formulas
        target_umf: target UMF formula
        material_names: names of the matrix columns
        available_materials: material dictionaries behind the columns; without
            them the resulting composition is only estimated from the matrix
        target_weights: umf_to_weights(target_umf), when the caller has already
            computed it. The target does not change while a search runs over
            hundreds of material subsets, so converting it once and passing it
            in saves the repeated conversion - and, when the target names an
            oxide with no molar mass, the repeated warning that comes with it.
            Left out, the conversion is done here, so a standalone call behaves
            exactly as before

    Returns:
        dictionary describing the solution, or {'error': ..., 'recipe': {}} when
        this particular material set cannot be solved

    Raises:
        ClassificationError: the oxide classification is unusable. Unlike every
            other failure here this one is not reported as a result: a corrupt
            database is not a material set that happens to have no solution
    """
    # Oxide list taken from the target_umf keys
    target_oxides = list(target_umf.keys())

    # Convert the UMF into weight percent
    if target_weights is None:
        target_weights = umf_to_weights(target_umf)
    weights_array = np.array([target_weights.get(oxide, 0.0) for oxide in target_oxides])

    try:
        # Solve the NNLS problem
        x, _residual = nnls(oxide_matrix, weights_array)

        # A solution too close to zero means the material is not used
        x[x < 1e-6] = 0

        # Normalize to 100%
        if np.sum(x) > 0:
            x = 100 * x / np.sum(x)

        # Build the recipe as a {material: percent} dictionary
        recipe = {}
        for i, name in enumerate(material_names):
            if x[i] > 0.1:  # Ignore materials weighing less than 0.1%
                recipe[name] = round(x[i], 2)

        # Without a material list the actual composition cannot be computed exactly
        if not available_materials:
            # Simplified estimate based on the oxide matrix
            composition = {}
            for i, oxide in enumerate(target_oxides):
                composition[oxide] = 0
                for j, _material in enumerate(material_names):
                    if x[j] > 0:
                        composition[oxide] += oxide_matrix[i, j] * (x[j] / 100)
        else:
            # Actual composition computed from the recipe
            composition = calculate_recipe_composition(available_materials, recipe)

        # Convert weight percent back into UMF. The result is already normalized
        # on the unity basis, so it is compared with the target as it is: any
        # extra rescaling would make the reported error something other than the
        # distance between the two formulas.
        actual_umf = weights_to_umf(composition)

        # Error between the target and the actual UMF
        error = calculate_umf_error(target_umf, actual_umf)
        
        return {
            'recipe': recipe,
            'error': round(error, 4),
            'target_composition': target_umf,
            'actual_composition': {oxide: round(value, 4) for oxide, value in actual_umf.items()},
            'weight_composition': {oxide: round(value, 2) for oxide, value in composition.items()},
            'materials_count': len(recipe)  # Number of materials in the solution
        }

    except ClassificationError:
        # Corrupt reference data, not a degenerate material set. The broad
        # handler below would turn it into "this subset has no solution", and
        # since solver_iterative._solve_material_set calls this function too,
        # both engines would report an empty answer for a broken database
        # instead of failing. The clause below is untouched; this one only
        # takes the one exception that must never be reported as a result.
        raise

    except Exception as e:
        return {
            'error': 'Ошибка решения: ' + str(e),
            'recipe': {}
        }


def _warn_no_usable_materials(inventory, materials_in_inventory):
    """
    Log why there is nothing to build a recipe from

    Three different situations end in the same error, and the log line has to
    say which one it was: an empty inventory is the caller's own doing, names
    that match nothing point at the inventory, and materials filtered out for
    an empty formula point at the database.

    Args:
        inventory: collection of material names the caller asked for
        materials_in_inventory: materials matched by those names, before the
            empty formula filter
    """
    if not inventory:
        logger.warning("the inventory is empty, there is nothing to build a recipe from")
    elif not materials_in_inventory:
        logger.warning(f"none of the {len(inventory)} inventory names matches a material of the database")
    else:
        logger.warning(f"all {len(materials_in_inventory)} materials of the inventory have an empty "
                       f"formula and cannot carry any oxide")


# Solve for a given target UMF using the whole inventory at once
def solve_glaze_recipe(target_umf, inventory_data=None):
    materials = load_materials(only_inventory=False, priority=True)
    inventory = resolve_inventory(inventory_data)

    # Keep only the materials available in the inventory that can carry oxides:
    # a material with an empty formula is a zero column of the NNLS matrix
    materials_in_inventory = filter_materials_by_inventory(materials, inventory)
    available_materials = filter_materials_with_formula(materials_in_inventory)

    if not available_materials:
        _warn_no_usable_materials(inventory, materials_in_inventory)
        return {'error': 'нет_доступных_материалов_в_инвентаре'}

    # All oxides of the target formula
    target_oxides = list(target_umf.keys())

    # Build the oxide matrix
    oxide_matrix, material_names = create_oxide_matrix(available_materials, target_oxides)

    # Check whether an exact solution is possible at all
    rank = np.linalg.matrix_rank(oxide_matrix)

    if rank < len(target_oxides):
        print(f"предупреждение: ранг матрицы ({rank}) меньше количества оксидов ({len(target_oxides)}). точное решение невозможно.")

    # The target does not change, so it is converted into weights once here
    target_weights = umf_to_weights(target_umf)

    # Solve
    solution = solve_recipe(oxide_matrix, target_umf, material_names, available_materials,
                            target_weights)

    return solution


# Search for several solutions built from different material subsets
def find_multiple_solutions(target_umf, max_solutions=5, min_materials=True, error_tolerance=1, logging=True, inventory_data=None, seed: int | None = 0):
    """
    Find several solutions for a given target UMF formula

    Args:
        target_umf: target UMF formula
        max_solutions: maximum number of solutions
        min_materials: if True, prefer solutions with fewer materials
        error_tolerance: acceptable error increase for solutions with fewer materials
        logging: enable logging of the search process
        inventory_data: optional inventory data instead of the default inventory
        seed: seed of the random generator drawing the material subsets; the
            default makes the search reproducible, seed=None makes it
            non-deterministic

    Returns:
        List of solutions sorted by preference
    """
    # A private generator, not the global numpy one: pinning np.random from the
    # outside must not be able to change what this search does
    rng = np.random.default_rng(seed)

    materials = load_materials(only_inventory=False, priority=True)
    inventory = resolve_inventory(inventory_data)

    # Keep only the materials available in the inventory that can carry oxides:
    # a material with an empty formula is a zero column of the NNLS matrix
    materials_in_inventory = filter_materials_by_inventory(materials, inventory)
    available_materials = filter_materials_with_formula(materials_in_inventory)

    if not available_materials:
        _warn_no_usable_materials(inventory, materials_in_inventory)
        return {'error': 'нет_доступных_материалов_в_инвентаре'}

    # All oxides of the target formula
    target_oxides = list(target_umf.keys())

    # The target is the same for every subset below, so it is converted into
    # weights once: hundreds of identical conversions per search otherwise, each
    # of them re-reporting the same unknown oxide
    target_weights = umf_to_weights(target_umf)

    # Build the full oxide matrix
    full_oxide_matrix, material_names = create_oxide_matrix(available_materials, target_oxides)

    # Base solution
    base_solution = solve_recipe(full_oxide_matrix, target_umf, material_names, available_materials,
                                 target_weights)
    solutions = [base_solution]

    # Look for alternative solutions by varying the set of materials used
    n_materials = len(available_materials)
    used_combinations = set()

    # Minimum number of materials worth trying
    min_required = max(3, len(target_oxides) - 3)  # Leave some room when choosing the minimum

    # Start with the SMALLEST material counts
    if min_materials:
        # Begin with very few materials and increase gradually

        # Try from min_required up to min_required + 5 materials (higher priority)
        for subset_size in range(min_required, min(n_materials, min_required + 5)):
            # Generate more combinations for small subsets
            attempts = min(200, n_materials * 3)

            if logging:
                print(f"Ищем решения с {subset_size} материалами...")
            
            for _attempt in range(attempts):
                subset_indices = rng.choice(n_materials, subset_size, replace=False)
                subset_key = tuple(sorted(subset_indices))
                
                if subset_key in used_combinations:
                    continue
                
                used_combinations.add(subset_key)
                
                # Build the submatrix for the selected materials
                subset_matrix = full_oxide_matrix[:, subset_indices]
                subset_names = [material_names[i] for i in subset_indices]
                subset_materials = [available_materials[i] for i in subset_indices]
                
                # Check the matrix rank (it must not be too low)
                rank = np.linalg.matrix_rank(subset_matrix)
                if rank < min_required - 1:  # Allow a small slack on the rank
                    continue

                solution = solve_recipe(subset_matrix, target_umf, subset_names, subset_materials,
                                        target_weights)

                # Allow a larger error for solutions built from fewer materials:
                # the fewer the materials, the wider the tolerance
                actual_error_tolerance = error_tolerance * (1 + (max(6, n_materials) - subset_size) * 0.05)

                # Acceptable solution found within the tolerance
                if solution['recipe'] and solution['error'] < base_solution['error'] * (1 + actual_error_tolerance):
                    solutions.append(solution)
                    if len(solutions) > 1:
                        if logging:
                            print(f"Найдено решение с {len(solution['recipe'])} материалами и ошибкой {solution['error']}")
    
    # If more solutions are still needed, search across varying material counts
    if len(solutions) < max_solutions:
        for subset_size in range(min_required, min(n_materials, 12)):
            # Plain search over random material subsets
            for _attempt in range(min(30, n_materials)):
                subset_indices = rng.choice(n_materials, subset_size, replace=False)
                subset_key = tuple(sorted(subset_indices))
                
                if subset_key in used_combinations:
                    continue
                
                used_combinations.add(subset_key)
                
                # Build the submatrix for the selected materials
                subset_matrix = full_oxide_matrix[:, subset_indices]
                subset_names = [material_names[i] for i in subset_indices]
                subset_materials = [available_materials[i] for i in subset_indices]
                
                solution = solve_recipe(subset_matrix, target_umf, subset_names, subset_materials,
                                        target_weights)
                
                # Store the solution if it is acceptable
                if solution['recipe'] and solution['error'] < base_solution['error'] * 3:
                    solutions.append(solution)

                    if len(solutions) >= max_solutions * 2:  # Collect extra solutions to sort later
                        break
            
            if len(solutions) >= max_solutions * 2:
                break
    
    # Sort the solutions by both error and material count
    if min_materials:
        # Composite sorting metric: solutions with fewer materials are preferred
        # as long as their error stays within error_tolerance of the best one
        best_error = min(solution['error'] for solution in solutions)

        def sort_key(solution):
            num_materials = solution['materials_count']
            err = solution['error']

            # Error tolerance multiplier depending on the material count:
            # the fewer the materials, the wider the tolerance
            error_multiplier = 1 + (0.1 * (8 - num_materials)) if num_materials < 8 else 1
            error_threshold = best_error * error_multiplier

            if err <= error_threshold:
                # Prefer solutions with fewer materials while the error
                # stays inside the widened tolerance
                return (0, num_materials, err)
            else:
                # Otherwise sort by error
                return (1, err, num_materials)

        solutions.sort(key=sort_key)
    else:
        # Sort by error only
        solutions.sort(key=lambda x: x['error'])

    # Drop duplicates by recipe composition
    unique_solutions = []
    seen_recipes = set()

    for sol in solutions:
        # Build a unique recipe identifier
        recipe_key = tuple(sorted((k, round(v, 1)) for k, v in sol['recipe'].items()))
        if recipe_key not in seen_recipes:
            seen_recipes.add(recipe_key)
            unique_solutions.append(sol)

            if len(unique_solutions) >= max_solutions:
                break

    # Return the requested number of best solutions
    return unique_solutions

# Command line entry point
def main():
    parser = argparse.ArgumentParser(description='Glaze Recipe Solver')
    parser.add_argument('--umf', type=str, required=True, help='Target UMF composition as JSON string, e.g., \'{"SiO2": 4, "Al2O3": 1, "Na2O": 0.5, "K2O": 0.5}\'')
    parser.add_argument('--solutions', type=int, default=3, help='Number of solutions to find (default: 3)')
    parser.add_argument('--min-materials', action='store_true', help='Prefer solutions with fewer materials')
    parser.add_argument('--error-tolerance', type=float, default=0.01, help='Acceptable error increase for solutions with fewer materials (default: 0.01)')
    parser.add_argument('--inventory', type=str, help='Custom inventory as a JSON list of material names, instead of the materials flagged inInventory in materials.json')
    
    args = parser.parse_args()
    
    try:
        target_umf = json.loads(args.umf)
        inventory_data = json.loads(args.inventory) if args.inventory else None
    except json.JSONDecodeError as e:
        if args.inventory and 'args.inventory' in str(e):
            print("ошибка: неверный формат JSON для инвентаря")
        else:
            print("ошибка: неверный формат JSON для UMF")
        return
    
    solutions = find_multiple_solutions(
        target_umf, 
        max_solutions=args.solutions,
        min_materials=args.min_materials,
        error_tolerance=args.error_tolerance,
        inventory_data=inventory_data
    )
    
    if isinstance(solutions, dict) and 'error' in solutions:
        print(f"ошибка: {solutions['error']}")
    else:
        print(f"\nнайдено {len(solutions)} решений для заданной UMF-формулы:")
        
        for i, solution in enumerate(solutions):
            print(f"\n[решение {i+1}] ошибка: {solution['error']} | материалов: {solution['materials_count']}")
            print("\nсостав рецепта (вес в %):")
            
            for material, percentage in solution['recipe'].items():
                print(f"  {material}: {percentage}%")
            
            print("\nцелевой состав (UMF):")
            for oxide, value in sorted(solution['target_composition'].items()):
                print(f"  {oxide}: {value}")
            
            print("\nфактический состав (UMF):")
            for oxide, value in sorted(solution['actual_composition'].items()):
                print(f"  {oxide}: {value}")
            
            print("-" * 50)

if __name__ == "__main__":
    main()
