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
Iterative glaze recipe solver.

The algorithm mimics the way a human ceramist works:

1. Start from the highest priority materials of the inventory (whole priority
   groups are taken one by one until the starting set is big enough).
2. Solve the mix with NNLS in WEIGHT space (the same math the classic solver
   uses: target UMF -> weights, NNLS over the material formulas, result back
   to UMF), drop materials weighing less than 0.1% and re-solve.
3. Look at the per-oxide residual, find the oxide that is the furthest away
   from the target.
4. Rank the materials that are not in the set yet by how well they cover that
   gap without contaminating the already matched oxides, then try them out:
   the candidate that brings the error down the most wins, and when several
   candidates end up equally good the higher priority (lower priority number)
   decides.
5. Add it to the set and go back to step 2.

A branch stops when its error drops below the threshold and the pool already
holds as many acceptable recipes as the caller asked for, when the material
limit is reached, when the iteration limit is reached, or when the error stops
improving (less than 1% per iteration). Nothing is ever lost: every solved set
goes into a pool and the best states are picked from it at the end, which is
the rollback to the best state found.

When more than one solution is requested the candidate step keeps the top-K
materials and the search turns into a beam search over several branches.
"""

import argparse
import json
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

from common import (
    DEFAULT_PRIORITY,
    filter_materials_by_inventory,
    load_materials,
    resolve_inventory,
    umf_to_weights,
    weights_to_umf,
)
from solver_classic import (
    calculate_recipe_composition,
    calculate_umf_error,
    create_oxide_matrix,
    solve_recipe,
)

# Recipe entries below this weight percent are considered noise and dropped
MIN_MATERIAL_WEIGHT = 0.1

# Whole priority groups are added to the starting set until it holds at least
# this many materials
DEFAULT_MIN_START_MATERIALS = 3

# A human converges in 5-6 steps, the default limit keeps some headroom
DEFAULT_MAX_ITERATIONS = 8

# A branch is considered stalled when one iteration improves the error by less
# than this relative amount
STALL_IMPROVEMENT = 0.01

# How many candidate materials are explored per iteration when several
# solutions are requested (beam search)
TOP_CANDIDATES = 3

# Maximum number of branches kept alive by the beam search
MAX_BEAM_WIDTH = 4

# Candidates whose score is within this relative distance from the best one are
# treated as equal, so that priority decides between them
CANDIDATE_SCORE_TIE = 0.05

# Solutions whose error is within this distance from the best one are treated as
# equally good, so that the material count decides between them
SOLUTION_ERROR_TIE_REL = 0.2
SOLUTION_ERROR_TIE_ABS = 0.01


def _expand_target(target_umf: Dict[str, float], materials: Sequence[Dict]) -> Dict[str, float]:
    """
    Extend the target UMF with explicit zeros for every oxide the available
    materials can bring in.

    An oxide missing from the target means "not wanted", so listing it with a
    zero makes both the NNLS and the error metric penalize contamination
    (P2O5 from bone ash, for example) instead of ignoring it.
    """
    full_target = dict(target_umf)

    for material in materials:
        for oxide in material.get('formula', {}):
            if oxide not in full_target:
                full_target[oxide] = 0.0

    return full_target


def _normalize_to_100(composition: Dict[str, float]) -> Dict[str, float]:
    """Scale an oxide composition so that its parts sum up to 100"""
    total = sum(composition.values())
    if total <= 0:
        return {oxide: 0.0 for oxide in composition}
    return {oxide: value * 100.0 / total for oxide, value in composition.items()}


def _priority_start_set(materials: Sequence[Dict], min_count: int) -> List[Dict]:
    """
    Build the starting material set out of whole priority groups.

    Groups are taken in order of increasing priority number (lower number =
    higher priority) until the set holds at least min_count materials.
    """
    ordered = sorted(materials, key=lambda m: (m.get('priority', DEFAULT_PRIORITY), m.get('name', '')))

    start_set: List[Dict] = []
    index = 0

    while index < len(ordered):
        current_priority = ordered[index].get('priority', DEFAULT_PRIORITY)

        # Take the whole priority group, never a part of it
        while index < len(ordered) and ordered[index].get('priority', DEFAULT_PRIORITY) == current_priority:
            start_set.append(ordered[index])
            index += 1

        if len(start_set) >= min_count:
            break

    return start_set


def _solve_material_set(material_set: Sequence[Dict], full_target: Dict[str, float],
                        target_umf: Dict[str, float]) -> Optional[Dict[str, Any]]:
    """
    Solve one material set with NNLS, dropping the materials weighing less than
    MIN_MATERIAL_WEIGHT and re-solving until the recipe is stable.

    The whole material set is kept in the state even when the solver does not
    use all of it: a material that is useless now may become useful once
    another one joins the set on a later iteration.

    Returns a state dictionary with the recipe, the resulting UMF and the error,
    or None when no recipe could be built.
    """
    oxides = list(full_target.keys())
    active = list(material_set)
    recipe: Dict[str, float] = {}

    # Dropping a material changes the optimum, so re-solve until the set settles
    for _ in range(len(material_set)):
        oxide_matrix, material_names = create_oxide_matrix(active, oxides)
        solution = solve_recipe(oxide_matrix, full_target, material_names, active)

        recipe = solution.get('recipe') or {}
        recipe = {name: weight for name, weight in recipe.items() if weight >= MIN_MATERIAL_WEIGHT}
        if not recipe:
            return None

        used = [material for material in active if material['name'] in recipe]
        if len(used) == len(active):
            break
        active = used

    # Normalize the recipe so that it sums up to exactly 100%
    total = float(sum(recipe.values()))
    if total <= 0:
        return None
    recipe = {name: round(float(weight) * 100.0 / total, 2) for name, weight in recipe.items()}

    composition = calculate_recipe_composition(active, recipe)
    result_umf = {oxide: float(value) for oxide, value in weights_to_umf(composition).items()}

    # full_target carries zeros for the unwanted oxides, so contamination counts
    error = float(calculate_umf_error(full_target, result_umf))

    return {
        'materials': list(material_set),
        'recipe': recipe,
        'result_umf': result_umf,
        'target_umf': dict(target_umf),
        'error': error,
        'weight_composition': composition,
        'materials_count': len(recipe),
    }


def _shrink_to_limit(state: Dict[str, Any], full_target: Dict[str, float],
                     target_umf: Dict[str, float], max_materials: int) -> Dict[str, Any]:
    """
    Bring a state down to the material limit by dropping the lightest material
    of the recipe and solving again.

    Needed when whole priority groups make the starting set larger than the
    caller allows.
    """
    while state['materials_count'] > max_materials:
        lightest = min(state['recipe'], key=lambda name: state['recipe'][name])
        reduced = [material for material in state['materials'] if material['name'] != lightest]
        if not reduced:
            break

        smaller = _solve_material_set(reduced, full_target, target_umf)
        if smaller is None:
            break
        smaller['iterations'] = state['iterations']
        state = smaller

    return state


def _focus_oxide(full_target: Dict[str, float], result_umf: Dict[str, float]) -> Optional[str]:
    """Return the oxide with the largest absolute deviation in UMF space"""
    worst_oxide = None
    worst_gap = 0.0

    for oxide in set(full_target) | set(result_umf):
        gap = abs(full_target.get(oxide, 0.0) - result_umf.get(oxide, 0.0))
        if gap > worst_gap:
            worst_gap = gap
            worst_oxide = oxide

    return worst_oxide


def _rank_candidates(candidates: Sequence[Dict], residual: Dict[str, float],
                     focus: Optional[str]) -> List[Tuple[float, Dict]]:
    """
    Rank the materials that are not in the set yet by how well they close the
    residual.

    The residual is expressed in weight percent (target weights minus the
    weights the current recipe produces), so a positive value means a deficit
    and a negative one an excess. The score of a material is the gain on the
    focus oxide minus the contamination it brings to the oxides that already
    match; ties are decided by priority.
    """
    scored: List[Tuple[float, Dict]] = []

    for material in candidates:
        formula = material.get('formula', {})
        total = sum(formula.values())
        if total <= 0:
            continue

        # Composition as fractions of the material weight
        fractions = {oxide: value / total for oxide, value in formula.items()}

        focus_gain = 0.0
        if focus is not None:
            focus_gain = residual.get(focus, 0.0) * fractions.get(focus, 0.0)

        # Every other oxide either helps (deficit) or pollutes (excess)
        cross_term = 0.0
        for oxide, fraction in fractions.items():
            if oxide == focus:
                continue
            cross_term += residual.get(oxide, 0.0) * fraction

        scored.append((focus_gain + cross_term, material))

    if not scored:
        return []

    scored.sort(key=lambda item: -item[0])

    # Candidates that score about the same are re-ordered by priority
    best_score = scored[0][0]
    tie_limit = best_score - abs(best_score) * CANDIDATE_SCORE_TIE
    near_best = [item for item in scored if item[0] >= tie_limit]
    rest = [item for item in scored if item[0] < tie_limit]
    near_best.sort(key=lambda item: (item[1].get('priority', DEFAULT_PRIORITY), -item[0]))

    return near_best + rest


def _expand_state(state: Dict[str, Any], available_materials: Sequence[Dict],
                  full_target: Dict[str, float], target_umf: Dict[str, float],
                  target_weights: Dict[str, float], seen_sets: set,
                  verbose: bool) -> List[Dict[str, Any]]:
    """
    Try to add one material to the set of a state.

    The candidates are ranked by the residual heuristic first and then actually
    tried out: a material is worth adding only if the recipe it produces is
    really better, which the heuristic alone cannot tell (wollastonite, for
    instance, can be replaced by chalk plus quartz without any visible change
    in the weight space residual).

    Returns the resulting states ordered from best to worst.
    """
    used_names = {material['name'] for material in state['materials']}
    candidates = [material for material in available_materials if material['name'] not in used_names]
    if not candidates:
        return []

    # Residual in weight space: what the current recipe fails to deliver
    actual_weights = _normalize_to_100(state['weight_composition'])
    residual = {}
    for oxide in set(target_weights) | set(actual_weights):
        residual[oxide] = target_weights.get(oxide, 0.0) - actual_weights.get(oxide, 0.0)

    focus = _focus_oxide(full_target, state['result_umf'])
    ranked = _rank_candidates(candidates, residual, focus)

    if verbose:
        logging.info(f"worst oxide {focus}, best candidates by heuristic: "
                     f"{[material['name'] for _, material in ranked[:TOP_CANDIDATES]]}")

    trials: List[Tuple[float, int, Dict[str, Any]]] = []

    for rank, (_score, candidate) in enumerate(ranked):
        new_set = list(state['materials']) + [candidate]
        set_names = frozenset(material['name'] for material in new_set)
        if set_names in seen_sets:
            continue
        seen_sets.add(set_names)

        new_state = _solve_material_set(new_set, full_target, target_umf)
        if new_state is None:
            continue

        new_state['set_names'] = set_names
        new_state['added'] = candidate['name']

        # Rounding the error keeps equally good candidates together, so that the
        # heuristic order (and through it the priority) decides between them
        trials.append((round(new_state['error'], 4), rank, new_state))

    trials.sort(key=lambda item: (item[0], item[1]))

    return [item[2] for item in trials]


def _recipe_key(recipe: Dict[str, float]) -> Tuple:
    """Composition based identity of a recipe, used to drop duplicates"""
    return tuple(sorted((name, round(weight, 1)) for name, weight in recipe.items()))


def _solution_sort_key(solution: Dict[str, Any], best_error: float):
    """
    Sort solutions by error, but prefer fewer materials when the errors are
    practically the same.
    """
    tie_limit = best_error + max(best_error * SOLUTION_ERROR_TIE_REL, SOLUTION_ERROR_TIE_ABS)

    if solution['error'] <= tie_limit:
        return (0, solution['materials_count'], solution['error'])
    return (1, solution['error'], solution['materials_count'])


def find_best_recipe(inventory, target_umf, min_materials=1, max_materials=10,
                     max_solutions=5, verbose=False, error_threshold=0.1) -> List[Dict[str, Any]]:
    """
    Find glaze recipes for a target UMF by adding materials one at a time.

    Args:
        inventory: list of available material names
        target_umf: target UMF formula as {oxide: value}
        min_materials: minimum number of materials in a returned recipe
        max_materials: maximum number of materials in a recipe
        max_solutions: how many solutions to return
        verbose: log the search process
        error_threshold: error below which a recipe is considered acceptable
            and its branch is not refined any further

    Returns:
        list of solutions, best first; every solution holds recipe, error,
        result_umf, target_umf, materials_count and iterations
    """
    if not target_umf:
        return []

    all_materials = load_materials(only_inventory=False, priority=True)
    available_materials = filter_materials_by_inventory(all_materials, resolve_inventory(inventory))

    if not available_materials:
        if verbose:
            logging.info("no materials available in the inventory")
        return []

    full_target = _expand_target(target_umf, available_materials)
    target_weights = umf_to_weights(full_target)

    min_start_materials = max(min_materials, DEFAULT_MIN_START_MATERIALS)
    max_materials = max(max_materials, min_start_materials)

    top_k = 1 if max_solutions <= 1 else TOP_CANDIDATES
    beam_width = 1 if max_solutions <= 1 else min(MAX_BEAM_WIDTH, max_solutions)

    start_set = _priority_start_set(available_materials, min_start_materials)
    if not start_set:
        return []

    if verbose:
        logging.info(f"starting set ({len(start_set)} materials): {[m['name'] for m in start_set]}")

    start_state = _solve_material_set(start_set, full_target, target_umf)
    if start_state is None:
        if verbose:
            logging.info("the starting set produced no recipe")
        return []

    start_state['iterations'] = 1
    start_state = _shrink_to_limit(start_state, full_target, target_umf, max_materials)
    start_state['set_names'] = frozenset(m['name'] for m in start_state['materials'])

    pool: List[Dict[str, Any]] = [start_state]
    beam: List[Dict[str, Any]] = [start_state]
    seen_sets = {start_state['set_names']}
    # Different recipes that already meet the requested quality
    found_recipes = set()
    if start_state['error'] <= error_threshold:
        found_recipes.add(_recipe_key(start_state['recipe']))

    for iteration in range(2, DEFAULT_MAX_ITERATIONS + 1):
        next_beam: List[Dict[str, Any]] = []

        for state in beam:
            # A branch that is good enough is dropped, but only once the pool
            # holds as many different recipes as the caller asked for
            if state['error'] <= error_threshold and len(found_recipes) >= max_solutions:
                if verbose:
                    logging.info(f"branch converged with error {state['error']:.4f}")
                continue

            if state['materials_count'] >= max_materials:
                continue

            children = _expand_state(state, available_materials, full_target, target_umf,
                                     target_weights, seen_sets, verbose)

            for child in children:
                child['iterations'] = iteration
                pool.append(child)
                if child['error'] <= error_threshold:
                    found_recipes.add(_recipe_key(child['recipe']))

            # Only the best branches are kept alive; a branch that stops
            # improving is abandoned and the pool keeps whatever it already found
            for child in children[:top_k]:
                improvement = state['error'] - child['error']
                if improvement <= state['error'] * STALL_IMPROVEMENT:
                    if verbose:
                        logging.info(f"branch stalled on {child['added']}: "
                                     f"{state['error']:.4f} -> {child['error']:.4f}")
                    continue

                if verbose:
                    logging.info(f"iteration {iteration}: added {child['added']}, "
                                 f"error {state['error']:.4f} -> {child['error']:.4f}")

                next_beam.append(child)

        if not next_beam:
            break

        next_beam.sort(key=lambda s: (s['error'], s['materials_count']))
        beam = next_beam[:beam_width]

    # Keep only recipes that respect the material limits
    solutions = [s for s in pool if min_materials <= s['materials_count'] <= max_materials]
    if not solutions:
        solutions = list(pool)

    best_error = min(s['error'] for s in solutions)
    solutions.sort(key=lambda s: _solution_sort_key(s, best_error))

    # Drop duplicates by recipe composition
    unique: List[Dict[str, Any]] = []
    seen_recipes = set()

    for solution in solutions:
        key = _recipe_key(solution['recipe'])
        if key in seen_recipes:
            continue
        seen_recipes.add(key)

        unique.append({
            'recipe': solution['recipe'],
            'error': solution['error'],
            'result_umf': solution['result_umf'],
            'target_umf': solution['target_umf'],
            'materials_count': solution['materials_count'],
            'iterations': solution['iterations'],
        })

        if len(unique) >= max_solutions:
            break

    if verbose and unique:
        logging.info(f"returning {len(unique)} solutions, best error {unique[0]['error']:.4f}")

    return unique


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description='Iterative Glaze Recipe Solver')
    parser.add_argument('--umf', type=str, help='Target UMF composition as JSON string')
    parser.add_argument('--solutions', type=int, default=5, help='Number of solutions to find (default: 5)')
    parser.add_argument('--max-materials', type=int, default=10, help='Maximum number of materials (default: 10)')
    parser.add_argument('--error-threshold', type=float, default=0.1, help='Error at which the search stops (default: 0.1)')
    parser.add_argument('--quiet', action='store_true', help='Do not log the search process')
    args = parser.parse_args()

    # Smoke run target: the transparent glaze made of the five base materials
    target_umf = {
        "Al2O3": 0.379,
        "B2O3": 0.266,
        "CaO": 0.718,
        "Fe2O3": 0.002,
        "K2O": 0.086,
        "MgO": 0.048,
        "Na2O": 0.143,
        "SiO2": 3.151,
        "SrO": 0.005,
        "TiO2": 0.003
    }

    original_recipe = {
        "Волластонит МИВОЛЛ": 20,
        "Каолин КЖФ-1": 15,
        "Кварцевая мука Кварцверке W12": 20,
        "Нефелин-сиенит VR13": 30,
        "Улексит (Химпэк)": 15
    }

    if args.umf:
        target_umf = json.loads(args.umf)
        original_recipe = None

    print("Target UMF:")
    for oxide, value in sorted(target_umf.items()):
        print(f"  {oxide}: {value}")

    if original_recipe:
        print("\nOriginal recipe:")
        for material, weight in original_recipe.items():
            print(f"  {material}: {weight}%")

    inventory = resolve_inventory()

    print("\nSearching for solutions...")
    solutions = find_best_recipe(
        inventory,
        target_umf,
        max_materials=args.max_materials,
        max_solutions=args.solutions,
        verbose=not args.quiet,
        error_threshold=args.error_threshold,
    )

    if not solutions:
        print("\nNo solutions found for the target UMF!")
        return

    print(f"\nFound {len(solutions)} solutions!")
    for index, solution in enumerate(solutions):
        print(f"\nSolution {index + 1}")
        print(f"Error: {solution['error']:.4f} | materials: {solution['materials_count']} | iterations: {solution['iterations']}")
        print("Recipe:")
        for material, weight in sorted(solution['recipe'].items(), key=lambda item: -item[1]):
            print(f"  {material}: {weight:.2f}%")

        print("Resulting UMF:")
        for oxide in sorted(set(solution['target_umf']) | set(solution['result_umf'])):
            actual = solution['result_umf'].get(oxide, 0.0)
            expected = solution['target_umf'].get(oxide, 0.0)
            if actual > 0.0005 or expected > 0.0005:
                print(f"  {oxide}: {actual:.3f} (target: {expected:.3f})")


if __name__ == "__main__":
    main()
