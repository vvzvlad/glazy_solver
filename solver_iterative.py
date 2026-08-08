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
   from the target - the focus oxide of this step.
4. Rank the materials that are not in the set yet by a score that is built from
   three separate terms: the gain on the focus oxide, a smaller gain on the
   remaining deficits and a penalty for contaminating the oxides that already
   match or are already in excess. Material priority is blended into that score,
   so a high priority material wins unless a low priority one is clearly better.
5. Actually solve the top candidates (see candidate_search) and keep the ones
   that really improve the recipe, then go back to step 2.

How much of step 4 survives depends on candidate_search, and the difference is
worth stating plainly:

* 'heuristic' solves only the TOP_CANDIDATES best ranked materials, so the focus
   oxide and the priority genuinely decide what is tried. This is the human
   procedure, and it costs a constant number of NNLS runs per step.
* 'exhaustive' (the default) solves every remaining material of the inventory,
   which makes the ranking a tie break rather than a filter: it decides between
   material sets whose error agrees to four decimals, and nothing more. The
   focus oxide and the priority therefore do not steer this mode.

The default is 'exhaustive' because on the reference set it is measurably more
accurate - the heuristic only matches it once K grows to about HALF the
inventory (see find_best_recipe for the measured numbers), at which point it has
stopped being a shortcut and costs about the same as the exhaustive pass anyway.
The honest summary is that greedy forward selection beats the human shortcut
here; the shortcut is kept, and named, for the cases where the inventory is
large enough that O(inventory) NNLS runs per step hurt.

A branch stops when its error drops below the threshold and the pool already
holds as many acceptable recipes as the caller asked for, when the material
limit is reached, when the iteration limit is reached, or when the error stops
improving (less than 1% per iteration). Nothing is ever lost: every solved set
goes into a pool and the best states are picked from it at the end, which is
the rollback to the best state found.

When more than one solution is requested the candidate step feeds several
children into the beam and the search turns into a beam search over several
branches.
"""

import argparse
import json
import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from common import (
    DEFAULT_PRIORITY,
    filter_materials_by_inventory,
    filter_materials_with_formula,
    flux_oxides,
    load_materials,
    load_molar_masses,
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

logger = logging.getLogger(__name__)

# Recipe entries below this weight percent are considered noise and dropped.
# One percent of a 100 g batch is 1 g, which is where a studio scale stops being
# trustworthy - below it the solver is fitting the fourth decimal of an oxide
# with an ingredient nobody can actually weigh out. The previous 0.1 let those
# through: on the eleven reference recipes it produced 6 such trace entries and
# recovered the original material set in 6 cases out of 11, while 1.0 produces
# none and recovers 7, with a slightly SMALLER total UMF error (115.5467 against
# 115.6420). Raising it further to 2.0 changes nothing, so the threshold is not
# a tuned number, it is a floor.
MIN_MATERIAL_WEIGHT = 1.0

# Whole priority groups are added to the starting set until it holds at least
# this many materials
DEFAULT_MIN_START_MATERIALS = 3

# A human converges in 5-6 steps, the default limit keeps some headroom
DEFAULT_MAX_ITERATIONS = 8

# A branch is considered stalled when one iteration improves the error by less
# than this relative amount
STALL_IMPROVEMENT = 0.01

# How many candidate materials are really solved per iteration in the
# 'heuristic' candidate search: the heuristic proposes, NNLS disposes
TOP_CANDIDATES = 3

# Maximum number of branches kept alive by the beam search
MAX_BEAM_WIDTH = 4

# Solutions whose error is within this distance from the best one are treated as
# equally good, so that the material count decides between them
SOLUTION_ERROR_TIE_REL = 0.2
SOLUTION_ERROR_TIE_ABS = 0.01

# --- unity basis ------------------------------------------------------------
#
# A UMF is a formula normalized so that the fluxes (R2O + RO) sum to 1, and
# weights_to_umf always produces such a vector for a recipe. A target does not
# have to be one: a target typed by hand can list no flux at all (SiO2 + Al2O3
# only), or list fluxes that add up to something other than 1. In that case the
# target and the recipe are normalized by two different quantities and comparing
# them oxide by oxide measures the difference of the two conventions, not the
# chemistry - the recipe for {SiO2: 3.0, Al2O3: 0.35} used to report SiO2 = 8.57
# and an error of 5.6 while having exactly the requested SiO2:Al2O3 ratio.
#
# The fix is to bring the recipe onto the basis of the target with a single
# scalar before comparing, and the two decisions behind it are:
#
# * WHEN. Only when the target is not a unity formula itself. If its fluxes do
#   sum to 1 the length of the target vector is meaningful and must not be
#   fitted away: a glaze carrying 1.5x the silica per unit of flux is a
#   different glaze, and scaling that difference out would hide it instead of
#   reporting it. So a proper UMF target is compared as is (scale 1.0), and only
#   a target that carries no basis of its own is fitted.
# * HOW. By least squares over the listed oxides: k = sum(target*result) /
#   sum(result^2) is the scalar that minimizes ||target - k*result||, and it
#   weights every oxide by its own magnitude. The alternative of pinning one
#   chosen oxide of the target puts the whole scale on that single component,
#   which is why it is not used: pinning a trace oxide such as Fe2O3 = 0.002
#   that the recipe misses by a factor of two would rescale the entire formula
#   by two. The classic solver used to do exactly that and no longer does.
UNITY_BASIS_TOLERANCE = 0.01

# --- candidate scoring ------------------------------------------------------
#
# The heuristic lives entirely in WEIGHT space, and so does the residual it
# works on. That is not an arbitrary choice:
#   * the NNLS problem itself is posed on weight percentages (the target UMF is
#     converted with umf_to_weights and the material formulas are weight
#     percent), so the residual of that very problem is a weight-space vector;
#   * a material formula answers "how many grams of oxide X does one gram of
#     this material carry", which is a weight-space quantity, while UMF is
#     renormalized by the flux sum and is therefore non-linear with respect to
#     mixing - "UMF gain per gram" is not even well defined;
#   * mixing the two spaces (focus picked in UMF, residual measured in weights)
#     was exactly the inconsistency the review found in the previous version.
# The reported error stays in UMF space, because that is the metric the callers
# and the acceptance tests speak; the heuristic only proposes candidates, the
# real NNLS solve decides.

# Deficits on oxides other than the focus one are worth less than the focus
# deficit: this step is about the focus oxide, the others get their own step
SECONDARY_GAIN_WEIGHT = 0.35

# Overshooting is asymmetrically bad: a deficit can be filled by adding another
# material later, an excess can never be subtracted, so contaminating an oxide
# that is already over the target costs more than filling a deficit gains
CONTAMINATION_WEIGHT = 1.5

# An oxide whose weight-percent residual is inside this band counts as matched
MATCHED_OXIDE_TOLERANCE = 0.5

# Bringing anything into an already matched oxide is a disturbance. Its residual
# is ~0 by definition, so without an explicit term the score would ignore it
# completely; MATCHED_OXIDE_TOLERANCE is used as the residual scale to keep the
# term dimensionally comparable with the gain terms
MATCHED_DISTURBANCE_WEIGHT = 0.5

# How strongly the material priority bends the chemical ranking. The chemical
# score is divided by the size of the focus gap, which is the score a material
# made purely of the focus oxide would get, so 0.25 means "a higher priority
# material wins unless the other candidate closes more than a quarter of the
# focus gap on top of what this one closes". That scale is absolute: it does not
# drift with the number of candidates the way a min-max normalization does.
PRIORITY_WEIGHT = 0.25

# Candidate search modes
SEARCH_HEURISTIC = 'heuristic'
SEARCH_EXHAUSTIVE = 'exhaustive'
CANDIDATE_SEARCH_MODES = (SEARCH_HEURISTIC, SEARCH_EXHAUSTIVE)


def _usable_target(target_umf: Dict[str, Any]) -> Dict[str, float]:
    """
    Keep only the target entries the math can actually work with: a known oxide
    (present in molar_masses.json) carrying a finite, non negative number.

    An oxide asked for as an explicit ZERO is kept, and that is deliberate. The
    UI sends the whole oxide table on every request, zeros included ('SrO': 0.0,
    'Fe2O3': 0.0, 'TiO2': 0.0), and "give me no iron" is a constraint, not the
    absence of an opinion. Dropping those entries used to move them into the
    unlisted group, where penalize_unlisted decides their fate - so with a soft
    weight "no iron please" silently turned into "iron is fine". A zero stays in
    the target, gets its NNLS row with a zero right hand side and is penalized
    like any other requested value, which is what the classic solver does too.

    Anything else is dropped: an unknown oxide, a non numeric value, a negative
    value, NaN and infinity.

    Note that a target of nothing but zeros is still unusable - umf_to_weights
    divides by the total molar weight - and find_best_recipe rejects it.
    """
    if not target_umf:
        return {}

    molar_masses = load_molar_masses()
    usable: Dict[str, float] = {}

    for oxide, value in target_umf.items():
        if oxide not in molar_masses:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number) and number >= 0.0:
            usable[oxide] = number

    return usable


def _flux_sum(umf: Dict[str, float]) -> float:
    """Sum of the fluxes (R2O + RO) of a formula - the UMF unity denominator"""
    fluxes = set(flux_oxides())
    return sum(float(value) for oxide, value in umf.items() if oxide in fluxes)


def _unity_scale(target_umf: Dict[str, float], result_umf: Dict[str, float]) -> float:
    """
    Scalar that brings the UMF of a recipe onto the normalization basis of the
    target, so that the two vectors can be compared oxide by oxide.

    Returns exactly 1.0 when both formulas are already unity formulas (their
    fluxes sum to 1 within UNITY_BASIS_TOLERANCE), which is the normal case: the
    target of a real recipe and every recipe the solver builds are both proper
    UMFs, so nothing is scaled and nothing is hidden.

    Otherwise - a target with no flux at all, a target whose fluxes do not add
    up to 1, or the rare recipe that carries no flux and is therefore normalized
    by weights_to_umf against its smallest oxide - only the direction of the
    target vector is meaningful, and the length is fitted by least squares:
    k = sum(target*result) / sum(result^2) minimizes ||target - k*result|| over
    the listed oxides. See the UNITY_BASIS_TOLERANCE block above for why the
    gate exists and why the least squares fit is preferred to pinning a single
    chosen oxide of the target.

    Falls back to 1.0 when the fit is degenerate (an empty or all zero result,
    or a result that shares nothing with the target): a non positive scale is
    not a formula.
    """
    target_fluxes = _flux_sum(target_umf)
    result_fluxes = _flux_sum(result_umf)

    if (abs(target_fluxes - 1.0) <= UNITY_BASIS_TOLERANCE
            and abs(result_fluxes - 1.0) <= UNITY_BASIS_TOLERANCE):
        return 1.0

    numerator = 0.0
    denominator = 0.0

    for oxide, expected in target_umf.items():
        actual = result_umf.get(oxide, 0.0)
        numerator += expected * actual
        denominator += actual * actual

    if denominator <= 0.0 or numerator <= 0.0:
        return 1.0

    return numerator / denominator


def _expand_target(target_umf: Dict[str, float], materials: Sequence[Dict]) -> Dict[str, float]:
    """
    Extend the target UMF with explicit zeros for every oxide the available
    materials can bring in.

    Whether those zeros are actually enforced is decided by the caller through
    penalize_unlisted: the expansion only builds the list of oxides, the weight
    attached to them says how much an unlisted oxide is allowed to appear.
    """
    full_target = dict(target_umf)

    for material in materials:
        for oxide in material.get('formula', {}):
            if oxide not in full_target:
                full_target[oxide] = 0.0

    return full_target


def _normalize_unlisted_weight(penalize_unlisted: Any) -> float:
    """
    Turn the penalize_unlisted argument into a weight in [0, 1].

    True/False are accepted as the hard 1.0 / 0.0 ends of the same scale, so
    that a boolean flag and a soft weight can be used interchangeably.

    Raises ValueError for anything that is not a finite number: a JSON null or a
    typo used to be silently turned into 1.0, which meant the caller got a
    completely different search than the one it asked for and never learned it.
    A finite number outside [0, 1] is clamped and logged - the ends of the scale
    are hard bounds ("must be zero" / "do not care"), there is nothing to
    extrapolate beyond them.
    """
    if penalize_unlisted is True:
        return 1.0
    if penalize_unlisted is False:
        return 0.0

    try:
        weight = float(penalize_unlisted)
    except (TypeError, ValueError):
        raise ValueError(f"penalize_unlisted must be a number in [0, 1] or a boolean, "
                         f"got {penalize_unlisted!r}")

    if not math.isfinite(weight):
        raise ValueError(f"penalize_unlisted must be a finite number in [0, 1], "
                         f"got {penalize_unlisted!r}")

    if weight < 0.0 or weight > 1.0:
        logger.warning(f"penalize_unlisted={weight} is outside [0, 1], clamped to the nearest bound")
        return min(max(weight, 0.0), 1.0)

    return weight


def _int_argument(value: Any, name: str) -> int:
    """
    Coerce one of the integer arguments, refusing what cannot be one.

    min_materials=None (a JSON null that made it through a caller) used to raise
    a bare TypeError from a comparison in the middle of the search; the caller
    now gets a ValueError that names the argument. A float is truncated, which
    is what int() has always done to max_materials here.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an integer, got {value!r}")


def _build_problem(target_umf: Dict[str, float], materials: Sequence[Dict],
                   unlisted_weight: float) -> Dict[str, Any]:
    """
    Pack everything the search needs to know about the target into one context.

    The unlisted oxides get their NNLS rows scaled by unlisted_weight. Their
    right hand side is zero by construction, so scaling the row alone is an
    exact weighted least squares: ||w*(A_i x) - w*0|| == w*||A_i x - 0||. That
    is how a soft "do not bring what I did not ask for" is expressed without
    touching the shared NNLS core.
    """
    full_target = _expand_target(target_umf, materials)
    oxides = list(full_target.keys())
    unlisted = tuple(oxide for oxide in oxides if oxide not in target_umf)

    row_weights = None
    if unlisted and unlisted_weight != 1.0:
        row_weights = np.array([1.0 if oxide in target_umf else unlisted_weight
                                for oxide in oxides])

    return {
        'target_umf': dict(target_umf),
        'full_target': full_target,
        'oxides': oxides,
        'unlisted': unlisted,
        'unlisted_weight': unlisted_weight,
        'row_weights': row_weights,
        'target_weights': umf_to_weights(full_target),
    }


def _objective_error(problem: Dict[str, Any], result_umf: Dict[str, float]) -> float:
    """
    The quantity the search minimizes: the plain UMF error on the requested
    oxides plus the contamination of the unlisted ones, damped by the weight.

    With unlisted_weight == 1.0 this is exactly calculate_umf_error against the
    fully expanded target; with 0.0 it is exactly calculate_umf_error against
    the requested target.
    """
    squared = 0.0

    for oxide, expected in problem['target_umf'].items():
        squared += (expected - result_umf.get(oxide, 0.0)) ** 2

    weight = problem['unlisted_weight']
    if weight > 0.0:
        for oxide in problem['unlisted']:
            squared += (weight * result_umf.get(oxide, 0.0)) ** 2

    return math.sqrt(squared)


def _normalize_to_100(composition: Dict[str, float]) -> Dict[str, float]:
    """Scale an oxide composition so that its parts sum up to 100"""
    total = sum(composition.values())
    if total <= 0:
        return {oxide: 0.0 for oxide in composition}
    return {oxide: value * 100.0 / total for oxide, value in composition.items()}


def _recipe_to_exactly_100(recipe: Dict[str, float]) -> Optional[Dict[str, float]]:
    """
    Scale a recipe to 100% and round it to two decimals so that the parts add up
    to exactly 100.

    Rounding every part on its own leaves a drift of up to half a hundredth per
    material (99.99 / 100.01 in practice); the drift is poured into the heaviest
    component, where it is relatively the least significant.
    """
    total = float(sum(recipe.values()))
    if total <= 0:
        return None

    scaled = {name: round(float(weight) * 100.0 / total, 2) for name, weight in recipe.items()}

    drift = round(100.0 - sum(scaled.values()), 2)
    if drift:
        # sorted() first, so that equal weights always pick the same material
        heaviest = max(sorted(scaled), key=lambda name: scaled[name])
        scaled[heaviest] = round(scaled[heaviest] + drift, 2)

    return scaled


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


def _solve_material_set(material_set: Sequence[Dict], problem: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Solve one material set with NNLS, dropping the materials weighing less than
    MIN_MATERIAL_WEIGHT and re-solving until the recipe is stable.

    The whole material set is kept in the state even when the solver does not
    use all of it: a material that is useless now may become useful once
    another one joins the set on a later iteration.

    Returns a state dictionary with the recipe, the resulting UMF and both error
    numbers, or None when no recipe could be built.
    """
    full_target = problem['full_target']
    oxides = problem['oxides']
    row_weights = problem['row_weights']

    active = list(material_set)
    recipe: Dict[str, float] = {}

    try:
        # Dropping a material changes the optimum, so re-solve until the set settles
        for _ in range(len(material_set)):
            oxide_matrix, material_names = create_oxide_matrix(active, oxides)
            if row_weights is not None:
                oxide_matrix = oxide_matrix * row_weights[:, None]

            solution = solve_recipe(oxide_matrix, full_target, material_names, active)

            recipe = solution.get('recipe') or {}
            recipe = {name: weight for name, weight in recipe.items() if weight >= MIN_MATERIAL_WEIGHT}
            if not recipe:
                return None

            used = [material for material in active if material['name'] in recipe]
            if len(used) == len(active):
                break
            active = used

        recipe = _recipe_to_exactly_100(recipe)
        if not recipe:
            return None

        composition = calculate_recipe_composition(active, recipe)
        recipe_umf = {oxide: float(value) for oxide, value in weights_to_umf(composition).items()}

        # Both errors below compare the target with the recipe oxide by oxide,
        # so the recipe first has to be put on the normalization basis of the
        # target. Normally the two already agree and the scale is exactly 1.0
        unity_scale = _unity_scale(problem['target_umf'], recipe_umf)
        if unity_scale == 1.0:
            result_umf = recipe_umf
        else:
            result_umf = {oxide: value * unity_scale for oxide, value in recipe_umf.items()}
    except (ValueError, ZeroDivisionError, ArithmeticError) as exc:
        # A degenerate material set (nothing convertible, zero total weight) is
        # not a server error, it simply produces no recipe
        logger.debug(f"material set produced no recipe: {exc}")
        return None

    return {
        'materials': list(material_set),
        'recipe': recipe,
        'result_umf': result_umf,
        # Scale that was applied to the UMF of the recipe to make it comparable
        # with the target; 1.0 means the two were already on the same basis
        'unity_scale': float(unity_scale),
        # Reported error: reproducible by the caller from target_umf/result_umf
        'error': float(calculate_umf_error(problem['target_umf'], result_umf)),
        # Search objective: the reported error plus the damped contamination
        'objective_error': float(_objective_error(problem, result_umf)),
        'weight_composition': composition,
        'materials_count': len(recipe),
    }


def _shrink_to_limit(state: Dict[str, Any], problem: Dict[str, Any],
                     max_materials: int) -> Dict[str, Any]:
    """
    Bring a state down to the material limit by dropping the lightest material
    of the recipe and solving again.

    Needed when whole priority groups make the starting set larger than the
    caller allows, and when max_materials is smaller than the starting set the
    priority rule produces (a two component recipe is a legitimate request).
    """
    while state['materials_count'] > max_materials:
        lightest = min(sorted(state['recipe']), key=lambda name: state['recipe'][name])
        reduced = [material for material in state['materials'] if material['name'] != lightest]
        if not reduced:
            break

        smaller = _solve_material_set(reduced, problem)
        if smaller is None:
            break
        smaller['iterations'] = state['iterations']
        state = smaller

    return state


def _weight_residual(problem: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, float]:
    """
    Residual of the current recipe in weight percent: target minus actual.

    A positive value is a deficit (the recipe delivers too little of that oxide)
    and a negative one an excess. Both sides are normalized to 100, so the two
    vectors are directly comparable.
    """
    target_weights = problem['target_weights']
    actual_weights = _normalize_to_100(state['weight_composition'])

    residual: Dict[str, float] = {}
    for oxide in sorted(set(target_weights) | set(actual_weights)):
        residual[oxide] = target_weights.get(oxide, 0.0) - actual_weights.get(oxide, 0.0)

    return residual


def _focus_oxide(residual: Dict[str, float]) -> Optional[str]:
    """
    Return the oxide the current recipe is the furthest away from, measured in
    weight percent - the oxide this iteration is about.

    The oxides are walked in sorted order so that an exact tie between two
    oxides always resolves the same way; iterating a set here used to make the
    choice depend on the hash seed.
    """
    worst_oxide = None
    worst_gap = 0.0

    for oxide in sorted(residual):
        gap = abs(residual[oxide])
        if gap > worst_gap:
            worst_gap = gap
            worst_oxide = oxide

    return worst_oxide


def _score_candidate(material: Dict, residual: Dict[str, float], focus: Optional[str]) -> Optional[float]:
    """
    Score one candidate material against the current residual.

    The score is built from four disjoint terms, every oxide of the material
    falling into exactly one of them:

      + focus gain          residual on the focus oxide times the share of that
                            oxide in the material (signed: a material rich in an
                            oxide that is already in excess scores negative)
      + secondary gain      the same product on the other oxides that are still
                            short, damped by SECONDARY_GAIN_WEIGHT
      - contamination       what the material adds to the oxides that are
                            already over the target, scaled up by
                            CONTAMINATION_WEIGHT because an excess cannot be
                            subtracted later
      - disturbance         what the material adds to the oxides that already
                            match; their residual is ~0, so this term needs its
                            own scale (MATCHED_OXIDE_TOLERANCE) to exist at all

    Returns None for a material with an empty formula.
    """
    formula = material.get('formula', {})
    total = sum(formula.values())
    if total <= 0:
        return None

    # Composition as fractions of the material weight
    fractions = {oxide: value / total for oxide, value in formula.items()}

    focus_gain = 0.0
    secondary_gain = 0.0
    contamination = 0.0
    disturbance = 0.0

    for oxide in sorted(fractions):
        fraction = fractions[oxide]
        gap = residual.get(oxide, 0.0)

        if oxide == focus:
            focus_gain = gap * fraction
        elif abs(gap) <= MATCHED_OXIDE_TOLERANCE:
            disturbance += MATCHED_OXIDE_TOLERANCE * fraction
        elif gap > 0.0:
            secondary_gain += gap * fraction
        else:
            contamination += -gap * fraction

    return (focus_gain
            + SECONDARY_GAIN_WEIGHT * secondary_gain
            - CONTAMINATION_WEIGHT * contamination
            - MATCHED_DISTURBANCE_WEIGHT * disturbance)


def _rank_candidates(candidates: Sequence[Dict], residual: Dict[str, float],
                     focus: Optional[str]) -> List[Tuple[float, Dict]]:
    """
    Rank the materials that are not in the set yet, best first.

    Two things decide the order and both of them really move it:

    * the chemical score of _score_candidate, divided by the size of the focus
      gap so that 1.0 means "closes the whole gap this step is about";
    * the material priority, folded in as a penalty of PRIORITY_WEIGHT times the
      normalized position of its priority group. This is not a tie break: a top
      priority material overtakes a chemically better one whenever the other one
      is ahead by less than PRIORITY_WEIGHT of the focus gap.

    Returns (chemical score, material) pairs in the blended order.
    """
    scored: List[Tuple[float, Dict]] = []

    for material in candidates:
        score = _score_candidate(material, residual, focus)
        if score is None:
            continue
        scored.append((score, material))

    if not scored:
        return []

    # A material made purely of the focus oxide would score exactly the focus
    # gap, which makes that gap the natural unit of this step. The fallbacks
    # only matter for an already converged recipe, where the order is moot
    reference = abs(residual.get(focus, 0.0)) if focus is not None else 0.0
    if reference <= 0.0:
        reference = max((abs(score) for score, _ in scored), default=0.0)
    if reference <= 0.0:
        reference = 1.0

    # Priority groups present in this step, mapped onto [0, 1]
    priorities = sorted({material.get('priority', DEFAULT_PRIORITY) for _, material in scored})
    priority_rank = {value: (index / (len(priorities) - 1) if len(priorities) > 1 else 0.0)
                     for index, value in enumerate(priorities)}

    def blended(item: Tuple[float, Dict]) -> Tuple[float, float, float, str]:
        score, material = item
        normalized = score / reference
        penalty = PRIORITY_WEIGHT * priority_rank[material.get('priority', DEFAULT_PRIORITY)]
        # Sorted ascending, hence the negated values; the name closes the last
        # possible tie so that the order never depends on the input order
        return (-(normalized - penalty),
                material.get('priority', DEFAULT_PRIORITY),
                -score,
                material.get('name', ''))

    scored.sort(key=blended)

    return scored


def _expand_state(state: Dict[str, Any], available_materials: Sequence[Dict],
                  problem: Dict[str, Any], seen_sets: set, candidate_limit: Optional[int],
                  verbose: bool) -> List[Dict[str, Any]]:
    """
    Try to add one material to the set of a state.

    The candidates are ranked by the residual heuristic first and only the best
    candidate_limit of them are really solved (None means "all of them", the
    exhaustive mode). Solving is what decides in the end: the heuristic cannot
    tell that wollastonite can be replaced by chalk plus quartz without any
    visible change in the weight space residual.

    Material sets that some other branch already solved are skipped without
    spending one of the candidate_limit slots on them, so a step always costs at
    most candidate_limit NNLS runs of new work.

    Returns the resulting states ordered from best to worst.
    """
    used_names = {material['name'] for material in state['materials']}
    candidates = [material for material in available_materials if material['name'] not in used_names]
    if not candidates:
        return []

    residual = _weight_residual(problem, state)
    focus = _focus_oxide(residual)
    ranked = _rank_candidates(candidates, residual, focus)

    if verbose:
        logger.info(f"worst oxide {focus}, best candidates by heuristic: "
                    f"{[material['name'] for _, material in ranked[:TOP_CANDIDATES]]}")

    trials: List[Tuple[float, int, Dict[str, Any]]] = []
    solved = 0

    for rank, (_score, candidate) in enumerate(ranked):
        if candidate_limit is not None and solved >= candidate_limit:
            break

        new_set = list(state['materials']) + [candidate]
        set_names = frozenset(material['name'] for material in new_set)
        if set_names in seen_sets:
            continue
        seen_sets.add(set_names)

        solved += 1
        new_state = _solve_material_set(new_set, problem)
        if new_state is None:
            continue

        new_state['set_names'] = set_names
        new_state['added'] = candidate['name']

        # Rounding the objective keeps equally good candidates together, so that
        # the heuristic order (and through it the priority) decides between them
        trials.append((round(new_state['objective_error'], 4), rank, new_state))

    trials.sort(key=lambda item: (item[0], item[1]))

    return [item[2] for item in trials]


def _recipe_key(recipe: Dict[str, float]) -> Tuple:
    """Composition based identity of a recipe, used to drop duplicates"""
    return tuple(sorted((name, round(weight, 1)) for name, weight in recipe.items()))


def _is_stalled(previous_error: float, next_error: float) -> bool:
    """
    Whether one step improved the objective by less than STALL_IMPROVEMENT of
    what there was to improve - the branch is then abandoned.

    The threshold is relative on purpose: an absolute one would keep polishing a
    recipe that is already at 0.001 and would give up on one that is at 10.
    """
    improvement = previous_error - next_error
    return improvement <= previous_error * STALL_IMPROVEMENT


def _solution_tie_limit(best_error: float) -> float:
    """
    Objective up to which a solution counts as "as good as the best one", so
    that the material count decides between them instead of the fourth decimal.
    """
    return best_error + max(best_error * SOLUTION_ERROR_TIE_REL, SOLUTION_ERROR_TIE_ABS)


def _solution_sort_key(solution: Dict[str, Any], best_error: float):
    """
    Order the solutions the way the caller is promised they are ordered:

      1. everything within the tie band of the best objective comes first, and
         inside that band FEWER MATERIALS WINS (the objective only breaks a tie
         between two recipes of the same size);
      2. everything outside the band follows, ordered by the objective.

    So the first solution is not necessarily the one with the smallest
    objective - by design. Two recipes whose errors agree to the third decimal
    are indistinguishable in the glaze bucket, and the shorter one is the better
    answer; find_best_recipe documents this and the tests pin it.
    """
    tie_limit = _solution_tie_limit(best_error)

    if solution['objective_error'] <= tie_limit:
        return (0, solution['materials_count'], solution['objective_error'])
    return (1, solution['objective_error'], solution['materials_count'])


def find_best_recipe(inventory, target_umf, min_materials=1, max_materials=10,
                     max_solutions=5, verbose=False, error_threshold=0.1,
                     penalize_unlisted=1.0,
                     candidate_search=SEARCH_EXHAUSTIVE) -> List[Dict[str, Any]]:
    """
    Find glaze recipes for a target UMF by adding materials one at a time.

    Args:
        inventory: list of available material names
        target_umf: target UMF formula as {oxide: value}. An oxide listed as an
            explicit zero is a constraint ("none of this"), not an omission:
            unknown oxides, negative and non numeric values are dropped, zeros
            are kept and penalized like any other requested value
        min_materials: minimum number of materials in a returned recipe; when no
            recipe reaches it the result is an empty list, the constraint is
            never silently broken
        max_materials: maximum number of materials in a recipe; the starting set
            built from whole priority groups is shrunk down to this limit, so
            values below DEFAULT_MIN_START_MATERIALS are honoured too
        max_solutions: how many solutions to return; 0 or less returns []
        verbose: log the search process
        error_threshold: objective error below which a recipe counts as
            acceptable. A branch that reached it is dropped only once the pool
            already holds max_solutions different acceptable recipes; until then
            the branch keeps being refined, because the extra recipes have to
            come from somewhere.
        penalize_unlisted: how hard an oxide that the target does not mention is
            pushed towards zero. 1.0 / True means "not listed = must be zero",
            0.0 / False means "not listed = do not care", anything in between is
            a soft weight applied both to the NNLS rows of those oxides and to
            their share of the search objective. Targets derived from a real
            recipe list every oxide it brings, so 1.0 is right for them; a
            target typed by hand in the UI lists only what the user cares about,
            and a hard 1.0 there makes the solver sacrifice the requested oxides
            to zero out the unmentioned ones.
        candidate_search: 'exhaustive' (default) solves every candidate material
            of every step. It costs O(len(inventory)) NNLS runs per step, it is
            the more accurate of the two, and it reduces the candidate ranking
            to a tie break between sets whose error agrees to four decimals -
            the focus oxide and the priority do not steer it.
            'heuristic' solves only the TOP_CANDIDATES best ranked candidates,
            which is what a human does: there the ranking, the focus oxide and
            the priority really pick what gets tried, and a step costs a
            constant number of NNLS runs.
            Measured by calling find_best_recipe once per reference recipe (11
            of them, 19 material inventory, otherwise default arguments) and
            counting the calls to scipy nnls: exhaustive 885 runs, recovering
            the original material set exactly on 10 of the 11 recipes;
            heuristic 271 runs and 5 of 11. The heuristic reaches the same 10 of
            11 at TOP_CANDIDATES = 9, and there it costs 855 runs - about half
            the inventory is tried per step, so it is no longer a shortcut and
            no longer cheaper. The 11th recipe is out of reach for both: it
            needs MnO2 and no material of the inventory carries any.

    Raises:
        ValueError: candidate_search is not one of CANDIDATE_SEARCH_MODES,
            penalize_unlisted is not a number or a boolean, or one of
            min_materials / max_materials / max_solutions is not an integer.

    Returns:
        list of solutions, best first, where "best" is the order documented in
        _solution_sort_key: the recipes whose objective is within the tie band
        of the best one come first and among those the SHORTEST one leads, the
        rest follow by increasing objective. The first solution therefore has an
        objective inside the tie band, not necessarily the smallest one in the
        list. Every solution holds:
            recipe          {material: weight percent}, adds up to exactly 100
            error           calculate_umf_error(target_umf, result_umf); it can
                            be recomputed from the two dictionaries below
            objective_error what the search minimized: error plus the damped
                            contamination of the unlisted oxides
            result_umf      UMF of the recipe, brought onto the normalization
                            basis of the target (see unity_scale)
            unity_scale     the scale that was applied to get there; 1.0 in the
                            normal case, where the target is a unity formula and
                            nothing needs scaling. The untouched UMF of the
                            recipe is result_umf divided by this
            target_umf      the requested target, cleaned of unusable entries
            effective_target_umf  target_umf plus a zero for every oxide the
                            inventory can bring, the oxides penalize_unlisted
                            talks about
            unlisted_weight the penalize_unlisted value actually applied
            materials_count number of materials in the recipe
            iterations      how many steps the recipe took
    """
    solution_limit = _int_argument(max_solutions, 'max_solutions')
    material_limit = _int_argument(max_materials, 'max_materials')
    material_floor = _int_argument(min_materials, 'min_materials')

    if solution_limit <= 0:
        return []

    if candidate_search not in CANDIDATE_SEARCH_MODES:
        raise ValueError(f"unknown candidate_search '{candidate_search}', "
                         f"expected one of: {', '.join(CANDIDATE_SEARCH_MODES)}")

    unlisted_weight = _normalize_unlisted_weight(penalize_unlisted)

    # A target of unknown oxides or of nothing but zeros cannot be converted
    # into weights at all (umf_to_weights would divide by a zero total weight);
    # catching it here keeps ZeroDivisionError out of the callers
    clean_target = _usable_target(target_umf)
    if not any(value > 0.0 for value in clean_target.values()):
        if verbose:
            logger.info("the target holds no usable oxide")
        return []

    if material_limit < 1:
        logger.warning(f"max_materials={max_materials} leaves no room for a recipe")
        return []

    if material_floor > material_limit:
        logger.warning(f"min_materials={material_floor} is above max_materials={material_limit}, "
                       f"no recipe can satisfy both")
        return []

    all_materials = load_materials(only_inventory=False, priority=True)
    # A material with an empty formula can never move the UMF. _rank_candidates
    # already skips it, but _priority_start_set does not, so it is dropped here
    available_materials = filter_materials_with_formula(
        filter_materials_by_inventory(all_materials, resolve_inventory(inventory)))

    if not available_materials:
        if verbose:
            logger.info("no materials available in the inventory")
        return []

    problem = _build_problem(clean_target, available_materials, unlisted_weight)

    candidate_limit = None if candidate_search == SEARCH_EXHAUSTIVE else TOP_CANDIDATES
    beam_width = 1 if solution_limit <= 1 else min(MAX_BEAM_WIDTH, solution_limit)
    # How many children of one state are allowed to stay in the beam
    beam_children = 1 if solution_limit <= 1 else TOP_CANDIDATES

    min_start_materials = max(material_floor, DEFAULT_MIN_START_MATERIALS)
    start_set = _priority_start_set(available_materials, min_start_materials)
    if not start_set:
        return []

    if verbose:
        logger.info(f"starting set ({len(start_set)} materials): {[m['name'] for m in start_set]}")

    start_state = _solve_material_set(start_set, problem)
    if start_state is None:
        if verbose:
            logger.info("the starting set produced no recipe")
        return []

    start_state['iterations'] = 1
    start_state = _shrink_to_limit(start_state, problem, material_limit)
    start_state['set_names'] = frozenset(m['name'] for m in start_state['materials'])

    pool: List[Dict[str, Any]] = [start_state]
    beam: List[Dict[str, Any]] = [start_state]
    seen_sets = {start_state['set_names']}
    # Different recipes that already meet the requested quality
    found_recipes = set()
    if start_state['objective_error'] <= error_threshold:
        found_recipes.add(_recipe_key(start_state['recipe']))

    for iteration in range(2, DEFAULT_MAX_ITERATIONS + 1):
        next_beam: List[Dict[str, Any]] = []

        for state in beam:
            # A branch that is good enough is dropped, but only once the pool
            # holds as many different recipes as the caller asked for
            if state['objective_error'] <= error_threshold and len(found_recipes) >= solution_limit:
                if verbose:
                    logger.info(f"branch converged with error {state['objective_error']:.4f}")
                continue

            if state['materials_count'] >= material_limit:
                continue

            children = _expand_state(state, available_materials, problem, seen_sets,
                                     candidate_limit, verbose)

            for child in children:
                child['iterations'] = iteration
                pool.append(child)
                if child['objective_error'] <= error_threshold:
                    found_recipes.add(_recipe_key(child['recipe']))

            # Only the best branches are kept alive; a branch that stops
            # improving is abandoned and the pool keeps whatever it already found
            for child in children[:beam_children]:
                if _is_stalled(state['objective_error'], child['objective_error']):
                    if verbose:
                        logger.info(f"branch stalled on {child['added']}: "
                                    f"{state['objective_error']:.4f} -> {child['objective_error']:.4f}")
                    continue

                if verbose:
                    logger.info(f"iteration {iteration}: added {child['added']}, "
                                f"error {state['objective_error']:.4f} -> {child['objective_error']:.4f}")

                next_beam.append(child)

        if not next_beam:
            break

        next_beam.sort(key=lambda s: (s['objective_error'], s['materials_count']))
        beam = next_beam[:beam_width]

    # Keep only recipes that respect the material limits. An empty result here
    # means the limits are unreachable with this inventory - the caller is told
    # so instead of being handed a recipe that breaks them
    solutions = [s for s in pool if material_floor <= s['materials_count'] <= material_limit]
    if not solutions:
        logger.warning(f"no recipe with {material_floor}..{material_limit} materials was reachable, "
                       f"the pool holds {len(pool)} states")
        return []

    best_error = min(s['objective_error'] for s in solutions)
    solutions.sort(key=lambda s: _solution_sort_key(s, best_error))

    # Drop duplicates by recipe composition
    unique: List[Dict[str, Any]] = []
    seen_recipes = set()

    for solution in solutions:
        if len(unique) >= solution_limit:
            break

        key = _recipe_key(solution['recipe'])
        if key in seen_recipes:
            continue
        seen_recipes.add(key)

        unique.append({
            'recipe': solution['recipe'],
            'error': solution['error'],
            'objective_error': solution['objective_error'],
            'result_umf': solution['result_umf'],
            'unity_scale': solution['unity_scale'],
            'target_umf': dict(problem['target_umf']),
            'effective_target_umf': dict(problem['full_target']),
            'unlisted_weight': unlisted_weight,
            'materials_count': solution['materials_count'],
            'iterations': solution['iterations'],
        })

    if verbose and unique:
        logger.info(f"returning {len(unique)} solutions, best error {unique[0]['error']:.4f}")

    return unique


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description='Iterative Glaze Recipe Solver')
    parser.add_argument('--umf', type=str, help='Target UMF composition as JSON string')
    parser.add_argument('--solutions', type=int, default=5, help='Number of solutions to find (default: 5)')
    parser.add_argument('--max-materials', type=int, default=10, help='Maximum number of materials (default: 10)')
    parser.add_argument('--error-threshold', type=float, default=0.1, help='Error at which the search stops (default: 0.1)')
    parser.add_argument('--penalize-unlisted', type=float, default=1.0,
                        help='How hard an oxide missing from the target is pushed to zero, 0.0..1.0 (default: 1.0)')
    parser.add_argument('--candidate-search', choices=CANDIDATE_SEARCH_MODES, default=SEARCH_EXHAUSTIVE,
                        help=f'Candidate search mode (default: {SEARCH_EXHAUSTIVE})')
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
        penalize_unlisted=args.penalize_unlisted,
        candidate_search=args.candidate_search,
    )

    if not solutions:
        print("\nNo solutions found for the target UMF!")
        return

    print(f"\nFound {len(solutions)} solutions!")
    for index, solution in enumerate(solutions):
        print(f"\nSolution {index + 1}")
        print(f"Error: {solution['error']:.4f} | objective: {solution['objective_error']:.4f} "
              f"| materials: {solution['materials_count']} | iterations: {solution['iterations']}")
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
