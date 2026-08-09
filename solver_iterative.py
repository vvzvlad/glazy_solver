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
   to UMF), drop materials weighing less than MIN_MATERIAL_WEIGHT and re-solve.
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

The search itself only ever ADDS materials, so the last thing that happens to a
recipe is _prune_solution(): greedy backward elimination that drops every
material whose removal does not make the fit worse. That is where noise
components go - not into a weight threshold, which cannot tell a rounding
artefact apart from half a percent of cobalt carbonate.
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
    load_molar_masses,
    resolve_inventory,
    resolve_material_pool,
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

# Numerical floor of the NNLS solution, NOT a way to keep the recipe short.
# A weight this small is the solver's own noise - a column that got a sliver of
# mass while fitting the fourth decimal of an oxide - and 0.1 is where
# solver_classic.solve_recipe already cuts, so the two agree.
#
# THIS NUMBER MUST NOT BE RAISED. It was 1.0 for one commit, on the argument
# that one gram of a 100 g batch is what a studio scale can still weigh, and it
# was reverted because a weight threshold cannot tell a numerical artefact apart
# from an ingredient that is genuinely tiny. Measured on a target built from
# feldspar 39.5 / quartz 30 / chalk 20 / kaolin 10 / cobalt carbonate 0.5, with
# the cobalt in the inventory:
#
#   MIN=0.1  err 0.0030  {kaolin 9.99, quartz 30.01, chalk 20.02,
#                         feldspar 39.48, CoCO3 0.5}
#   MIN=1.0  err 0.0575  {kaolin 10.04, quartz 30.16, chalk 20.12, feldspar 39.68}
#
# 0.0575 is below the default error_threshold of 0.1, so the second answer is
# returned as acceptable and nothing in it says that an ingredient was dropped:
# a blue glaze silently becomes a clear one. Colourants are exactly the class of
# material that weighs the least and matters the most. The cut was fuzzy on top
# of that - a component genuinely needed at 1.00% went too, because the first
# NNLS estimate landed just below the floor and _solve_material_set then removed
# the material from `active` for good.
#
# Keeping a recipe free of pointless components is the job of _prune_solution(),
# which asks whether a material is the only source of something the target asked
# for, and failing that whether removing it makes the fit worse - two questions
# that have nothing to do with how much it weighs.
MIN_MATERIAL_WEIGHT = 0.1

# How much a removal may cost, checked per removal against the state it starts
# from, on BOTH error numbers (see _prune_solution for why both).
#
# What this tolerance is for is recognising FIT NOISE: a material the search
# added to shave a fourth decimal off one oxide, which by construction is not
# the only thing in the recipe carrying that oxide. 0.005 of UMF error is
# invisible in a fired glaze - smaller than the 0.0033 that the published
# percentages of a textbook recipe already cost by being rounded to one decimal.
#
# What it is NOT for, at any calibration, is protecting a colourant, and it is
# worth being explicit about that because the first version of this pass claimed
# otherwise. Removing 0.5% of cobalt carbonate costs 0.0545, ten times the
# tolerance, which looked like proof that the test protects colourants. It is
# not: cobalt is a FLUX, so losing it drags the unity denominator of the whole
# formula and inflates the measured cost. Colourants that are not fluxes get no
# such amplification, and the same measurement on the same base says so:
#
#   material              oxide          1.0%    0.5%    0.3%    0.2%
#   Карбонат кобальта     CoO (flux)   0.1196  0.0545  0.0371  0.0242
#   Оксид никеля          NiO (flux)   0.1844  0.0954  0.0555  0.0313
#   Оксид хрома           Cr2O3        0.0240  0.0110  0.0070  0.0051
#   Хромат железа         Cr2O3/Fe2O3  0.0152  0.0067  0.0034  0.0030
#   Оксид железа красный  Fe2O3        0.0212  0.0099  0.0050  0.0045
#
# Everything at or below the tolerance in that table is a colourant this test
# throws away: 0.4% of iron chromate, 0.25% of red iron oxide, 0.15% of chrome
# oxide - and 0.4% is four grams in a kilogram batch, an ordinary weighable
# addition. Raising the number to catch them would only move the line to some
# other colourant, because the quantity being measured is the wrong one. A
# colourant works OPTICALLY and contributes almost nothing to the chemistry;
# that is what makes it a colourant. 0.5% of cobalt is the difference between a
# blue glaze and a clear one and 0.15% of chrome oxide between a green one and a
# clear one, and no threshold on UMF error can see that difference at any
# setting.
#
# So colourants are not protected by this number at all. They are protected by
# the sole carrier rule in _prune_solution, which asks a different question
# entirely and has no quantity in it.
PRUNE_ERROR_TOLERANCE = 0.005

# How many candidate recipes the pruning pass may work through, as a multiple of
# max_solutions. The pass must run before the sort and before the max_solutions
# cut (see find_best_recipe), which in principle means pruning every distinct
# recipe the pool holds - neither the objective nor the material count moves
# monotonically under pruning, so there is no admissible way to prove in advance
# that a candidate cannot reach the top max_solutions.
#
# In practice that is unaffordable. The pool holds one state per material set
# TRIED, and while many of those collapse onto the same recipe, what is left
# still scales with the catalogue: over the eleven reference recipes the pool
# holds 2 to 48 distinct recipes on the 19 material inventory and 14 to 267 on
# the whole 216 material one. Measured on recipe 03 over the full catalogue at
# max_solutions=5, pruning every distinct recipe costs 3959 scipy nnls calls
# against the 1369 of the unpruned search, and takes a POST /api/solve from
# 236 ms to 619 ms.
#
# The margin is the compromise: candidates are pruned in the order the UNPRUNED
# sort key gives, and the pass stops after max_solutions * this many of them
# (with one exception, see find_best_recipe). Three leaves room for a recipe to
# shrink past two others and still be returned, which a plain cut at
# max_solutions could not do. Same measurement with the margin: 1453 calls and
# 250 ms, so the pass costs about 6% over not pruning at all instead of 3x.
PRUNE_CANDIDATE_MARGIN = 3

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


def _sole_carriers(used: Sequence[Dict], target_umf: Dict[str, float]) -> set:
    """
    Names of the materials that are the ONLY source of a requested oxide.

    "Requested" means the target asks for a positive amount of it. An oxide the
    target lists as an explicit zero is deliberately excluded: there "no other
    material carries it" is a reason to remove the carrier, not to keep it. So
    is an oxide the target never mentions, which is contamination by definition.

    Args:
        used: the material records the recipe actually uses
        target_umf: the requested target, as cleaned by _usable_target

    Returns:
        set of material names that must not be pruned away
    """
    carriers: Dict[str, List[str]] = {}

    for material in used:
        name = material.get('name', '')
        for oxide, amount in material.get('formula', {}).items():
            if amount > 0.0 and target_umf.get(oxide, 0.0) > 0.0:
                carriers.setdefault(oxide, []).append(name)

    return {names[0] for names in carriers.values() if len(names) == 1}


def _prune_solution(state: Dict[str, Any], problem: Dict[str, Any],
                    min_materials: int) -> Dict[str, Any]:
    """
    Drop from a solved state every material the recipe turns out not to need.

    Greedy backward elimination: each round re-solves the recipe once per
    removable material with that material taken out, keeps the removals that
    pass the test below, applies the one that ends up with the lowest objective
    and starts again. It stops when no single removal qualifies any more.

    A removal has to clear TWO gates, and they answer different questions.

    1. THE SOLE CARRIER RULE, which is structural and has no quantity in it. A
       material that is the only thing in the recipe carrying an oxide the
       target asked for is never removed. Not because removing it would score
       badly - because the result would be a recipe that does not answer the
       request. The target said CoO, one material carries CoO, so that material
       stays.

       This is what protects colourants, opacifiers and every other ingredient
       that is present in a small amount for a reason, and it is the only thing
       that can. A colourant works optically and contributes almost nothing to
       the chemistry - that is what makes it a colourant - so no threshold on
       chemical error can tell one from fit noise at any calibration. The table
       in PRUNE_ERROR_TOLERANCE is the demonstration: cobalt survived the
       error test only because it happens to be a flux and its removal drags the
       unity denominator, while chrome oxide at 0.15% and iron chromate at 0.4%
       are just as much colourants, are not fluxes, and were being thrown away.

    2. THE ERROR TOLERANCE, for everything else. Fit noise - a material the
       search added to shave a fourth decimal off one oxide - is by construction
       not the sole carrier of anything requested, so gate 1 lets it through and
       this one measures it: the removal is accepted when it grows the error by
       at most PRUNE_ERROR_TOLERANCE.

       BOTH error numbers are checked, not just the objective the search
       minimizes. With penalize_unlisted > 0 - the default, and what the API
       sends - the objective folds in the contamination of the unlisted oxides,
       so removing a material that brought unrequested oxides shrinks that term
       and can pay for a rise in `error`, which is the number the caller
       actually receives. Checking the objective alone therefore has no bound on
       `error` at all. tests/test_solver_inverse.py TestPruningChecksBothErrors
       builds the trade explicitly and pins that the removal is refused.

    Only the materials the recipe actually USES are candidates and only they are
    re-solved: the state carries the whole material set it was built from, and
    putting a material that NNLS already zeroed back into the matrix would turn
    a removal into a swap.

    The floor is max(min_materials, 1): the caller's minimum is never broken
    from this side, and the last material is never taken away even when the
    caller asked for a minimum of zero.

    Args:
        state: a state as returned by _solve_material_set
        problem: the problem context the state was solved against
        min_materials: the caller's min_materials

    Returns:
        the pruned state, or the very same object when nothing could go
    """
    current = state
    floor = max(int(min_materials), 1)
    target_umf = problem['target_umf']

    while current['materials_count'] > floor:
        used = [material for material in current['materials']
                if material['name'] in current['recipe']]
        protected = _sole_carriers(used, target_umf)

        objective_limit = current['objective_error'] + PRUNE_ERROR_TOLERANCE
        error_limit = current['error'] + PRUNE_ERROR_TOLERANCE
        best: Optional[Dict[str, Any]] = None

        # sorted() so that two equally good removals always resolve the same
        # way, whatever order the material set happens to be in
        for dropped in sorted(used, key=lambda material: material.get('name', '')):
            if dropped['name'] in protected:
                continue

            reduced = [material for material in used if material['name'] != dropped['name']]
            if not reduced:
                continue

            candidate = _solve_material_set(reduced, problem)
            if candidate is None or candidate['materials_count'] < floor:
                continue
            if candidate['objective_error'] > objective_limit or candidate['error'] > error_limit:
                continue

            if best is None:
                best = candidate
            elif candidate['objective_error'] < best['objective_error']:
                best = candidate

        if best is None:
            break

        # _solve_material_set knows nothing about the search, so the bookkeeping
        # of the state being replaced is carried over by hand
        best['iterations'] = current.get('iterations', 1)
        best['set_names'] = frozenset(material['name'] for material in best['materials'])
        current = best

    return current


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
                     candidate_search=SEARCH_EXHAUSTIVE,
                     materials=None) -> List[Dict[str, Any]]:
    """
    Find glaze recipes for a target UMF by adding materials one at a time.

    Every recipe is put through _prune_solution() before it is ranked and
    returned: the search only ever adds materials, and the pruning pass is what
    takes back the ones that turned out not to be needed. A material is dropped
    only when it is not the sole source of a requested oxide AND its removal
    costs at most PRUNE_ERROR_TOLERANCE on both error numbers, so an ingredient
    the recipe genuinely answers the request with survives however little it
    weighs - see MIN_MATERIAL_WEIGHT for what happens when smallness is used as
    the criterion instead.

    Two consequences of the pass are worth knowing before reading the numbers
    below, because neither is visible in the returned shape:

    * error_threshold is NOT rechecked after pruning. A branch stops when its
      objective reaches the threshold, and a later removal can push the returned
      error past it by up to one tolerance per removal. Measured over the 300
      recipe Glazy corpus, one top-1 answer of 300 crosses the default 0.1, at
      0.09850 -> 0.10901. The number in "error" is always the true error of the
      recipe returned; what no longer holds exactly is the sentence "the search
      stopped because this recipe was under the threshold".
    * pruning can produce a one material recipe, and on an UNREACHABLE target
      the relative tie band of _solution_sort_key can rank it first, because it
      is the shortest of a set of equally hopeless answers. Measured over the
      same 300 targets: solved against their own materials, where the answer is
      reachable by construction, the share of single material top-1 answers is
      0.3% with or without the pass. Solved against the 19 material inventory,
      where most of them are unreachable, it is 0.3% unpruned and 1.7% pruned -
      it was 4.7% before the sole carrier rule, which blocks most of these
      collapses because a hopeless target usually has exactly one carrier left
      for something it asked for. A caller showing a headline recipe may still
      want to read materials_count together with the error.

    Args:
        inventory: list of available material names
        target_umf: target UMF formula as {oxide: value}. An oxide listed as an
            explicit zero is a constraint ("none of this"), not an omission:
            unknown oxides, negative and non numeric values are dropped, zeros
            are kept and penalized like any other requested value
        min_materials: minimum number of materials in a returned recipe; when no
            recipe reaches it the result is an empty list, the constraint is
            never silently broken. The pruning pass respects it too and never
            takes the last material away even when it is 0
        max_materials: maximum number of materials in a recipe; the starting set
            built from whole priority groups is shrunk down to this limit, so
            values below DEFAULT_MIN_START_MATERIALS are honoured too
        max_solutions: upper bound on how many solutions to return; 0 or less
            returns []. Fewer can come back than asked for, and the pruning pass
            made that more common: several recipes of the search can prune onto
            the same answer, and they are merged rather than back-filled. The
            merged_variants field of each solution says how many.
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
            of them, the 19 material inventory of database/materials.json,
            max_solutions=5 and every other argument at its default) and
            counting the calls to scipy.optimize.nnls, the pruning pass
            included: exhaustive 1680 runs, recovering the original material set
            exactly on 9 of the 11 recipes; heuristic 624 runs and 7 of 11. The
            heuristic gets to 8 of 11 at TOP_CANDIDATES = 9, and there it costs
            1448 runs - about half the inventory is tried per step, so it is no
            longer a shortcut and no longer cheaper, and it still does not catch
            up. The 11th recipe is out of reach for every mode: it needs MnO2
            and no material of the inventory carries any.
            The configuration matters to these numbers and used to be left out
            of this paragraph, which is a good way to mislead yourself: the run
            count scales with max_solutions through the beam width AND through
            the pruning budget, and the pass costs 787 -> 1680 runs on this
            sweep. On the full 216 material catalogue the worst single call of
            the eleven is 1851 runs against 1706 unpruned.
        materials: optional material records to use as the database, same shape
            as database/materials.json entries. Meant for the tests and for
            callers carrying their own catalogue; when it is given together with
            inventory=None it bypasses the inventory resolution and every
            injected material is available. See common.resolve_material_pool().
            "priority" is optional in those records and defaults to
            DEFAULT_PRIORITY, which puts every injected material in one group -
            so _priority_start_set() starts from the whole catalogue at once
            unless the records carry explicit priorities.

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
        list. That order is established AFTER the pruning pass, so it holds on
        the recipes actually returned rather than on the ones the search built.
        Every solution holds:
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
            materials_count number of materials in the recipe, after pruning
            merged_variants how many OTHER distinct recipes of the search pruned
                            onto this same one and are therefore not listed
                            separately; 0 when nothing collapsed onto it. Counted
                            over the candidates the pruning pass actually looked
                            at, which is PRUNE_CANDIDATE_MARGIN * max_solutions
                            of them at most, so it is a floor rather than an
                            exhaustive census of the pool
            iterations      how many search steps the recipe took; the pruning
                            pass is not a step and does not raise it
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

    all_materials, available_names = resolve_material_pool(materials, inventory)
    # A material with an empty formula can never move the UMF. _rank_candidates
    # already skips it, but _priority_start_set does not, so it is dropped here
    available_materials = filter_materials_with_formula(
        filter_materials_by_inventory(all_materials, available_names))

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

    # Backward elimination, and it happens HERE - after the search, before the
    # sort and before the max_solutions cut. Three decisions are packed into
    # that placement:
    #
    # * NOT inside the beam loop. Pruning one state costs O(materials) NNLS runs
    #   per round, the loop solves tens to hundreds of states per call, and the
    #   beam is going to add materials on top of whatever was pruned anyway.
    # * BEFORE the sort, because pruning changes materials_count, and
    #   _solution_sort_key ranks by materials_count inside the tie band. Sorting
    #   first would order the recipes by a size they no longer have.
    # * BEFORE the max_solutions cut, for the same reason: a recipe that prunes
    #   from seven materials down to four has to be able to overtake one that
    #   started at five and stayed there, and one that prunes onto a recipe
    #   already in the list has to lose its slot to the next distinct answer.
    #
    # HOW MANY candidates are pruned is a cost decision, and the honest version
    # of it is that pruning every distinct recipe of the pool is what the
    # placement above really wants and what nobody can afford. Neither the
    # objective nor the material count moves monotonically under pruning (a
    # removal may grow the objective, and on the reference set it sometimes
    # shrinks it - recipe 06 went from 5.8311 on three materials to 3.2224 on
    # one), so no candidate can be proved irrelevant in advance. But the pool
    # holds 2 to 48 distinct recipes on the 19 material inventory and 14 to 267
    # on the whole 216 material catalogue, and pruning all of them took one
    # /api/solve request from 236 ms to 619 ms. So the candidates are taken in
    # the order the UNPRUNED sort key gives and the pass stops after
    # PRUNE_CANDIDATE_MARGIN * max_solutions of them - a bounded overshoot
    # rather than an exact cut at max_solutions, so that a recipe still has room
    # to shrink past two others and be returned.
    #
    # The budget yields to one thing: having max_solutions DISTINCT answers to
    # return. Several candidates can prune onto the same recipe - that is the
    # point of the pass - and stopping at the budget while the list is still
    # short would silently under-deliver alternatives that do exist further down
    # the pool. Measured over the eleven references on the full catalogue, a
    # hard stop at the budget returns 35 alternatives against the 55 of the
    # unpruned search; letting it run on until the list is full returns 44, and
    # the extra work is only ever done when the collapse actually happened.
    pre_prune_best = min(s['objective_error'] for s in solutions)
    ordered = sorted(solutions, key=lambda s: _solution_sort_key(s, pre_prune_best))
    prune_budget = max(solution_limit * PRUNE_CANDIDATE_MARGIN, 1)

    seen_before_pruning = set()
    distinct_pruned = set()
    pruned: List[Dict[str, Any]] = []

    for solution in ordered:
        if len(pruned) >= prune_budget and len(distinct_pruned) >= solution_limit:
            break
        key = _recipe_key(solution['recipe'])
        if key in seen_before_pruning:
            continue
        seen_before_pruning.add(key)
        pruned_state = _prune_solution(solution, problem, material_floor)
        pruned.append(pruned_state)
        distinct_pruned.add(_recipe_key(pruned_state['recipe']))

    best_error = min(s['objective_error'] for s in pruned)
    pruned.sort(key=lambda s: _solution_sort_key(s, best_error))

    # Two different recipes can prune onto the same one - that is exactly what
    # happens when both of them carry the same answer plus one redundant
    # material each - so the list is deduplicated again here.
    #
    # The collapse is counted rather than back-filled. Topping the list up with
    # the UNPRUNED states would restore the very recipes the pass just judged to
    # be this same answer plus noise - the junk the pass exists to remove,
    # dressed up as an alternative. Measured over the eleven references on the
    # full catalogue, pruning takes 55 alternatives down to 44, and recipe 08
    # collapses 5 -> 1 because all five were one four component core plus one or
    # two percent of kaolin or alum. Handing those five back would be a worse
    # answer honestly counted, so each returned solution reports merged_variants
    # instead and a caller can show "4 near-identical variants were merged".
    merged_counts: Dict[Tuple, int] = {}
    for solution in pruned:
        key = _recipe_key(solution['recipe'])
        merged_counts[key] = merged_counts.get(key, 0) + 1

    unique: List[Dict[str, Any]] = []
    seen_recipes = set()

    for solution in pruned:
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
            'merged_variants': merged_counts[key] - 1,
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
