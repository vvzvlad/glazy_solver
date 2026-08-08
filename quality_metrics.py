#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# flake8: noqa
# pylint: disable=broad-exception-raised, raise-missing-from, too-many-arguments, redefined-outer-name
# pylint: disable=multiple-statements, logging-fstring-interpolation, trailing-whitespace, line-too-long
# pylint: disable=broad-exception-caught, missing-function-docstring, missing-class-docstring
# pylint: disable=f-string-without-interpolation
# pylance: disable=reportMissingImports, reportMissingModuleSource

"""
Quality metrics of a solver solution measured against an original recipe

The inverse glaze problem has no unique solution, so "the chemistry matches" is
a necessary condition and not a measure of quality - the solvers already pass
it. This module is the other half of the judgement: it compares a produced
recipe with an ORIGINAL recipe that is known to yield the same chemistry.

The original is not "the right answer" (there are many). It is an EXISTENCE
PROOF: it demonstrates that this chemistry is reachable with that many
materials, in those proportions, at that cost. The requirement on the solver is
therefore "no worse than the original on every axis", never "identical to the
original".

Every metric reports the value of the solution, the value of the original and
the delta or ratio between them, so a caller can see WHY a rule was violated and
not merely that it was. Thresholds are named module constants, each with the
reason for its number.

The module is pure: no Flask, no network, and the only file it ever reads is
database/prices.json (through load_prices()).

Return shape of solution_quality():

    {
      # number of components; the solution may use one material more
      "count": {"solution": 5, "original": 5, "delta": 0, "ok": True},
      # components lighter than JUNK_WEIGHT_PERCENT
      "junk": {"solution": 0, "original": 1, "delta": -1, "ok": True},
      # smallest component; "required" tells whether the rule applied at all
      "min_portion": {"solution": 4.2, "original": 3.0,
                      "required": True, "ok": True},
      # roubles per kg of dry batch; see the prices discussion below
      "cost": {"solution": None, "original": None, "ratio": None,
               "coverage": 0.0, "cost_abs": None, "ok": None},
      # weighted priority, lower is more basic; None without a priority mapping
      "priority": {"solution": None, "original": None, "ratio": None,
                   "ok": None},
      # conditioning of the oxides x used-materials matrix - the degeneracy gate
      "conditioning": {"cond": 15.1, "rank": 4, "redundancy": 0,
                       "rank_deficient": False, "original": 15.1, "ok": True},
      # UMF error introduced by rounding every share to ROUNDING_STEP_PERCENT.
      # DIAGNOSTIC ONLY - it is not a degeneracy detector, see below
      "rounding_drift": {"value": 0.004, "original": 0.003, "ok": True},
      # "how easy is this to put together" in roubles x pieces; tracked against
      # a baseline, not a threshold, so it carries no "ok" and never fails.
      # None on either side when that side's cost is unknown - never a bare
      # component count, and callers must filter the Nones out of aggregates
      "assembly_score": {"value": None, "cost_abs": None, "original": None},
      "set_jaccard": 0.67,
      "share_delta": 12.4,
      "clay_content": {"solution": 15.0, "original": 15.0,
                       "threshold": 5.0, "ok": True},
      "loi": {"solution": 11.2, "original": 10.4, "delta": 0.8},
      "loi_delta": 0.8,
      "unknown_materials": [],
      "failures": ["junk"],          # rules that gate, and were violated
      "warnings": ["clay_content"],  # rules that only warn
    }

"ok" is None for a metric that could not be computed (no prices, no priorities,
an empty recipe). Such a metric never appears in "failures": "we could not
measure it" is not the same statement as "it is worse than the original". The
one deliberate exception is a rank-deficient material set, where conditioning
reports cond=None with ok=False: there the missing number means "infinitely
ill-conditioned", which is the worst case rather than an absent measurement.

Which metric detects degeneracy: "conditioning", and only that one. The earlier
design gave that job to "rounding_drift" and was wrong - see the warning on
_rounding_drift() and TZ_SOLVER_V2.md 10.9. The drift is kept as a diagnostic of
whether the recipe can be weighed out on a coarse scale, its "ok" flag is
informational, and it can never put a solution into "failures".

database/prices.json format - roubles per kg of material:

    {"Каолин КЖФ-1": 90}

Filling that file is a separate, human-verified data task and it is only
partially done, so the module owns the format and its consumption and must
behave correctly at any coverage - including an empty file, where the cost ratio
is None and the coverage 0.0.
"""

import json
import os

import numpy as np

from common import (
    DEFAULT_PRIORITY,
    NON_OXIDE_KEYS,
    load_molar_masses,
    weights_to_umf,
)
from solver_classic import (
    calculate_recipe_composition,
    calculate_umf_error,
    create_oxide_matrix,
)


# A component lighter than this is "junk": below roughly two percent a raw
# material stops being a recipe ingredient and becomes a rounding artefact of
# the fit - it is hard to weigh, easy to forget, and rarely changes the fired
# result enough to justify a separate bag on the shelf.
JUNK_WEIGHT_PERCENT = 2.0

# Nothing below one percent should appear in a 100 g batch at all: pottery
# scales read to 0.1-0.5 g, so a sub-percent component is mostly weighing error.
MIN_PORTION_PERCENT = 1.0

# The solution is allowed to use one material more than the original. Zero would
# demand that the solver find exactly the original decomposition, which the
# inverse problem does not guarantee; two extra materials already means a
# different, more complicated recipe.
MAX_COUNT_DELTA = 1

# Twenty percent over the original price is the usual noise band of supplier
# quotes, so a smaller excess is not evidence that the solution is worse.
MAX_COST_RATIO = 1.2

# Same reasoning as the cost ratio, applied to the priority proxy: a fifth of
# the weighted priority is within the arbitrariness of the priority numbers.
MAX_PRIORITY_RATIO = 1.2

# Studio scales weigh a 100 g batch to about half a gram, so a recipe has to
# survive its shares being rounded to this grid.
ROUNDING_STEP_PERCENT = 0.5

# Informational band for the rounding drift - NOT a gate, see the warning on
# _rounding_drift(). It is roughly where a recipe stops being reproducible on a
# coarse scale, but it also flags perfectly healthy recipes whose shares merely
# happen to sit far from the grid, so nothing may fail on it.
MAX_ROUNDING_DRIFT = 0.02

# Condition number above which the set of materials the recipe uses counts as
# degenerate: the solution is then held together by a compensating pair rather
# than by chemistry, and small errors anywhere move the weights a long way. The
# same 1e3 is used by matrix_diagnostics of stage 2, and the two populations sit
# far apart around it - measured on the same three recipes, an honest feldspar /
# silica / whiting / kaolin set gives 15.1 while the same chemistry rebuilt on a
# near-collinear pair of feldspars gives 3668.8 (see TZ_SOLVER_V2.md 10.9).
MAX_CONDITION_NUMBER = 1e3

# A glaze with less clay than this will not stay in suspension no matter how
# perfect its chemistry: it settles into a brick at the bottom of the bucket.
MIN_CLAY_PERCENT = 5.0

# ... but the demand is capped at half of what the original used: if the
# original itself is clay-poor, the chemistry evidently does not allow more, and
# blaming the solution for it would be unfair.
CLAY_ORIGINAL_FRACTION = 0.5

# Name fragments that mark a plastic, suspension-keeping material. Matching is
# done on the lower-cased material name, which works for Cyrillic as well.
# "глина" deliberately does not match "глинозём" (alumina), which is not a clay.
CLAY_KEYWORDS = (
    'каолин',
    'глина',
    'бентонит',
    'kaolin',
    'clay',
    'bentonite',
    'ball clay',
)

# Metrics that gate: a False "ok" is a failure of the "no worse than the
# original" requirement. Order fixes the order of the "failures" list.
# rounding_drift is deliberately absent - it reports an "ok" flag for
# information only and can never fail a solution.
GATED_METRICS = ('count', 'junk', 'min_portion', 'cost', 'priority', 'conditioning')

# Metrics that only warn: technologically suspicious, but not a reason to
# declare the solution worse than the original.
WARNING_METRICS = ('clay_content',)

# Default location of the price list, next to the other reference data.
PRICES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'database', 'prices.json')


def load_prices(path=None):
    """
    Load the material price list

    Args:
        path: path to the JSON file; None means database/prices.json next to
              this module

    Returns:
        dictionary {material name: price per kg}; an empty dictionary when the
        file is missing, empty or holds a bare null - the price list is optional
        data and its absence must never break a metric run
    """
    if path is None:
        path = PRICES_PATH

    if not os.path.isfile(path):
        return {}

    with open(path, 'r', encoding='utf-8') as f:
        text = f.read().strip()

    if not text:
        return {}

    return json.loads(text) or {}


def _material_names(materials):
    """Set of the names of the given material records"""
    return {material.get('name') for material in materials}


def _is_clay(material_name):
    """Whether the material name marks a plastic, suspension-keeping material"""
    lowered = material_name.lower()
    return any(keyword in lowered for keyword in CLAY_KEYWORDS)


def _clay_content(recipe):
    """Total share of clay-like materials in the recipe"""
    return sum(weight for name, weight in recipe.items() if _is_clay(name))


def _weighted_sum(recipe, values, default=None):
    """
    Sum of value * share / 100 over the recipe

    Args:
        recipe: {material: weight percent}
        values: {material: number}
        default: value for a material missing from the mapping; None means the
                 sum is undefined without full coverage

    Returns:
        the weighted sum, or None when a material has no value and no default
    """
    total = 0.0

    for name, weight in recipe.items():
        value = values.get(name, default)
        if value is None:
            return None
        total += value * weight / 100.0

    return total


def _umf_or_none(weight_composition):
    """
    UMF of a weight composition, or None when there is nothing to normalize

    weights_to_umf() falls back to the smallest positive molar amount when a
    composition carries no fluxes, so a composition with no convertible oxide at
    all would raise instead of returning an empty formula. "No oxides" is a
    legitimate state here (a recipe made entirely of materials we do not know),
    and it has no UMF.
    """
    molar_masses = load_molar_masses()
    convertible = [
        weight for oxide, weight in weight_composition.items()
        if weight > 0 and oxide not in NON_OXIDE_KEYS and oxide in molar_masses
    ]

    if not convertible:
        return None

    return weights_to_umf(weight_composition)


def _round_to_step(recipe):
    """
    Round every share to ROUNDING_STEP_PERCENT and renormalize the result to 100

    A component small enough to round down to zero is dropped by the caller's
    chemistry calculation, which is exactly the failure the drift metric is
    meant to expose - it is not special-cased away. Only the renormalization is
    guarded, so that a recipe that rounds away entirely cannot divide by zero.
    """
    rounded = {
        name: round(weight / ROUNDING_STEP_PERCENT) * ROUNDING_STEP_PERCENT
        for name, weight in recipe.items()
    }

    total = sum(rounded.values())
    if total <= 0:
        return rounded

    return {name: weight * 100.0 / total for name, weight in rounded.items()}


def _rounding_drift(materials, recipe):
    """
    UMF error a recipe picks up when its shares are rounded to the scale grid

    What this measures is whether the recipe can be weighed out on a coarse
    scale and still fire the same. That is all it measures.

    THIS IS NOT A DEGENERACY DETECTOR. Do not re-derive the opposite conclusion:
    with non-negative weights and non-negative analyses the map from weights to
    chemistry is Lipschitz with a modest constant, so a bounded weight
    perturbation moves the chemistry by a bounded amount however ill-conditioned
    the inverse problem is - collinearity makes the WEIGHTS ill-determined while
    leaving the CHEMISTRY robust. Measured exhaustively over a chemically null
    family (83 845 splits of one feldspar mass across near-collinear twins) the
    drift of a degenerate solution never exceeded 0.0153, while an honest recipe
    whose shares merely sit far from the grid reaches 0.0271 - the metric is
    anti-correlated with degeneracy and dominated by the solver's print
    precision. See conditioning() for the real measure and TZ_SOLVER_V2.md 10.9
    for the measurement.

    Args:
        materials: material records
        recipe: {material: weight percent}

    Returns:
        the UMF error between the recipe as given and the same recipe rounded to
        ROUNDING_STEP_PERCENT, or None when the recipe has no UMF to compare
        (empty recipe, or nothing but unknown materials)

    Note that the renormalization to 100 inside _round_to_step() cannot change
    the number by itself: UMF is scale invariant. It is kept because the metric
    is defined on a renormalized batch and the invariance is a property of the
    current conversion, not a promise of it.
    """
    umf_before = _umf_or_none(calculate_recipe_composition(materials, recipe))
    if umf_before is None:
        return None

    rounded = _round_to_step(recipe)
    umf_after = _umf_or_none(calculate_recipe_composition(materials, rounded))
    if umf_after is None:
        # Everything rounded away: report the full distance from the target
        # rather than pretending the recipe survived.
        umf_after = {}

    return float(calculate_umf_error(umf_before, umf_after))


def _condition_number(materials, recipe):
    """
    Conditioning of the "oxides x used materials" matrix behind a recipe

    This is the anti-degeneracy measure. A recipe standing on a compensating
    pair of near-identical materials has an almost singular material matrix: the
    weights that produce its chemistry are barely determined, so any error
    anywhere sends them a long way, and the recipe is a numerical accident
    rather than a piece of chemistry.

    The matrix is built from the materials the recipe actually uses, over the
    union of the oxides those materials carry. Both are sorted, so the number
    does not depend on dictionary order. A material named by the recipe but
    missing from the records contributes no column - it is already reported in
    "unknown_materials" - but it still counts towards the redundancy below,
    because a material we cannot analyse is a material we cannot justify.

    Note the metric judges the SET of materials, not the amounts: two recipes
    built from the same materials in different proportions have the same
    condition number. That is exactly the invariance the rounding drift lacks.

    Returns:
        (cond, rank, rank_deficient) where cond is None for a linearly dependent
        or empty material set. Such a set is infinitely ill-conditioned, and None
        says so honestly; a huge float would invite comparisons ("1e17 is worse
        than 1e16") between numbers that are pure floating point noise.
    """
    used = sorted(
        (material for material in materials if material.get('name') in recipe),
        key=lambda material: material.get('name'),
    )
    oxides = sorted({
        oxide
        for material in used
        for oxide, content in material.get('formula', {}).items()
        if content and oxide not in NON_OXIDE_KEYS
    })

    if not used or not oxides:
        return None, 0, True

    matrix, _ = create_oxide_matrix(used, oxides)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    rank = int(np.linalg.matrix_rank(matrix))

    # A wide matrix (more materials than oxides) keeps only min(rows, columns)
    # singular values, so a dependent column set can still show a finite ratio.
    # Comparing the rank with the number of columns catches that case too.
    if rank < matrix.shape[1] or singular_values[-1] <= 0:
        return None, rank, True

    return float(singular_values[0] / singular_values[-1]), rank, False


def _conditioning_metric(recipe, original, materials):
    """
    Condition number, rank and redundancy of the recipe's material set

    The original's condition number comes along so that a caller can see whether
    the solver did worse than the existence proof it was measured against.
    """
    cond, rank, rank_deficient = _condition_number(materials, recipe)
    original_cond, _, _ = _condition_number(materials, original)

    return {
        'cond': cond,
        'rank': rank,
        'redundancy': len(recipe) - rank,
        'rank_deficient': rank_deficient,
        'original': original_cond,
        # A dependent set is not "unmeasurable", it is the worst possible case,
        # so it fails rather than abstaining.
        'ok': False if cond is None else cond <= MAX_CONDITION_NUMBER,
    }


def _loi(materials, recipe):
    """
    Loss on ignition of the batch, in percent of the raw weight

    100 g of raw materials leave behind the sum of their oxide contributions;
    the rest went up the chimney as CO2 and water. Loss-on-ignition keys of the
    analyses are bookkeeping, not oxides, and are left out of the sum.

    Two materials of the project database have analyses summing to more than 100
    (Cryolite 122.90, Zircon 135.22), so this number can legitimately come out
    negative. It is a diagnostic and is never clamped.
    """
    composition = calculate_recipe_composition(materials, recipe)
    oxide_total = sum(
        weight for oxide, weight in composition.items() if oxide not in NON_OXIDE_KEYS
    )
    return 100.0 - oxide_total


def _ratio(solution_value, original_value):
    """Ratio of two metric values, or None when it is not defined"""
    if solution_value is None or original_value is None:
        return None
    if original_value == 0:
        return None
    return solution_value / original_value


def _count_metric(recipe, original):
    """Number of components: the solution may use MAX_COUNT_DELTA more"""
    delta = len(recipe) - len(original)
    return {
        'solution': len(recipe),
        'original': len(original),
        'delta': delta,
        'ok': delta <= MAX_COUNT_DELTA,
    }


def _junk_metric(recipe, original):
    """Components too light to be worth a separate bag"""
    solution_junk = sum(1 for weight in recipe.values() if weight < JUNK_WEIGHT_PERCENT)
    original_junk = sum(1 for weight in original.values() if weight < JUNK_WEIGHT_PERCENT)
    return {
        'solution': solution_junk,
        'original': original_junk,
        'delta': solution_junk - original_junk,
        'ok': solution_junk <= original_junk,
    }


def _min_portion_metric(recipe, original):
    """
    Smallest component of each side

    The MIN_PORTION_PERCENT rule is only enforced when the original itself keeps
    every component at or above the limit: an original built on a sub-percent
    colorant is proof that this chemistry cannot be had without one.
    """
    solution_min = min(recipe.values()) if recipe else None
    original_min = min(original.values()) if original else None

    if solution_min is None:
        return {'solution': None, 'original': original_min, 'required': None, 'ok': None}

    required = original_min is not None and original_min >= MIN_PORTION_PERCENT
    return {
        'solution': solution_min,
        'original': original_min,
        'required': required,
        'ok': solution_min >= MIN_PORTION_PERCENT if required else True,
    }


def _cost_metric(recipe, original, prices):
    """
    Cost per kg of dry batch of both sides, their ratio and the price coverage

    A ratio is only meaningful when the prices cover every material of both
    sides, because prices are region dependent: US glaze practice is frit based
    and Russian practice is raw-material based, so a ratio against a foreign
    original compares supply chains rather than recipes. The absolute cost of
    our own solution stays meaningful either way and is always reported as
    cost_abs whenever the solution's own materials are fully priced.
    """
    both_sides = set(recipe) | set(original)
    if both_sides:
        coverage = sum(1 for name in both_sides if name in prices) / len(both_sides)
    else:
        coverage = 0.0

    solution_cost = _weighted_sum(recipe, prices) if recipe else None
    original_cost = _weighted_sum(original, prices) if original else None
    ratio = _ratio(solution_cost, original_cost)

    return {
        'solution': solution_cost,
        'original': original_cost,
        'ratio': ratio,
        'coverage': coverage,
        'cost_abs': solution_cost,
        'ok': None if ratio is None else ratio <= MAX_COST_RATIO,
    }


def _priority_metric(recipe, original, priorities):
    """
    Weighted priority of both sides and their ratio, lower being more basic

    Priorities are our own supply reality, so they are never loaded implicitly:
    a corpus of foreign recipes has none, and silently scoring it against
    priorities.json would invent a verdict. Without a mapping the metric is
    simply not computed.
    """
    if priorities is None:
        return {'solution': None, 'original': None, 'ratio': None, 'ok': None}

    solution_priority = _weighted_sum(recipe, priorities, default=DEFAULT_PRIORITY) if recipe else None
    original_priority = _weighted_sum(original, priorities, default=DEFAULT_PRIORITY) if original else None
    ratio = _ratio(solution_priority, original_priority)

    return {
        'solution': solution_priority,
        'original': original_priority,
        'ratio': ratio,
        'ok': None if ratio is None else ratio <= MAX_PRIORITY_RATIO,
    }


def _assembly_score(recipe, cost_abs, original, original_cost):
    """
    "How easy is this to put together": cost per kg times the number of bags

    Without a known cost the score is None on that side, never the bare number
    of components. Falling back to the count would put roubles-times-pieces and
    pieces into one tracked field, and a baseline diff cannot tell them apart:
    measured on reference recipe 04, the same solution scored 5.0 while its
    priced twin scored 995.7 against an original of 431.2, which a diff would
    have read as an 86-fold improvement (TZ_SOLVER_V2.md 10.10).

    Nothing is lost by the None. The number of components is already tracked by
    the "count" metric, so a caller without prices simply tracks count, junk,
    min_portion and conditioning, and this case drops out of the assembly-score
    aggregate. Do not "fix" this by restoring the fallback.

    Consumers must filter the Nones out before sorting or averaging, and should
    compare percentiles only over the cases where the score is defined in BOTH
    runs - the same intersection rule the solved set already uses.

    The metric has no absolute threshold - it is tracked against a baseline run
    - and therefore carries no "ok" and can never appear in "failures".
    """
    return {
        'value': cost_abs * len(recipe) if cost_abs is not None else None,
        'cost_abs': cost_abs,
        'original': original_cost * len(original) if original_cost is not None else None,
    }


def _set_jaccard(recipe, original):
    """Jaccard index of the two material sets; two empty recipes are identical"""
    solution_set = set(recipe)
    original_set = set(original)
    union = solution_set | original_set

    if not union:
        return 1.0

    return len(solution_set & original_set) / len(union)


def _share_delta(recipe, original):
    """Sum of the absolute share differences over the materials common to both"""
    return sum(abs(recipe[name] - original[name]) for name in set(recipe) & set(original))


def _clay_metric(recipe, original):
    """
    Total share of clay-like materials

    The demand is the lower of MIN_CLAY_PERCENT and half of what the original
    used, so an inherently clay-poor chemistry does not produce a permanent
    warning. Detection runs on the recipe's material names, which is also what
    the material records are keyed by, so a material missing from the records
    is still recognised as clay.
    """
    solution_clay = _clay_content(recipe)
    original_clay = _clay_content(original)
    threshold = min(MIN_CLAY_PERCENT, CLAY_ORIGINAL_FRACTION * original_clay)

    return {
        'solution': solution_clay,
        'original': original_clay,
        'threshold': threshold,
        'ok': solution_clay >= threshold,
    }


def solution_quality(recipe, original, materials, prices=None, priorities=None):
    """
    Compare a produced recipe with an original one that yields the same chemistry

    Args:
        recipe: solution as a flat {material: weight percent} dict summing to ~100
        original: the reference recipe, same shape; an existence proof rather
                  than a correct answer (see the module docstring)
        materials: material records, same structure as database/materials.json
                   entries - at least "name" and "formula". Used for chemistry,
                   loss on ignition and clay detection. A material named by a
                   recipe but absent here contributes no oxides and is listed in
                   "unknown_materials", so that a caller can tell "cheap recipe"
                   apart from "half of it could not be analysed".
        prices: optional {material: price per kg}
        priorities: optional {material: int}, lower being more basic; a material
                    absent from the mapping gets common.DEFAULT_PRIORITY

    Returns:
        dictionary of metrics, plus the "failures" and "warnings" lists; see the
        module docstring for the full shape
    """
    prices = prices or {}

    known = _material_names(materials)
    unknown_materials = sorted((set(recipe) | set(original)) - known)

    cost = _cost_metric(recipe, original, prices)
    solution_loi = _loi(materials, recipe)
    original_loi = _loi(materials, original)
    drift = _rounding_drift(materials, recipe)
    original_drift = _rounding_drift(materials, original)

    result = {
        'count': _count_metric(recipe, original),
        'junk': _junk_metric(recipe, original),
        'min_portion': _min_portion_metric(recipe, original),
        'cost': cost,
        'priority': _priority_metric(recipe, original, priorities),
        'conditioning': _conditioning_metric(recipe, original, materials),
        # Informational only: this "ok" says "can be weighed on a coarse scale",
        # it is not part of GATED_METRICS and never fails a solution.
        'rounding_drift': {
            'value': drift,
            'original': original_drift,
            'ok': None if drift is None else drift <= MAX_ROUNDING_DRIFT,
        },
        'assembly_score': _assembly_score(recipe, cost['cost_abs'], original, cost['original']),
        'set_jaccard': _set_jaccard(recipe, original),
        'share_delta': _share_delta(recipe, original),
        'clay_content': _clay_metric(recipe, original),
        'loi': {
            'solution': solution_loi,
            'original': original_loi,
            'delta': solution_loi - original_loi,
        },
        'loi_delta': abs(solution_loi - original_loi),
        'unknown_materials': unknown_materials,
    }

    # A metric that could not be computed reports ok=None and must not be
    # counted as a violation: "unknown" is not "worse".
    result['failures'] = [key for key in GATED_METRICS if result[key]['ok'] is False]
    result['warnings'] = [key for key in WARNING_METRICS if result[key]['ok'] is False]

    return result
