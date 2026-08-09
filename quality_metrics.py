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
      # small components that carry nothing unique - the gated one
      "junk": {"solution": 0, "original": 1, "delta": -1, "ok": True},
      # every component lighter than JUNK_WEIGHT_PERCENT, load bearing or not;
      # diagnostic, so that "how many light components" is still answerable
      "small_components": {"solution": 1, "original": 2, "delta": -1},
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
      # share of the batch we have no chemistry for, in either sense below;
      # gated absolutely, because nothing can be claimed about that mass
      "unanalysed_share": {"solution": 1.0, "original": 1.0,
                           "threshold": 20.0, "ok": True},
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
      # named by a recipe but absent from the material records
      "unknown_materials": [],
      # in the records, but with an analysis carrying no oxide at all
      "unanalysed_materials": ["Кобальт голубой пигмент 6226"],
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
import logging
import math
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

logger = logging.getLogger(__name__)


# Below roughly two percent a raw material is hard to weigh, easy to forget and
# rarely worth a separate bag on the shelf. That makes a component SMALL. It does
# not by itself make it junk.
#
# Weight says nothing about importance: 0.5% of cobalt carbonate is the whole
# difference between a blue glaze and a clear one. Measured over the Glazy dump,
# 19862 of 35097 recipes (56.6%) carry a component under 2%, and 17378 of those
# 33301 components (52.2%) are the ONLY source of an oxide in their own recipe -
# cobalt carbonate at 0.19% for CoO, copper carbonate at 0.02% for CuO, zircopax
# at 0.40% for ZrO2, lithium carbonate at 1.10% for Li2O. Counting those as junk
# calls the reason a recipe exists an artefact of the fit.
#
# So a small component is junk only when it is NOT the sole carrier, within its
# own recipe, of an oxide the original's chemistry contains - the same "load
# bearing" test the solver's pruning pass applies before dropping a component,
# so that the two agree on what may be thrown away. The metric keeps its teeth:
# the other 47.8% of small components in the dump carry nothing unique and stay
# junk, among them the 0.37% of feldspar that a solver adds to shave a fourth
# decimal off an oxide every other material already supplies.
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
# threshold is shared with matrix_diagnostics of stage 2.
#
# Where the number comes from, and what it is worth. The often quoted pair of
# figures - 15.1 for an honest feldspar / silica / whiting / kaolin set against
# 3668.8 for the same chemistry rebuilt on a collinear pair of feldspars
# (TZ_SOLVER_V2.md 10.9) - is measured against a SYNTHETIC twin that exists only
# in tests/test_quality_metrics.py. It shows the metric responds to degeneracy;
# it is not a calibration against real materials. Measured on the real database:
#   - all 169 pairs of the 19 inventory materials that carry oxides score
#     1.0 to 34.8 on their own, nowhere near the threshold;
#   - of the 120 ways to add two inventory materials to a quartz / chalk /
#     kaolin base, 118 stay at or below 300, and two cross: wollastonite with
#     talc (5.6e3) and wollastonite with dolomite (1.5e4), because wollastonite
#     is very nearly chalk plus quartz in oxide space - the same substitution
#     the solver makes in reference recipe 03;
#   - the two exact duplicates of the inventory - aluminium powder with alumina,
#     and zinc carbonate with zinc oxide - are caught through the rank instead,
#     as cond=None.
# So the gate catches genuine linear dependence and near-duplicates, and leaves
# merely similar materials alone: the closest ordinary pair in the inventory,
# nepheline syenite VR13 with feldspar FFF, sits at 34.8. Do not move the number
# without a measurement that justifies a different one.
MAX_CONDITION_NUMBER = 1e3

# Share of the batch that may consist of material with no usable analysis -
# either absent from the records or present with an oxide-free analysis. Such
# material is excluded from the conditioning matrix, because an empty column
# would condemn every pigmented recipe as degenerate; this rule is what stops
# that exclusion from quietly becoming a licence. Nothing computes the chemistry
# of that mass, so the UMF describes only the rest of the bucket, and past some
# fraction the whole chemistry claim stops being a description of the fired
# glaze.
#
# 20% is measured, not chosen by taste. Of the 34778 usable Glazy corpus cases,
# 8.4% carry an oxide-free ingredient at all (9.7% of the standard 300-case
# sample, matching the corpus benchmark), and among those carriers the median
# share is 4.4% and the 90th percentile 19.7% - ordinary stain, grog, Darvan and
# CMC practice lives below a fifth of the batch. Above it sits a different
# animal: the 269 cases (0.77% of the corpus) over 20% are recipes built on
# water, grog, terra sigillata or a ready-made commercial glaze, where the UMF
# is a statement about a minority of the bucket. All eleven of our own reference
# recipes are at 0%, so the rule costs nothing at home. A tighter 10% would fire
# on 1.40% of the corpus, much of it legitimate stain-heavy work.
#
# There is deliberately no "no worse than the original" waiver here, unlike
# min_portion. A sub-percent colorant in the original is evidence that the
# chemistry needs one; 80% of water in the original is evidence that the record
# is junk, and copying it does not make the copy judgeable.
MAX_UNANALYSED_SHARE_PERCENT = 20.0

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
GATED_METRICS = ('count', 'junk', 'min_portion', 'cost', 'priority', 'conditioning',
                 'unanalysed_share')

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
        dictionary {material name: price per kg}, carrying only the entries
        whose price is a finite number; an empty dictionary when the file is
        missing, empty, holds a bare null, cannot be read or decoded, is not
        valid JSON or does not hold an object. The price list is optional data:
        neither its absence nor its corruption may break a metric run, and a
        typo in one entry must cost that entry's coverage rather than crash a
        corpus run four calls later. Anything a file loses this way is logged,
        because silently pricing nothing looks exactly like having no prices.
    """
    if path is None:
        path = PRICES_PATH

    if not os.path.isfile(path):
        return {}

    try:
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        prices = json.loads(text) if text else None
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as error:
        logger.warning(f"{path}: cannot be read as JSON ({error}) - continuing without prices")
        return {}

    if prices is None:
        return {}

    if not isinstance(prices, dict):
        logger.warning(
            f"{path}: expected an object of {{material: price}}, got {type(prices).__name__}"
            " - continuing without prices"
        )
        return {}

    # A price that is not a finite number would not fail here but four calls
    # later, inside the cost metric, halfway through a corpus run. Dropping it
    # degrades the coverage instead, which the cost metric already reports.
    # bool is an int in Python and is never a price.
    usable = {}
    rejected = []
    for material, price in prices.items():
        if isinstance(price, bool) or not isinstance(price, (int, float)) or not math.isfinite(price):
            rejected.append(material)
        else:
            usable[material] = price

    if rejected:
        logger.warning(
            f"{path}: dropping {len(rejected)} entries whose price is not a finite number: "
            f"{', '.join(sorted(rejected))}"
        )

    return usable


def _material_names(materials):
    """Set of the names of the given material records"""
    return {material.get('name') for material in materials}


def _carries_oxides(material):
    """
    Whether a material record has an analysis a matrix can use

    37 of the 216 materials of the database carry no oxide at all: every
    pigment, all six silicon-carbide fractions, silicon nitride, silver nitrate,
    cadmium carbonate, the phosphor, water, CMC, charcoal, gypsum and alum. They
    are legal recipe entries - a glaze really does contain 1% of cobalt pigment
    - but they are empty columns for any chemistry calculation.
    """
    formula = material.get('formula', {})
    return sum(
        content for oxide, content in formula.items() if oxide not in NON_OXIDE_KEYS
    ) > 0


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
    does not depend on dictionary order.

    Two kinds of material are left out of the matrix. One is a material missing
    from the records, reported in "unknown_materials". The other, 37 times more
    common, is a material that IS in the records but whose analysis carries no
    oxide - a pigment, silicon carbide, water, CMC - reported in
    "unanalysed_materials". Neither may be given a column: an empty column is
    all zeros, which drops the rank below the number of columns and would
    condemn every pigmented recipe as degenerate.

    That exclusion is bounded by the "unanalysed_share" metric, NOT by the
    redundancy reported below. Redundancy is diagnostic output that nothing
    gates on, so leaning on it would mean the exclusion has no limit at all and
    a recipe that is four fifths unanalysable would pass this metric clean.

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
        (
            material for material in materials
            if material.get('name') in recipe and _carries_oxides(material)
        ),
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


def _chemistry_oxides(materials, recipe):
    """
    The oxides a recipe's chemistry actually contains

    Read from the weight composition rather than from the UMF on purpose.
    weights_to_umf() rounds to three decimals, and a colorant dosed at a few
    hundredths of a percent - copper carbonate at 0.02% is a real Glazy
    specimen - contributes a UMF value that rounds to zero. Its oxide would then
    not count as part of the chemistry, its sole carrier would be junk again,
    and the rule would fail on precisely the case it exists for.
    """
    composition = calculate_recipe_composition(materials, recipe)
    return {
        oxide for oxide, weight in composition.items()
        if weight > 0 and oxide not in NON_OXIDE_KEYS
    }


def _carried_oxides(material):
    """Oxides a material record actually brings to a batch"""
    return {
        oxide for oxide, content in material.get('formula', {}).items()
        if content > 0 and oxide not in NON_OXIDE_KEYS
    }


def _sole_carrier_oxides(recipe, index):
    """
    For each material of the recipe, the oxides it is the only source of

    Sole-carrier status is a property of THIS recipe, not of the database: chalk
    is the only source of CaO in one recipe and one of three in the next, and
    only the first makes it load bearing.
    """
    sources = {}
    for name, weight in recipe.items():
        if weight <= 0:
            continue
        for oxide in _carried_oxides(index.get(name, {})):
            sources.setdefault(oxide, []).append(name)

    sole = {}
    for oxide, names in sources.items():
        if len(names) == 1:
            sole.setdefault(names[0], set()).add(oxide)

    return sole


def _count_small(recipe):
    """Components lighter than JUNK_WEIGHT_PERCENT, load bearing or not"""
    return sum(1 for weight in recipe.values() if weight < JUNK_WEIGHT_PERCENT)


def _count_junk(recipe, index, chemistry_oxides):
    """
    Small components that carry nothing the chemistry could not get elsewhere

    A material missing from the records or carrying no oxide brings nothing
    unique by definition, so it is never exempt - being unanalysable is not a
    reason to keep a component.
    """
    sole = _sole_carrier_oxides(recipe, index)
    return sum(
        1 for name, weight in recipe.items()
        if weight < JUNK_WEIGHT_PERCENT and not (sole.get(name, frozenset()) & chemistry_oxides)
    )


def _junk_metric(recipe, original, index, chemistry_oxides):
    """
    Small components that are not the sole carrier of an oxide of the chemistry

    Both sides are judged by the same rule and against the same set of oxides -
    the original's - so that the comparison stays like for like. See the comment
    on JUNK_WEIGHT_PERCENT for why weight alone is not the test.
    """
    solution_junk = _count_junk(recipe, index, chemistry_oxides)
    original_junk = _count_junk(original, index, chemistry_oxides)
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

    An EMPTY mapping abstains for the same reason. Letting it through would send
    every material to DEFAULT_PRIORITY, which cancels in the ratio and returns a
    confident 1.0 with ok=True - a verdict manufactured out of no data at all,
    which is worse than either abstaining or failing.
    """
    if not priorities:
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


def _unanalysed_share_metric(recipe, original, unanalysable):
    """
    Share of the batch made of material whose chemistry we do not have

    Both classes count: a material absent from the records and a material whose
    analysis carries no oxide. Neither contributes to the computed chemistry, so
    both are the same hole from this metric's point of view - which of the two
    it was stays visible in "unknown_materials" and "unanalysed_materials".

    This is a validity precondition rather than a comparison, so it is gated
    absolutely and the original's share is reported for context only. A solver
    solution scores 0 here by construction (the solvers drop oxide-free
    materials from the inventory before they start), so the rule only ever
    speaks about a recipe someone handed us.
    """
    solution_share = sum(weight for name, weight in recipe.items() if name in unanalysable)
    original_share = sum(weight for name, weight in original.items() if name in unanalysable)

    return {
        'solution': solution_share,
        'original': original_share,
        'threshold': MAX_UNANALYSED_SHARE_PERCENT,
        'ok': solution_share <= MAX_UNANALYSED_SHARE_PERCENT,
    }


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
                   loss on ignition and clay detection. Two kinds of material
                   contribute no oxides and are reported separately, so that a
                   caller can tell "cheap recipe" apart from "half of it could
                   not be analysed": one absent from these records entirely
                   ("unknown_materials"), and one present but with an analysis
                   carrying no oxide at all - a pigment, silicon carbide, water
                   ("unanalysed_materials"). Neither is allowed to fail the
                   conditioning metric.
        prices: optional {material: price per kg}
        priorities: optional {material: int}, lower being more basic; a material
                    absent from the mapping gets common.DEFAULT_PRIORITY

    Returns:
        dictionary of metrics, plus the "failures" and "warnings" lists; see the
        module docstring for the full shape
    """
    prices = prices or {}

    named = set(recipe) | set(original)
    known = _material_names(materials)
    unknown_materials = sorted(named - known)
    unanalysed_materials = sorted(
        material.get('name') for material in materials
        if material.get('name') in named and not _carries_oxides(material)
    )

    index = {material.get('name'): material for material in materials}
    chemistry_oxides = _chemistry_oxides(materials, original)
    small_solution = _count_small(recipe)
    small_original = _count_small(original)

    cost = _cost_metric(recipe, original, prices)
    solution_loi = _loi(materials, recipe)
    original_loi = _loi(materials, original)
    drift = _rounding_drift(materials, recipe)
    original_drift = _rounding_drift(materials, original)

    result = {
        'count': _count_metric(recipe, original),
        'junk': _junk_metric(recipe, original, index, chemistry_oxides),
        'small_components': {
            'solution': small_solution,
            'original': small_original,
            'delta': small_solution - small_original,
        },
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
        'unanalysed_share': _unanalysed_share_metric(
            recipe, original, set(unknown_materials) | set(unanalysed_materials)
        ),
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
        'unanalysed_materials': unanalysed_materials,
    }

    # A metric that could not be computed reports ok=None and must not be
    # counted as a violation: "unknown" is not "worse".
    result['failures'] = [key for key in GATED_METRICS if result[key]['ok'] is False]
    result['warnings'] = [key for key in WARNING_METRICS if result[key]['ok'] is False]

    return result
