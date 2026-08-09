#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# flake8: noqa
# pylint: disable=broad-exception-raised, raise-missing-from, too-many-arguments, redefined-outer-name
# pylint: disable=multiple-statements, logging-fstring-interpolation, trailing-whitespace, line-too-long
# pylint: disable=broad-exception-caught, missing-function-docstring, missing-class-docstring
# pylint: disable=f-string-without-interpolation
# pylance: disable=reportMissingImports, reportMissingModuleSource
# type: ignore

# "Is this target reachable from these materials at all, and if not, which oxide
# is to blame" - the feasibility layer, plus the achievable ranges the interface
# asks for ("how much ulexite can this recipe take" -> "9-19%").
#
# WHY THIS IS A LINEAR PROGRAM AT ALL
#
# A UMF value is a ratio - moles of the oxide divided by the sum of the fluxes -
# so it is fraction-linear in the material amounts, and a constraint on it looks
# non linear. It is not, because both the numerator and the denominator are
# HOMOGENEOUS in x: scaling the whole batch scales them together and leaves the
# UMF unchanged. Two consequences are what this whole module rests on:
#
#   1. Any UMF constraint can be written homogeneously and then holds at every
#      scale:  lo <= n_i / S_flux <= hi   <=>   n_i - hi*S_flux <= 0
#                                          and  lo*S_flux - n_i <= 0.
#   2. Fixing the scale with ONE equality - S_flux(x) = 1 - makes the UMF value
#      of oxide i literally equal to (A x)_i, a linear function of x.
#
# So the work happens in MOLAR space:
#
#   A[i, j]  = weight% of oxide i in material j / molar_mass(oxide i)
#              (moles of oxide i per 100 g of raw material j)
#   f[i]     = 1 if oxide i is in the unity basis (common.flux_oxides()) else 0
#   S_flux(x) = f . A . x
#
# The matrix does not care what a material analysis sums to. Most rows of
# database/materials.json sum to less than 100 because the remainder is the
# implicit loss on ignition, but two real entries sum to MORE (Cryolite 122.90,
# Zircon 135.22) and those are correct analyses, not broken records. Nothing
# here normalizes a formula, so both cases are handled by the same arithmetic:
# a column is "moles per 100 g as weighed out", whatever the analysis totals.
#
# WHAT THIS MODULE MUST NEVER DO
#
# Raise into a solver path. Feasibility is diagnostics: when it fails the
# solvers must keep working, so every public function catches its own failures
# and reports them in the returned dictionary instead of propagating them.

import logging
import math

import numpy as np
from scipy.optimize import linprog

from common import (NON_OXIDE_KEYS, filter_materials_with_formula, flux_oxides,
                    load_molar_masses)

logger = logging.getLogger(__name__)

# How far the closest reachable formula may sit from the target before the
# target is declared unreachable, as a fraction of the oxide scale below.
# TUNABLE POLICY, NOT PHYSICS: nothing in the chemistry says 5% is the line
# between "reachable" and "not". It is the number the verdict in the interface
# is drawn with, and moving it moves the verdict, not the chemistry.
DEFAULT_FEASIBILITY_TOL = 0.05

# The scale an oxide's deviation is measured against: max(target, this). Without
# a floor the relative deviation of a trace oxide dominates everything - missing
# MgO 0.05 by 0.04 would read as 80% while missing SiO2 3.0 by 0.04 reads as 1%,
# and the minimax LP would spend the whole material budget chasing the trace.
# TUNABLE POLICY, NOT PHYSICS, and deliberately the same value as the floor in
# sensitivity.py, which ranks contributions on the same footing.
OXIDE_SCALE_FLOOR = 0.1

# Materials below this share are dropped from "closest_recipe": the LP happily
# returns a vertex with a 0.004% column, which is not something anyone can weigh
# out. The survivors are rescaled back to 100%.
MIN_RECIPE_PERCENT = 0.1

# Above this condition number the material set is called ill conditioned. Same
# threshold and therefore the same policy as quality_metrics.MAX_CONDITION_NUMBER
# - the two are meant to agree on what "these materials are nearly collinear"
# means, and a second opinion here would be a second definition.
MAX_CONDITION_NUMBER = 1e3

# Numerical slack when comparing an LP result against a bound. HiGHS returns a
# primal feasible point to about 1e-9, and a passenger sitting exactly on its
# ceiling must not be reported as having broken it.
BOUND_EPS = 1e-7

# linprog statuses we know how to read. 0 optimal, 2 infeasible, 3 unbounded -
# the last two are answers about the problem, not failures of the solve. 1
# (iteration limit) and 4 (numerical trouble) are failures.
LP_OPTIMAL = 0
LP_INFEASIBLE = 2
LP_UNBOUNDED = 3


def usable_oxides(oxides):
    """
    Keep the oxide names the molar matrix can actually carry a row for

    Two kinds of name are dropped. Loss on ignition (common.NON_OXIDE_KEYS) is
    bookkeeping and is dropped silently - it is not a lost oxide. Anything else
    without an entry in database/molar_masses.json cannot be converted to moles,
    so it is dropped WITH a warning: the usual source is a typo in a target, and
    silently ignoring it would answer "reachable" about a formula the caller did
    not ask for.

    Args:
        oxides: iterable of oxide names, duplicates allowed

    Returns:
        list of usable oxide names, first occurrence order preserved
    """
    molar_masses = load_molar_masses()
    kept = []
    unknown = []

    for oxide in dict.fromkeys(oxides):
        if oxide in NON_OXIDE_KEYS:
            continue
        if oxide not in molar_masses:
            unknown.append(str(oxide))
            continue
        kept.append(oxide)

    if unknown:
        logger.warning(f"feasibility: dropped {len(unknown)} oxides with no molar mass: "
                       f"{', '.join(unknown)}")

    return kept


def build_molar_matrix(materials, oxides):
    """
    Build the "moles of oxide per 100 g of material" matrix

    A[i, j] = formula[oxide_i] of material j / molar_mass(oxide_i).

    Materials whose analysis carries no oxide at all - water, CMC, silicon
    carbide, the 21 pigments, 37 records in total - are excluded through
    common.filter_materials_with_formula(). They are legal entries, but as a
    column they are all zeros: they can never move the chemistry, and they would
    only make the matrix rank deficient.

    Args:
        materials: list of material records, as in database/materials.json
        oxides: oxide names, one row each. Names this module cannot use are
            dropped first, so the rows correspond to usable_oxides(oxides) and
            NOT necessarily to the argument. Callers that index rows by oxide
            must call usable_oxides() themselves and pass its result

    Returns:
        (A, material_names): the matrix of shape (len(usable_oxides(oxides)),
        len(kept materials)) and the names of the columns, in order
    """
    rows = usable_oxides(oxides)
    kept = filter_materials_with_formula(materials)
    molar_masses = load_molar_masses()

    matrix = np.zeros((len(rows), len(kept)))
    material_names = []

    for j, material in enumerate(kept):
        material_names.append(material.get('name'))
        formula = material.get('formula') or {}
        for i, oxide in enumerate(rows):
            content = formula.get(oxide, 0.0)
            matrix[i, j] = float(content) / molar_masses[oxide]

    return matrix, material_names


def flux_row(oxides):
    """
    The 0/1 indicator of the unity basis over the given oxide rows

    The list of fluxes comes from common.flux_oxides() on every call and is
    never cached here: which oxides form the unity denominator is a convention
    that lives in database/oxide_classification.json, and a local copy is
    exactly how the four copies the first stage removed came to disagree.

    Args:
        oxides: oxide names in row order, as returned by usable_oxides()

    Returns:
        numpy array of len(oxides) floats, 1.0 for a flux and 0.0 otherwise
    """
    fluxes = set(flux_oxides())
    return np.array([1.0 if oxide in fluxes else 0.0 for oxide in oxides])


def matrix_diagnostics(materials, oxides):
    """
    Conditioning of the material set: can these materials be told apart at all

    A set standing on two near identical feldspars has an almost singular
    matrix, so the amounts that produce a given chemistry are barely determined
    and any error anywhere sends them a long way. The number is the ratio of the
    largest to the smallest singular value of the "oxides x materials" matrix.

    The matrix here is the WEIGHT matrix (weight% of oxide i in material j), not
    the molar one the LP works on, so that this number and the one in
    quality_metrics._condition_number are the same measurement of the same
    thing; MAX_CONDITION_NUMBER is likewise the same threshold and therefore the
    same policy. The molar matrix differs from this one by a positive row
    scaling (1/molar mass, a factor of about 14 between the extremes), which
    does move the condition number - the LP is conditioned slightly differently
    than this reports, and that is the price of having one comparable number
    instead of two incomparable ones.

    Args:
        materials: list of material records
        oxides: oxide names to build the rows from

    Returns:
        {"cond": float or None, "ill_conditioned": bool, "rank": int,
         "n_oxides": int}. cond is None for a rank deficient or empty set, which
        is infinitely ill conditioned rather than unmeasured - the same
        convention quality_metrics uses, and the reason ill_conditioned is True
        there instead of being derived from a comparison with None
    """
    try:
        rows = usable_oxides(oxides)
        kept = filter_materials_with_formula(materials)

        matrix = np.zeros((len(rows), len(kept)))
        for j, material in enumerate(kept):
            formula = material.get('formula') or {}
            for i, oxide in enumerate(rows):
                matrix[i, j] = float(formula.get(oxide, 0.0))

        if matrix.size == 0:
            return {"cond": None, "ill_conditioned": True, "rank": 0, "n_oxides": len(rows)}

        singular_values = np.linalg.svd(matrix, compute_uv=False)
        rank = int(np.linalg.matrix_rank(matrix))

        # Degenerate means "lost rank it could have had", min(rows, columns),
        # and NOT "fewer independent columns than columns" as in
        # quality_metrics._condition_number. The two judge different objects:
        # there the columns are the materials of ONE RECIPE, where a column
        # beyond the oxide count is the compensating pair the metric is hunting
        # for, while here they are a whole INVENTORY - 19 materials over 12
        # oxides in the shipped one - which cannot be independent and is not
        # broken for it. Keeping their rule would report cond=None,
        # ill_conditioned=True for every real inventory, which is not a
        # diagnostic but a constant.
        if rank < min(matrix.shape) or singular_values[-1] <= 0:
            return {"cond": None, "ill_conditioned": True, "rank": rank, "n_oxides": len(rows)}

        cond = float(singular_values[0] / singular_values[-1])
        return {"cond": cond, "ill_conditioned": cond > MAX_CONDITION_NUMBER,
                "rank": rank, "n_oxides": len(rows)}

    except Exception as exc:
        # Diagnostics never break the caller; an unmeasurable matrix reports
        # itself as the worst case, exactly like a rank deficient one
        logger.warning(f"feasibility: matrix diagnostics failed: {exc}")
        return {"cond": None, "ill_conditioned": True, "rank": 0, "n_oxides": 0}


def _usable_target(target_umf):
    """
    Keep the target entries the math can work with

    Same rule as solver_iterative._usable_target, deliberately: a known oxide
    carrying a finite, non negative number. An explicit zero is KEPT - the
    interface sends the whole oxide table on every request and "give me no iron"
    is a constraint, not the absence of an opinion.

    Returns:
        (usable, dropped): the cleaned target and the names that were refused
    """
    if not target_umf:
        return {}, []

    molar_masses = load_molar_masses()
    usable = {}
    dropped = []

    for oxide, value in target_umf.items():
        if oxide in NON_OXIDE_KEYS:
            continue
        if oxide not in molar_masses:
            dropped.append(str(oxide))
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            dropped.append(str(oxide))
            continue
        if math.isfinite(number) and number >= 0.0:
            usable[oxide] = number
        else:
            dropped.append(str(oxide))

    return usable, dropped


def _normalized_target(target):
    """
    Put the target on the unity basis the LP normalization assumes

    S_flux(x) = 1 fixes the scale of the answer, so a target whose own fluxes
    sum to 2 can never be matched oxide for oxide however good the materials
    are - and "unreachable" would be the wrong word for it, because a UMF and
    the same UMF scaled are the same glaze. The iterative solver has the mirror
    image of this in _unity_scale, which rescales the RESULT onto the basis of
    the target; here the target is brought onto the basis of the answer, which
    comes to the same comparison.

    A target with no fluxes at all is refused rather than rescaled. It is not a
    UMF - there is no unity to normalize by - and inventing a basis for it (the
    "smallest oxide" branch of weights_to_umf) would answer a question nobody
    asked. See TZ_SOLVER_V2.md 10.17.2.

    Returns:
        (target, flux_sum): the rescaled target and the sum it was divided by,
        or (None, 0.0) when the target carries no flux
    """
    fluxes = set(flux_oxides())
    flux_sum = sum(value for oxide, value in target.items() if oxide in fluxes)

    if flux_sum <= 0.0:
        return None, 0.0
    if abs(flux_sum - 1.0) <= 1e-9:
        return dict(target), 1.0

    return {oxide: value / flux_sum for oxide, value in target.items()}, flux_sum


def _usable_passengers(passengers, target_umf):
    """
    Clean the passenger ceilings, and take them out of the two sided target

    An oxide named in both places is a passenger here: the upper bound is the
    weaker and the more honest statement of the two, and a caller sending both
    has said "I do not really want this value, just keep it under control".

    Returns:
        (ceilings, dropped, overridden)
    """
    if not passengers:
        return {}, [], []

    molar_masses = load_molar_masses()
    ceilings = {}
    dropped = []
    overridden = []

    for oxide, value in passengers.items():
        if oxide in NON_OXIDE_KEYS or oxide not in molar_masses:
            dropped.append(str(oxide))
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            dropped.append(str(oxide))
            continue
        if not math.isfinite(number) or number < 0.0:
            dropped.append(str(oxide))
            continue
        ceilings[oxide] = number
        if oxide in target_umf:
            overridden.append(str(oxide))

    return ceilings, dropped, overridden


def _material_oxides(materials):
    """Every oxide the given materials can bring in, in a stable order"""
    seen = []
    for material in materials:
        for oxide in (material.get('formula') or {}):
            if oxide not in seen:
                seen.append(oxide)
    return sorted(seen)


def _fmt(value):
    """Short human readable number for a message: 0.0812 -> '0.0812', 3.0 -> '3'"""
    return f"{float(value):.3g}"


def _matmul(left, right):
    """
    A matrix product with the SPURIOUS floating point flags of numpy 2 silenced

    On a matrix of about 50 x 180 - the whole database rather than an inventory
    - "flux @ A" raises "divide by zero", "overflow" and "invalid value" at
    once, while both operands are finite and the product agrees with a hand
    written loop to 6e-17. The flags are left behind by the vectorized BLAS
    kernel, not by our arithmetic, and numpy 2.2 reports them as ours.

    Silencing them is only safe because the operands are checked separately:
    _finite_matrix() refuses a matrix with a NaN or an infinity in it before any
    of this runs, so a real non-finite value is reported rather than hidden.
    """
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        return left @ right


def _finite_matrix(A):
    """True when the molar matrix has no NaN and no infinity in it"""
    return bool(np.isfinite(A).all())


def _linprog(c, A_ub, b_ub, A_eq, b_eq, bounds):
    """
    One LP through HiGHS, with the argument shuffling kept in one place

    Returns the OptimizeResult; the callers read .status themselves because
    "infeasible" and "unbounded" are answers here, not errors.
    """
    return linprog(
        c=c,
        A_ub=np.array(A_ub) if len(A_ub) else None,
        b_ub=np.array(b_ub) if len(b_ub) else None,
        A_eq=np.array(A_eq) if len(A_eq) else None,
        b_eq=np.array(b_eq) if len(b_eq) else None,
        bounds=bounds,
        method='highs',
    )


def check_feasibility(target_umf, materials, tol=DEFAULT_FEASIBILITY_TOL, passengers=None):
    """
    Can this target be reached from these materials, and if not, which oxide fails

    The Chebyshev (minimax) LP of TZ_SOLVER_V2.md 2.1. Variables are x (grams of
    each material) and t (the worst relative deviation over the oxides):

        minimize   t
        subject to f.A.x = 1                       (scale: UMF values ARE A x)
                   (A x)_i - b_i <=  t * s_i       for every measured oxide i
                   b_i - (A x)_i <=  t * s_i
                   (A x)_i - hi_i * f.A.x <= 0     for every passenger oxide i
                   x >= 0, t >= 0

    with s_i = max(b_i, OXIDE_SCALE_FLOOR). The equality is not cosmetic: it is
    what turns a ratio constraint into a plain linear one, and it is why the
    whole question is an LP and not a search.

    A SECOND LP FOLLOWS, and the answer is unusable without it. A minimax has a
    large flat optimum: once t* is paid for by the one oxide that cannot be hit,
    every other oxide may drift anywhere inside t* * s_i for free, and HiGHS
    returns whichever vertex it lands on. Measured on the reference clear glaze
    with a ZrO2 of 0.2 added - nothing in the stock carries zirconium - the bare
    minimax called SEVEN oxides unreachable, six of which are perfectly
    reachable. The verdict named the wrong oxides. So t is then frozen at t* and
    a second LP minimizes the SUM of the deviations, sum |(A x)_i - b_i| / s_i,
    over that optimal face. The verdict itself cannot change (t* is a hard
    constraint of the second LP), and the per-oxide numbers become the least-bad
    point instead of an arbitrary one: on that target the list becomes ZrO2 and
    nothing else.

    What the polish CANNOT do is hide a real coupling, and it should not. Ask
    the same glaze for Li2O 0.5 and the list is Li2O AND CaO, because lithium is
    a FLUX: half the unity budget was asked of a material that does not exist,
    so whatever fluxes remain have to fill the whole of it and the largest of
    them cannot stay where it was. That second name is part of the answer.

    MEASURED OXIDES are the ones the target names plus, with b_i = 0, every
    oxide the materials could bring but the target does not mention - the same
    expansion solver_iterative._expand_target does. Contamination is part of the
    verdict: a target that can only be hit by dragging in half a percent of
    something nobody asked for has not really been hit.

    PASSENGERS are the exception to that, and the reason the parameter exists.
    An oxide the user did not ask for is not an error to be driven to zero; it
    arrives with the materials whether anyone likes it or not - Fe2O3 with the
    kaolin, TiO2 with the wollastonite, SrO with the ulexite. Given a two sided
    target of 0.012 the solver spends real effort matching a number nobody
    chose, and pays for it by dosing 0.3% of an iron oxide it cannot weigh out:
    that is where junk components come from. So a passenger gets a ONE SIDED
    CEILING - Fe2O3 <= 0.03 - with no lower bound and no term in t. It is a
    constraint on the answer, not a goal of it (UI_DECISIONS.md 2.2).

    Args:
        target_umf: {oxide: value} in UMF units. Unknown names, negatives, NaN
            and infinities are dropped with a warning; an explicit 0.0 is kept
            and means "none of this, please"
        materials: list of material records to build from - already filtered to
            the caller's inventory. Records with an empty formula are dropped
        tol: the verdict line. t* <= tol means reachable
        passengers: {oxide: upper_bound} in UMF units. An oxide listed here is
            taken out of the two sided treatment even if the target names it

    Returns:
        {
          "feasible": bool,               # t* <= tol
          "max_relative_deviation": float,
          "per_oxide": [{"oxide", "target", "closest", "delta", "relative",
                         "reachable"}, ...],   # sorted by |delta| descending
          "unreachable_oxides": [str, ...],
          "why": {oxide: "short reason, in Russian"},
          "passengers": [{"oxide", "limit", "closest", "within_limit"}, ...],
          "closest_recipe": {material: percent},  # renormalized to 100
          "warnings": [str, ...],
        }

        per_oxide covers the target oxides always, and a contamination oxide
        only when the closest formula actually carries it - listing the sixty
        oxides of the whole database at 0.0 would bury the answer.

        On an LP that did not converge: {"feasible": None, "error": ...}. Same
        shape for the cases where there is nothing to solve at all (no usable
        target, no usable material, no flux carrier in the set). "feasible":
        False with an empty per_oxide means the constraints contradict each
        other - only passengers can do that here.
    """
    try:
        return _check_feasibility(target_umf, materials, tol, passengers)
    except Exception as exc:
        # Feasibility is diagnostics. Whatever went wrong in here, the solvers
        # above must keep working, so nothing leaves this module as an exception
        logger.exception(f"feasibility: check failed: {exc}")
        return {"feasible": None, "error": "feasibility_failed", "message": str(exc)}


def _check_feasibility(target_umf, materials, tol, passengers):
    warnings = []

    target, dropped_target = _usable_target(target_umf)
    if dropped_target:
        warnings.append(f"оксиды цели не распознаны и не учтены: {', '.join(dropped_target)}")

    ceilings, dropped_passengers, overridden = _usable_passengers(passengers, target)
    if dropped_passengers:
        warnings.append(f"пассажиры не распознаны и не учтены: {', '.join(dropped_passengers)}")
    if overridden:
        warnings.append(f"оксиды заданы и целью, и пассажиром — взята верхняя граница: "
                        f"{', '.join(overridden)}")

    if not target:
        return {"feasible": None, "error": "empty_target",
                "message": "target UMF has no usable oxide", "warnings": warnings}

    target, flux_sum = _normalized_target(target)
    if target is None:
        return {"feasible": None, "error": "no_target_fluxes",
                "message": "target UMF carries no flux, there is no unity to normalize by",
                "warnings": warnings}
    if flux_sum != 1.0:
        warnings.append(f"сумма плавней цели {_fmt(flux_sum)}, а не 1 — "
                        f"формула приведена к единице перед проверкой")

    kept_materials = filter_materials_with_formula(materials or [])
    if not kept_materials:
        return {"feasible": None, "error": "no_usable_materials",
                "message": "no material with an oxide analysis in the set",
                "warnings": warnings}

    # Row order: the target first, in the order it was given, then everything
    # the materials can add. Deterministic, and it keeps the interesting rows on
    # top when the matrix is printed while debugging.
    oxides = usable_oxides(list(target) + list(ceilings) + _material_oxides(kept_materials))
    A, material_names = build_molar_matrix(kept_materials, oxides)

    if not _finite_matrix(A):
        return {"feasible": None, "error": "nonfinite_analysis",
                "message": "a material analysis contains NaN or infinity",
                "warnings": warnings}

    flux = flux_row(oxides)
    flux_vec = _matmul(flux, A)

    if not np.any(flux_vec > 0):
        return {"feasible": None, "error": "no_fluxes",
                "message": "no material of the set carries a flux, UMF is undefined",
                "warnings": warnings}

    n = A.shape[1]
    index = {oxide: i for i, oxide in enumerate(oxides)}

    # Measured = two sided, with a term in t. Everything the target names, plus
    # a zero for every oxide the materials can bring; passengers are excluded.
    measured = {}
    for oxide in oxides:
        if oxide in ceilings:
            continue
        measured[oxide] = float(target.get(oxide, 0.0))

    A_ub = []
    b_ub = []
    for oxide, value in measured.items():
        i = index[oxide]
        scale = max(value, OXIDE_SCALE_FLOOR)
        row = np.zeros(n + 1)
        row[:n] = A[i]
        row[n] = -scale
        A_ub.append(row)
        b_ub.append(value)

        row = np.zeros(n + 1)
        row[:n] = -A[i]
        row[n] = -scale
        A_ub.append(row)
        b_ub.append(-value)

    for oxide, ceiling in ceilings.items():
        i = index[oxide]
        row = np.zeros(n + 1)
        row[:n] = A[i] - ceiling * flux_vec
        A_ub.append(row)
        b_ub.append(0.0)

    A_eq = [np.concatenate([flux_vec, [0.0]])]
    b_eq = [1.0]

    c = np.zeros(n + 1)
    c[n] = 1.0

    result = _linprog(c, A_ub, b_ub, A_eq, b_eq, [(0.0, None)] * (n + 1))

    if result.status == LP_INFEASIBLE:
        # Only the passenger ceilings can do this: the two sided rows always
        # admit a big enough t, and the normalization is satisfiable whenever
        # some material carries a flux
        warnings.append("ограничения по пассажирам несовместимы с нормировкой "
                        "по плавням: ни один допустимый состав не найден")
        return {"feasible": False, "max_relative_deviation": None, "per_oxide": [],
                "unreachable_oxides": sorted(target), "why": {},
                "passengers": [{"oxide": oxide, "limit": ceiling, "closest": None,
                                "within_limit": False}
                               for oxide, ceiling in ceilings.items()],
                "closest_recipe": {}, "warnings": warnings}

    if result.status != LP_OPTIMAL:
        logger.warning(f"feasibility: LP did not converge, status={result.status} "
                       f"({result.message})")
        return {"feasible": None, "error": "lp_not_converged",
                "message": f"status {result.status}: {result.message}",
                "warnings": warnings}

    t_star = float(result.x[n])
    x = _polish_on_optimal_face(A, index, flux_vec, n, measured, ceilings, t_star,
                                np.asarray(result.x[:n]))
    achieved = _matmul(A, x)

    per_oxide = []
    unreachable = []
    worst = 0.0
    for oxide, value in measured.items():
        i = index[oxide]
        closest = float(achieved[i])
        scale = max(value, OXIDE_SCALE_FLOOR)
        delta = closest - value
        relative = abs(delta) / scale
        reachable = relative <= tol + BOUND_EPS
        worst = max(worst, relative)
        if not reachable:
            unreachable.append(oxide)
        if oxide not in target and closest <= 0.0:
            # A contamination oxide the closest formula does not carry at all;
            # reporting "TiO2: asked 0, got 0" sixty times hides the answer
            continue
        per_oxide.append({"oxide": oxide, "target": value, "closest": closest,
                          "delta": delta, "relative": relative, "reachable": reachable})

    per_oxide.sort(key=lambda row: abs(row["delta"]), reverse=True)

    passenger_rows = []
    for oxide, ceiling in ceilings.items():
        closest = float(achieved[index[oxide]])
        passenger_rows.append({"oxide": oxide, "limit": ceiling, "closest": closest,
                               "within_limit": closest <= ceiling + BOUND_EPS})
    passenger_rows.sort(key=lambda row: row["closest"], reverse=True)

    why = _explain_unreachable(unreachable, target, achieved, A, index, flux_vec, n)

    # Read off the reported point rather than taken from the minimax variable,
    # so that "max_relative_deviation" and the rows of per_oxide can never
    # disagree. The polish keeps every deviation within t*, so the two differ
    # only when the polish shaved the worst one - which it can, when several
    # points share t* and one of them is strictly better everywhere
    return {
        "feasible": worst <= tol + BOUND_EPS,
        "max_relative_deviation": worst,
        "per_oxide": per_oxide,
        "unreachable_oxides": unreachable,
        "why": why,
        "passengers": passenger_rows,
        "closest_recipe": _as_recipe(x, material_names),
        "warnings": warnings,
    }


def _polish_on_optimal_face(A, index, flux_vec, n, measured, ceilings, t_star, fallback):
    """
    Pick the least-bad point among the many that share the minimax optimum

    The minimax says how bad the worst oxide has to be; it says nothing about
    the others, which are free to drift anywhere within t* * s_i at no cost to
    the objective. This LP freezes that budget and spends as little of it as
    possible:

        minimize   sum_i d_i / s_i
        subject to |(A x)_i - b_i| <= d_i,  0 <= d_i <= t* * s_i
                   f.A.x = 1, the passenger ceilings, x >= 0

    The verdict cannot move - t* is a hard bound here - so this is presentation,
    not a second opinion. Deviations are summed in the same relative units the
    minimax used, otherwise SiO2 would buy its way out at the expense of every
    trace oxide.

    Args:
        fallback: the minimax point, returned unchanged if this LP does not
            solve. A worse looking answer is better than no answer, and the
            verdict is the same either way

    Returns:
        the material amounts x, still on the S_flux(x) = 1 scale
    """
    oxides = list(measured)
    m = len(oxides)
    if m == 0:
        return fallback

    scales = [max(measured[oxide], OXIDE_SCALE_FLOOR) for oxide in oxides]

    A_ub = []
    b_ub = []
    for k, oxide in enumerate(oxides):
        i = index[oxide]
        value = measured[oxide]

        row = np.zeros(n + m)
        row[:n] = A[i]
        row[n + k] = -1.0
        A_ub.append(row)
        b_ub.append(value)

        row = np.zeros(n + m)
        row[:n] = -A[i]
        row[n + k] = -1.0
        A_ub.append(row)
        b_ub.append(-value)

    for oxide, ceiling in ceilings.items():
        row = np.zeros(n + m)
        row[:n] = A[index[oxide]] - ceiling * flux_vec
        A_ub.append(row)
        b_ub.append(0.0)

    A_eq = [np.concatenate([flux_vec, np.zeros(m)])]
    b_eq = [1.0]

    c = np.zeros(n + m)
    for k, scale in enumerate(scales):
        c[n + k] = 1.0 / scale

    # A hair of slack on the frozen budget: t* comes back from HiGHS with its
    # own rounding, and re-imposing it to the last bit can make this LP
    # infeasible for no reason anyone would recognise
    bounds = [(0.0, None)] * n + [(0.0, t_star * scale + BOUND_EPS) for scale in scales]

    result = _linprog(c, A_ub, b_ub, A_eq, b_eq, bounds)

    if result.status != LP_OPTIMAL:
        logger.warning(f"feasibility: polish LP did not converge, status={result.status}; "
                       f"reporting the minimax point")
        return fallback

    return np.asarray(result.x[:n])


def _as_recipe(x, material_names):
    """
    Turn the LP point into a recipe in weight percent

    x is in whatever grams the normalization implies, so it is rescaled to sum
    100. Traces below MIN_RECIPE_PERCENT are dropped and the survivors rescaled
    again, so the result still sums to 100 - dropping without the second pass
    would quietly lose a percent or two on a wide inventory.
    """
    total = float(np.sum(x))
    if total <= 0:
        return {}

    shares = {name: 100.0 * float(value) / total for name, value in zip(material_names, x)}
    kept = {name: value for name, value in shares.items() if value >= MIN_RECIPE_PERCENT}
    if not kept:
        return {}

    kept_total = sum(kept.values())
    return {name: round(100.0 * value / kept_total, 2) for name, value in kept.items()}


def _oxide_extreme(A, index, flux_vec, n, oxide, maximize):
    """
    The largest (or smallest) UMF value of one oxide this material set can reach

    Nothing but the normalization and x >= 0: no other oxide of the target is
    constrained. That is deliberate - it answers "could this number happen at
    all", which is the question behind an unreachable oxide, and it separates
    "the materials cannot do this" from "the materials cannot do this AND
    everything else at once".

    Returns:
        the value, or None when it is unbounded (a real answer: with pure quartz
        in the set and nothing capping SiO2, there is no maximum)
    """
    i = index[oxide]
    c = -A[i] if maximize else A[i]
    result = _linprog(c, [], [], [np.array(flux_vec)], [1.0], [(0.0, None)] * n)

    if result.status == LP_UNBOUNDED:
        return None
    if result.status != LP_OPTIMAL:
        return None
    return float(-result.fun if maximize else result.fun)


def _explain_unreachable(unreachable, target, achieved, A, index, flux_vec, n):
    """
    Why each unreachable oxide is unreachable, derived and never guessed

    Three cases, in the order they are checked:

      1. NO CARRIER. The row of the matrix is all zeros - not one material of
         the set contains the oxide. Read straight off the matrix, costs
         nothing, and covers the most common real case (asking for lithium
         without a lithium material).
      2. CANNOT BE PUSHED FAR ENOUGH. The oxide IS carried, and its extreme on
         its own - one extra LP, the same machinery achievable_ranges uses -
         still falls short of (or, for an overshoot, stays above) the target.
         The message states that extreme: "достижимо максимум 0.08".
      3. EVERYTHING ELSE falls through to a generic line saying the oxide is
         individually reachable but not together with the rest of the target.
         That is the honest limit of what two cheap checks can establish: naming
         WHICH other oxide it fights with would need the LP duals, and the dual
         of a Chebyshev LP names the binding constraints of the minimax
         solution, not the chemistry.
    """
    why = {}

    for oxide in unreachable:
        i = index[oxide]
        wanted = float(target.get(oxide, 0.0))
        got = float(achieved[i])

        if not np.any(A[i] > 0):
            why[oxide] = f"ни один материал набора не содержит {oxide}"
            continue

        if got < wanted:
            extreme = _oxide_extreme(A, index, flux_vec, n, oxide, maximize=True)
            if extreme is not None and extreme < wanted:
                why[oxide] = (f"{oxide} есть в наборе, но его максимум — "
                              f"{_fmt(extreme)} против нужных {_fmt(wanted)}")
                continue
        else:
            extreme = _oxide_extreme(A, index, flux_vec, n, oxide, maximize=False)
            if extreme is not None and extreme > wanted:
                why[oxide] = (f"{oxide} приезжает с материалами набора: меньше "
                              f"{_fmt(extreme)} не получится, нужно {_fmt(wanted)}")
                continue

        why[oxide] = (f"{oxide} достижим сам по себе, но не одновременно "
                      f"с остальными оксидами цели")

    return why


def achievable_ranges(target_umf, materials, oxide_constraints=None,
                      material_constraints=None, tol=DEFAULT_FEASIBILITY_TOL):
    """
    How far each oxide and each material can move while the target still holds

    "How much ulexite can this recipe take" -> "9-19%". This is the surviving
    half of the interval mode of TZ_SOLVER_V2.md 10.16: the loop of narrowing
    ranges by hand was dropped, the achievable shares were not.

    Every constraint is written homogeneously, so it holds at any scale, and
    each end of each interval is one small LP with the scale pinned by the
    normalization that makes its objective linear:

        oxide i in [lo, hi]:      (A x)_i - hi * f.A.x <= 0
                                  lo * f.A.x - (A x)_i <= 0
        material j in [lo, hi]%:  100 * x_j - hi * sum(x) <= 0
                                  lo * sum(x) - 100 * x_j <= 0

        min/max of oxide i:       f.A.x = 1,     objective +-(A x)_i
        min/max of material j:    sum(x) = 100,  objective +-x_j

    WHICH CONSTRAINTS THE TARGET IMPOSES. Every oxide the target names gets the
    box check_feasibility(tol) would accept it in: [b - tol*s, b + tol*s] with
    s = max(b, OXIDE_SCALE_FLOOR), clipped at zero. So "achievable" here means
    exactly "reachable without breaking the verdict", and the two functions
    cannot disagree about the same target.

    An oxide the target does NOT name is left free, unlike in check_feasibility,
    where it is measured against zero. That is the passenger rule again: here
    there is no minimax to protect, so an unnamed oxide is not a goal at all,
    and its own interval is reported anyway - creeping contamination in numbers.
    A caller that does want a ceiling on one passes it in oxide_constraints as
    [None, hi], which is what the API does with its "passengers" parameter.

    Args:
        target_umf: {oxide: value}, the formula the ranges are measured around
        materials: list of material records, already filtered to the inventory
        oxide_constraints: {oxide: [lo, hi]} in UMF units, either end None for
            "no bound". Overrides the box derived from the target for that oxide
        material_constraints: {material: [lo, hi]} in weight percent, either end
            None for "no bound"
        tol: width of the box derived from the target, as in check_feasibility

    Returns:
        {
          "feasible": True,
          "oxide_ranges": {"SiO2": [3.05, 3.42], "MgO": [0.0, None], ...},
          "material_ranges": {"Каолин КЖФ-1": [0.0, 55.3], ...},
          "example_recipe": {material: percent},
          "lp_count": int,
          "warnings": [str, ...],
        }

        None as the upper end means unbounded, and it is a real answer, not a
        failure: while pure quartz is in the set and nothing caps SiO2, there is
        no largest SiO2. The interface prints it as infinity.

        {"feasible": False, ...} means the constraints contradict each other. It
        is decided on the first LP and the remaining ones are not run, so
        lp_count says 1 - grinding through seventy LPs to repeat "infeasible"
        seventy times tells the caller nothing it did not already know.

        {"feasible": None, "error": ...} when an LP did not converge or there is
        nothing to solve.

    One edge worth knowing about. A flux free mixture - 100% quartz, 100%
    alumina - satisfies EVERY homogeneous UMF constraint vacuously, because with
    S_flux(x) = 0 both sides read 0 <= 0. Such a mixture has no UMF at all, so
    it is not really a member of the region, and it is admitted here anyway. In
    practice it is excluded the moment the target names one non flux oxide with
    an upper bound (SiO2 <= 3.3 * 0 = 0 fails for any silica at all), which
    every real target does; it survives only for a target made of fluxes alone,
    where it can widen a material range. Fixing it would take a lower bound on
    the flux sum, and there is no non arbitrary number to put there.
    """
    try:
        return _achievable_ranges(target_umf, materials, oxide_constraints,
                                  material_constraints, tol)
    except Exception as exc:
        logger.exception(f"feasibility: achievable ranges failed: {exc}")
        return {"feasible": None, "error": "ranges_failed", "message": str(exc)}


def _bounds_pair(value, warnings, what):
    """
    Read a [lo, hi] constraint, tolerating the shapes a JSON request arrives in

    Returns (lo, hi) with None for an absent bound, or None when the value
    cannot be read as a pair at all.
    """
    if isinstance(value, dict):
        pair = (value.get('min'), value.get('max'))
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        pair = (value[0], value[1])
    else:
        warnings.append(f"ограничение {what} не распознано и пропущено")
        return None

    cleaned = []
    for bound in pair:
        if bound is None:
            cleaned.append(None)
            continue
        try:
            number = float(bound)
        except (TypeError, ValueError):
            warnings.append(f"ограничение {what} не распознано и пропущено")
            return None
        if not math.isfinite(number):
            cleaned.append(None)
        else:
            cleaned.append(number)

    return cleaned[0], cleaned[1]


def _achievable_ranges(target_umf, materials, oxide_constraints, material_constraints, tol):
    warnings = []

    target, dropped_target = _usable_target(target_umf)
    if dropped_target:
        warnings.append(f"оксиды цели не распознаны и не учтены: {', '.join(dropped_target)}")

    if target:
        # Same rescaling as check_feasibility, and for the same reason: the box
        # below is derived from the target, and the LPs live on S_flux(x) = 1
        normalized, flux_sum = _normalized_target(target)
        if normalized is None:
            return {"feasible": None, "error": "no_target_fluxes",
                    "message": "target UMF carries no flux, there is no unity to normalize by",
                    "warnings": warnings}
        target = normalized
        if flux_sum != 1.0:
            warnings.append(f"сумма плавней цели {_fmt(flux_sum)}, а не 1 — "
                            f"формула приведена к единице перед расчётом диапазонов")

    kept_materials = filter_materials_with_formula(materials or [])
    if not kept_materials:
        return {"feasible": None, "error": "no_usable_materials",
                "message": "no material with an oxide analysis in the set",
                "warnings": warnings}

    requested_oxides = list(oxide_constraints or {})
    oxides = usable_oxides(list(target) + requested_oxides + _material_oxides(kept_materials))
    A, material_names = build_molar_matrix(kept_materials, oxides)

    if not _finite_matrix(A):
        return {"feasible": None, "error": "nonfinite_analysis",
                "message": "a material analysis contains NaN or infinity",
                "warnings": warnings}

    flux = flux_row(oxides)
    flux_vec = _matmul(flux, A)

    if not np.any(flux_vec > 0):
        return {"feasible": None, "error": "no_fluxes",
                "message": "no material of the set carries a flux, UMF is undefined",
                "warnings": warnings}

    n = A.shape[1]
    index = {oxide: i for i, oxide in enumerate(oxides)}
    material_index = {name: j for j, name in enumerate(material_names)}

    # The box the target itself imposes, then the caller's overrides on top
    bounds_by_oxide = {}
    for oxide, value in target.items():
        scale = max(value, OXIDE_SCALE_FLOOR)
        bounds_by_oxide[oxide] = (max(0.0, value - tol * scale), value + tol * scale)

    for oxide, raw in (oxide_constraints or {}).items():
        if oxide not in index:
            warnings.append(f"ограничение по оксиду {oxide} пропущено: оксид не распознан")
            continue
        pair = _bounds_pair(raw, warnings, f"по оксиду {oxide}")
        if pair is not None:
            bounds_by_oxide[oxide] = pair

    bounds_by_material = {}
    for name, raw in (material_constraints or {}).items():
        if name not in material_index:
            warnings.append(f"ограничение по материалу «{name}» пропущено: "
                            f"материала нет в наборе")
            continue
        pair = _bounds_pair(raw, warnings, f"по материалу «{name}»")
        if pair is not None:
            bounds_by_material[name] = pair

    ones = np.ones(n)
    A_ub = []
    b_ub = []

    for oxide, (lo, hi) in bounds_by_oxide.items():
        i = index[oxide]
        if hi is not None:
            A_ub.append(A[i] - hi * flux_vec)
            b_ub.append(0.0)
        if lo is not None and lo > 0.0:
            A_ub.append(lo * flux_vec - A[i])
            b_ub.append(0.0)

    for name, (lo, hi) in bounds_by_material.items():
        column = np.zeros(n)
        column[material_index[name]] = 100.0
        if hi is not None:
            A_ub.append(column - hi * ones)
            b_ub.append(0.0)
        if lo is not None and lo > 0.0:
            A_ub.append(lo * ones - column)
            b_ub.append(0.0)

    zero_objective = np.zeros(n)
    variable_bounds = [(0.0, None)] * n
    lp_count = 0

    # One probe first, on the sum(x) = 100 normalization: it answers "is the
    # feasible region empty" once, so that a contradiction costs one LP instead
    # of seventy. It also hands back the example recipe for free.
    probe = _linprog(zero_objective, A_ub, b_ub, [ones], [100.0], variable_bounds)
    lp_count += 1

    if probe.status == LP_INFEASIBLE:
        return {"feasible": False, "oxide_ranges": {}, "material_ranges": {},
                "example_recipe": {}, "lp_count": lp_count, "warnings": warnings}

    if probe.status != LP_OPTIMAL:
        logger.warning(f"feasibility: range probe did not converge, status={probe.status} "
                       f"({probe.message})")
        return {"feasible": None, "error": "lp_not_converged",
                "message": f"status {probe.status}: {probe.message}",
                "lp_count": lp_count, "warnings": warnings}

    example_recipe = _as_recipe(np.asarray(probe.x), material_names)

    oxide_ranges = {}
    for oxide in oxides:
        i = index[oxide]
        interval = []
        for maximize in (False, True):
            c = -A[i] if maximize else A[i]
            result = _linprog(c, A_ub, b_ub, [flux_vec], [1.0], variable_bounds)
            lp_count += 1
            if result.status == LP_INFEASIBLE:
                return {"feasible": False, "oxide_ranges": {}, "material_ranges": {},
                        "example_recipe": {}, "lp_count": lp_count, "warnings": warnings}
            if result.status == LP_UNBOUNDED:
                interval.append(None)
                continue
            if result.status != LP_OPTIMAL:
                logger.warning(f"feasibility: range LP for {oxide} did not converge, "
                               f"status={result.status}")
                return {"feasible": None, "error": "lp_not_converged",
                        "message": f"{oxide}: status {result.status}: {result.message}",
                        "lp_count": lp_count, "warnings": warnings}
            interval.append(float(-result.fun if maximize else result.fun))
        oxide_ranges[oxide] = interval

    material_ranges = {}
    for name in material_names:
        j = material_index[name]
        column = np.zeros(n)
        column[j] = 1.0
        interval = []
        for maximize in (False, True):
            c = -column if maximize else column
            result = _linprog(c, A_ub, b_ub, [ones], [100.0], variable_bounds)
            lp_count += 1
            if result.status == LP_INFEASIBLE:
                return {"feasible": False, "oxide_ranges": {}, "material_ranges": {},
                        "example_recipe": {}, "lp_count": lp_count, "warnings": warnings}
            if result.status != LP_OPTIMAL:
                # sum(x) = 100 with x >= 0 bounds every share into [0, 100], so
                # unbounded cannot happen here - unlike for an oxide
                logger.warning(f"feasibility: range LP for material {name} did not "
                               f"converge, status={result.status}")
                return {"feasible": None, "error": "lp_not_converged",
                        "message": f"{name}: status {result.status}: {result.message}",
                        "lp_count": lp_count, "warnings": warnings}
            interval.append(float(-result.fun if maximize else result.fun))
        material_ranges[name] = interval

    return {"feasible": True, "oxide_ranges": oxide_ranges,
            "material_ranges": material_ranges, "example_recipe": example_recipe,
            "lp_count": lp_count, "warnings": warnings}
