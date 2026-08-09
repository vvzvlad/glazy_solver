#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# flake8: noqa
# pylint: disable=broad-exception-raised, raise-missing-from, too-many-arguments, redefined-outer-name
# pylint: disable=multiple-statements, logging-fstring-interpolation, trailing-whitespace, line-too-long
# pylint: disable=broad-exception-caught, missing-function-docstring, missing-class-docstring
# pylint: disable=f-string-without-interpolation
# pylance: disable=reportMissingImports, reportMissingModuleSource
# type: ignore

# "What does this recipe rest on" - sensitivity of the UMF of a recipe to the
# uncertainty of the material analyses it is built from.
#
# The question this answers is the one a direct calculation never does: the
# formula is computed from database/materials.json as if those numbers were
# exact, while in reality datasheets disagree with each other and batches drift.
# One nepheline can come with two different certificates, frits vary from melt
# to melt, and the degree of hydration of ulexite or borax moves the B2O3 it
# actually brings.
#
# The ranking is by sensitivity TIMES uncertainty, not by sensitivity alone.
# Quartz has an enormous lever on SiO2, but its analysis (99-100% SiO2) never
# lies; ulexite has a smaller lever and a much wider spread - and it is the
# dangerous one. A ranking by lever alone would name quartz every single time
# and would be useless.
#
# Method (one sigma, linear propagation, no Monte Carlo):
#   1. u0 = weights_to_umf(calculate_recipe_composition(...)) - the base formula.
#   2. For every material j of the recipe and every oxide i it carries, perturb
#      that ONE cell of the analysis: A[i][j] -> A[i][j] * (1 + sigma_ij), and
#      recompute the whole UMF. The response delta_u^(ij) = u' - u0 is already
#      the response to exactly one sigma, so no separate derivative step exists.
#   3. Spread of the result:  sigma(u_k) = sqrt(sum_ij (delta_u_k^(ij))^2),
#      treating the perturbations as independent.
#   4. Contribution of a material:
#         contribution_j = sum_i sum_k (delta_u_k^(ij) / s_k)^2,
#      with s_k = max(u0_k, OXIDE_SCALE_FLOOR). The floor is what keeps a trace
#      oxide from dominating the ranking: without it a CoO of 0.001 moving by
#      50% would outweigh SiO2 moving by 1%.
#
# The perturbed material formula is deliberately NOT renormalized. Material
# formulas in our database sum to less than 100 - the remainder is the implicit
# LOI - so raising one oxide simply eats into that remainder. That is exactly
# the physical model wanted here: "this batch carries a bit more boron and a bit
# less water". Renormalizing would instead model "more boron and correspondingly
# less of everything else", which is a different and wrong statement.
#
# This module stays Flask free, like feasibility/glazy_import: it returns plain
# dictionaries and the API layer decides what to do with an "error" in them.

import json
import logging
import math
import os

from common import (
    load_molar_masses,
    oxides_classification,
    weights_to_umf,
)
from solver_classic import calculate_recipe_composition

logger = logging.getLogger(__name__)

# Same floor as the feasibility LP of TZ_SOLVER_V2.md, section 2.1: an oxide is
# scored against max(its own value, 0.1), so that SiO2 ~ 3 and MgO ~ 0.05 are
# compared on a common footing instead of the trace oxide taking over.
OXIDE_SCALE_FLOOR = 0.1

# Fallback when material_tolerance.json cannot be read at all. Deliberately the
# same value as the "default_relative" of the shipped file: a missing tolerance
# database degrades the answer to "every material is equally trustworthy", which
# still ranks by lever, instead of failing the request.
FALLBACK_RELATIVE = 0.05

# ... but that degradation changes the ranking qualitatively - on the reference
# clear glaze ulexite drops from 0.70 to 0.27 and stops being the leader - and a
# flat sigma leaves exactly the "by lever alone" answer the header of this file
# calls useless. A log line is not enough: the caller sees a perfectly valid
# looking response and cannot tell the two apart, so the fact travels in
# "warnings" as well.
DEGRADED_TOLERANCES_WARNING = (
    "база допусков недоступна, все материалы считаются одинаково надёжными — "
    "ранжирование только по плечу")

# The same for a recipe no material of which can move the formula at all: the
# shares are then honestly zero and do NOT sum to 1.0, so a consumer normalizing
# by their sum would divide by zero without ever being told why.
ZERO_CONTRIBUTION_WARNING = (
    "формулу рецепта не сдвигает ни один материал: unity-базис UMF задан "
    "единственным оксидом, любое отклонение паспорта уходит в сам базис. "
    "Все доли равны нулю и в сумме дают 0, а не 1.0")

# Below this share of fluxes among all the moles of the recipe, the unity basis
# of the UMF rests on traces and the whole formula is numerically unstable. Real
# glazes sit around 0.19-0.21 (measured on the reference recipes); the manganese
# metallic of DATA_NOTES.md, section 2, where MnO2 falls outside every group of
# oxides_classification(), sits at 0.0037 and blows the UMF up by ~117x.
LOW_FLUX_FRACTION = 0.02

# Fluxes at or below this many moles per 100 g of recipe mean there is no unity
# basis at all: weights_to_umf would silently fall back to "the smallest oxide
# is unity", an arbitrary basis on which a sensitivity number means nothing.
ZERO_FLUX_MOLES = 1e-9

# How many oxides of the RESULT are named as the ones a material moves, and how
# small a share of that material's own contribution still earns a mention. The
# cut keeps the list honest: an oxide moved by 1% of what the leader is moved by
# is not something the material "affects" in any useful sense.
AFFECTS_LIMIT = 3
AFFECTS_MIN_SHARE = 0.05

# Decimals kept on a material share, see the comment at the normalization step
SHARE_DIGITS = 9


def _default_tolerances_path():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, 'database', 'material_tolerance.json')


def load_tolerances(path=None):
    """
    Read database/material_tolerance.json with class defaults and overrides

    The file is re-read on every call and deliberately not cached, unlike the
    molar masses of common.py: material_tolerance.md tells the user to edit it
    by hand ("set a small sigma and the material drops down the ranking"), and
    a process wide cache would make that edit take effect only after a restart
    of the server. It is ~1.4 KB against a full UMF recalculation per (material,
    oxide) pair, so the read is not what this endpoint spends its time on.

    Args:
        path: optional path to an alternative tolerance file

    Returns:
        dictionary {"default_relative": float, "classes": {...}, "materials": {...},
        "degraded": bool}; "degraded" is True when the file could not be read and
        every material fell back to one flat sigma
    """
    return _read_tolerances(path if path is not None else _default_tolerances_path())


def _fallback_tolerances():
    return {"default_relative": FALLBACK_RELATIVE, "classes": {}, "materials": {},
            "degraded": True}


def _read_tolerances(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (OSError, ValueError) as exc:
        # A broken tolerance file must not take the whole endpoint down: the
        # ranking is still meaningful with one flat sigma for everything - as
        # long as the answer says so, see DEGRADED_TOLERANCES_WARNING.
        logger.warning(f"material_tolerance_unreadable: {path}: {exc}")
        return _fallback_tolerances()

    if not isinstance(data, dict):
        logger.warning(f"material_tolerance_malformed: {path}: expected an object")
        return _fallback_tolerances()

    return {
        "default_relative": _positive_float(data.get('default_relative'), FALLBACK_RELATIVE),
        "classes": data.get('classes') if isinstance(data.get('classes'), dict) else {},
        "materials": data.get('materials') if isinstance(data.get('materials'), dict) else {},
        "degraded": False,
    }


def _positive_float(value, fallback):
    """A tolerance is a relative spread: anything not a positive number is junk"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback

    if not math.isfinite(number) or number <= 0:
        return fallback

    return number


def material_sigma(material, tolerances):
    """
    Relative spread of every oxide of one material record

    Resolution order, most specific first: an explicit per-oxide override, the
    tolerance of the class the material is assigned to, the global default.

    Args:
        material: material dictionary from materials.json
        tolerances: dictionary as returned by load_tolerances()

    Returns:
        dictionary {oxide: relative_sigma} covering the oxides the material
        actually carries; empty for a material with an empty formula
    """
    formula = material.get('formula') or {}
    if not formula:
        return {}

    default_relative = _positive_float(tolerances.get('default_relative'), FALLBACK_RELATIVE)
    classes = tolerances.get('classes') or {}
    entry = (tolerances.get('materials') or {}).get(material.get('name'))
    if not isinstance(entry, dict):
        entry = {}

    class_name = entry.get('class')
    base_sigma = _positive_float(classes.get(class_name), default_relative)

    overrides = entry.get('oxides') if isinstance(entry.get('oxides'), dict) else {}

    sigmas = {}
    for oxide, content in formula.items():
        try:
            content_value = float(content)
        except (TypeError, ValueError):
            continue
        if content_value == 0:
            # A zero cell cannot be perturbed multiplicatively, and an oxide the
            # material does not carry has no spread to speak of
            continue
        sigmas[oxide] = _positive_float(overrides.get(oxide), base_sigma)

    return sigmas


def _flux_moles(weight_composition, molar_masses, flux_oxides):
    """Moles of fluxes and total moles of a weight composition"""
    flux_sum = 0.0
    total_sum = 0.0

    for oxide, weight in weight_composition.items():
        molar_mass = molar_masses.get(oxide)
        if not molar_mass:
            continue
        moles = weight / molar_mass
        total_sum += moles
        if oxide in flux_oxides:
            flux_sum += moles

    return flux_sum, total_sum


def _empty_result(warnings, error, message):
    return {
        "umf": {},
        "per_oxide": [],
        "by_material": [],
        "warnings": warnings,
        "error": error,
        "message": message,
    }


def recipe_sensitivity(recipe, materials, tolerances=None):
    """
    Rank materials by their contribution to the spread of the recipe UMF

    Args:
        recipe: dictionary {material_name: weight_percent}
        materials: list of material dictionaries to resolve the names against
        tolerances: optional dictionary from load_tolerances(); read from the
            default file when omitted. A "degraded" one turns into the first
            warning of the answer: the numbers are then a ranking by lever only

    Returns:
        {
          "umf": {oxide: value},          # base formula of the recipe
          "per_oxide": [{"oxide", "value", "sigma", "relative"}, ...],
          "by_material": [{"material", "share", "via_oxide", "sigma_used",
                           "affects"}, ...],
          "warnings": [str, ...],
          "error": None                   # a code when nothing could be computed
        }

        The shares of "by_material" sum to 1.0, except when no material can move
        the formula at all - then every share is 0.0 and ZERO_CONTRIBUTION_WARNING
        is in "warnings".
    """
    if tolerances is None:
        tolerances = load_tolerances()

    warnings = []

    if tolerances.get('degraded'):
        warnings.append(DEGRADED_TOLERANCES_WARNING)

    if not recipe:
        return _empty_result(warnings, "empty_recipe", "the recipe carries no materials")

    by_name = {}
    for material in materials or []:
        name = material.get('name')
        if name is not None and name not in by_name:
            by_name[name] = material

    # Resolve the recipe once. A name the database does not know is skipped with
    # a warning rather than raising: the rest of the recipe is still worth an
    # answer, and the warning says why the numbers cover only part of it.
    used = []
    for material_name, percentage in recipe.items():
        try:
            amount = float(percentage)
        except (TypeError, ValueError):
            warnings.append(f"«{material_name}»: доля «{percentage}» не число, материал пропущен")
            logger.warning(f"sensitivity_invalid_percentage: {material_name}")
            continue

        if not math.isfinite(amount):
            # 1e400 is valid JSON and Python parses it into inf; the "amount <= 0"
            # test below lets both inf and nan through (nan <= 0 is False) and
            # every number downstream would come out as nan, which serializes
            # into a "NaN" string sitting in a documented numeric field.
            warnings.append(f"«{material_name}»: доля «{percentage}» не конечное число, "
                            f"материал пропущен")
            logger.warning(f"sensitivity_nonfinite_percentage: {material_name}")
            continue

        material = by_name.get(material_name)
        if material is None:
            warnings.append(f"«{material_name}» нет в базе материалов, материал пропущен")
            logger.warning(f"sensitivity_unknown_material: {material_name}")
            continue

        if amount <= 0:
            warnings.append(f"«{material_name}»: доля {amount}, материал не влияет на формулу")
            continue

        used.append((material_name, material, amount))

    if not used:
        return _empty_result(warnings, "no_known_materials",
                             "ни один материал рецепта не найден в базе")

    resolved_recipe = {name: amount for name, _material, amount in used}
    resolved_materials = [material for _name, material, _amount in used]

    base_composition = calculate_recipe_composition(resolved_materials, resolved_recipe)
    if not base_composition:
        return _empty_result(warnings, "empty_composition",
                             "у материалов рецепта нет ни одного оксида в формуле")

    molar_masses = load_molar_masses()
    classes = oxides_classification()
    flux_oxides = set(classes['r2o'] + classes['ro'])

    flux_moles, total_moles = _flux_moles(base_composition, molar_masses, flux_oxides)

    if flux_moles <= ZERO_FLUX_MOLES:
        # weights_to_umf would fall back to "the smallest oxide is unity" here.
        # That basis is arbitrary and a sensitivity computed on it would be a
        # number without a meaning, so nothing is returned instead.
        logger.warning("sensitivity_no_fluxes: the recipe carries no fluxes of oxides_classification()")
        return _empty_result(
            warnings, "no_fluxes",
            "в рецепте нет флюсов (R2O/RO), unity-базис UMF не определён — "
            "чувствительность посчитать нельзя")

    if total_moles > 0 and flux_moles / total_moles < LOW_FLUX_FRACTION:
        message = (f"сумма флюсов подозрительно мала: {flux_moles / total_moles:.1%} "
                   f"от всех молей рецепта, UMF стоит на следовых оксидах и сильно раздут")
        warnings.append(message)
        logger.warning(f"sensitivity_low_flux_sum: flux_fraction={flux_moles / total_moles:.5f}")

    base_umf = weights_to_umf(base_composition, round_digits=None)
    if not base_umf:
        # Unreachable at the current order of checks: the flux check above only
        # passes when at least one flux oxide is both in molar_masses.json and
        # carried in a non-zero amount, and weights_to_umf keeps every such
        # oxide. Kept as a guard, and deliberately NOT advertised in API.md as a
        # response a caller can receive.
        return _empty_result(warnings, "empty_umf",
                             "ни один оксид рецепта не найден в таблице молярных масс")

    result_oxides = list(base_umf.keys())
    scales = {oxide: max(base_umf[oxide], OXIDE_SCALE_FLOOR) for oxide in result_oxides}

    # Accumulated squared response per result oxide, over every perturbation
    variance = {oxide: 0.0 for oxide in result_oxides}

    material_rows = []
    for material_name, material, amount in used:
        sigmas = material_sigma(material, tolerances)
        formula = material.get('formula') or {}

        if not sigmas:
            # An empty formula is a legal record in this database (pigments,
            # SiC, CMC, water, gypsum): it carries no oxides, so it cannot move
            # the formula at all.
            material_rows.append({
                "material": material_name,
                "contribution": 0.0,
                "via_oxide": None,
                "sigma_used": None,
                "affects": [],
            })
            continue

        total_contribution = 0.0
        best_oxide = None
        best_oxide_contribution = -1.0
        best_oxide_sigma = None
        per_result_oxide = {oxide: 0.0 for oxide in result_oxides}

        for oxide, sigma in sigmas.items():
            if oxide not in molar_masses:
                # An oxide missing from molar_masses.json never reaches the UMF,
                # so perturbing it provably changes nothing
                continue

            # Only one cell of the analysis moves, so only one entry of the
            # weight composition changes: A[i][j] * sigma * (percent / 100).
            delta_weight = float(formula[oxide]) * sigma * (amount / 100.0)
            perturbed_composition = dict(base_composition)
            perturbed_composition[oxide] = perturbed_composition.get(oxide, 0.0) + delta_weight

            perturbed_umf = weights_to_umf(perturbed_composition, round_digits=None)

            oxide_contribution = 0.0
            for result_oxide in result_oxides:
                delta = perturbed_umf.get(result_oxide, 0.0) - base_umf[result_oxide]
                variance[result_oxide] += delta * delta
                scaled = delta / scales[result_oxide]
                oxide_contribution += scaled * scaled
                per_result_oxide[result_oxide] += scaled * scaled

            total_contribution += oxide_contribution

            if oxide_contribution > best_oxide_contribution:
                best_oxide_contribution = oxide_contribution
                best_oxide = oxide
                best_oxide_sigma = sigma

        material_rows.append({
            "material": material_name,
            "contribution": total_contribution,
            "via_oxide": best_oxide,
            "sigma_used": best_oxide_sigma,
            "affects": _top_affected(per_result_oxide, total_contribution),
        })

    # A total of zero means nothing in the recipe can move the formula at all:
    # either every material has an empty formula, or the only oxide the recipe
    # carries is a flux and so IS the unity basis by itself (a recipe of pure
    # chalk is always exactly CaO 1.0). Every share is then honestly zero and
    # the "shares sum to 1" invariant does not apply - which is worth saying out
    # loud, because a consumer normalizing by that sum divides by zero.
    #
    # SHARE_DIGITS and not something shorter because the shares are documented
    # to sum to 1.0 within 1e-6: rounding to 6 decimals lets the per-row error
    # accumulate to 3e-6 on a ten material recipe (measured), which would break
    # exactly the invariant a consumer would rely on.
    total = sum(row["contribution"] for row in material_rows)
    if total <= 0:
        warnings.append(ZERO_CONTRIBUTION_WARNING)
        logger.warning("sensitivity_zero_total_contribution: no material moves the UMF")
    by_material = [{
        "material": row["material"],
        "share": round(row["contribution"] / total, SHARE_DIGITS) if total > 0 else 0.0,
        "via_oxide": row["via_oxide"],
        "sigma_used": row["sigma_used"],
        "affects": row["affects"],
    } for row in material_rows]

    by_material.sort(key=lambda row: row["share"], reverse=True)

    per_oxide = []
    for oxide in result_oxides:
        value = base_umf[oxide]
        sigma = math.sqrt(variance[oxide])
        per_oxide.append({
            "oxide": oxide,
            "value": round(value, 4),
            "sigma": round(sigma, 5),
            "relative": round(sigma / value, 4) if value > 0 else None,
        })

    # Sorted by relative spread: the point of the list is "which oxide of the
    # result is the least trustworthy", and that is a relative question
    per_oxide.sort(key=lambda item: item["relative"] if item["relative"] is not None else -1.0,
                   reverse=True)

    return {
        "umf": {oxide: round(value, 3) for oxide, value in base_umf.items()},
        "per_oxide": per_oxide,
        "by_material": by_material,
        "warnings": warnings,
        "error": None,
    }


def _top_affected(per_result_oxide, total_contribution):
    """Oxides of the result this material moves the most, strongest first"""
    if total_contribution <= 0:
        return []

    ranked = sorted(per_result_oxide.items(), key=lambda item: item[1], reverse=True)

    affected = []
    for oxide, contribution in ranked[:AFFECTS_LIMIT]:
        if contribution / total_contribution < AFFECTS_MIN_SHARE and affected:
            break
        affected.append(oxide)

    return affected
