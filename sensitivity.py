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

# Fallback for a material_tolerance.json that cannot be read at all, and for a
# "default_relative" that is not a positive number. Deliberately the
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
#
# "Unavailable" here is about the CONTENT and not about the file handle: a file
# that parses into an object but carries no usable sigma at all (an empty object,
# a "classes"/"materials" typo, a section of the wrong type) gives numerically
# the same flat answer as a missing file, so it is the same degradation.
DEGRADED_TOLERANCES_WARNING = (
    "база допусков недоступна или непригодна, все материалы считаются одинаково "
    "надёжными — ранжирование только по плечу")

# The same for a recipe no material of which can move the formula at all: the
# shares are then honestly zero and do NOT sum to 1.0, so a consumer normalizing
# by their sum would divide by zero without ever being told why.
ZERO_CONTRIBUTION_WARNING = (
    "формулу рецепта не сдвигает ни один материал: unity-базис UMF задан "
    "единственным оксидом, любое отклонение паспорта уходит в сам базис. "
    "Все доли равны нулю и в сумме дают 0, а не 1.0")

# And the same shape of answer - every share 0.0 - for a total that came out as
# inf or nan. It is NOT the degenerate case above and must not be reported as
# one: nothing here says the materials cannot move the formula, only that the
# numbers stopped meaning anything.
NONFINITE_CONTRIBUTION_WARNING = (
    "сумма вкладов материалов не является конечным числом (переполнение при "
    "возведении отклика в квадрат), все доли обнулены — числам в этом ответе "
    "верить нельзя")

# Returned instead of a result any number of which is inf or nan. The guards on
# the input cannot cover every path to an overflow - the share is bounded above,
# but the sigma comes from a hand edited file and the variance squares whatever
# response it produces - so the invariant is checked where it actually holds or
# fails: on the way out. "Infinity" and "NaN" in a field documented as a number
# are worse than a refusal, because make_json_safe turns them into strings and
# the caller sees error: null right next to them.
NONFINITE_RESULT_MESSAGE = (
    "расчёт дал бесконечность или NaN: числа не имеют смысла и не возвращаются. "
    "Проверьте доли рецепта и сигмы в database/material_tolerance.json")

# Below this share of fluxes among all the moles of the recipe, the unity basis
# of the UMF rests on traces and the whole formula is numerically unstable. Real
# glazes sit around 0.19-0.21 (measured on the reference recipes); a body that
# carries no flux of its own - kaolin КЖФ-1 60 / quartz W12 40, whose only R2O/RO
# are the traces of the kaolin analysis - sits at 0.0064 and stands its SiO2 at
# 131 instead of ~3. The manganese metallic of DATA_NOTES.md, section 2, used to
# be the example here at 0.0037 and ~117x, but that was an artifact of MnO2
# falling outside every group of oxides_classification(): MnO2 is a flux now, the
# recipe measures 0.427 and is perfectly ordinary - do not restore the old
# numbers from the history.
LOW_FLUX_FRACTION = 0.02

# Fluxes at or below this many moles per 100 g of recipe mean there is no unity
# basis at all: weights_to_umf would silently fall back to "the smallest oxide
# is unity", an arbitrary basis on which a sensitivity number means nothing.
ZERO_FLUX_MOLES = 1e-9

# A share of the recipe is a weight percent, so anything above a million is not a
# recipe any more. The bound is not cosmetic: the variance accumulates SQUARES of
# the response, so a share of ~1e157 already squares into inf while the share
# itself is still a perfectly finite float - the "is it finite" test on the input
# passes and the answer comes back with "sigma": Infinity under error: null.
MAX_PERCENTAGE = 1e6

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
        "degraded": bool, "issues": [str, ...]}

        "degraded" is True when nothing usable came out of the file and every
        material fell back to one flat sigma - whether because the file could not
        be read at all or because its content turned out to carry no sigma
        (see _usable_sigma_count). "issues" describes the parts that were dropped
        while the rest still worked, in the language of the response.
    """
    return _read_tolerances(path if path is not None else _default_tolerances_path())


def _fallback_tolerances():
    return {"default_relative": FALLBACK_RELATIVE, "classes": {}, "materials": {},
            "degraded": True, "issues": []}


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

    issues = []

    default_relative = _positive_float(data.get('default_relative'), FALLBACK_RELATIVE)
    if 'default_relative' in data and not _is_usable_sigma(data.get('default_relative')):
        # Substituting a value silently is what made a file with a single typo
        # indistinguishable from the shipped one
        logger.warning(f"material_tolerance_bad_default: {path}: "
                       f"{data.get('default_relative')!r}, using {FALLBACK_RELATIVE}")
        issues.append(f"в базе допусков значение default_relative непригодно, "
                      f"взято {FALLBACK_RELATIVE}")

    classes = _object_section(data, 'classes', path, issues)
    materials = _object_section(data, 'materials', path, issues)

    usable = _usable_sigma_count(classes, materials, path, issues)

    if not usable:
        # The file parses, but not one sigma of it survives: every material ends
        # up on the same default and the answer is bit for bit the answer of a
        # missing file. That is the same degradation and gets the same flag - the
        # numbers, not the exception, are what the caller has to be told about.
        # What was parsed is still returned as it stands: it changes nothing (it
        # is unreachable by construction), and "default_relative" is the sigma
        # every material will actually be given, so replacing it with the
        # built-in fallback would answer with a spread the file never asked for.
        logger.warning(f"material_tolerance_unusable: {path}: no per class or per "
                       f"oxide sigma survived parsing")

    return {
        "default_relative": default_relative,
        "classes": classes,
        "materials": materials,
        "degraded": not usable,
        "issues": issues,
    }


def _object_section(data, key, path, issues):
    """
    One object valued section of the tolerance file

    A section of the wrong type used to be replaced by {} without a word, so a
    "materials" written as a list came back as a healthy looking answer computed
    from nothing.
    """
    value = data.get(key)

    if value is None:
        return {}

    if not isinstance(value, dict):
        logger.warning(f"material_tolerance_section_ignored: {path}: '{key}' is a "
                       f"{type(value).__name__}, expected an object")
        issues.append(f"в базе допусков секция «{key}» неверного типа и не учтена")
        return {}

    return value


def _usable_sigma_count(classes, materials, path, issues):
    """
    How many sigmas of the file actually differ from the flat default

    This is what tells "the tolerance database works" from "the tolerance
    database parses": material_sigma() only ever reads a sigma through a class a
    material is assigned to or through an explicit per oxide override, so a file
    where neither resolves gives every material the same default_relative - the
    ranking by lever alone this module exists to avoid.

    Returns:
        number of usable sigma sources; appends to "issues" the parts that were
        dropped while others survived
    """
    resolved_classes = 0
    unresolved_classes = 0
    broken_entries = 0
    overrides = 0

    for name, entry in materials.items():
        if not isinstance(entry, dict):
            logger.warning(f"material_tolerance_entry_ignored: {path}: '{name}' is a "
                           f"{type(entry).__name__}, expected an object")
            broken_entries += 1
            continue

        class_name = entry.get('class')
        if class_name is not None:
            if _is_usable_sigma(classes.get(class_name)):
                resolved_classes += 1
            else:
                unresolved_classes += 1

        oxides = entry.get('oxides')
        if isinstance(oxides, dict):
            overrides += sum(1 for value in oxides.values() if _is_usable_sigma(value))

    usable = resolved_classes + overrides

    # Only worth saying when something else still works: with usable == 0 the
    # caller gets DEGRADED_TOLERANCES_WARNING, which says strictly more.
    if usable and unresolved_classes:
        logger.warning(f"material_tolerance_unresolved_classes: {path}: "
                       f"{unresolved_classes} entries point at a class without a sigma")
        issues.append(f"в базе допусков {unresolved_classes} материалов ссылаются на "
                      f"класс без пригодной сигмы, для них взята сигма по умолчанию")

    if usable and broken_entries:
        issues.append(f"в базе допусков {broken_entries} записей материалов неверного "
                      f"типа и не учтены")

    return usable


def _is_usable_sigma(value):
    """True when _positive_float() keeps the value instead of falling back"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False

    return math.isfinite(number) and number > 0


def _positive_float(value, fallback):
    """A tolerance is a relative spread: anything not a positive number is junk"""
    return float(value) if _is_usable_sigma(value) else fallback


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
        if not math.isfinite(content_value):
            # materials.json is written by glazy_import and edited by hand too;
            # an inf cell would be perturbed into an inf weight and poison every
            # oxide of the result, not only its own
            logger.warning(f"sensitivity_nonfinite_formula_cell: "
                           f"{material.get('name')}: {oxide}={content!r}")
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
        tolerances: a dictionary from load_tolerances() and nothing else; read
            from the default file when omitted. The state of the tolerance
            database travels in its "degraded"/"issues" keys and turns into the
            first warnings of the answer, so a hand built dictionary carrying
            neither is taken at face value and reports no degradation - which is
            fine for a caller that built it on purpose and wrong for one that
            hoped this function would check the file for it.

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
        is in "warnings". Every number of the result is finite: an overflow
        anywhere is answered with the "nonfinite_result" error instead.
    """
    if tolerances is None:
        tolerances = load_tolerances()

    warnings = []

    if tolerances.get('degraded'):
        warnings.append(DEGRADED_TOLERANCES_WARNING)

    # A partly usable file is not degraded - most of the ranking still stands -
    # but the part that was dropped changed the numbers and cannot stay silent
    warnings.extend(tolerances.get('issues') or [])

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

        if abs(amount) > MAX_PERCENTAGE:
            # Being finite is not enough: the variance below squares the response
            # to this share, and a finite 1e160 squares into inf. See MAX_PERCENTAGE.
            warnings.append(f"«{material_name}»: доля {amount} выходит за разумные пределы "
                            f"(больше {MAX_PERCENTAGE:g}%), материал пропущен")
            logger.warning(f"sensitivity_percentage_out_of_range: {material_name}: {amount}")
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
        # Deliberately not "not found in the database": a material that IS in the
        # database and was dropped for an unusable share lands here too, and the
        # old wording sent the caller looking for a name that is perfectly fine
        return _empty_result(warnings, "no_known_materials",
                             "ни один материал рецепта не пригоден для расчёта, "
                             "подробности в warnings")

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

    by_material, share_warning = _material_shares(material_rows)
    if share_warning:
        warnings.append(share_warning)

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

    result = {
        "umf": {oxide: round(value, 3) for oxide, value in base_umf.items()},
        "per_oxide": per_oxide,
        "by_material": by_material,
        "warnings": warnings,
        "error": None,
    }

    # The invariant, checked where it can actually be checked. Every guard above
    # sits on an input and none of them can promise this one: the response is
    # squared, summed, divided and rooted on the way here, and any of those steps
    # can leave the finite range. See NONFINITE_RESULT_MESSAGE.
    if not _all_finite(result):
        logger.error("sensitivity_nonfinite_result: the computed answer carries inf or nan")
        return _empty_result(warnings, "nonfinite_result", NONFINITE_RESULT_MESSAGE)

    return result


def _material_shares(material_rows):
    """
    Normalize the contributions into shares, and say so when they do not add up

    A total of zero means nothing in the recipe can move the formula at all:
    either every material has an empty formula, or the only oxide the recipe
    carries is a flux and so IS the unity basis by itself (a recipe of pure chalk
    is always exactly CaO 1.0). Every share is then honestly zero and the "shares
    sum to 1" invariant does not apply - which is worth saying out loud, because
    a consumer normalizing by that sum divides by zero.

    A total that is inf or nan produces the same shape of answer and must go down
    the same branch: "nan <= 0" is False, so a test on "<= 0" alone would call a
    poisoned total healthy, and "nan > 0" is False as well, so the shares would
    come out zero anyway - the exact degenerate answer with no explanation
    attached. It is reported apart from the zero case because it means something
    entirely different: not "no material moves the formula" but "the numbers
    stopped meaning anything".

    SHARE_DIGITS and not something shorter because the shares are documented to
    sum to 1.0 within 1e-6: rounding to 6 decimals lets the per-row error
    accumulate to 3e-6 on a ten material recipe (measured), which would break
    exactly the invariant a consumer would rely on.

    Returns:
        (rows, warning) - the "by_material" list sorted by share, and the warning
        the answer needs, or None when the shares do sum to 1.0
    """
    total = sum(row["contribution"] for row in material_rows)
    usable = math.isfinite(total) and total > 0

    rows = [{
        "material": row["material"],
        "share": round(row["contribution"] / total, SHARE_DIGITS) if usable else 0.0,
        "via_oxide": row["via_oxide"],
        "sigma_used": row["sigma_used"],
        "affects": row["affects"],
    } for row in material_rows]

    rows.sort(key=lambda row: row["share"], reverse=True)

    if usable:
        return rows, None

    if math.isfinite(total):
        logger.warning("sensitivity_zero_total_contribution: no material moves the UMF")
        return rows, ZERO_CONTRIBUTION_WARNING

    logger.error(f"sensitivity_nonfinite_total_contribution: total={total}")
    return rows, NONFINITE_CONTRIBUTION_WARNING


def _all_finite(value):
    """Every number anywhere in a nested result structure is a finite float"""
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())

    if isinstance(value, list):
        return all(_all_finite(item) for item in value)

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return True

    return math.isfinite(value)


def _top_affected(per_result_oxide, total_contribution):
    """Oxides of the result this material moves the most, strongest first"""
    if not (math.isfinite(total_contribution) and total_contribution > 0):
        # Same reason as in _material_shares: with a nan total every comparison
        # below is False, so the "share of the leader" cut never fires and the
        # material comes back claiming to affect three oxides while its own share
        # is 0.0 and its via_oxide is null
        return []

    ranked = sorted(per_result_oxide.items(), key=lambda item: item[1], reverse=True)

    affected = []
    for oxide, contribution in ranked[:AFFECTS_LIMIT]:
        if contribution / total_contribution < AFFECTS_MIN_SHARE and affected:
            break
        affected.append(oxide)

    return affected
