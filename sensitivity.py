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
# "default_relative" that is not a sane positive number. Deliberately the
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
# The fact is established by OBSERVATION, on the finished rows, and never by
# looking at the tolerance file and guessing what it will do. Three rounds of
# fixes tried the second way - each round a smarter count of what the file
# contains - and each round left an input that walked past it, because whether a
# sigma reaches the calculation depends on the recipe, on materials.json and on
# which oxides the material actually carries, none of which the file knows. The
# renamed material and the override on an oxide the material does not have are
# the two that survived the last count, and both produce the answer of a missing
# file bit for bit. What counts as flat - every applied sigma equal to every
# OTHER applied sigma, and not to default_relative - is at the check itself, in
# recipe_sensitivity().
#
# The message says the FACT and stops there. Its predecessor went on to name
# three causes - "the tolerance database is unavailable, does not describe these
# materials or has drifted from the names in database/materials.json" - and on
# the shipped file all three are false at once: clay is 0.05 and so is
# default_relative, so a recipe of clays gets the flat answer out of a file that
# is present, complete and spelled exactly like materials.json. Whoever went to
# fix that file found nothing to fix. A cause is named below only where it was
# observed here, next to the answer it explains.
FLAT_SIGMA_WARNING = (
    "ранжирование идёт только по плечу: разброс паспортов не различает материалы "
    "этого рецепта")

# What was observed, appended to the line above. The number is part of the fact
# and worth naming: "все получили 0.02" is a sentence its reader can check
# against material_tolerance.json in a second, while "все получили одинаковую"
# leaves them to work out which one.
#
# "Материалы, способные сдвинуть формулу" and not "все материалы рецепта",
# because the two differ inside the very same answer: a material with an empty
# formula (a pigment, SiC, CMC, water, gypsum - 37 of the 216 entries) is
# perturbed nowhere, so it contributes no sigma to the set this message reports,
# and its by_material row says so with sigma_used: null. The wider wording was
# refuted by the row printed next to it.
FLAT_SIGMA_OBSERVED = (
    "все материалы, способные сдвинуть формулу, получили одну и ту же "
    "сигму {sigma:g}")
NO_SIGMA_OBSERVED = "ни одна сигма не вошла в расчёт"

# The one cause this module can actually see from here, and it is only said when
# it was seen: the tolerance file does describe materials, and not one of them is
# in this recipe. That is the renamed supplier material of the commit before, and
# it is a statement about these two files together - which is why it cannot be
# made at load time and is not in "issues".
FLAT_SIGMA_NO_NAME_MATCHED = (
    "Ни одно имя из базы допусков не совпало с материалами рецепта: либо она их "
    "не описывает, либо имена разошлись с database/materials.json")

# Said when the file itself never became a dictionary. It is a fact about the
# file and NOT a statement about the answer: what that costs the ranking is for
# FLAT_SIGMA_WARNING to say, after the ranking exists.
UNREADABLE_TOLERANCES_ISSUE = (
    "база допусков не прочитана: файл недоступен или испорчен, "
    "взяты значения по умолчанию")

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

# Returned instead of a result any number of which is inf or nan. The bounds on
# the share and on the sigma each close the path they know about and neither can
# promise this: an analysis cell of materials.json is a hand edited number too,
# the variance squares whatever response it produces, and the next such number is
# not on the list of the ones already thought of. So the invariant is checked
# where it actually holds or fails: on the way out. "Infinity" and "NaN" in a
# field documented as a number are worse than a refusal, because make_json_safe
# turns them into strings and the caller sees error: null right next to them.
NONFINITE_RESULT_MESSAGE = (
    "расчёт дал бесконечность или NaN: числа не имеют смысла и не возвращаются. "
    "Проверьте доли рецепта, формулы материалов в database/materials.json и "
    "сигмы в database/material_tolerance.json")

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

# A share of the recipe is a weight percent, so anything above a million is not a
# recipe any more. The bound is not cosmetic: the variance accumulates SQUARES of
# the response, so a share of ~1e157 already squares into inf while the share
# itself is still a perfectly finite float - the "is it finite" test on the input
# passes and the answer comes back with "sigma": Infinity under error: null.
MAX_PERCENTAGE = 1e6

# The same bound, on the other hand-edited number of this calculation. A sigma is
# a RELATIVE spread and material_tolerance.md describes it as "порядка сотых" -
# the loosest shipped class is wood ash at 0.20. At 1.0 the passport is declared
# wrong by 100%, which is already outside anything the file documents, and above
# it the value is a typo rather than a tolerance: a 5.0 quietly turns the ranking
# over (kaolin takes the lead with 0.589 on the reference glaze) and a 1e308 used
# to take the whole answer down with nonfinite_result because one class was
# mistyped. Such a value is not a usable sigma: the material falls back to the
# default, the answer stays alive, and the dropped value is named in "issues".
# _all_finite() on the result stays as the last net behind this.
MAX_SIGMA = 1.0

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
        "issues": [str, ...]}

        "issues" lists what the file lost on the way in - a section of the wrong
        type, an entry that is not an object, a sigma outside the sane range -
        in the language of the response, so that a partly broken file still says
        what it dropped.

        There is deliberately no flag here saying the ranking will come out flat.
        This function sees the file and nothing else; whether any of its sigmas
        reaches a given recipe depends on materials.json and on the recipe, and
        every attempt to predict that from the file alone left a hole. The
        question is answered in recipe_sensitivity(), on the finished rows.
    """
    return _read_tolerances(path if path is not None else _default_tolerances_path())


def _fallback_tolerances():
    return {"default_relative": FALLBACK_RELATIVE, "classes": {}, "materials": {},
            "issues": [UNREADABLE_TOLERANCES_ISSUE]}


def _read_tolerances(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (OSError, ValueError) as exc:
        # A broken tolerance file must not take the whole endpoint down: the
        # ranking is still meaningful with one flat sigma for everything - as
        # long as the answer says so, see FLAT_SIGMA_WARNING.
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

    _report_dropped_sigmas(classes, materials, path, issues)

    return {
        "default_relative": default_relative,
        "classes": classes,
        "materials": materials,
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


def _report_dropped_sigmas(classes, materials, path, issues):
    """
    Name every value of the file that the resolution will not use

    Strictly a description of the CONTENT, and no conclusion drawn from it. Its
    predecessor, _usable_sigma_count(), added up the same numbers and decided
    from them whether the answer would be flat - which it cannot know, and got
    wrong on every input where the file was healthy but disconnected from
    materials.json. Counting what was dropped is a fact; counting what survived
    and calling it a working database is a guess.

    A dropped value is silent damage in the most literal sense: the material
    falls back a level (an override to the class, a class to the default) and
    the answer that comes back is a smaller number for that material with
    nothing in it saying why.

    Every line below therefore describes the CONTENT of the file and stops
    before the calculation. The earlier wording finished each of them with what
    the materials then took ("эти материалы взяли сигму уровнем выше", "для них
    взята сигма по умолчанию"), and that half was a guess of exactly the kind
    the commit before this one removed from the flat check: an unused class with
    a typo in it, or a broken entry for a material this recipe does not contain,
    produced "1 value dropped, those materials fell back a level" with no such
    material anywhere in the request.
    """
    unresolved_classes = 0
    broken_entries = 0
    broken_oxide_sections = 0
    broken_overrides = 0
    oversized = 0

    for name, value in classes.items():
        if _is_oversized_sigma(value):
            logger.warning(f"material_tolerance_sigma_out_of_range: {path}: class "
                           f"'{name}' = {value!r}, above {MAX_SIGMA}")
            oversized += 1

    for name, entry in materials.items():
        if not isinstance(entry, dict):
            logger.warning(f"material_tolerance_entry_ignored: {path}: '{name}' is a "
                           f"{type(entry).__name__}, expected an object")
            broken_entries += 1
            continue

        class_name = entry.get('class')
        if class_name is not None and not _is_usable_sigma(classes.get(class_name)):
            unresolved_classes += 1

        oxides = entry.get('oxides')
        if oxides is not None and not isinstance(oxides, dict):
            # Used to be replaced by {} inside material_sigma() without a word,
            # which is how "oxides": [0.10] moved ulexite from 0.700 to 0.618
            # under warnings: []
            logger.warning(f"material_tolerance_oxides_ignored: {path}: '{name}': "
                           f"'oxides' is a {type(oxides).__name__}, expected an object")
            broken_oxide_sections += 1
            continue

        for oxide, value in (oxides or {}).items():
            if _is_usable_sigma(value):
                continue
            if _is_oversized_sigma(value):
                logger.warning(f"material_tolerance_sigma_out_of_range: {path}: "
                               f"'{name}': {oxide}={value!r}, above {MAX_SIGMA}")
                oversized += 1
            else:
                # An unusable value INSIDE oxides used to be the one kind of
                # damage nothing reported: a bad class sigma was counted, a bad
                # override was simply not counted as usable and vanished
                logger.warning(f"material_tolerance_override_ignored: {path}: "
                               f"'{name}': {oxide}={value!r}")
                broken_overrides += 1

    if unresolved_classes:
        logger.warning(f"material_tolerance_unresolved_classes: {path}: "
                       f"{unresolved_classes} entries point at a class without a sigma")
        issues.append(f"в базе допусков {unresolved_classes} записей материалов ссылаются "
                      f"на класс без пригодной сигмы")

    if broken_entries:
        issues.append(f"в базе допусков {broken_entries} записей материалов неверного "
                      f"типа и не учтены")

    if broken_oxide_sections:
        issues.append(f"в базе допусков не учтена секция «oxides» неверного типа: таких "
                      f"записей {broken_oxide_sections}")

    if broken_overrides:
        issues.append(f"в базе допусков не учтены непригодные переопределения сигмы по "
                      f"оксиду: таких значений {broken_overrides}")

    if oversized:
        issues.append(f"в базе допусков не учтены сигмы больше {MAX_SIGMA:.1f} (100%) — "
                      f"паспорт не врёт настолько: таких значений {oversized}")


def _is_usable_sigma(value):
    """True when _positive_float() keeps the value instead of falling back"""
    if isinstance(value, bool):
        # float(True) is 1.0, so a "clay": true typed into the file used to pass
        # for a sigma of 100% and hand the class the widest spread in the answer.
        # _all_finite() has had this check from the start; this one did not.
        return False

    try:
        number = float(value)
    except (TypeError, ValueError):
        return False

    return math.isfinite(number) and 0 < number <= MAX_SIGMA


def _is_oversized_sigma(value):
    """A number, and a plausible sigma in everything but its size"""
    if isinstance(value, bool):
        return False

    try:
        number = float(value)
    except (TypeError, ValueError):
        return False

    return math.isfinite(number) and number > MAX_SIGMA


def _positive_float(value, fallback):
    """A tolerance is a relative spread: anything but a sane positive number is junk"""
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

    Every fallback below is silent by design: this function has no channel to
    report through and is called once per material. What was dropped and why is
    said once, by _report_dropped_sigmas() at load time, over the whole file.
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
            # Reachable only on a direct call to material_sigma(). Through
            # recipe_sensitivity() it is dead code: calculate_recipe_composition()
            # multiplies the same cell by the share several steps earlier and
            # raises the TypeError there, which the endpoint answers with a 500.
            # That hole is not this module's to close - it belongs to a
            # validation of materials.json on the way in - and this guard is
            # deliberately left in place for whoever does close it.
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
            from the default file when omitted. What the file lost on the way in
            travels in its "issues" key and turns into the first warnings of the
            answer, so a hand built dictionary carrying none is taken at face
            value on that point. Whether the sigmas actually did anything is not
            taken on trust from anybody: it is read off the rows below.

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

    # What the file dropped on the way in. Not a verdict on the answer - most of
    # the ranking usually still stands - but the dropped part changed the numbers
    # and cannot stay silent
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

    # Every sigma that actually entered the variance. Not "does the file contain
    # a sigma" - that question was asked three times and answered wrongly three
    # times - but "what got used", which is the thing the answer rests on and
    # which is only knowable here. The whole set is watched and not just the
    # "sigma_used" of the rows: that field reports the leading oxide of a
    # material alone, so an override on a secondary oxide - a real move away from
    # the flat answer - would not show up in it.
    #
    # The plane is "all of them equal to EACH OTHER" and not "all of them equal
    # to default_relative": what makes a ranking flat is that nothing tells the
    # materials apart, and which number they all landed on is beside the point.
    # material_tolerance.json groups materials into classes, so several of them
    # sharing one number is the ordinary case and not an exotic one - feldspar
    # and silica are both 0.02 in the shipped file, ash covers both ashes at
    # 0.20, carbonate both carbonates at 0.01. Swept over all 5016 combinations
    # of 2-4 of the 19 inventory materials (11 of them refused for having no
    # fluxes at all), the shipped file answers 103 flat, and the rule this
    # replaces - equality to default_relative - saw 11 of those 103.
    applied_sigmas = set()

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

            applied_sigmas.add(sigma)

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

    if len(applied_sigmas) <= 1:
        # Everything that reached the calculation was one and the same number, so
        # this IS the flat answer - whatever the tolerance file says about itself
        # and whatever level that number came from. A file that never mentions
        # these materials, a name a supplier changed, an override on an oxide the
        # material does not carry, a missing file and a recipe whose materials
        # all resolved to one and the same number land here, and they land here
        # for the same reason: what the file contains was never the question,
        # what got used is. Sharing a class is not by itself enough and is not
        # claimed to be: an override lifts one oxide of one material off the
        # class number, and the shipped file does exactly that to B2O3 of ulexite
        # and borax, so a recipe of either alone has TWO applied sigmas.
        #
        # The log line is worded neutrally: the set is empty when nothing was
        # perturbed at all - an inf cell in a formula gets there - and "one sigma
        # for the whole recipe: []" was a line contradicted by its own payload.
        logger.warning(f"sensitivity_flat_sigmas: applied sigmas: {sorted(applied_sigmas)}")
        warnings.append(_flat_sigma_warning(applied_sigmas, tolerances, used))

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
    nonfinite = _first_nonfinite(result)
    if nonfinite is not None:
        # With the offending row named: everything needed for it is already
        # computed here, and a bare "the answer carries inf or nan" left whoever
        # reads the log to reproduce the whole recipe to find out which material
        # or oxide it was
        logger.error(f"sensitivity_nonfinite_result: the computed answer carries inf or "
                     f"nan: {nonfinite}")
        return _empty_result(warnings, "nonfinite_result", NONFINITE_RESULT_MESSAGE)

    return result


def _flat_sigma_warning(applied_sigmas, tolerances, used):
    """
    The message of a flat ranking: what was observed, and why only when it is known

    Two parts, and the boundary between them is the point of this function. The
    first is a fact read off the calculation that just happened and is true every
    time it is said. The second is a cause, and a cause is a claim about two
    files - it is appended only where this code has just checked it, and stays
    absent otherwise. The line these two used to share told the reader that the
    tolerance database was unavailable, silent about these materials, or out of
    sync with materials.json, on an answer where all three were false.
    """
    if applied_sigmas:
        (sigma,) = applied_sigmas  # a set of one is what brought us here
        text = f"{FLAT_SIGMA_WARNING} — {FLAT_SIGMA_OBSERVED.format(sigma=sigma)}"
    else:
        # No perturbation ran at all: every material of the recipe carries a
        # formula the resolution found nothing usable in. The shares are zero
        # anyway and ZERO_CONTRIBUTION_WARNING says that part.
        text = f"{FLAT_SIGMA_WARNING} — {NO_SIGMA_OBSERVED}"

    named = tolerances.get('materials')
    if isinstance(named, dict) and named and not any(name in named for name, _m, _a in used):
        text = f"{text}. {FLAT_SIGMA_NO_NAME_MATCHED}"

    return text


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
    return _first_nonfinite(value) is None


def _first_nonfinite(value, path="result"):
    """
    Where the first inf or nan of a nested structure sits, None when there is none

    The path is what goes into the log, so a row of a list is labelled by its own
    "material" or "oxide" rather than by an index nobody can resolve afterwards:
    "result.by_material[Улексит (Химпэк)].share=nan" says what happened, while
    "by_material[3]" says that something did.
    """
    if isinstance(value, dict):
        for key, item in value.items():
            found = _first_nonfinite(item, f"{path}.{key}")
            if found is not None:
                return found
        return None

    if isinstance(value, list):
        for index, item in enumerate(value):
            found = _first_nonfinite(item, f"{path}[{_row_label(item, index)}]")
            if found is not None:
                return found
        return None

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None

    return None if math.isfinite(value) else f"{path}={value}"


def _row_label(item, index):
    """What a row of by_material/per_oxide calls itself, its index otherwise"""
    if isinstance(item, dict):
        for key in ("material", "oxide"):
            name = item.get(key)
            if isinstance(name, str):
                return name

    return index


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
