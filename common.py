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
import os
import numpy as np
from scipy.optimize import nnls
import math

logger = logging.getLogger(__name__)

# Keys that may appear in a material formula or a weight composition without
# being an oxide at all. Loss on ignition is bookkeeping, not a lost oxide, so
# it must not be reported as one.
NON_OXIDE_KEYS = frozenset({"Loi", "LOI"})

# Structural oxide groups of the classification, in the order the UMF columns
# are printed. "unity" is a meta key (it names the groups that form the unity
# basis), not a group of oxides, and therefore never leaves this module as one.
OXIDE_GROUP_NAMES = ('r2o', 'ro', 'r2o3', 'ro2')

# Keys of the classification file that are not oxide groups. "unity" holds the
# names of the groups forming the normalization basis and "unity_presets" holds
# named flux conventions, so neither can be named by "unity" as a group.
CLASSIFICATION_META_KEYS = frozenset({'unity', 'unity_presets'})

# The classification is read-only reference data that never changes while the
# process runs, and it is consulted on every UMF conversion, so the file is
# parsed once and kept in memory - same reasoning as the molar masses below.
_OXIDE_CLASSIFICATION_CACHE = None


class ClassificationError(Exception):
    """
    The oxide classification file cannot be used

    Deliberately not a subclass of ValueError: the solvers catch ValueError (and
    the classic one catches Exception) around their numeric work and turn it
    into "no solution", which is exactly the wrong answer for corrupt reference
    data. Keeping this class off the ValueError branch lets it travel past
    solver_iterative._solve_material_set instead of being reported as an empty
    result. The API validates the file at import so that it never has to travel
    that far.
    """


def _validate_oxide_classification(classification, source):
    """
    Check that a parsed classification file can actually be used

    This is corrupt reference data, not a recoverable condition, so a problem
    raises instead of degrading quietly. The bar is not "the file parses" but
    "flux_oxides() returns a non-empty list of oxide names": anything short of
    that silently sends weights_to_umf into its "no fluxes, normalize by the
    smallest oxide" branch and turns every UMF the application computes into a
    different number, or - for entries that are not even strings - throws a
    TypeError that the solvers report as "no solutions found".

    So the checks are: the four display groups exist; "unity" is a non-empty
    list of names of real, non-meta groups; every group that feeds the unity
    basis is a non-empty list; and every entry of every checked group is a
    string.

    Args:
        classification: the parsed contents of the classification file
        source: path of the file, for the error message

    Raises:
        ClassificationError: the file is not usable as a classification
    """
    if not isinstance(classification, dict):
        raise ClassificationError(
            f"{source}: expected a JSON object, got {type(classification).__name__}")

    missing_groups = [group for group in OXIDE_GROUP_NAMES if group not in classification]
    if missing_groups:
        raise ClassificationError(f"{source}: missing oxide groups: {', '.join(missing_groups)}")

    unity = classification.get('unity')
    if not isinstance(unity, list) or not unity:
        raise ClassificationError(f"{source}: \"unity\" must be a non-empty list of group names, "
                                  f"otherwise the UMF normalization basis is undefined")

    # Checked before the lookups below: an unhashable entry would blow up the
    # "in classification" test with a TypeError instead of a readable message
    non_string_unity = [repr(group) for group in unity if not isinstance(group, str)]
    if non_string_unity:
        raise ClassificationError(f"{source}: \"unity\" must hold group names as strings, got: "
                                  f"{', '.join(non_string_unity)}")

    reserved_unity = [group for group in unity if group in CLASSIFICATION_META_KEYS]
    if reserved_unity:
        raise ClassificationError(f"{source}: \"unity\" names reserved keys instead of oxide "
                                  f"groups: {', '.join(reserved_unity)}")

    unknown_unity = [group for group in unity if group not in classification]
    if unknown_unity:
        raise ClassificationError(f"{source}: \"unity\" names groups that the file does not define: "
                                  f"{', '.join(unknown_unity)}")

    # The groups that have to be well formed: the four the API exposes, plus any
    # further group the unity basis happens to be built from
    checked_groups = list(dict.fromkeys(list(OXIDE_GROUP_NAMES) + unity))

    malformed_groups = [group for group in checked_groups
                        if not isinstance(classification[group], list)]
    if malformed_groups:
        raise ClassificationError(f"{source}: these oxide groups are not lists of oxide names: "
                                  f"{', '.join(malformed_groups)}")

    for group in checked_groups:
        non_string_oxides = [repr(oxide) for oxide in classification[group]
                             if not isinstance(oxide, str)]
        if non_string_oxides:
            raise ClassificationError(f"{source}: oxide group \"{group}\" must hold oxide names as "
                                      f"strings, got: {', '.join(non_string_oxides)}")

    empty_unity_groups = [group for group in unity if not classification[group]]
    if empty_unity_groups:
        raise ClassificationError(f"{source}: \"unity\" names empty oxide groups, which would leave "
                                  f"the UMF normalization basis empty: "
                                  f"{', '.join(empty_unity_groups)}")

    _validate_unity_presets(classification, source)


def _validate_unity_presets(classification, source):
    """
    Check the optional "unity_presets" block of a classification file

    A preset is an alternative flux convention: a flat list of oxide names that
    replaces the group-derived basis when a caller asks for it by name. The bar
    is the same as for the groups above - flux_oxides(convention) has to return
    a non-empty list of oxide names - because a preset that expands to nothing
    sends weights_to_umf into its "no fluxes" branch and quietly renormalizes
    every UMF computed under that convention.

    The key is optional: a file without it simply has no named conventions, and
    flux_oxides(convention) then rejects every name it is given.

    Args:
        classification: the parsed contents of the classification file
        source: path of the file, for the error message

    Raises:
        ClassificationError: "unity_presets" is present but unusable
    """
    if 'unity_presets' not in classification:
        return

    presets = classification['unity_presets']
    if not isinstance(presets, dict):
        raise ClassificationError(f"{source}: \"unity_presets\" must be an object mapping a "
                                  f"convention name to a list of oxide names, got "
                                  f"{type(presets).__name__}")

    non_string_names = [repr(name) for name in presets if not isinstance(name, str)]
    if non_string_names:
        raise ClassificationError(f"{source}: \"unity_presets\" must be keyed by convention names "
                                  f"as strings, got: {', '.join(non_string_names)}")

    for name in sorted(presets):
        oxides = presets[name]
        if not isinstance(oxides, list) or not oxides:
            raise ClassificationError(f"{source}: unity preset \"{name}\" must be a non-empty list "
                                      f"of oxide names, otherwise the UMF normalization basis of "
                                      f"that convention is empty")

        non_string_oxides = [repr(oxide) for oxide in oxides if not isinstance(oxide, str)]
        if non_string_oxides:
            raise ClassificationError(f"{source}: unity preset \"{name}\" must hold oxide names as "
                                      f"strings, got: {', '.join(non_string_oxides)}")


def _oxide_classification():
    """
    Return the cached oxide classification table

    The returned dictionary is the cache itself and must never be mutated;
    external callers should use oxides_classification() or flux_oxides().
    """
    global _OXIDE_CLASSIFICATION_CACHE

    if _OXIDE_CLASSIFICATION_CACHE is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        classification_file = os.path.join(script_dir, 'database', 'oxide_classification.json')

        with open(classification_file, 'r', encoding='utf-8') as f:
            classification = json.load(f)

        _validate_oxide_classification(classification, classification_file)
        _OXIDE_CLASSIFICATION_CACHE = classification

    return _OXIDE_CLASSIFICATION_CACHE


def oxides_classification():
    """
    Load the oxide classification by structural group

    Returns:
        dictionary {'r2o': [...], 'ro': [...], 'r2o3': [...], 'ro2': [...]};
        fresh lists on every call, so callers are free to modify them
    """
    classification = _oxide_classification()
    return {group: list(classification[group]) for group in OXIDE_GROUP_NAMES}


def load_oxide_classification():
    """
    Load the whole validated classification file, meta keys included

    oxides_classification() answers "which group is this oxide in" and hides the
    meta keys; this one hands out the file as it is, for the callers that have
    to serve or inspect it. Going through here rather than opening the file
    again is what keeps them on the validated, cached copy.

    Returns:
        dictionary as stored in database/oxide_classification.json; a fresh copy
        on every call, so callers are free to modify it. "unity_presets" is a
        nested object, so it is copied one level deeper than the flat groups -
        otherwise a caller editing a preset would edit the cache itself

    Raises:
        FileNotFoundError: the classification file is missing and nothing has
            been cached yet; the API turns this one into a 404
        ClassificationError: the file is there but is not usable as a
            classification
    """
    def copy_value(value):
        if isinstance(value, list):
            return list(value)
        if isinstance(value, dict):
            return {key: copy_value(item) for key, item in value.items()}
        return value

    classification = _oxide_classification()
    return {key: copy_value(value) for key, value in classification.items()}


def flux_oxides(convention=None):
    """
    Load the list of oxides that form the UMF unity basis (the fluxes)

    This function is the single place the unity convention lives: the groups
    taking part in the normalization are named by the "unity" key of
    database/oxide_classification.json, and every consumer of the flux list
    calls this function instead of keeping a copy of its own.

    Which oxides count as fluxes is a convention, not a fact, and the schools
    disagree: ours is ceramicscalc-2018 ("unity" groups, the default here),
    while modern Glazy leaves FeO/CoO/NiO/CuO out of the basis. On a recipe
    carrying a colorant the two conventions inflate the unity denominator
    differently and every oxide of the resulting UMF moves at once, so a test
    comparing against published Glazy numbers has to ask for "glazy" explicitly.
    The named alternatives live in the "unity_presets" key of the same file:
    "legacy" is the default basis written out flat, "glazy" is the Glazy
    convention, and "segerlab" is the convention of segerlab.ru - the source our
    material and recipe data comes from - which additionally counts Fe2O3, SnO2,
    Cu2O, CdO and V2O5 as fluxes. Fe2O3 is the whole of the constant 1.0022
    factor between their published numbers and ours; the preset exists so their
    formulas can be reproduced digit for digit when cross-checking, not because
    the default changes.

    MnO2 lives in "ro" and therefore in the basis, which is a deliberate call
    and not the structural reading of the formula: the manganese colorant oxides
    act as divalent fluxes in the melt - MnO2 gives up its oxygen well below
    cone 6 and enters the glass as MnO - upstream classifies it as "AEarth" (a
    flux), and the alternative is not merely a different opinion but a wrong
    number.
    Two materials of the database are almost pure MnO2, and with MnO2 outside
    the basis the flux sum of a manganese glaze collapses to the trace K2O and
    CaO of its kaolin (~0.0086) - the reference "Марганцевый металлик △6" then
    comes out as MnO2 115.5 / SiO2 131.2 instead of MnO2 0.99 / SiO2 1.13, off
    by two orders of magnitude. Grouping it with the other RO fluxes is what
    the JSON cannot say for itself, so it is said here.

    An oxide is listed once even if it belongs to several unity groups: the
    callers sum over the returned list, and a duplicate would count that oxide
    twice in the unity denominator.

    Args:
        convention: name of a preset from "unity_presets", or None for the
            group-derived basis named by "unity". None is the default and the
            behaviour of every existing caller

    Returns:
        flat list of oxide names without duplicates - in the order of the unity
        groups for the default, in the order of the preset otherwise; a fresh
        list on every call

    Raises:
        ClassificationError: the requested convention is not defined by the
            classification file
    """
    classification = _oxide_classification()

    if convention is not None:
        presets = classification.get('unity_presets') or {}
        if convention not in presets:
            available = ', '.join(sorted(presets)) if presets else 'none'
            raise ClassificationError(f"unknown flux convention '{convention}', "
                                      f"available: {available}")
        return list(dict.fromkeys(presets[convention]))

    fluxes = []
    for group in classification['unity']:
        fluxes.extend(classification[group])
    return list(dict.fromkeys(fluxes))


def format_umf(umf):
    """
    Format UMF values for readability in columns based on oxide classes
    
    Args:
        umf: Dictionary of UMF values

    Returns:
        formatted text for console output with oxides grouped by type
    """
    classes = oxides_classification()
    
    # Prepare data for each column
    r2o_ro_data = []
    for oxide in classes['r2o'] + classes['ro']:
        if oxide in umf and umf[oxide] > 0:
            r2o_ro_data.append((oxide, umf[oxide]))
    
    r2o3_data = []
    for oxide in classes['r2o3']:
        if oxide in umf and umf[oxide] > 0:
            r2o3_data.append((oxide, umf[oxide]))
    
    ro2_data = []
    for oxide in classes['ro2']:
        if oxide in umf and umf[oxide] > 0:
            ro2_data.append((oxide, umf[oxide]))
    
    # Find oxides that don't belong to any of the standard groups
    all_standard_oxides = classes['r2o'] + classes['ro'] + classes['r2o3'] + classes['ro2']
    other_data = []
    for oxide, value in umf.items():
        if oxide not in all_standard_oxides and value > 0:
            other_data.append((oxide, value))
    
    lines = []
    
    # Header line
    lines.append("    R₂O/RO     |       R₂O₃       |       RO₂      ")
    
    # Calculate max rows
    max_rows = max(len(r2o_ro_data), len(r2o3_data), len(ro2_data) + len(other_data))
    
    # Create rows
    for i in range(max_rows):
        col1 = " " * 16
        col2 = " " * 18
        col3 = " " * 16
        
        # R2O/RO column
        if i < len(r2o_ro_data):
            oxide, value = r2o_ro_data[i]
            if oxide == "Na2O":
                col1 = f"{oxide}     {value:.3f} "
            else:
                col1 = f"{oxide}      {value:.3f} "
            
        # R2O3 column
        if i < len(r2o3_data):
            oxide, value = r2o3_data[i]
            col2 = f" {oxide}      {value:.3f} "
            
        # RO2 column and Other
        if i < len(ro2_data):
            oxide, value = ro2_data[i]
            col3 = f" {oxide}      {value:.3f}"
        elif i - len(ro2_data) < len(other_data):
            idx = i - len(ro2_data)
            oxide, value = other_data[idx]
            col3 = f" {oxide}      {value:.3f}"
            
        lines.append(f"{col1}|{col2}|{col3}")
    
    return "\n".join(lines)
    

def format_weight_composition(weight_composition):
    """
    Format weight composition for readability
    
    Args:
        weight_composition: Dictionary of weight composition
    
    Returns:
        formatted text for console output
    """
    return "\n".join([f"{oxide}: {value:.2f}%" for oxide, value in weight_composition.items()])

def calc_error(umf, target_umf):
    """
    Calculate error between UMF and target UMF
    
    Args:
        umf: Dictionary of UMF values
        target_umf: Dictionary of target UMF values
    
    Returns:
        UMF data: [{"oxide": umf_value, "target_umf_value": target_umf_value, "abs_error": abs_error}, ...]
        Stats: {total_error, max_error}
    """
    all_oxides = set(list(umf.keys()) + list(target_umf.keys()))
    data = []
    total_error = 0
    max_error = 0
    
    for oxide in all_oxides:
        umf_value = umf.get(oxide, 0)
        target_umf_value = target_umf.get(oxide, 0)
        abs_error = abs(umf_value - target_umf_value)
        
        total_error += abs_error
        max_error = max(max_error, abs_error)
        
        data.append({
            "oxide": oxide,
            "umf_value": umf_value,
            "target_umf_value": target_umf_value,
            "abs_error": abs_error
        })
    
    data.sort(key=lambda x: x["abs_error"], reverse=True)
    stats = {"total_error": total_error, "max_error": max_error}
    
    return data, stats

def _warn_about_unknown_oxides(unknown_oxides, where):
    """
    Report the oxides that were silently dropped by a conversion

    An oxide with no entry in molar_masses.json cannot be converted and is left
    out of the math. That used to happen without a trace; one warning per call
    makes the loss visible without flooding the log from inside a solver loop.

    Args:
        unknown_oxides: collection of oxide names that were dropped
        where: name of the conversion, for the log message
    """
    if not unknown_oxides:
        return

    names = ', '.join(sorted(unknown_oxides))
    logger.warning(f"{where}: no molar mass for {names} - dropped from the conversion")


def weights_to_umf(weight_composition, *, convention=None, round_digits=3):
    """
    Converts weight fractions to UMF (Unity Molecular Formula)

    Args:
        weight_composition: dictionary {oxide: weight_fraction}
        convention: name of a flux convention from "unity_presets", or None for
            the default basis; see flux_oxides(). Only the set of oxides in the
            unity denominator changes, everything else is untouched
        round_digits: number of decimals the result is rounded to, 3 by default
            (the readable form every existing caller expects). Pass None to get
            the raw values: rounding to 3 decimals quantizes the result with a
            step of 0.001, which destroys any response smaller than that -
            sensitivity.py compares formulas that differ by a fraction of a
            percent and needs the unrounded numbers.
            Also needed by quality_metrics, to see a trace oxide that is
            really in the batch and rounds to 0.000 in the formula.

    Returns:
        dictionary {oxide: umf_value}
    """
    molar_masses = _molar_masses()

    # Convert to molar amounts
    molar_amounts = {}
    unknown_oxides = []
    for oxide, weight in weight_composition.items():
        if oxide in molar_masses:
            molar_amounts[oxide] = weight / molar_masses[oxide]
        elif oxide not in NON_OXIDE_KEYS:
            unknown_oxides.append(oxide)

    _warn_about_unknown_oxides(unknown_oxides, 'weights_to_umf')

    # Calculate the sum of the fluxes (the unity basis)
    sum_fluxes = sum(molar_amounts.get(oxide, 0) for oxide in flux_oxides(convention))

    # Normalize relative to the sum of fluxes
    if sum_fluxes == 0:
        # If no fluxes, use the minimum value as unity
        min_value = min(v for v in molar_amounts.values() if v > 0)
        unity_factor = 1 / min_value
    else:
        unity_factor = 1 / sum_fluxes
    
    umf = {oxide: amount * unity_factor for oxide, amount in molar_amounts.items()}

    # Round values for readability
    if round_digits is not None:
        umf = {oxide: round(value, round_digits) for oxide, value in umf.items()}

    return umf

def umf_to_weights(umf):
    """
    Converts UMF (Unity Molecular Formula) to weight fractions
    
    Args:
        umf: dictionary {oxide: umf_value}
    
    Returns:
        dictionary {oxide: weight_fraction}
    """
    # Convert to molar_weights
    molar_masses = _molar_masses()

    molar_weights = {}
    unknown_oxides = []
    for oxide, umf_value in umf.items():
        if oxide in molar_masses:
            molar_weights[oxide] = umf_value * molar_masses[oxide]
        elif oxide not in NON_OXIDE_KEYS:
            unknown_oxides.append(oxide)

    _warn_about_unknown_oxides(unknown_oxides, 'umf_to_weights')

    # Calculate total weight
    total_weight = sum(molar_weights.values())
    
    # Normalize to weight percentages
    weight_percentages = {oxide: (weight / total_weight) * 100 for oxide, weight in molar_weights.items()}
    
    # Round values
    weight_percentages = {oxide: round(value, 2) for oxide, value in weight_percentages.items()}
    
    return weight_percentages


# Priority semantics: lower number = higher priority. Materials missing from
# priorities.json get the lowest possible priority, so that explicitly listed
# base materials always win over unlisted ones.
DEFAULT_PRIORITY = 100


def load_materials(only_inventory=True, priority=True):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    materials_file = os.path.join(script_dir, 'database', 'materials.json')

    with open(materials_file, 'r', encoding='utf-8') as f:
        materials = json.load(f)

    if priority is True:
        priorities_file = os.path.join(script_dir, 'database', 'priorities.json')
        with open(priorities_file, 'r', encoding='utf-8') as f:
            priorities = json.load(f)
        for material in materials:
            material['priority'] = priorities.get(material['name'], DEFAULT_PRIORITY)

    if only_inventory is True:
        filtered_materials = []
        for material in materials:
            if material.get('inInventory') is True:
                filtered_materials.append(material)
        return filtered_materials
    else:
        return materials

def resolve_inventory(inventory_data=None):
    """
    Resolve the list of available material names

    Args:
        inventory_data: optional explicit list of material names; returned as is

    Returns:
        list of available material names (materials flagged inInventory in
        materials.json when no explicit inventory is given)
    """
    if inventory_data is not None:
        return inventory_data

    materials = load_materials(only_inventory=True, priority=False)
    return [material['name'] for material in materials]


def resolve_material_pool(materials=None, inventory_data=None):
    """
    Decide which material records a solver works from, and which of them it may use

    This is the material injection seam. It exists for the tests and for any
    caller carrying its own catalogue (an idealised textbook material set, an
    imported dump), so that a solver run can be pinned to known numbers instead
    of whatever database/materials.json currently holds.

    Injected records bypass the inventory resolution entirely when no inventory
    is given: resolve_inventory() would go back to database/materials.json for
    the inInventory flags and filter the injected catalogue against the real
    stock, which would usually leave nothing at all. "Here is my catalogue" is
    taken to mean "and all of it is available".

    Args:
        materials: optional list of material records, same structure as the
            entries of database/materials.json - at least "name" and "formula".
            "priority" is optional and defaults to DEFAULT_PRIORITY wherever the
            solvers read it. None loads the database as before
        inventory_data: optional explicit list of available material names. When
            it is given the records are filtered by it as usual, injected or not

    Returns:
        (records, inventory): the material records to build the matrix from and
        the names available out of them. The caller still applies
        filter_materials_by_inventory() and filter_materials_with_formula(), so
        an injected material with an empty formula is dropped like any other
    """
    if materials is None:
        return load_materials(only_inventory=False, priority=True), resolve_inventory(inventory_data)

    records = list(materials)
    if inventory_data is not None:
        return records, inventory_data

    return records, [record.get('name') for record in records]


def filter_materials_by_inventory(materials, inventory):
    """
    Keep only materials whose name is present in the inventory

    Args:
        materials: list of material dictionaries
        inventory: collection of available material names

    Returns:
        list of material dictionaries available in the inventory
    """
    available_materials = []
    for material in materials:
        if material.get('name') in inventory:
            available_materials.append(material)
    return available_materials


def filter_materials_with_formula(materials):
    """
    Keep only materials whose formula actually carries oxides

    Water, CMC, silicon carbide, pigments and the like are legal entries of the
    database, but their oxide formula sums to zero, so for a solver they are
    dead columns that can never move the UMF. Loss on ignition is not an oxide
    and does not count towards the sum.

    Args:
        materials: list of material dictionaries

    Returns:
        list of material dictionaries whose formula sums to more than zero
    """
    with_formula = []
    dropped = []

    for material in materials:
        formula = material.get('formula') or {}
        total = sum(value for oxide, value in formula.items() if oxide not in NON_OXIDE_KEYS)
        if total > 0:
            with_formula.append(material)
        else:
            dropped.append(str(material.get('name')))

    if dropped:
        logger.debug(f"dropped {len(dropped)} materials with an empty formula: {', '.join(dropped)}")

    return with_formula


def make_json_safe(obj):
    """
    Convert an object to a JSON-serializable form, replacing infinite and NaN
    floats with their string representations

    Args:
        obj: source object (dict, list or scalar)

    Returns:
        object safe for JSON serialization
    """
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    elif isinstance(obj, float) and (math.isinf(obj) or math.isnan(obj)):
        return "Infinity" if obj > 0 else "-Infinity" if obj < 0 else "NaN"
    else:
        return obj


def load_recipes():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    recipes_file = os.path.join(script_dir, 'database', 'recipes.json')
    
    with open(recipes_file, 'r', encoding='utf-8') as f:
        recipes = json.load(f)
    return recipes

# Molar masses are read-only reference data that never changes while the process
# runs, so the file is parsed once and kept in memory: the conversion helpers are
# called hundreds of times per solver run and re-reading the file every time cost
# most of the request latency.
_MOLAR_MASSES_CACHE = None


def _molar_masses():
    """
    Return the cached molar mass table

    The returned dictionary is the cache itself and must never be mutated;
    external callers should use load_molar_masses() instead.
    """
    global _MOLAR_MASSES_CACHE

    if _MOLAR_MASSES_CACHE is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        molar_masses_file = os.path.join(script_dir, 'database', 'molar_masses.json')

        with open(molar_masses_file, 'r', encoding='utf-8') as f:
            _MOLAR_MASSES_CACHE = json.load(f)

    return _MOLAR_MASSES_CACHE


def load_molar_masses():
    """
    Load the molar mass table

    Returns:
        dictionary {oxide: molar_mass}; a fresh copy on every call, so callers
        are free to modify it without corrupting the cache
    """
    return dict(_molar_masses())


def calc_ratios_umf(umf):
    """
    Perform analysis on UMF values to derive useful metrics
    
    Args:
        umf: Dictionary of UMF values
    
    Returns:
        Dictionary with analysis results
    """
    analysis = {}
    
    # Get total silica and alumina
    silica = umf.get('SiO2', 0)
    alumina = umf.get('Al2O3', 0)
    
    # Calculate ratios
    analysis['SiO2:Al2O3'] = round(silica / alumina, 2) if alumina > 0 else "∞"
    
    # Calculate flux ratios over the same groups the UMF normalization uses
    classes = oxides_classification()
    r2o_sum = sum(umf.get(oxide, 0) for oxide in classes['r2o'])
    ro_sum = sum(umf.get(oxide, 0) for oxide in classes['ro'])

    # Avoid division by zero
    analysis['R2O:RO'] = round(r2o_sum / ro_sum, 2) if ro_sum > 0 else "∞"
    analysis['RO:R2O'] = round(ro_sum / r2o_sum, 2) if r2o_sum > 0 else "∞"
    
    return analysis




def calculate_umf_from_recipe(weight_composition, convention=None):
    """
    Calculate UMF from weight composition with proper normalization 
    based on RO+R2O oxides
    
    Args:
        weight_composition: Dictionary of oxide weights
        convention: name of a flux convention from "unity_presets", or None for
            the default basis; see flux_oxides()
    
    Returns:
        Dictionary of UMF values
    """

    molar_masses = _molar_masses()

    # Convert to molar amounts
    molar_amounts = {}
    for oxide, weight in weight_composition.items():
        if oxide in molar_masses:
            molar_amounts[oxide] = weight / molar_masses[oxide]
    
    # Calculate the sum of the fluxes (the unity basis)
    sum_fluxes = sum(molar_amounts.get(oxide, 0) for oxide in flux_oxides(convention))

    # Normalize relative to the sum of fluxes
    if sum_fluxes == 0:
        # If no fluxes, use the minimum value as unity
        min_value = min(v for v in molar_amounts.values() if v > 0)
        unity_factor = 1 / min_value
    else:
        unity_factor = 1 / sum_fluxes
    
    umf = {}
    for oxide, amount in molar_amounts.items():
        umf[oxide] = amount * unity_factor
    
    # Save unrounded values
    raw_umf = umf.copy()
    
    # Round values for readability
    umf = {oxide: round(value, 3) for oxide, value in umf.items()}
    
    return umf, raw_umf