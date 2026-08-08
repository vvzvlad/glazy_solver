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
import os
import numpy as np
from scipy.optimize import nnls
import math

def oxides_classification():
    oxides = {}
    oxides['r2o'] = ['Na2O', 'K2O', 'Li2O']
    oxides['ro'] = ['MgO', 'CaO', 'SrO', 'BaO', 'ZnO', 'MnO', 'FeO', 'CoO', 'NiO', 'CuO']
    oxides['r2o3'] = ['Al2O3', 'B2O3', 'Fe2O3', 'Cr2O3', 'Mn2O3']
    oxides['ro2'] = ['SiO2', 'TiO2', 'ZrO2', 'SnO2']
    return oxides
    
    


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

def weights_to_umf(weight_composition):
    """
    Converts weight fractions to UMF (Unity Molecular Formula)
    
    Args:
        weight_composition: dictionary {oxide: weight_fraction}
    
    Returns:
        dictionary {oxide: umf_value}
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    molar_masses_file = os.path.join(script_dir, 'database', 'molar_masses.json')

    with open(molar_masses_file, 'r', encoding='utf-8') as f:
        molar_masses = json.load(f)

    # Convert to molar amounts
    molar_amounts = {}
    for oxide, weight in weight_composition.items():
        if oxide in molar_masses:
            molar_amounts[oxide] = weight / molar_masses[oxide]
    
    classes = oxides_classification()
    
    # Calculate sum of R2O and RO (fluxes)
    sum_r2o = sum(molar_amounts.get(oxide, 0) for oxide in classes['r2o'])
    sum_ro = sum(molar_amounts.get(oxide, 0) for oxide in classes['ro'])
    sum_fluxes = sum_r2o + sum_ro
    
    # Normalize relative to the sum of fluxes
    if sum_fluxes == 0:
        # If no fluxes, use the minimum value as unity
        min_value = min(v for v in molar_amounts.values() if v > 0)
        unity_factor = 1 / min_value
    else:
        unity_factor = 1 / sum_fluxes
    
    umf = {oxide: amount * unity_factor for oxide, amount in molar_amounts.items()}
    
    # Round values for readability
    umf = {oxide: round(value, 3) for oxide, value in umf.items()}
    
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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    molar_masses_file = os.path.join(script_dir, 'database', 'molar_masses.json')
    
    with open(molar_masses_file, 'r', encoding='utf-8') as f:
        molar_masses = json.load(f)
    
    molar_weights = {}
    for oxide, umf_value in umf.items():
        if oxide in molar_masses:
            molar_weights[oxide] = umf_value * molar_masses[oxide]
    
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

def load_molar_masses():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    molar_masses_file = os.path.join(script_dir, 'database', 'molar_masses.json')
    
    with open(molar_masses_file, 'r', encoding='utf-8') as f:
        molar_masses = json.load(f)
    return molar_masses


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
    
    # Calculate flux ratios
    r2o_sum = sum(umf.get(oxide, 0) for oxide in ['Na2O', 'K2O', 'Li2O'])
    ro_sum = sum(umf.get(oxide, 0) for oxide in ['MgO', 'CaO', 'SrO', 'BaO', 'ZnO'])
    
    # Avoid division by zero
    analysis['R2O:RO'] = round(r2o_sum / ro_sum, 2) if ro_sum > 0 else "∞"
    analysis['RO:R2O'] = round(ro_sum / r2o_sum, 2) if r2o_sum > 0 else "∞"
    
    return analysis




def calculate_umf_from_recipe(weight_composition):
    """
    Calculate UMF from weight composition with proper normalization 
    based on RO+R2O oxides
    
    Args:
        weight_composition: Dictionary of oxide weights
    
    Returns:
        Dictionary of UMF values
    """

    molar_masses = load_molar_masses()
    
    # Convert to molar amounts
    molar_amounts = {}
    for oxide, weight in weight_composition.items():
        if oxide in molar_masses:
            molar_amounts[oxide] = weight / molar_masses[oxide]
    
    # Classification of oxides
    r2o = ['Na2O', 'K2O', 'Li2O']
    ro = ['MgO', 'CaO', 'SrO', 'BaO', 'ZnO', 'MnO', 'FeO', 'CoO', 'NiO', 'CuO']
    
    # Calculate sum of R2O and RO (fluxes)
    sum_r2o = sum(molar_amounts.get(oxide, 0) for oxide in r2o)
    sum_ro = sum(molar_amounts.get(oxide, 0) for oxide in ro)
    sum_fluxes = sum_r2o + sum_ro
    
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