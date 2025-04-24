#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# flake8: noqa
# pylint: disable=broad-exception-raised, raise-missing-from, too-many-arguments, redefined-outer-name
# pylint: disable=multiple-statements, logging-fstring-interpolation, trailing-whitespace, line-too-long
# pylint: disable=broad-exception-caught, missing-function-docstring, missing-class-docstring
# pylint: disable=f-string-without-interpolation
# pylance: disable=reportMissingImports, reportMissingModuleSource

import json
import os

def calculate_umf_from_recipe(weight_composition):
    """
    Calculate UMF from weight composition with proper normalization 
    based on RO+R2O oxides
    
    Args:
        weight_composition: Dictionary of oxide weights
    
    Returns:
        Dictionary of UMF values
    """
    # Определяем путь к файлу с молярными массами
    script_dir = os.path.dirname(os.path.abspath(__file__))
    molar_masses_file = os.path.join(script_dir, 'database', 'molar_masses.json')
    
    # Load molar masses
    with open(molar_masses_file, 'r', encoding='utf-8') as f:
        molar_masses = json.load(f)
    
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

def analyze_umf(umf):
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

def analyze_recipe(recipe_ingredients):
    """
    Analyze the recipe by calculating UMF (Unity Molecular Formula)
    and weight composition
    
    Args:
        recipe_ingredients: словарь {название_материала: процент}
    
    Returns:
        Dictionary with recipe analysis results
    """
    # Определяем путь к файлу с материалами
    script_dir = os.path.dirname(os.path.abspath(__file__))
    materials_file = os.path.join(script_dir, 'database', 'materials.json')
    
    # Load materials
    with open(materials_file, 'r', encoding='utf-8') as f:
        materials_data = json.load(f)
    
    # Initialize recipe weight composition
    weight_composition = {}
    
    # Process each ingredient
    matching_materials = {}
    for ingredient_name, amount in recipe_ingredients.items():
        # Ищем материал по точному или частичному совпадению имени
        material = None
        for m in materials_data:
            if m['name'] == ingredient_name or ingredient_name in m['name'] or m['name'] in ingredient_name:
                material = m
                matching_materials[ingredient_name] = m['name']
                break
                
        if material:
            for oxide, percentage in material['formula'].items():
                if oxide in weight_composition:
                    weight_composition[oxide] += percentage * amount / 100
                else:
                    weight_composition[oxide] = percentage * amount / 100
        else:
            matching_materials[ingredient_name] = 'NOT FOUND'
    
    # Calculate UMF
    umf, raw_umf = calculate_umf_from_recipe(weight_composition)
    
    # Calculate proportions
    result = {
        'weight_composition': weight_composition,
        'umf': umf,
        'raw_umf': raw_umf,
        'analysis': analyze_umf(umf),
        'materials_used': matching_materials
    }
    
    return result


def print_recipe_analysis(analysis):    
    print(f"\n===== Recipe Analysis =====")
        
    print("\nUMF (Unity Molecular Formula):")
    for oxide, value in sorted(analysis['umf'].items()):
        print(f"  {oxide}: {value}")
    
    print("\nRatio Analysis:")
    for name, value in analysis['analysis'].items():
        print(f"  {name}: {value}")

if __name__ == "__main__":
    test_recipe = {
        "Нефелин-сиенит VR13": 30,
        "Кварцевая мука Кварцверке W12": 20,
        "Волластонит МИВОЛЛ": 20,
        "Улексит (Химпэк)": 15,
        "Каолин КЖФ-1": 15
    }
    analysis = analyze_recipe(test_recipe)
    print_recipe_analysis(analysis) 