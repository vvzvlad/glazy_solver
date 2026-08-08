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
import logging
import itertools
from typing import Dict, List, Tuple, Any, Optional, Set

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def find_variants(materials: List[Dict], target_umf: Dict[str, float], max_solutions: int = 10, verbose: bool = False,
                  error_threshold: float = 0.05) -> List[Dict]:
    """
    Find optimal glaze recipes using non-negative least squares optimization.
    
    Args:
        materials: List of material dictionaries with composition
        target_umf: Target UMF composition to match
        max_solutions: Maximum number of solutions to return
        verbose: Whether to print detailed debug information
        error_threshold: Maximum acceptable error threshold
        
    Returns:
        List of solutions sorted by similarity to target UMF
    """
    all_solutions = []
    
    # Extract all unique oxides from materials and target
    all_oxides = set(target_umf.keys())
    for material in materials:
        for oxide in material.get('umf', {}).keys():
            all_oxides.add(oxide)
    
    # Sort materials by priority (higher priority = lower number)
    sorted_materials = sorted(materials, key=lambda x: x.get('priority', 1))
    
    # Group materials by priority
    priority_groups = {}
    for material in sorted_materials:
        priority = material.get('priority', 1)
        if priority not in priority_groups:
            priority_groups[priority] = []
        priority_groups[priority].append(material)
    
    # Sort priorities from highest to lowest (lowest number first)
    sorted_priorities = sorted(priority_groups.keys())
    
    # Start with minimum possible components (2) and increase until we find solutions
    for num_components in range(2, len(sorted_materials) + 1):
        if verbose:
            logging.info(f"trying to find solutions with {num_components} components")
        
        # Try with each priority group first, then mix if needed
        for current_priority_idx in range(len(sorted_priorities)):
            current_priority = sorted_priorities[current_priority_idx]
            current_materials = []
            
            # Add all materials from current and higher priorities
            for priority_idx in range(current_priority_idx + 1):
                priority = sorted_priorities[priority_idx]
                current_materials.extend(priority_groups[priority])
            
            if len(current_materials) < num_components:
                continue  # Not enough materials at this priority level
            
            # Generate all possible combinations of materials for this number of components
            for materials_subset in itertools.combinations(current_materials, num_components):
                try:
                    solution = solve_recipe(list(materials_subset), target_umf, all_oxides, verbose)
                    
                    # Check if the solution is valid (all materials have positive weights)
                    if solution and min(solution['weights']) > 0.001:  # Minimum weight threshold
                        solution['num_components'] = num_components
                        solution['priority_level'] = current_priority
                        all_solutions.append(solution)
                except Exception as e:
                    if verbose:
                        logging.error(f"error solving recipe: {e}")
                    continue
        
        # If we found solutions, stop looking for more components
        if all_solutions:
            # Sort solutions by total error
            all_solutions.sort(key=lambda x: x['total_error'])
            
            # Filter out solutions with error above threshold
            valid_solutions = [s for s in all_solutions if s['total_error'] <= error_threshold]
            
            # If we have valid solutions, return them, otherwise continue searching
            if valid_solutions:
                if verbose:
                    logging.info(f"found {len(valid_solutions)} valid solutions with {num_components} components")
                return valid_solutions[:max_solutions]
    
    # If we get here and have some solutions (even if above threshold), return them
    if all_solutions:
        all_solutions.sort(key=lambda x: x['total_error'])
        if verbose:
            logging.info(f"found {len(all_solutions)} solutions, but all exceed error threshold {error_threshold}")
        return all_solutions[:max_solutions]
    
    if verbose:
        logging.info("no solutions found")
    return []


def solve_recipe(materials: List[Dict], target_umf: Dict[str, float], all_oxides: Set[str], verbose: bool = False) -> Optional[Dict]:
    """
    Solve a recipe using non-negative least squares optimization
    
    Args:
        materials: List of material dictionaries to use
        target_umf: Target UMF to match
        all_oxides: Set of all oxides to consider
        verbose: Whether to print detailed info
        
    Returns:
        Solution dictionary or None if no solution found
    """
    # Create the A matrix (materials composition)
    A = []
    material_names = []
    
    for material in materials:
        material_names.append(material['name'])
        material_umf = material.get('umf', {})
        
        # Add a row for each oxide
        material_row = []
        for oxide in all_oxides:
            material_row.append(material_umf.get(oxide, 0.0))
        A.append(material_row)
    
    # Transpose A to get the right shape for NNLS
    A = np.array(A).T
    
    # Create the b vector (target values)
    b = []
    for oxide in all_oxides:
        b.append(target_umf.get(oxide, 0.0))
    b = np.array(b)
    
    # Solve using non-negative least squares
    try:
        weights, residual = nnls(A, b)
        
        # If all weights are 0, the solution is invalid
        if np.sum(weights) < 0.001:
            if verbose:
                logging.warning("all material weights are near zero - invalid solution")
            return None
        
        # Normalize weights to sum to 100
        total_weight = np.sum(weights)
        normalized_weights = (weights / total_weight) * 100
        
        # Calculate the resulting UMF
        result_umf = {}
        for i, oxide in enumerate(all_oxides):
            result_umf[oxide] = 0
            for j, material in enumerate(materials):
                material_umf = material.get('umf', {})
                result_umf[oxide] += normalized_weights[j] * material_umf.get(oxide, 0) / 100
        
        # Calculate error between target and result
        total_error = 0
        max_error = 0
        for oxide in all_oxides:
            target_val = target_umf.get(oxide, 0)
            result_val = result_umf.get(oxide, 0)
            error = abs(target_val - result_val)
            total_error += error
            max_error = max(max_error, error)
        
        return {
            'materials': material_names,
            'weights': normalized_weights.tolist(),
            'result_umf': result_umf,
            'total_error': total_error,
            'max_error': max_error,
            'residual': float(residual)
        }
    
    except Exception as e:
        if verbose:
            logging.error(f"nnls solver error: {e}")
        return None


def main():
    # Хардкод целевого UMF для тестирования
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
    
    # Оригинальный рецепт для сравнения
    original_recipe = {
        "Волластонит МИВОЛЛ": 20,
        "Каолин КЖФ-1": 15,
        "Кварцевая мука Кварцверке W12": 20,
        "Нефелин-сиенит VR13": 30,
        "Улексит (Химпэк)": 15
    }
    
    print("Target UMF:")
    for oxide, value in target_umf.items():
        print(f"  {oxide}: {value}")
    
    print("\nOriginal Recipe:")
    for material, weight in original_recipe.items():
        print(f"  {material}: {weight}%")
    
    # Загрузка материалов
    from common import load_materials
    materials = load_materials(only_inventory=True, priority=True)
    
    verbose = True
    max_solutions = 10
    error_threshold = 0.05
    
    print("\nSearching for solutions...")
    
    # Найти варианты
    solutions = find_variants(
        materials, 
        target_umf, 
        max_solutions=max_solutions,
        verbose=verbose,
        error_threshold=error_threshold
    )
    
    # Вывести решения
    if solutions:
        print(f"\nFound {len(solutions)} solutions!")
        for i, solution in enumerate(solutions):
            print(f"\nSolution {i+1}")
            print(f"Error: {solution['total_error']:.4f}")
            print(f"Components: {solution['num_components']}")
            print("Recipe:")
            for j, (material, weight) in enumerate(zip(solution['materials'], solution['weights'])):
                print(f"  {material}: {weight:.2f}%")
            
            print("Resulting UMF:")
            for oxide, value in solution['result_umf'].items():
                if value > 0.001:  # Show only non-zero values
                    print(f"  {oxide}: {value:.3f} (target: {target_umf.get(oxide, 0):.3f})")
    else:
        print("\nNo solutions found for the target UMF!")


if __name__ == "__main__":
    main()
