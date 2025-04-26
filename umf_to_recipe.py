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
    
    # Classification of oxides
    r2o = ['Na2O', 'K2O', 'Li2O']
    ro = ['MgO', 'CaO', 'SrO', 'BaO', 'ZnO', 'MnO', 'FeO', 'CoO', 'NiO', 'CuO']
    r2o3 = ['Al2O3', 'B2O3', 'Fe2O3', 'Cr2O3', 'Mn2O3']
    ro2 = ['SiO2', 'TiO2', 'ZrO2', 'SnO2']
    
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


# Загрузка данных из materials.json
def load_materials():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    materials_file = os.path.join(script_dir, 'database', 'materials.json')
    
    with open(materials_file, 'r', encoding='utf-8') as f:
        materials = json.load(f)
    return materials

# Загрузка доступных материалов из inventory.json
def load_inventory(inventory_data=None):
    """
    Загружает инвентарь материалов из переданных данных или из файла inventory.json
    
    Args:
        inventory_data: опциональный словарь или список с данными инвентаря
    
    Returns:
        список или словарь имен доступных материалов
    """
    if inventory_data is not None:
        return inventory_data
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    inventory_file = os.path.join(script_dir, 'database', 'inventory.json')
    
    with open(inventory_file, 'r', encoding='utf-8') as f:
        inventory = json.load(f)
    return inventory

# Фильтрация материалов по доступным именам
def filter_materials_by_inventory(materials, inventory):
    available_materials = []
    for material in materials:
        if material.get('name') in inventory:
            available_materials.append(material)
    return available_materials

# Расчет матрицы оксидов для всех доступных материалов
def create_oxide_matrix(materials, target_oxides):
    n_materials = len(materials)
    n_oxides = len(target_oxides)
    
    oxide_matrix = np.zeros((n_oxides, n_materials))
    material_names = []
    
    for j, material in enumerate(materials):
        material_names.append(material['name'])
        formula = material.get('formula', {})
        
        for i, oxide in enumerate(target_oxides):
            oxide_matrix[i, j] = formula.get(oxide, 0.0)
    
    return oxide_matrix, material_names

# Расчет ошибки между целевым и фактическим UMF
def calculate_umf_error(target_umf, actual_umf):
    # Используем только оксиды из целевого UMF
    squared_error = 0.0
    
    for oxide in target_umf.keys():
        target_value = target_umf.get(oxide, 0.0)
        actual_value = actual_umf.get(oxide, 0.0)
        squared_error += (target_value - actual_value) ** 2
    
    return np.sqrt(squared_error)

# Расчет фактического состава в весовых процентах на основе рецепта
def calculate_recipe_composition(materials, recipe):
    composition = {}
    
    for material_name, percentage in recipe.items():
        # Найдем материал по имени
        material = None
        for m in materials:
            if m['name'] == material_name:
                material = m
                break
        
        if material is None:
            continue
        
        # Добавляем вклад каждого оксида из материала
        for oxide, content in material.get('formula', {}).items():
            if oxide not in composition:
                composition[oxide] = 0.0
            composition[oxide] += content * (percentage / 100.0)
    
    return composition

# Решение задачи методом неотрицательных наименьших квадратов
def solve_recipe(oxide_matrix, target_umf, material_names, available_materials=None):
    # Получаем список оксидов из ключей target_umf
    target_oxides = list(target_umf.keys())
    
    # Преобразование UMF в весовые проценты
    target_weights = umf_to_weights(target_umf)
    weights_array = np.array([target_weights.get(oxide, 0.0) for oxide in target_oxides])
    
    try:
        # Решение задачи NNLS
        x, _residual = nnls(oxide_matrix, weights_array)
        
        # Если решение слишком близко к нулю, считаем что материал не используется
        x[x < 1e-6] = 0
        
        # Нормализация к 100%
        if np.sum(x) > 0:
            x = 100 * x / np.sum(x)
        
        # Формирование рецепта в виде словаря {материал: процент}
        recipe = {}
        for i, name in enumerate(material_names):
            if x[i] > 0.1:  # Игнорируем материалы с весом менее 0.05%
                recipe[name] = round(x[i], 2)
        
        # Если нет доступных материалов, не можем рассчитать фактический состав
        if not available_materials:
            # Упрощенный расчет на основе матрицы оксидов
            composition = {}
            for i, oxide in enumerate(target_oxides):
                composition[oxide] = 0
                for j, _material in enumerate(material_names):
                    if x[j] > 0:
                        composition[oxide] += oxide_matrix[i, j] * (x[j] / 100)
        else:
            # Расчет фактического состава на основе рецепта
            composition = calculate_recipe_composition(available_materials, recipe)
        
        # Преобразование весовых процентов в UMF
        actual_umf = weights_to_umf(composition)
        
        # Нормализация UMF относительно одного из оксидов (обычно самого маленького)
        base_oxide = None
        min_value = float('inf')
        
        for oxide, value in target_umf.items():
            if value < min_value and value > 0:
                min_value = value
                base_oxide = oxide
        
        if base_oxide and base_oxide in actual_umf and actual_umf[base_oxide] > 0:
            scale_factor = target_umf[base_oxide] / actual_umf[base_oxide]
            actual_umf = {oxide: value * scale_factor for oxide, value in actual_umf.items()}
        elif base_oxide:
            # Если базовый оксид отсутствует или равен нулю, ищем другой оксид для нормализации
            for oxide in target_umf:
                if oxide in actual_umf and actual_umf[oxide] > 0 and target_umf[oxide] > 0:
                    scale_factor = target_umf[oxide] / actual_umf[oxide]
                    actual_umf = {oxide: value * scale_factor for oxide, value in actual_umf.items()}
                    break
        
        # Вычисление ошибки между целевым и фактическим UMF
        error = calculate_umf_error(target_umf, actual_umf)
        
        return {
            'recipe': recipe,
            'error': round(error, 4),
            'target_composition': target_umf,
            'actual_composition': {oxide: round(value, 4) for oxide, value in actual_umf.items()},
            'weight_composition': {oxide: round(value, 2) for oxide, value in composition.items()},
            'materials_count': len(recipe)  # Добавляем количество материалов в решении
        }
    
    except Exception as e:
        return {
            'error': 'Ошибка решения: ' + str(e),
            'recipe': {}
        }

# Основная функция решения для заданной UMF-формулы
def solve_glaze_recipe(target_umf, inventory_data=None):
    materials = load_materials()
    inventory = load_inventory(inventory_data)
    
    # Фильтрация материалов по инвентарю
    available_materials = filter_materials_by_inventory(materials, inventory)
    
    if not available_materials:
        return {'error': 'нет_доступных_материалов_в_инвентаре'}
    
    # Определение всех оксидов в целевой формуле
    target_oxides = list(target_umf.keys())
    
    # Создание матрицы оксидов
    oxide_matrix, material_names = create_oxide_matrix(available_materials, target_oxides)
    
    # Проверка возможности решения
    rank = np.linalg.matrix_rank(oxide_matrix)
    
    if rank < len(target_oxides):
        print(f"предупреждение: ранг матрицы ({rank}) меньше количества оксидов ({len(target_oxides)}). точное решение невозможно.")
    
    # Решение задачи
    solution = solve_recipe(oxide_matrix, target_umf, material_names, available_materials)
    
    return solution

# Функция для преобразования результатов к безопасному для JSON формату
def make_json_safe(obj):
    """
    Преобразует объект для безопасной сериализации в JSON,
    заменяя бесконечные значения на строки 'Infinity'
    
    Args:
        obj: исходный объект (словарь, список или простой тип)
    
    Returns:
        объект, безопасный для сериализации в JSON
    """
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    elif isinstance(obj, float) and (math.isinf(obj) or math.isnan(obj)):
        return "Infinity" if obj > 0 else "-Infinity" if obj < 0 else "NaN"
    else:
        return obj

# Функция для поиска нескольких решений с различными комбинациями материалов
def find_multiple_solutions(target_umf, max_solutions=5, min_materials=True, error_tolerance=0.01, logging=False, inventory_data=None):
    """
    Найти несколько решений для заданной UMF-формулы
    
    Args:
        target_umf: целевая UMF-формула
        max_solutions: максимальное количество решений
        min_materials: если True, предпочитать решения с меньшим количеством материалов
        error_tolerance: допустимое увеличение ошибки для решений с меньшим числом материалов
        logging: включить логирование процесса поиска
        inventory_data: опциональные данные инвентаря вместо загрузки из файла
    
    Returns:
        Список решений, отсортированный по предпочтительности
    """
    materials = load_materials()
    inventory = load_inventory(inventory_data)
    
    # Фильтрация материалов по инвентарю
    available_materials = filter_materials_by_inventory(materials, inventory)
    
    if not available_materials:
        return {'error': 'нет_доступных_материалов_в_инвентаре'}
    
    # Определение всех оксидов в целевой формуле
    target_oxides = list(target_umf.keys())
    
    # Создание полной матрицы оксидов
    full_oxide_matrix, material_names = create_oxide_matrix(available_materials, target_oxides)
    
    # Базовое решение
    base_solution = solve_recipe(full_oxide_matrix, target_umf, material_names, available_materials)
    solutions = [base_solution]
    
    # Попробуем найти альтернативные решения, меняя набор используемых материалов
    n_materials = len(available_materials)
    used_combinations = set()
    
    # Определяем минимально необходимое количество материалов
    min_required = max(3, len(target_oxides) - 3)  # Даем себе больше свободы в выборе минимума
    
    # Сначала попробуем решения с МИНИМАЛЬНЫМ количеством материалов
    if min_materials:
        # Начнем с очень малого количества материалов и постепенно увеличиваем
        
        # Пробуем от min_required до min_required + 5 материалов (более приоритетно)
        for subset_size in range(min_required, min(n_materials, min_required + 5)):
            # Генерируем больше комбинаций для маленьких подмножеств
            attempts = min(200, n_materials * 3)
            
            if logging:
                print(f"Ищем решения с {subset_size} материалами...")
            
            for _attempt in range(attempts):
                subset_indices = np.random.choice(n_materials, subset_size, replace=False)
                subset_key = tuple(sorted(subset_indices))
                
                if subset_key in used_combinations:
                    continue
                
                used_combinations.add(subset_key)
                
                # Создание подматрицы для выбранных материалов
                subset_matrix = full_oxide_matrix[:, subset_indices]
                subset_names = [material_names[i] for i in subset_indices]
                subset_materials = [available_materials[i] for i in subset_indices]
                
                # Проверка ранга матрицы (должен быть не слишком малым)
                rank = np.linalg.matrix_rank(subset_matrix)
                if rank < min_required - 1:  # Даем небольшой запас для ранга
                    continue
                
                solution = solve_recipe(subset_matrix, target_umf, subset_names, subset_materials)
                
                # Используем более высокий допуск ошибки для решений с меньшим количеством материалов
                # Чем меньше материалов, тем больше допуск
                actual_error_tolerance = error_tolerance * (1 + (max(6, n_materials) - subset_size) * 0.05)
                
                # Если найдено приемлемое решение с допустимой ошибкой
                if solution['recipe'] and solution['error'] < base_solution['error'] * (1 + actual_error_tolerance):
                    solutions.append(solution)
                    if len(solutions) > 1:
                        if logging:
                            print(f"Найдено решение с {len(solution['recipe'])} материалами и ошибкой {solution['error']}")
    
    # Если все еще нужны решения, ищем с разным количеством материалов
    if len(solutions) < max_solutions:
        for subset_size in range(min_required, min(n_materials, 12)):
            # Простой поиск решений для случайных подмножеств материалов
            for _attempt in range(min(30, n_materials)):
                subset_indices = np.random.choice(n_materials, subset_size, replace=False)
                subset_key = tuple(sorted(subset_indices))
                
                if subset_key in used_combinations:
                    continue
                
                used_combinations.add(subset_key)
                
                # Создание подматрицы для выбранных материалов
                subset_matrix = full_oxide_matrix[:, subset_indices]
                subset_names = [material_names[i] for i in subset_indices]
                subset_materials = [available_materials[i] for i in subset_indices]
                
                solution = solve_recipe(subset_matrix, target_umf, subset_names, subset_materials)
                
                # Если найдено приемлемое решение, добавляем в список
                if solution['recipe'] and solution['error'] < base_solution['error'] * 3:
                    solutions.append(solution)
                    
                    if len(solutions) >= max_solutions * 2:  # Генерируем больше решений для последующей сортировки
                        break
            
            if len(solutions) >= max_solutions * 2:
                break
    
    # Сортировка решений с учетом как ошибки, так и количества материалов
    if min_materials:
        # Создаем композитную метрику для сортировки:
        # решения с меньшим количеством материалов предпочтительнее,
        # если их ошибка не превышает ошибку лучшего решения более чем на error_tolerance
        best_error = min(solution['error'] for solution in solutions)
        
        def sort_key(solution):
            num_materials = solution['materials_count']
            err = solution['error']
            
            # Коэффициент увеличения допустимой ошибки в зависимости от кол-ва материалов
            # Чем меньше материалов, тем выше допуск
            error_multiplier = 1 + (0.1 * (8 - num_materials)) if num_materials < 8 else 1
            error_threshold = best_error * error_multiplier
            
            if err <= error_threshold:
                # Приоритизируем решения с меньшим числом материалов,
                # если ошибка в пределах увеличенного допуска
                return (0, num_materials, err)
            else:
                # Иначе сортируем по ошибке
                return (1, err, num_materials)
        
        solutions.sort(key=sort_key)
    else:
        # Сортировка только по ошибке
        solutions.sort(key=lambda x: x['error'])
    
    # Удаляем дубликаты по составу рецептов
    unique_solutions = []
    seen_recipes = set()
    
    for sol in solutions:
        # Создаем уникальный идентификатор рецепта
        recipe_key = tuple(sorted((k, round(v, 1)) for k, v in sol['recipe'].items()))
        if recipe_key not in seen_recipes:
            seen_recipes.add(recipe_key)
            unique_solutions.append(sol)
            
            if len(unique_solutions) >= max_solutions:
                break
    
    # Возвращаем указанное количество лучших решений
    return unique_solutions

# Команда для использования из консоли
def main():
    parser = argparse.ArgumentParser(description='Glaze Recipe Solver')
    parser.add_argument('--umf', type=str, required=True, help='Target UMF composition as JSON string, e.g., \'{"SiO2": 4, "Al2O3": 1, "Na2O": 0.5, "K2O": 0.5}\'')
    parser.add_argument('--solutions', type=int, default=3, help='Number of solutions to find (default: 3)')
    parser.add_argument('--min-materials', action='store_true', help='Prefer solutions with fewer materials')
    parser.add_argument('--error-tolerance', type=float, default=0.01, help='Acceptable error increase for solutions with fewer materials (default: 0.01)')
    parser.add_argument('--inventory', type=str, help='Custom inventory as JSON string, instead of using inventory.json')
    
    args = parser.parse_args()
    
    try:
        target_umf = json.loads(args.umf)
        inventory_data = json.loads(args.inventory) if args.inventory else None
    except json.JSONDecodeError as e:
        if args.inventory and 'args.inventory' in str(e):
            print("ошибка: неверный формат JSON для инвентаря")
        else:
            print("ошибка: неверный формат JSON для UMF")
        return
    
    solutions = find_multiple_solutions(
        target_umf, 
        max_solutions=args.solutions,
        min_materials=args.min_materials,
        error_tolerance=args.error_tolerance,
        inventory_data=inventory_data
    )
    
    if isinstance(solutions, dict) and 'error' in solutions:
        print(f"ошибка: {solutions['error']}")
    else:
        print(f"\nнайдено {len(solutions)} решений для заданной UMF-формулы:")
        
        for i, solution in enumerate(solutions):
            print(f"\n[решение {i+1}] ошибка: {solution['error']} | материалов: {solution['materials_count']}")
            print("\nсостав рецепта (вес в %):")
            
            for material, percentage in solution['recipe'].items():
                print(f"  {material}: {percentage}%")
            
            print("\nцелевой состав (UMF):")
            for oxide, value in sorted(solution['target_composition'].items()):
                print(f"  {oxide}: {value}")
            
            print("\nфактический состав (UMF):")
            for oxide, value in sorted(solution['actual_composition'].items()):
                print(f"  {oxide}: {value}")
            
            print("-" * 50)

if __name__ == "__main__":
    main()
