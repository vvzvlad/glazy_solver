#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# flake8: noqa
# pylint: disable=broad-exception-raised, raise-missing-from, too-many-arguments, redefined-outer-name
# pylint: disable=multiple-statements, logging-fstring-interpolation, trailing-whitespace, line-too-long
# pylint: disable=broad-exception-caught, missing-function-docstring, missing-class-docstring
# pylint: disable=f-string-without-interpolation
# pylance: disable=reportMissingImports, reportMissingModuleSource

import unittest
import sys
import os
import json

# Fix imports by adding parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from recipe_to_umf import analyze_recipe, print_recipe_analysis
from umf_to_recipe import solve_glaze_recipe, find_multiple_solutions

class TestRecipesFromJson(unittest.TestCase):
    

    def test_recipe_reverse_engineering(self):
        """
        Проверяет процесс обратного восстановления рецепта:
        1. Берем рецепт
        2. Рассчитываем его UMF
        3. Используем UMF для восстановления рецепта через солвер
        4. Сравниваем полученный рецепт с оригинальным
        """
        # Загрузка рецептов из JSON-файла
        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'database', 'recipes.json'), 'r', encoding='utf-8') as f:
            recipes_data = json.load(f)
        
        test_recipes = recipes_data 
        
        print(f"\n\n=== ТЕСТИРОВАНИЕ ОБРАТНОГО ВОССТАНОВЛЕНИЯ РЕЦЕПТОВ ===")
        print(f"Тестируем {len(test_recipes)} рецептов")
        
        for recipe_entry in test_recipes:
            recipe_name = recipe_entry.get('name', 'Unnamed Recipe')
            print(f"\n=== Рецепт: {recipe_name} ===")
            
            # Преобразуем формат рецепта в ожидаемый формат словаря
            original_recipe = {}
            for ingredient in recipe_entry.get('recipe', []):
                # Пропускаем добавки, если они не включены в расчеты
                if ingredient.get('isAddition', False) and not recipe_entry.get('includeAdditionsIntoCalculations', False):
                    continue
                
                material_name = ingredient.get('material', {}).get('name', '')
                if material_name:
                    original_recipe[material_name] = ingredient.get('value', 0)
            
            # Проверяем, есть ли ингредиенты в рецепте
            if not original_recipe:
                print(f"Пропускаем: нет ингредиентов в рецепте")
                continue
            
            # Шаг 1: Анализ рецепта для получения UMF
            try:
                print(f"Оригинальный рецепт:")
                total_original = sum(original_recipe.values())
                for material, value in sorted(original_recipe.items()):
                    percentage = (value / total_original) * 100
                    print(f"  {material}: {value} ({percentage:.2f}%)")

                print("\n")

                analysis = analyze_recipe(original_recipe)
                if 'umf' not in analysis or not analysis['umf']:
                    print(f"Пропускаем: не удалось рассчитать UMF для рецепта")
                    continue
                
                target_umf = analysis['umf']
                print(f"Рассчитанная UMF:")
                for oxide, value in sorted(target_umf.items()):
                    print(f"  {oxide}: {value}")
                
                # Шаг 2: Использование UMF для восстановления рецепта
                solutions = find_multiple_solutions(target_umf, min_materials=True, error_tolerance=0.1)
                
                # Проверяем, что solutions - это не словарь с ошибкой
                if isinstance(solutions, dict) and 'error' in solutions:
                    print(f"Ошибка при решении: {solutions['error']}")
                    continue
                
                # Берем первое (лучшее) решение
                if not solutions or len(solutions) == 0:
                    print("Решения не найдены")
                    continue
                
                best_solution = solutions[0]
                solved_recipe = best_solution['recipe']
                
                print(f"\nВосстановленный рецепт:")
                for material, percentage in sorted(solved_recipe.items()):
                    print(f"  {material}: {percentage:.2f}%")
                
                # Анализ различий
                print(f"\nАнализ различий:")
                
                # Объединяем все ключи из обоих рецептов
                all_materials = set(original_recipe.keys()) | set(solved_recipe.keys())
                
                # Нормализуем оригинальный рецепт к процентам
                original_percentage = {}
                for material, value in original_recipe.items():
                    original_percentage[material] = (value / total_original) * 100
                
                # Различия в материалах
                missing_in_solved = []
                missing_in_original = []
                different_values = []
                
                for material in all_materials:
                    if material in original_percentage and material not in solved_recipe:
                        missing_in_solved.append((material, original_percentage[material]))
                    elif material not in original_percentage and material in solved_recipe:
                        missing_in_original.append((material, solved_recipe[material]))
                    elif material in original_percentage and material in solved_recipe:
                        diff = abs(original_percentage[material] - solved_recipe[material])
                        if diff > 1.0:  # Разница более 1%
                            different_values.append((material, original_percentage[material], solved_recipe[material], diff))
                
                # Вывод различий
                if missing_in_solved:
                    print("\nМатериалы, отсутствующие в восстановленном рецепте:")
                    for material, value in missing_in_solved:
                        print(f"  {material}: {value:.2f}%")
                
                if missing_in_original:
                    print("\nДополнительные материалы в восстановленном рецепте:")
                    for material, value in missing_in_original:
                        print(f"  {material}: {value:.2f}%")
                
                if different_values:
                    print("\nМатериалы с разными пропорциями:")
                    for material, orig, solved, diff in different_values:
                        print(f"  {material}: оригинал {orig:.2f}%, решение {solved:.2f}%, разница {diff:.2f}%")
                
                # Проверка ошибки в UMF
                print(f"\nОшибка в UMF: {best_solution['error']}")
                    
            except Exception as e:
                print(f"Ошибка при обратном тестировании: {e}")
                import traceback
                traceback.print_exc()
                continue

if __name__ == "__main__":
    unittest.main()