#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# flake8: noqa
# pylint: disable=broad-exception-raised, raise-missing-from, too-many-arguments, redefined-outer-name
# pylint: disable=multiple-statements, logging-fstring-interpolation, trailing-whitespace, line-too-long
# pylint: disable=broad-exception-caught, missing-function-docstring, missing-class-docstring
# pylint: disable=f-string-without-interpolation
# pylance: disable=reportMissingImports, reportMissingModuleSource

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from umf_to_recipe import solve_glaze_recipe

# Тестовые данные на основе изображения из пользовательского ввода
test_umf = {
    "Na2O": 0.143,
    "K2O": 0.086,
    "MgO": 0.048,
    "CaO": 0.717,
    "SrO": 0.005,
    "Fe2O3": 0.002,
    "Al2O3": 0.378,
    "B2O3": 0.265,
    "TiO2": 0.003,
    "SiO2": 3.144
}

expected_materials = [
    "Нефелин-сиенит VR",
    "Кварцевая мука Кв",
    "Волластонит МИВО",
    "Улексит (Химпэк)",
    "Каолин КЖФ-1"
]

def test_solve_glaze_recipe():
    """Интеграционный тест основной функции решения с реальными данными из базы"""
    solution = solve_glaze_recipe(test_umf)
    
    print(f"\n[Интеграционный тест] Ошибка решения: {solution['error']}")
    
    # Проверка структуры решения
    if 'recipe' not in solution or 'error' not in solution:
        print("FAILED: Решение не содержит ожидаемых полей")
        return False
    
    # Проверка содержимого рецепта
    if len(solution['recipe']) == 0:
        print("FAILED: Рецепт пуст")
        return False
    
    # Вывод найденного рецепта
    print("\nНайденный рецепт:")
    for material, percentage in solution['recipe'].items():
        print(f"  {material}: {percentage}%")
    
    # Проверка наличия ожидаемых материалов
    found_expected = 0
    for material in expected_materials:
        if material in solution['recipe']:
            found_expected += 1
            print(f"FOUND: {material}")
    
    # Должно быть найдено хотя бы 3 из ожидаемых материалов
    if found_expected < 3:
        print(f"FAILED: Найдено только {found_expected} из ожидаемых материалов")
        return False
    
    # Суммарный процент должен быть около 100%
    total = sum(solution['recipe'].values())
    if abs(total - 100.0) > 0.1:
        print(f"FAILED: Сумма процентов {total} отличается от 100%")
        return False
    
    # Вывод сравнения целевой и фактической формул
    print("\nСравнение целевой и фактической UMF:")
    for oxide in sorted(set(solution['target_composition']) | set(solution['actual_composition'])):
        target = solution['target_composition'].get(oxide, 0)
        actual = solution['actual_composition'].get(oxide, 0)
        diff = abs(target - actual)
        print(f"  {oxide}: цель={target}, факт={actual}, разница={diff:.4f}")
    
    print("\nТест решения успешно пройден!")
    return True


if __name__ == "__main__":
    
    success = True
    
    print("\n========== Тест одиночного решения ==========")
    if not test_solve_glaze_recipe():
        success = False
    
    if success:
        print("\nТесты успешно пройдены!")
    else:
        print("\nЕсть ошибки в тестах!")
        exit(1) 