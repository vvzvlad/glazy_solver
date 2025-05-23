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
import logging
import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from umf_to_recipe import find_multiple_solutions, weights_to_umf, umf_to_weights, load_materials, make_json_safe

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('glaze_recipe_api')

app = Flask(__name__, static_folder=None)
CORS(app)  # Разрешаем CORS для всех маршрутов

# Путь к директории UI
UI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'UI')
# Путь к директории с данными
DATABASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'database')

@app.route('/api/solve', methods=['POST'])
def solve_recipe():
    """
    API endpoint для расчета рецепта глазури на основе UMF формулы
    
    POST JSON параметры:
    {
        "umf": {"SiO2": 4, "Al2O3": 1, "Na2O": 0.5, "K2O": 0.5},
        "max_solutions": 3,  // опционально, по умолчанию 3
        "min_materials": true,  // опционально, по умолчанию true
        "error_tolerance": 0.01,  // опционально, по умолчанию 0.01
        "inventory": ["Material1", "Material2", ...]  // опционально, список доступных материалов
    }
    
    Возвращает:
    [
        {
            "recipe": {"Material1": 45.2, "Material2": 54.8},
            "error": 0.0123,
            "target_composition": {"SiO2": 4, "Al2O3": 1, ...},
            "actual_composition": {"SiO2": 3.98, "Al2O3": 1.02, ...},
            "weight_composition": {"SiO2": 65.2, "Al2O3": 18.1, ...},
            "materials_count": 2,
            "recipe_umf": {"SiO2": 3.98, "Al2O3": 1.02, ...}  // UMF для конкретного рецепта
        },
        ...
    ]
    """
    try:
        data = request.get_json()
        
        if not data or 'umf' not in data:
            logger.warning("missing_umf parameter in request")
            return jsonify({"error": "missing_umf", "message": "umf parameter is required"}), 400
        
        umf = data['umf']
        max_solutions = data.get('max_solutions', 3)
        min_materials = data.get('min_materials', True)
        error_tolerance = data.get('error_tolerance', 0.01)
        inventory_data = data.get('inventory', None)
        
        logger.info(f"solving recipe for umf: {umf}, max_solutions: {max_solutions}, min_materials: {min_materials}")
        
        solutions = find_multiple_solutions(
            umf, 
            max_solutions=max_solutions,
            min_materials=min_materials,
            error_tolerance=error_tolerance,
            inventory_data=inventory_data
        )
        
        if isinstance(solutions, dict) and 'error' in solutions:
            logger.error(f"calculation_error: {solutions['error']}")
            return jsonify({"error": "calculation_error", "message": solutions['error']}), 500
        
        # Добавляем информацию о UMF для каждого рецепта
        for solution in solutions:
            # UMF для конкретного рецепта уже содержится в actual_composition, но также добавим его как отдельное поле
            # для удобства использования на фронтенде
            solution['recipe_umf'] = solution['actual_composition']
        
        # Обрабатываем результаты для безопасной сериализации в JSON
        safe_solutions = make_json_safe(solutions)
        
        logger.info(f"found {len(solutions)} solutions")
        return jsonify(safe_solutions)
    
    except Exception as e:
        logger.exception(f"server_error: {str(e)}")
        return jsonify({"error": "server_error", "message": str(e)}), 500

@app.route('/api/molar_masses', methods=['GET'])
def get_molar_masses():
    """
    API endpoint для получения списка оксидов и их молярных масс
    
    Возвращает:
    {
        "SiO2": 60.084,
        "Al2O3": 101.961,
        ...
    }
    """
    try:
        molar_masses_path = os.path.join(DATABASE_DIR, 'molar_masses.json')
        
        if not os.path.exists(molar_masses_path):
            logger.error(f"molar_masses_file_not_found: {molar_masses_path}")
            return jsonify({"error": "file_not_found", "message": "Molar masses file not found"}), 404
        
        with open(molar_masses_path, 'r', encoding='utf-8') as f:
            molar_masses = json.load(f)
        
        logger.info(f"returning {len(molar_masses)} molar masses")
        return jsonify(molar_masses)
    
    except Exception as e:
        logger.exception(f"molar_masses_error: {str(e)}")
        return jsonify({"error": "server_error", "message": str(e)}), 500

@app.route('/api/umf_to_weights', methods=['POST'])
def convert_umf_to_weights():
    """
    API endpoint для конвертации UMF формулы в весовые проценты
    
    POST JSON параметры:
    {
        "umf": {"SiO2": 4, "Al2O3": 1, "Na2O": 0.5, "K2O": 0.5}
    }
    
    Возвращает:
    {
        "weights": {"SiO2": 65.2, "Al2O3": 18.1, "Na2O": 8.4, "K2O": 8.3}
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'umf' not in data:
            logger.warning("missing_umf parameter in umf_to_weights request")
            return jsonify({"error": "missing_umf", "message": "umf parameter is required"}), 400
        
        umf = data['umf']
        logger.info(f"converting umf to weights: {umf}")
        
        weights = umf_to_weights(umf)
        
        return jsonify({"weights": weights})
    
    except Exception as e:
        logger.exception(f"umf_to_weights_error: {str(e)}")
        return jsonify({"error": "server_error", "message": str(e)}), 500

@app.route('/api/weights_to_umf', methods=['POST'])
def convert_weights_to_umf():
    """
    API endpoint для конвертации весовых процентов в UMF формулу
    
    POST JSON параметры:
    {
        "weights": {"SiO2": 65.2, "Al2O3": 18.1, "Na2O": 8.4, "K2O": 8.3}
    }
    
    Возвращает:
    {
        "umf": {"SiO2": 4, "Al2O3": 1, "Na2O": 0.5, "K2O": 0.5}
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'weights' not in data:
            logger.warning("missing_weights parameter in weights_to_umf request")
            return jsonify({"error": "missing_weights", "message": "weights parameter is required"}), 400
        
        weights = data['weights']
        logger.info(f"converting weights to umf: {weights}")
        
        umf = weights_to_umf(weights)
        
        return jsonify({"umf": umf})
    
    except Exception as e:
        logger.exception(f"weights_to_umf_error: {str(e)}")
        return jsonify({"error": "server_error", "message": str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """
    API endpoint для проверки работоспособности сервера
    """
    logger.debug("health check requested")
    return jsonify({"status": "ok"})

@app.route('/api/materials', methods=['GET'])
def get_materials():
    """
    API endpoint для получения списка всех доступных материалов
    
    GET параметры:
        inventory_only (bool, optional): Если True, возвращает только материалы из инвентаря
    
    Возвращает:
    [
        {
            "name": "Материал 1",
            "formula": {"SiO2": 65.2, "Al2O3": 18.1, ...},
            "description": "Описание материала",
            "id": 123,
            ...
        },
        ...
    ]
    """
    try:
        inventory_only = request.args.get('inventory_only', 'false').lower() == 'true'
        
        # Загружаем все материалы
        materials = load_materials()
        
        if inventory_only:
            # Фильтруем материалы по флагу inInventory
            materials = [material for material in materials if material.get('inInventory', False)]
        
        logger.info(f"returning {len(materials)} materials, inventory_only={inventory_only}")
        return jsonify(materials)
    
    except Exception as e:
        logger.exception(f"materials_error: {str(e)}")
        return jsonify({"error": "server_error", "message": str(e)}), 500

# Отдача UI статических файлов
@app.route('/', defaults={'path': 'index.html'})
@app.route('/<path:path>')
def serve_ui(path):
    """
    Отдача статических файлов UI
    """
    if path.startswith('api/'):
        return jsonify({"error": "not_found", "message": "API endpoint not found"}), 404
    
    logger.debug(f"serving ui file: {path}")
    return send_from_directory(UI_DIR, path)

if __name__ == '__main__':
    logger.info("starting glaze recipe api server on 0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False) 