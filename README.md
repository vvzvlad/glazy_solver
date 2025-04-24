# Glaze Recipe Solver API

Веб-сервер, предоставляющий API для расчета рецептов керамических глазурей на основе UMF формулы.

## Установка и запуск

1. Установите зависимости:
   ```
   pip install -r requirements.txt
   ```

2. Запустите сервер:
   ```
   python api_server.py
   ```

Сервер будет доступен по адресу http://localhost:5000

## API Спецификация

### Расчет рецепта глазури

**Endpoint:** `POST /api/solve`

**Input:**
```json
{
  "umf": {
    "SiO2": 4,
    "Al2O3": 1,
    "Na2O": 0.5,
    "K2O": 0.5
  },
  "max_solutions": 3,
  "min_materials": true,
  "error_tolerance": 0.01
}
```

Параметры:
- `umf` (обязательный): UMF формула глазури в виде словаря оксид:значение
- `max_solutions` (опциональный, по умолчанию 3): максимальное количество возвращаемых решений
- `min_materials` (опциональный, по умолчанию true): предпочитать решения с меньшим количеством материалов
- `error_tolerance` (опциональный, по умолчанию 0.01): допустимое увеличение ошибки для решений с меньшим числом материалов

**Output:**
```json
[
  {
    "recipe": {
      "Custer Feldspar": 45.2,
      "Silica": 54.8
    },
    "error": 0.0123,
    "target_composition": {
      "SiO2": 4,
      "Al2O3": 1,
      "Na2O": 0.5,
      "K2O": 0.5
    },
    "actual_composition": {
      "SiO2": 3.98,
      "Al2O3": 1.02,
      "Na2O": 0.49,
      "K2O": 0.51
    },
    "weight_composition": {
      "SiO2": 65.2,
      "Al2O3": 18.1,
      "Na2O": 8.4,
      "K2O": 8.3
    },
    "materials_count": 2
  },
  {...}
]
```

### Конвертация UMF в весовые проценты

**Endpoint:** `POST /api/umf_to_weights`

**Input:**
```json
{
  "umf": {
    "SiO2": 4,
    "Al2O3": 1,
    "Na2O": 0.5,
    "K2O": 0.5
  }
}
```

**Output:**
```json
{
  "weights": {
    "SiO2": 65.2,
    "Al2O3": 18.1,
    "Na2O": 8.4,
    "K2O": 8.3
  }
}
```

### Конвертация весовых процентов в UMF

**Endpoint:** `POST /api/weights_to_umf`

**Input:**
```json
{
  "weights": {
    "SiO2": 65.2,
    "Al2O3": 18.1,
    "Na2O": 8.4,
    "K2O": 8.3
  }
}
```

**Output:**
```json
{
  "umf": {
    "SiO2": 4,
    "Al2O3": 1,
    "Na2O": 0.5,
    "K2O": 0.5
  }
}
```

### Проверка работоспособности сервера

**Endpoint:** `GET /api/health`

**Output:**
```json
{
  "status": "ok"
}
```

## Коды ошибок

- `400 Bad Request`: Отсутствуют обязательные параметры
- `500 Internal Server Error`: Ошибка при расчете рецепта или конвертации

## Пример использования (curl)

```bash
curl -X POST http://localhost:5000/api/solve \
  -H "Content-Type: application/json" \
  -d '{"umf": {"SiO2": 4, "Al2O3": 1, "Na2O": 0.5, "K2O": 0.5}}'
``` 