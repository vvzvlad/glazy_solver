# API документация Glazy Solver

## Базовая информация

API-сервер доступен по адресу `http://localhost:5000/api` при локальном запуске.

Все API-методы принимают и возвращают данные в формате JSON.

## Endpoints

### Расчет рецепта глазури

**Endpoint:** `POST /api/solve`

**Описание:** Calculates recipes for the given UMF formula using available materials.

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

**Параметры:**
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

**Пример запроса с cURL:**
```bash
curl -X POST http://localhost:5000/api/solve \
  -H "Content-Type: application/json" \
  -d '{"umf": {"SiO2": 4, "Al2O3": 1, "Na2O": 0.5, "K2O": 0.5}}'
```

### Конвертация UMF в весовые проценты

**Endpoint:** `POST /api/umf_to_weights`

**Описание:** Преобразует UMF-формулу в весовые проценты.

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

**Пример запроса с cURL:**
```bash
curl -X POST http://localhost:5000/api/umf_to_weights \
  -H "Content-Type: application/json" \
  -d '{"umf": {"SiO2": 4, "Al2O3": 1, "Na2O": 0.5, "K2O": 0.5}}'
```

### Конвертация весовых процентов в UMF

**Endpoint:** `POST /api/weights_to_umf`

**Описание:** Преобразует весовые проценты оксидов в UMF-формулу.

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

**Пример запроса с cURL:**
```bash
curl -X POST http://localhost:5000/api/weights_to_umf \
  -H "Content-Type: application/json" \
  -d '{"weights": {"SiO2": 65.2, "Al2O3": 18.1, "Na2O": 8.4, "K2O": 8.3}}'
```

### Проверка работоспособности сервера

**Endpoint:** `GET /api/health`

**Описание:** Проверяет работоспособность API-сервера.

**Output:**
```json
{
  "status": "ok"
}
```

**Пример запроса с cURL:**
```bash
curl -X GET http://localhost:5000/api/health
```

## Коды ошибок

### HTTP Status Codes

- `200 OK` - Запрос выполнен успешно
- `400 Bad Request` - Отсутствуют обязательные параметры или неправильный формат запроса 
- `404 Not Found` - Запрашиваемый ресурс не найден
- `500 Internal Server Error` - Ошибка при выполнении запроса на сервере

### Error Response Format

```json
{
  "error": "error_code",
  "message": "Detailed error message"
}
```

Возможные значения `error_code`:
- `missing_umf` - Отсутствует обязательный параметр UMF
- `missing_weights` - Отсутствует обязательный параметр weights
- `calculation_error` - Ошибка при расчете или конвертации
- `server_error` - Внутренняя ошибка сервера
- `not_found` - Запрашиваемый ресурс не найден 