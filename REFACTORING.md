# ТЗ: завершение рефакторинга glazy_solver

## 1. Контекст и текущее состояние

Проект — решатель рецептов керамических глазурей: по целевой UMF-формуле подбирает
рецепт из материалов (NNLS-оптимизация), Flask API + веб-UI.

В мае 2025 (коммит `eddc7bb` «refactoring wip») был начат и брошен на середине
рефакторинг. Итог — **не работает ни один путь исполнения**:

| # | Проблема | Где |
|---|----------|-----|
| 1 | `api_server.py` импортирует модуль `umf_to_recipe`, переименованный в `umf_to_recipe_old.py` → сервер не стартует | `api_server.py:17` |
| 2 | Из `umf_to_recipe_old.py` вырезаны `load_materials`, `umf_to_weights`, `weights_to_umf` (переехали в `common.py`), но вызовы остались без импортов → NameError | `umf_to_recipe_old.py:78,112,169` |
| 3 | Функции `make_json_safe`, `load_inventory`, `filter_materials_by_inventory`, `solve_glaze_recipe` потеряны совсем: вырезаны из old, в common не добавлены | история `eddc7bb^:umf_to_recipe.py` |
| 4 | `tests/test_recipe_to_umf.py` импортирует удалённый модуль `recipe_to_umf` | `tests/test_recipe_to_umf.py:18` |
| 5 | `tests/test_umf_to_recipe.py` импортирует переименованный модуль и удалённую `solve_glaze_recipe` | `tests/test_umf_to_recipe.py:16` |
| 6 | Новый тест `tests/test_individual_recipes.py` (11 эталонных рецептов) ожидает функцию `find_best_recipe`, которой нет нигде, и вызывает неопределённый метод `create_inventory_from_materials()` | `tests/test_individual_recipes.py:19,205` |
| 7 | Недописанный новый решатель `find_variants` в `solve.py` (незакоммичен) читает `material['umf']`, но в `materials.json` у всех 216 материалов есть только `formula` (весовые %) → решатель всегда получает нули и неработоспособен в принципе | `solve.py:48` |
| 8 | `database/inventory.json` удалён, но старый `load_inventory()` читал именно его | история |

**Решения владельца проекта (зафиксированы, не обсуждаются):**
- Стратегия: обе ступени — сначала восстановить рабочее состояние на старом ядре,
  затем достроить новый решатель, затем сравнить оба на эталонных рецептах.
- Незакоммиченные правки `database/priorities.json` (приоритеты 2/5/10) и
  `database/molar_masses.json` (округление до 2 знаков) — **намеренные, сохранить**.
- Приоритеты — фича нового решателя: поиск ведётся от приоритетных (базовых)
  материалов, низкоприоритетные добираются «подгоном».
- Алгоритм нового решателя — **итеративный, как решает человек**: 90% рецептов
  сходятся за 5–6 итераций добавления материала под дефицитный оксид.

## 2. Целевая архитектура

```
common.py            — данные и математика, общие для всех решателей:
                       загрузка (materials + priorities + inventory, recipes,
                       molar_masses), UMF↔веса, расчёт ошибок, make_json_safe,
                       форматирование вывода
solver_classic.py    — старое проверенное ядро (бывш. umf_to_recipe_old.py):
                       NNLS по случайным подмножествам материалов
solver_iterative.py  — новый решатель (бывш. solve.py): итеративный,
                       приоритетный, human-like
api_server.py        — Flask API + отдача UI; движок по умолчанию classic,
                       опциональный параметр "solver" для выбора движка
compare_solvers.py   — скрипт сравнения движков на эталонных рецептах
tests/
  fixtures/reference_recipes.json — 11 эталонных пар (UMF + оригинальный рецепт),
                       вынесены из test_individual_recipes.py
  test_common.py     — тесты UMF-математики (замена test_recipe_to_umf.py)
  test_solver_classic.py — интеграционный тест классики (замена test_umf_to_recipe.py)
  test_individual_recipes.py — 11 эталонных рецептов через новый решатель
```

Удаляются: `umf_to_recipe_old.py` (после переноса), `solve.py` (после переноса),
`tests/test_recipe_to_umf.py`, `tests/test_umf_to_recipe.py` (после замены).

## 3. Ступень 1 — восстановить старое ядро

### 3.1 `common.py` — дополнить
- Добавить `make_json_safe(obj)` — восстановить из `git show eddc7bb^:umf_to_recipe.py`
  (строки ~300–320): рекурсивная замена inf/nan на строки для JSON.
- Добавить `resolve_inventory(inventory_data=None)`:
  - если `inventory_data` передан (список имён) — вернуть его;
  - иначе — имена материалов с `inInventory: true` из `materials.json`
    (замена удалённого `inventory.json`).
- Добавить `filter_materials_by_inventory(materials, inventory)` — восстановить
  из истории (фильтр списка материалов по списку имён).
- Существующие функции не менять (обратная совместимость с их вызовами).

### 3.2 `solver_classic.py` — переименовать и починить
- `git mv umf_to_recipe_old.py solver_classic.py`.
- Добавить `from common import umf_to_weights, weights_to_umf, make_json_safe,
  resolve_inventory, filter_materials_by_inventory` и
  `load_materials` (вызывать как `load_materials(only_inventory=False, priority=True)` —
  старая семантика «все материалы», приоритеты не мешают).
- В `find_multiple_solutions`: заменить `load_inventory(inventory_data)` на
  `resolve_inventory(inventory_data)`; остальную логику **не менять** — алгоритм
  проверенный.
- Восстановить `solve_glaze_recipe(target_umf, inventory_data=None)` из истории
  (строки ~272–297) с адаптацией к новым именам загрузчиков.
- Сохранить CLI `main()` с argparse.

### 3.3 `api_server.py` — починить импорты
- Строку 17 заменить на:
  `from solver_classic import find_multiple_solutions` +
  `from common import weights_to_umf, umf_to_weights, load_materials, make_json_safe`.
- В `get_materials()` вызов `load_materials()` → `load_materials(only_inventory=False,
  priority=True)` (эндпоинт отдаёт все материалы, фильтрует сам).
- Больше ничего не трогать (контракт API и UI сохраняется байт-в-байт).

### 3.4 Тесты ступени 1
- `tests/test_common.py` (новый, вместо `test_recipe_to_umf.py`): проверки
  `weights_to_umf`/`umf_to_weights` round-trip на 2–3 составах,
  `calculate_umf_from_recipe`, `make_json_safe` (inf/nan/вложенность).
- `tests/test_solver_classic.py` (новый, вместо `test_umf_to_recipe.py`):
  перенести существующий интеграционный сценарий (`test_umf_to_recipe.py:19–88`)
  на `solver_classic.solve_glaze_recipe` — целевая UMF прозрачной глазури,
  проверка структуры решения, суммы 100%, наличия ≥3 ожидаемых материалов.
- Удалить оба старых файла тестов.

### 3.5 Критерий готовности ступени 1
- `.venv/bin/python -c "import api_server"` — без ошибок.
- Сервер стартует, `curl POST /api/solve_recipe` с UMF прозрачной глазури
  возвращает непустые решения; `/api/materials`, `/api/molar_masses`,
  `/api/umf_to_weights`, `/api/weights_to_umf`, `/api/health` отвечают.
- Новые тесты ступени 1 зелёные.

## 4. Ступень 2 — новый итеративный решатель

### 4.1 Принцип (как решает человек)
1. Взять **базовый набор**: материалы наивысшей приоритетной группы инвентаря
   (минимальное число в `priorities.json`; сейчас это 5 базовых с приоритетом 2).
2. Решить NNLS на текущем наборе **в весовом пространстве** (та же математика,
   что в классике: target UMF → веса `umf_to_weights`, NNLS по `formula`-матрице,
   результат → `weights_to_umf`, ошибка по оксидам). Никакой линейной алгебры
   над UMF напрямую — UMF нелинейна относительно смешивания.
3. Посмотреть остаточную ошибку по оксидам: найти самый дефицитный/избыточный оксид.
4. Выбрать из оставшегося инвентаря кандидата, лучше всех закрывающего дефицит
   (наибольшая доля нужного оксида в `formula` при минимуме «загрязнения» уже
   сошедшихся оксидов). При равных кандидатах — предпочесть высокий приоритет
   (меньшее число).
5. Добавить кандидата в набор, вернуться к шагу 2.
6. Остановка: ошибка ≤ `error_threshold`, или `max_materials`, или
   `max_iterations` (default 8 — с запасом к человеческим 5–6), или ошибка
   перестала улучшаться (менее чем на 1% за итерацию).
7. Для `max_solutions > 1` — ветвление: на шаге 4 брать top-K кандидатов
   (K=2..3, beam search), собирать пул решений, дедуплицировать по составу.
8. Материалы с весом < 0.1% выбрасывать из рецепта и пересчитывать.

### 4.2 Контракт API (диктуется `tests/test_individual_recipes.py`)
```python
def find_best_recipe(inventory, target_umf, min_materials=1, max_materials=10,
                     max_solutions=5, verbose=False, error_threshold=0.1) -> list[dict]
```
- `inventory` — список имён материалов.
- Каждое решение — dict с ключами: `recipe` ({имя: вес%, сумма = 100}),
  `error` (float, метрика `calculate_umf_error`), `result_umf`, `target_umf`,
  `materials_count`, `iterations` (сколько шагов потребовалось).
- Список отсортирован: лучший — первый (меньше ошибка; при близкой ошибке —
  меньше материалов).

### 4.3 Файл `solver_iterative.py`
- Основа — текущий незакоммиченный `solve.py`, но: комбинаторный перебор
  `find_variants` **удалить** (неработоспособен, см. проблему №7), заменить
  итеративным алгоритмом из 4.1.
- Использовать общую математику из `common.py` и, где уместно, NNLS-ядро
  `solver_classic.solve_recipe`/`create_oxide_matrix` (не дублировать).
- CLI `main()`: захардкоженный тестовый рецепт из текущего `solve.py` оставить
  как smoke-run.

### 4.4 Тесты ступени 2
- В `tests/test_individual_recipes.py`:
  - импорт: `from solver_iterative import find_best_recipe`;
  - дописать недостающий `create_inventory_from_materials()`: имена материалов
    с `inInventory: true`;
  - 11 эталонных пар (UMF + рецепт) вынести в
    `tests/fixtures/reference_recipes.json`, тесты читают фикстуры;
  - логику проверок (`check_recipe`) не менять — она осмысленная.
- Допустимо, если часть из 11 рецептов не сойдётся идеально — это вход для
  ступени 3; но тест 01 (прозрачная глазурь: 5 базовых материалов с приоритетом 2)
  обязан сходиться — это прямая проверка приоритетного поиска.

## 5. Ступень 3 — сравнение движков

### 5.1 `compare_solvers.py`
- Читает `tests/fixtures/reference_recipes.json`, каждый рецепт прогоняет через
  оба движка (classic: `find_multiple_solutions`, iterative: `find_best_recipe`)
  с одинаковым инвентарём (inInventory-материалы).
- Метрики на лучший вариант каждого движка:
  суммарная и максимальная по-оксидная ошибка UMF, `calculate_umf_error`,
  число материалов, совпадение состава с оригиналом (общие материалы / объединение,
  сумма |Δдолей| по общим), время решения, число итераций (для iterative).
- Вывод: таблица в консоль + `comparison_results.md` (генерируется, в git не
  коммитится — добавить в `.gitignore`).
- classic недетерминирован (`np.random.choice`) — фиксировать seed для
  воспроизводимости.

### 5.2 Переключатель движка в API (опционально, дёшево)
- В `POST /api/solve_recipe` опциональный параметр `"solver": "classic" |
  "iterative"` (default `"classic"`); iterative-решения приводить к формату
  ответа classic (ключи `recipe`, `error`, `actual_composition` и т.д.),
  чтобы UI работал без изменений.

## 6. База данных
- `database/priorities.json` — оставить как в рабочем дереве (базовые = 2).
  Семантика: меньше число = выше приоритет; отсутствующий материал = 1
  (наивысший) — **проверить согласованность**: сейчас `common.load_materials`
  ставит default 1, при этом «базовым» дали 2 — значит материалы без записи
  в priorities.json оказались бы приоритетнее базовых. Исправить default на
  максимальный (низший) приоритет, например 100, с комментарием.
- `database/molar_masses.json` — оставить округление как есть.
- `database/recipes.json` — не трогать.

## 7. Правила выполнения
- Работа в отдельной ветке `refactoring/solver` от `main`; `main` не трогать.
- Первым коммитом на ветке зафиксировать текущий незакоммиченный WIP как
  baseline (чтобы ничего не потерять), дальше — атомарные коммиты по ступеням.
- Комментарии в коде — только на английском. Русские комментарии в
  переносимом старом коде — перевести в затрагиваемых местах.
- Таргетные правки, никаких попутных переделок вне ТЗ.
- Python 3.13, зависимости не добавлять (numpy, scipy, flask, flask-cors уже
  в `requirements.txt`).

## 8. Критерии приёмки (Definition of Done)
1. `import api_server` проходит, сервер стартует, все 6 эндпоинтов отвечают,
   UI получает решения.
2. `python -m unittest discover tests` — все тесты зелёные.
3. `compare_solvers.py` выдаёт сравнительную таблицу по 11 рецептам; оба движка
   возвращают решения; результаты приложены к финальному отчёту.
4. В репозитории нет мёртвого кода: `umf_to_recipe_old.py`, старый `solve.py`,
   сломанные тесты удалены; ни одного импорта несуществующих модулей/функций
   (проверка: `grep -rn "umf_to_recipe\|recipe_to_umf\|find_variants" --include="*.py"`
   возвращает пусто).
5. Тест 01 (прозрачная глазурь) сходится итеративным решателем к 5 базовым
   материалам.
