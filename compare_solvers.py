#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# flake8: noqa
# pylint: disable=broad-exception-raised, raise-missing-from, too-many-arguments, redefined-outer-name
# pylint: disable=too-many-positional-arguments, too-many-locals
# pylint: disable=multiple-statements, logging-fstring-interpolation, trailing-whitespace, line-too-long
# pylint: disable=broad-exception-caught, missing-function-docstring, missing-class-docstring
# pylint: disable=f-string-without-interpolation
# pylance: disable=reportMissingImports, reportMissingModuleSource
# type: ignore

"""
Compare the two glaze solvers on the reference recipes.

    classic   -> solver_classic.find_multiple_solutions (NNLS over random
                 subsets of the inventory, drawn from a seeded generator, so
                 the run is reproducible for a given --seed)
    iterative -> solver_iterative.find_best_recipe (priority driven, adds one
                 material at a time, deterministic)

Both engines are run on the same inventory (the materials flagged inInventory
in database/materials.json) and on the same 11 reference recipes taken from
tests/fixtures/reference_recipes.json.

Why the UMF is recomputed here instead of being read from the engines:

    Both engines now report the plain UMF of their recipe, so the two formulas
    are the same quantity.  What is still not shared is the TARGET each engine
    measures itself against: classic scores over the oxides the target names,
    while iterative scores over its cleaned target extended with zeros for the
    oxides nobody asked for, and it reports that cleaned target rather than the
    dictionary of the request.

    Every metric in this report is therefore computed on a UMF recomputed from
    the recipe itself with calculate_recipe_composition + weights_to_umf and
    scored the exact same way for both engines.  The errors the engines report
    themselves are still shown, in the 'native' column, where classic agrees
    with 'umfErr' and iterative answers its own question.
"""

import argparse
import contextlib
import io
import json
import os
import statistics
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from common import (
    filter_materials_by_inventory,
    load_materials,
    resolve_inventory,
    weights_to_umf,
)
from solver_classic import (
    calculate_recipe_composition,
    calculate_umf_error,
    find_multiple_solutions,
)
from solver_iterative import find_best_recipe


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIXTURES_PATH = os.path.join(SCRIPT_DIR, 'tests', 'fixtures', 'reference_recipes.json')
DEFAULT_OUTPUT = os.path.join(SCRIPT_DIR, 'comparison_results.md')

# The classic solver draws random material subsets, so the seed has to be
# passed to it to make the whole comparison reproducible
DEFAULT_SEED = 42

# Both engines are asked for the same number of solutions; only the best one
# of each is measured
MAX_SOLUTIONS = 5

# Recipe names are long, the console table truncates them
NAME_WIDTH = 24

ENGINE_CLASSIC = 'classic'
ENGINE_ITERATIVE = 'iterative'

TIE = 'tie'

MISSING = '-'


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def load_reference_recipes(path: str = FIXTURES_PATH) -> List[Dict[str, Any]]:
    """Read the reference recipe fixtures"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def select_recipes(recipes: Sequence[Dict[str, Any]], wanted: Optional[str]) -> List[Dict[str, Any]]:
    """
    Filter the fixtures down to a single recipe.

    The selector accepts the full fixture id, any substring of it, or just the
    number of the recipe ('3', '03').
    """
    if not wanted:
        return list(recipes)

    needle = wanted.strip().lower()
    exact = [recipe for recipe in recipes if recipe['id'].lower() == needle]
    if exact:
        return exact

    padded = needle.zfill(2)
    matches = [
        recipe for recipe in recipes
        if needle in recipe['id'].lower() or recipe['id'].lower().startswith(f'recipe_{padded}')
    ]
    return matches


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def normalize_recipe(recipe: Dict[str, float]) -> Dict[str, float]:
    """Scale a recipe so that its material weights sum up to 100"""
    total = float(sum(recipe.values()))
    if total <= 0:
        return {}
    return {name: float(value) * 100.0 / total for name, value in recipe.items()}


def recompute_umf(recipe: Dict[str, float], available_materials: Sequence[Dict]) -> Dict[str, float]:
    """
    Recompute the UMF of a recipe the same way for both engines: recipe ->
    weight composition -> UMF, with no extra normalization on top.
    """
    composition = calculate_recipe_composition(available_materials, recipe)
    return {oxide: float(value) for oxide, value in weights_to_umf(composition).items()}


def oxide_errors(target_umf: Dict[str, float], actual_umf: Dict[str, float]) -> Tuple[float, float, Optional[str]]:
    """
    Total and worst per oxide absolute error.

    The union of the target and the actual oxide sets is used on purpose, so
    that the oxides the materials drag in on top of the target (contamination)
    are penalized as well.

    The oxides are walked in sorted order so that ties on the worst oxide are
    broken the same way on every run: plain set iteration depends on the string
    hash seed and would make the report irreproducible.
    """
    total_error = 0.0
    worst_error = 0.0
    worst_oxide = None

    for oxide in sorted(set(target_umf) | set(actual_umf)):
        diff = abs(float(target_umf.get(oxide, 0.0)) - float(actual_umf.get(oxide, 0.0)))
        total_error += diff
        if diff > worst_error:
            worst_error = diff
            worst_oxide = oxide

    return total_error, worst_error, worst_oxide


def compare_composition(solved_recipe: Dict[str, float],
                        original_recipe: Dict[str, float]) -> Dict[str, Any]:
    """
    Compare the material set of a solution with the original recipe.

    The original recipe is normalized to 100 first, so that the share
    difference is measured on the same scale as the solved one.
    """
    original = normalize_recipe(original_recipe)
    solved = normalize_recipe(solved_recipe)

    common = set(original) & set(solved)
    extra = set(solved) - set(original)
    lost = set(original) - set(solved)

    share_delta = sum(abs(original[name] - solved[name]) for name in common)

    return {
        'same': len(common),
        'extra': len(extra),
        'lost': len(lost),
        'share_delta': share_delta,
        'exact_set': not extra and not lost,
        'extra_names': sorted(extra),
        'lost_names': sorted(lost),
    }


# --------------------------------------------------------------------------
# engine runners
# --------------------------------------------------------------------------

def run_classic(target_umf: Dict[str, float], inventory: Sequence[str],
                seed: int) -> Tuple[Optional[Dict[str, Any]], float, str]:
    """
    Run the classic solver and return (best solution, seconds, status).

    The seed is handed to the solver itself - it draws its material subsets from
    its own generator, so pinning the global numpy one would do nothing - which
    is what makes a single recipe run and a full run produce the same classic
    result.  Its stdout is captured on top of logging=False, just in case the
    engine prints anything else.
    """
    sink = io.StringIO()
    start = time.perf_counter()

    try:
        with contextlib.redirect_stdout(sink):
            solutions = find_multiple_solutions(
                target_umf,
                max_solutions=MAX_SOLUTIONS,
                min_materials=True,
                logging=False,
                inventory_data=list(inventory),
                seed=seed,
            )
    except Exception as exc:
        return None, time.perf_counter() - start, f'failed: {exc}'

    elapsed = time.perf_counter() - start

    # The engine reports inventory problems as a dict instead of a list
    if isinstance(solutions, dict):
        return None, elapsed, f"failed: {solutions.get('error', 'unknown error')}"
    if not solutions:
        return None, elapsed, 'no solutions'
    if not solutions[0].get('recipe'):
        return None, elapsed, 'empty recipe'

    return solutions[0], elapsed, 'ok'


def run_iterative(target_umf: Dict[str, float],
                  inventory: Sequence[str]) -> Tuple[Optional[Dict[str, Any]], float, str]:
    """Run the iterative solver and return (best solution, seconds, status)"""
    sink = io.StringIO()
    start = time.perf_counter()

    try:
        with contextlib.redirect_stdout(sink):
            solutions = find_best_recipe(
                list(inventory),
                target_umf,
                max_solutions=MAX_SOLUTIONS,
                verbose=False,
            )
    except Exception as exc:
        return None, time.perf_counter() - start, f'failed: {exc}'

    elapsed = time.perf_counter() - start

    if not solutions:
        return None, elapsed, 'no solutions'
    if not solutions[0].get('recipe'):
        return None, elapsed, 'empty recipe'

    return solutions[0], elapsed, 'ok'


def evaluate(reference: Dict[str, Any], engine: str, solution: Optional[Dict[str, Any]],
             elapsed: float, status: str,
             available_materials: Sequence[Dict]) -> Dict[str, Any]:
    """Turn one engine run into a row of the comparison table"""
    row: Dict[str, Any] = {
        'id': reference['id'],
        'name': reference['name'],
        'engine': engine,
        'status': status,
        'seconds': elapsed,
        'iterations': solution.get('iterations') if solution else None,
        'native_error': solution.get('error') if solution else None,
    }

    if solution is None:
        return row

    recipe = solution['recipe']
    target_umf = reference['umf']
    actual_umf = recompute_umf(recipe, available_materials)

    total_error, max_error, worst_oxide = oxide_errors(target_umf, actual_umf)
    composition = compare_composition(recipe, reference['recipe'])

    row.update({
        'recipe': recipe,
        'actual_umf': actual_umf,
        'sum_error': total_error,
        'max_error': max_error,
        'worst_oxide': worst_oxide,
        'umf_error': float(calculate_umf_error(target_umf, actual_umf)),
        'materials_count': len(recipe),
    })
    row.update(composition)

    return row


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

TABLE_HEADERS = [
    '#', 'Recipe', 'Engine', 'Status',
    'sumErr', 'maxErr', 'worst', 'umfErr', 'native*',
    'mats', 'same', 'extra', 'lost', 'dShare', 'time,s', 'iter',
]

TABLE_ALIGN = ['r', 'l', 'l', 'l', 'r', 'r', 'l', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r']


def _number(value: Optional[float], digits: int) -> str:
    return MISSING if value is None else f'{value:.{digits}f}'


def _integer(value: Optional[int]) -> str:
    return MISSING if value is None else str(value)


def _short_name(name: str) -> str:
    if len(name) <= NAME_WIDTH:
        return name
    return name[:NAME_WIDTH - 1] + '…'


def build_cells(rows: Sequence[Dict[str, Any]]) -> List[List[str]]:
    """Format every row of the comparison into table cells"""
    cells = []

    for row in rows:
        number = row['id'].split('_')[1] if '_' in row['id'] else row['id']
        cells.append([
            number,
            _short_name(row['name']),
            row['engine'],
            row['status'],
            _number(row.get('sum_error'), 4),
            _number(row.get('max_error'), 4),
            row.get('worst_oxide') or MISSING,
            _number(row.get('umf_error'), 4),
            _number(row.get('native_error'), 4),
            _integer(row.get('materials_count')),
            _integer(row.get('same')),
            _integer(row.get('extra')),
            _integer(row.get('lost')),
            _number(row.get('share_delta'), 2),
            _number(row.get('seconds'), 3),
            _integer(row.get('iterations')),
        ])

    return cells


def render_console_table(headers: Sequence[str], cells: Sequence[Sequence[str]],
                         align: Sequence[str]) -> str:
    """Render an aligned plain text table"""
    widths = [len(header) for header in headers]
    for row in cells:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def format_row(row: Sequence[str]) -> str:
        parts = []
        for index, cell in enumerate(row):
            parts.append(cell.ljust(widths[index]) if align[index] == 'l' else cell.rjust(widths[index]))
        return '  '.join(parts).rstrip()

    lines = [format_row(headers), '  '.join('-' * width for width in widths)]
    lines.extend(format_row(row) for row in cells)
    return '\n'.join(lines)


def render_markdown_table(headers: Sequence[str], cells: Sequence[Sequence[str]],
                          align: Sequence[str]) -> str:
    """Render the same table in markdown"""
    separator = ['---' if kind == 'l' else '---:' for kind in align]
    lines = ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(separator) + ' |']
    lines.extend('| ' + ' | '.join(row) + ' |' for row in cells)
    return '\n'.join(lines)


# --------------------------------------------------------------------------
# summary
# --------------------------------------------------------------------------

SUMMARY_HEADERS = [
    'Engine', 'solved', 'sumErr avg', 'sumErr med', 'maxErr avg', 'umfErr avg',
    'umfErr med', 'mats avg', 'exact sets', 'dShare avg', 'total time,s',
]

SUMMARY_ALIGN = ['l', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r']


def summarize(rows: Sequence[Dict[str, Any]], engine: str, total_recipes: int) -> Dict[str, Any]:
    """Aggregate the rows of one engine"""
    engine_rows = [row for row in rows if row['engine'] == engine]
    solved = [row for row in engine_rows if row['status'] == 'ok']

    def mean(key: str) -> Optional[float]:
        values = [row[key] for row in solved if row.get(key) is not None]
        return statistics.fmean(values) if values else None

    def median(key: str) -> Optional[float]:
        values = [row[key] for row in solved if row.get(key) is not None]
        return statistics.median(values) if values else None

    return {
        'engine': engine,
        'solved': len(solved),
        'total': total_recipes,
        'sum_error_avg': mean('sum_error'),
        'sum_error_med': median('sum_error'),
        'max_error_avg': mean('max_error'),
        'umf_error_avg': mean('umf_error'),
        'umf_error_med': median('umf_error'),
        'materials_avg': mean('materials_count'),
        'exact_sets': sum(1 for row in solved if row.get('exact_set')),
        'share_delta_avg': mean('share_delta'),
        'total_seconds': sum(row['seconds'] for row in engine_rows),
    }


def build_summary_cells(summaries: Sequence[Dict[str, Any]]) -> List[List[str]]:
    cells = []
    for item in summaries:
        cells.append([
            item['engine'],
            f"{item['solved']}/{item['total']}",
            _number(item['sum_error_avg'], 4),
            _number(item['sum_error_med'], 4),
            _number(item['max_error_avg'], 4),
            _number(item['umf_error_avg'], 4),
            _number(item['umf_error_med'], 4),
            _number(item['materials_avg'], 1),
            str(item['exact_sets']),
            _number(item['share_delta_avg'], 2),
            _number(item['total_seconds'], 3),
        ])
    return cells


DIVERGENCE_HEADERS = ['#', 'Recipe', 'classic sumErr', 'iterative sumErr', 'better', 'delta']
DIVERGENCE_ALIGN = ['r', 'l', 'r', 'r', 'l', 'r']


def build_divergence(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Per recipe head to head comparison on the uniformly recomputed error"""
    by_id: Dict[str, Dict[str, Dict[str, Any]]] = {}
    order: List[str] = []

    for row in rows:
        if row['id'] not in by_id:
            by_id[row['id']] = {}
            order.append(row['id'])
        by_id[row['id']][row['engine']] = row

    result = []
    for recipe_id in order:
        classic = by_id[recipe_id].get(ENGINE_CLASSIC, {})
        iterative = by_id[recipe_id].get(ENGINE_ITERATIVE, {})

        classic_error = classic.get('sum_error')
        iterative_error = iterative.get('sum_error')

        if classic_error is None and iterative_error is None:
            better, delta = MISSING, None
        elif iterative_error is None:
            better, delta = ENGINE_CLASSIC, None
        elif classic_error is None:
            better, delta = ENGINE_ITERATIVE, None
        else:
            delta = abs(classic_error - iterative_error)
            if delta == 0.0:
                better = TIE
            else:
                better = ENGINE_CLASSIC if classic_error < iterative_error else ENGINE_ITERATIVE

        result.append({
            'id': recipe_id,
            'name': classic.get('name') or iterative.get('name') or recipe_id,
            'classic': classic_error,
            'iterative': iterative_error,
            'better': better,
            'delta': delta,
        })

    return result


def build_divergence_cells(divergence: Sequence[Dict[str, Any]]) -> List[List[str]]:
    cells = []
    for item in divergence:
        number = item['id'].split('_')[1] if '_' in item['id'] else item['id']
        cells.append([
            number,
            _short_name(item['name']),
            _number(item['classic'], 4),
            _number(item['iterative'], 4),
            item['better'],
            _number(item['delta'], 4),
        ])
    return cells


def build_conclusions(summaries: Sequence[Dict[str, Any]],
                      divergence: Sequence[Dict[str, Any]]) -> List[str]:
    """Short automatic read of the numbers"""
    by_engine = {item['engine']: item for item in summaries}
    classic = by_engine[ENGINE_CLASSIC]
    iterative = by_engine[ENGINE_ITERATIVE]
    lines = []

    def compare(label: str, key: str, digits: int, lower_is_better: bool = True) -> None:
        left, right = classic.get(key), iterative.get(key)
        if left is None or right is None:
            lines.append(f'- {label}: not enough data')
            return
        if left == right:
            lines.append(f'- {label}: tie ({left:.{digits}f})')
            return
        winner = ENGINE_CLASSIC if ((left < right) == lower_is_better) else ENGINE_ITERATIVE
        lines.append(f'- {label}: **{winner}** wins '
                     f'(classic {left:.{digits}f} vs iterative {right:.{digits}f})')

    compare('accuracy, mean total per oxide error', 'sum_error_avg', 4)
    compare('accuracy, median total per oxide error', 'sum_error_med', 4)
    compare('accuracy, mean worst per oxide error', 'max_error_avg', 4)
    compare('accuracy, mean calculate_umf_error', 'umf_error_avg', 4)
    compare('speed, total run time', 'total_seconds', 3)
    compare('recipe size, mean material count', 'materials_avg', 1)
    compare('closeness to the original recipe, mean sum of share differences', 'share_delta_avg', 2)

    lines.append(f'- exact material set recovered: classic {classic["exact_sets"]}, '
                 f'iterative {iterative["exact_sets"]} (out of {classic["total"]} recipes)')
    lines.append(f'- solutions returned: classic {classic["solved"]}/{classic["total"]}, '
                 f'iterative {iterative["solved"]}/{iterative["total"]}')

    wins_classic = [item for item in divergence if item['better'] == ENGINE_CLASSIC]
    wins_iterative = [item for item in divergence if item['better'] == ENGINE_ITERATIVE]
    ties = [item for item in divergence if item['better'] == TIE]
    lines.append(f'- per recipe wins by total per oxide error: classic {len(wins_classic)}, '
                 f'iterative {len(wins_iterative)}, ties {len(ties)}')

    biggest = [item for item in divergence if item['delta'] and item['better'] != TIE]
    biggest.sort(key=lambda item: -item['delta'])
    for item in biggest[:3]:
        lines.append(f'- biggest gap on `{item["id"]}` ({item["name"]}): '
                     f'classic {item["classic"]:.4f} vs iterative {item["iterative"]:.4f}, '
                     f'delta {item["delta"]:.4f} in favour of {item["better"]}')

    lines.append('- classic draws random material subsets from a seeded generator, so it is reproducible '
                 'but its numbers only hold for the seed printed above; iterative does not depend on a seed')

    return lines


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

COLUMN_LEGEND = [
    '`sumErr` - sum of absolute per oxide errors over the union of the target and the resulting UMF oxides',
    '`maxErr` / `worst` - the largest per oxide error and the oxide it belongs to',
    '`umfErr` - solver_classic.calculate_umf_error on the uniformly recomputed UMF (same formula for both engines)',
    '`native*` - the error the engine reports itself. For classic it is now the same quantity as `umfErr` by '
    'construction: it measures the plain UMF of its recipe against the target, exactly as this report does. '
    'Iterative scores the same UMF against its cleaned target extended with zeros for the oxides nobody asked '
    'for, so it is measuring something else and only coincides with `umfErr` while the target already names '
    'every oxide the recipe brings - which is the case on all the reference recipes',
    '`mats` - materials in the recipe, `same` / `extra` / `lost` - materials shared with, added to and missing '
    'from the original recipe',
    '`dShare` - sum of absolute differences of the material shares over the shared materials, original recipe '
    'normalized to 100',
    '`iter` - iterations spent by the iterative solver, meaningless for classic',
]


def build_report(rows: Sequence[Dict[str, Any]], summaries: Sequence[Dict[str, Any]],
                 divergence: Sequence[Dict[str, Any]], seed: int,
                 inventory: Sequence[str], markdown: bool) -> str:
    """Assemble the whole report either as plain text or as markdown"""
    table = render_markdown_table if markdown else render_console_table
    heading = (lambda text: f'## {text}') if markdown else (lambda text: f'{text}\n' + '=' * len(text))

    cells = build_cells(rows)
    parts: List[str] = []

    if markdown:
        parts.append('# Solver comparison')
        parts.append('')
        parts.append(f'Generated by `compare_solvers.py`, seed `{seed}`, '
                     f'{len(inventory)} materials in the inventory, '
                     f'{len(divergence)} reference recipes, best solution of each engine.')
    else:
        parts.append(f'SOLVER COMPARISON | seed {seed} | inventory {len(inventory)} materials | '
                     f'{len(divergence)} reference recipes')
    parts.append('')

    parts.append(heading('Per recipe'))
    parts.append('')
    parts.append(table(TABLE_HEADERS, cells, TABLE_ALIGN))
    parts.append('')

    parts.append(heading('Columns'))
    parts.append('')
    parts.extend(f'- {line}' if markdown else f'  {line}' for line in COLUMN_LEGEND)
    parts.append('')

    parts.append(heading('Summary'))
    parts.append('')
    parts.append(table(SUMMARY_HEADERS, build_summary_cells(summaries), SUMMARY_ALIGN))
    parts.append('')

    parts.append(heading('Head to head'))
    parts.append('')
    parts.append(table(DIVERGENCE_HEADERS, build_divergence_cells(divergence), DIVERGENCE_ALIGN))
    parts.append('')

    parts.append(heading('Conclusions'))
    parts.append('')
    parts.extend(build_conclusions(summaries, divergence))
    parts.append('')

    return '\n'.join(parts)


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def compare(recipes: Sequence[Dict[str, Any]], inventory: Sequence[str],
            available_materials: Sequence[Dict], seed: int) -> List[Dict[str, Any]]:
    """Run both engines over every reference recipe"""
    rows: List[Dict[str, Any]] = []

    for reference in recipes:
        target_umf = reference['umf']

        solution, elapsed, status = run_classic(target_umf, inventory, seed)
        rows.append(evaluate(reference, ENGINE_CLASSIC, solution, elapsed, status, available_materials))

        solution, elapsed, status = run_iterative(target_umf, inventory)
        rows.append(evaluate(reference, ENGINE_ITERATIVE, solution, elapsed, status, available_materials))

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description='Compare the classic and the iterative glaze solvers')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                        help=f'random seed pinned for the classic solver (default: {DEFAULT_SEED})')
    parser.add_argument('--recipe', type=str, default=None,
                        help='run a single reference recipe (fixture id, part of it, or its number)')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT,
                        help='path of the generated markdown report (default: comparison_results.md)')
    parser.add_argument('--fixtures', type=str, default=FIXTURES_PATH,
                        help='path of the reference recipes fixture file')
    args = parser.parse_args()

    recipes = load_reference_recipes(args.fixtures)
    selected = select_recipes(recipes, args.recipe)

    if not selected:
        print(f"no reference recipe matches '{args.recipe}'")
        print('available ids: ' + ', '.join(recipe['id'] for recipe in recipes))
        raise SystemExit(1)

    inventory = resolve_inventory()
    all_materials = load_materials(only_inventory=False, priority=True)
    available_materials = filter_materials_by_inventory(all_materials, inventory)

    rows = compare(selected, inventory, available_materials, args.seed)

    summaries = [
        summarize(rows, ENGINE_CLASSIC, len(selected)),
        summarize(rows, ENGINE_ITERATIVE, len(selected)),
    ]
    divergence = build_divergence(rows)

    print(build_report(rows, summaries, divergence, args.seed, inventory, markdown=False))

    report = build_report(rows, summaries, divergence, args.seed, inventory, markdown=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f'markdown report written to {args.output}')


if __name__ == '__main__':
    main()
