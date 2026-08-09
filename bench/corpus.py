#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# flake8: noqa
# pylint: disable=broad-exception-raised, raise-missing-from, too-many-arguments, redefined-outer-name
# pylint: disable=too-many-positional-arguments, too-many-locals
# pylint: disable=multiple-statements, logging-fstring-interpolation, trailing-whitespace, line-too-long
# pylint: disable=broad-exception-caught, missing-function-docstring, missing-class-docstring
# pylint: disable=f-string-without-interpolation
# pylance: disable=reportMissingImports, reportMissingModuleSource

"""
The Glazy corpus: fetching the dump, parsing it, sampling it and running a case

This module is the shared half of stage 7.6 / 7.7. Both consumers need exactly
the same corpus, the same sample and the same per-case run, and they must not
drift apart:

    tests/test_inverse_corpus.py   the pass/fail gate of scenarios A and B
    bench/diff_baseline.py         the baseline snapshot and its regression diff
    bench/history.py               the append only log of the runs worth keeping

The last two also share the roll-up itself - METRICS, profile() and
engine_profile() live here so that the diff and the history cannot end up
computing "the median" two different ways.

TWO SCENARIOS, ONE MACHINERY

Both scenarios parse the same dump, build the same cases and aim at the same
target - our own forward calculation of the dump recipe, which is what keeps the
flux convention and Glazy's normalization out of the measurement. They differ in
one line: what is on the shelf.

    A   inventory = the recipe's OWN materials, injected through the materials=
        seam. A perfect answer exists by construction, so what is measured is the
        quality of the answer that comes back.
    B   inventory = OUR inInventory materials, the 19 the workshop actually has,
        with no injection at all - the literal production call. This is the real
        use case: you saw a recipe made of American frits and you want that
        chemistry out of what is on your shelf in Russia. The answer may not
        exist, so feasibility.check_feasibility runs FIRST and a target our stock
        genuinely cannot reach goes into an "honestly unreachable" bucket, which
        is a correct answer and not a failure.

Scenario B is where the cost metrics stop abstaining. Glazy material names are
absent from database/prices.json, so cost_abs and assembly_score are None on
every scenario A case; scenario B builds from our own inventory, 18 of whose 19
materials are priced - everything except wood ash, which nobody sells - so on
that side they are real numbers. What is NOT computed in scenario B is any
comparison of cost with the original: a ratio against a frit-based American
recipe measures the distance between two countries' supply chains, not the
quality of the solver. Absolute cost only. The ratio abstains by itself, because
the dump's materials carry no price, and that is left as the structural
guarantee it is rather than being re-implemented as a special case here.

The dump itself is CC BY-NC-SA licensed data and is never written into the
repository: it lives in ~/.cache/glazy_solver/ (a location outside any git work
tree, which Glazy's own terms explicitly allow), together with the derived slim
cache. Nothing in bench/ writes to the project directory except the baseline
snapshot, which holds only numbers we computed ourselves.

Why a slim cache exists. Parsing the 8 MB gzip of YAML costs about 11 seconds
with PyYAML's C loader and about ten times that with the Python one; a benchmark
that is re-run while a solver is being tuned cannot pay it on every run. The
first parse therefore writes bench-shaped JSON next to the dump - materials with
a cleaned formula and recipes with their ingredient ids - and every later run
loads that instead. The cache is keyed by the dump's file name AND its size, so
a new dump, a truncated download or a change of the slim format all invalidate
it rather than being silently reused.

PyYAML is a development dependency (requirements-dev.txt) and is imported lazily
inside the parser, so importing this module - which the test module does
unconditionally - never requires it.
"""

import gzip
import hashlib
import json
import logging
import os
import random
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import flux_oxides, load_materials, weights_to_umf
from solver_classic import (
    calculate_recipe_composition,
    calculate_umf_error,
    find_multiple_solutions,
)
from solver_iterative import find_best_recipe
import feasibility
import quality_metrics as qm

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINE_PATH = os.path.join(BENCH_DIR, 'quality_baseline.json')
HISTORY_PATH = os.path.join(BENCH_DIR, 'history.jsonl')

# The dump lives on the "master" branch of the repository. "main" answers 404,
# and with the wrong URL the corpus test would skip itself forever - the most
# expensive kind of failure there is, because it looks like everything is fine.
DUMP_REPO_RAW = 'https://raw.githubusercontent.com/derekphilipau/glazy-data/master'
LATEST_URL = f'{DUMP_REPO_RAW}/LATEST'

DEFAULT_CACHE_DIR = os.path.join(os.path.expanduser('~'), '.cache', 'glazy_solver')

# Bump when the shape of the slim cache changes, so that an old cache is
# rebuilt instead of being read with the wrong expectations.
SLIM_FORMAT_VERSION = 2

# Loss on ignition is not an oxide: it must never reach a formula, or every
# material would look like it carried an extra 10-15% of something.
LOI_KEYS = frozenset({'LOI', 'Loi', 'loi'})

# Sampling. The seed is part of the baseline and must not be changed casually:
# a different seed is a different corpus and makes every stored number
# incomparable.
DEFAULT_SEED = 20260531
DEFAULT_SAMPLE_SIZE = 300
DEFAULT_CLASSIC_SUBSAMPLE = 50

# Scenario B draws its targets out of the scenario A sample with the same seed,
# so every B case is also an A case with the same glazy_id and the two scenarios
# can be read side by side, case by case.
DEFAULT_SCENARIO_B_SUBSAMPLE = 100

# Size buckets of the stratification, as (label, low, high) with both ends
# inclusive; "high" of None means "and up". Recipes of a single ingredient are
# deliberately outside every bucket: there is nothing to solve there.
SIZE_BUCKETS = (
    ('2-4', 2, 4),
    ('5-7', 5, 7),
    ('8+', 8, None),
)

ENGINE_ITERATIVE = 'iterative'
ENGINE_CLASSIC = 'classic'

SCENARIO_A = 'A'
SCENARIO_B = 'B'
SCENARIOS = (SCENARIO_A, SCENARIO_B)

# The reachability verdict of scenario B is drawn with the production line, not
# with one chosen for the benchmark: feasibility.DEFAULT_FEASIBILITY_TOL is what
# POST /api/feasibility uses, and no passengers are declared because a target
# built by forward-calculating a real recipe asks for every oxide it names.
#
# That the ceiling is not the driver here was measured rather than assumed: with
# every oxide our stock can bring but the target does not name declared as a
# passenger at 0.05, the verdict moved on 2 of the 100 cases. The unreachable
# bucket is not made of contamination.
FEASIBILITY_TOL = feasibility.DEFAULT_FEASIBILITY_TOL

# The configuration the tests and the API actually run in. Numbers measured at
# max_solutions=1 do not carry over: the beam width and the number of children
# are both 1 there, which is a different search (TZ_SOLVER_V2.md 10.8).
MAX_SOLUTIONS = 5
CLASSIC_SEED = 42

# Level 1 of the two-level criterion of 7.1: the chemistry has to match.
MAX_UMF_ERROR = 0.1

# The raw components a run is measured by, and the direction that counts as
# better. Everything is "smaller is better" except the smallest portion of a
# recipe, where a larger value means the recipe is easier to weigh out.
#
# This list lives here rather than in bench/diff_baseline.py because two
# consumers roll the same numbers up and must not drift apart: the regression
# diff, which profiles them over the intersection of two runs, and the run
# history (bench/history.py), which profiles them over one run.
METRICS: Tuple[Tuple[str, str], ...] = (
    ('umf_error', 'lower'),
    ('count', 'lower'),
    ('cost_abs', 'lower'),
    ('assembly_score', 'lower'),
    ('min_portion', 'higher'),
    ('junk_count', 'lower'),
    ('rounding_drift', 'lower'),
    ('cond', 'lower'),
)

# The aggregates of a distribution profile, in the order they are printed. The
# whole distribution rather than a mean: a change can improve the median while
# wrecking a handful of cases, and only p90 / p99 / max show it.
PROFILE_KEYS = ('min', 'p10', 'median', 'mean', 'p90', 'p99', 'max')

# Data files whose content decides what the solver can answer. Their hashes go
# into the baseline so that "the solver changed" can be told apart from "the
# data changed" - an updated price list moves every cost metric without a line
# of solver code being touched, and that is not a regression.
HASHED_DATA_FILES = (
    os.path.join('database', 'materials.json'),
    os.path.join('database', 'prices.json'),
    os.path.join('database', 'oxide_classification.json'),
)


class CorpusUnavailable(Exception):
    """
    The corpus cannot be obtained: no dump on disk and no way to fetch one

    Callers turn this into a skip, never into a failure. A benchmark that needs
    an 8 MB download must not turn a missing network into a red test suite.
    """


# --------------------------------------------------------------------------
# fetching
# --------------------------------------------------------------------------

def cache_dir() -> str:
    """Directory of the dump and of the derived slim cache"""
    return os.environ.get('GLAZY_CACHE_DIR') or DEFAULT_CACHE_DIR


def _download(url: str, timeout: int = 120) -> bytes:
    """Fetch a URL, translating every failure into CorpusUnavailable"""
    try:
        import requests  # already a runtime dependency
    except ImportError as exc:
        raise CorpusUnavailable(f'requests is not importable: {exc}')

    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
    except Exception as exc:
        raise CorpusUnavailable(f'cannot fetch {url}: {exc}')

    return response.content


def resolve_dump_path(allow_download: bool = True) -> str:
    """
    Locate the Glazy dump, downloading it into the cache when it is missing

    Resolution order: GLAZY_DUMP_PATH, then the cache directory, then the
    network. The pinned file name comes from the LATEST file of the same
    repository, cached next to the dump so that an offline run keeps using the
    same dump instead of failing on the name lookup.

    Raises:
        CorpusUnavailable: no dump and no way to get one
    """
    explicit = os.environ.get('GLAZY_DUMP_PATH')
    if explicit:
        if not os.path.exists(explicit):
            raise CorpusUnavailable(f'GLAZY_DUMP_PATH points at a missing file: {explicit}')
        return explicit

    directory = cache_dir()
    latest_path = os.path.join(directory, 'LATEST')

    name = None
    if os.path.exists(latest_path):
        with open(latest_path, 'r', encoding='utf-8') as f:
            name = f.read().strip()

    if not name:
        if not allow_download:
            raise CorpusUnavailable(f'no pinned dump name in {latest_path} and downloading is disabled')
        os.makedirs(directory, exist_ok=True)
        name = _download(LATEST_URL).decode('utf-8').strip()
        if not name:
            raise CorpusUnavailable(f'{LATEST_URL} returned an empty file name')
        with open(latest_path, 'w', encoding='utf-8') as f:
            f.write(name + '\n')

    # The name comes from a remote file and is used to build a local path:
    # refuse anything that is not a plain file name.
    if os.path.basename(name) != name or name in ('.', '..'):
        raise CorpusUnavailable(f'LATEST holds an unusable dump name: {name!r}')

    path = os.path.join(directory, name)
    if os.path.exists(path):
        return path

    if not allow_download:
        raise CorpusUnavailable(f'{path} is missing and downloading is disabled')

    os.makedirs(directory, exist_ok=True)
    payload = _download(f'{DUMP_REPO_RAW}/{name}')
    # Write through a temporary name: a half written dump left behind by an
    # interrupted download would be indistinguishable from a good one.
    temporary = path + '.part'
    with open(temporary, 'wb') as f:
        f.write(payload)
    os.replace(temporary, path)

    return path


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------

def _clean_formula(analysis: Dict[str, Any]) -> Dict[str, float]:
    """
    Turn a "Percent Analysis" into one of our formulas

    Loss on ignition is dropped: materials in the dump are stored as calcined
    oxides plus LOI, and LOI is not an oxide. Everything else is kept as is -
    every other key of the dump exists in database/molar_masses.json, so nothing
    is silently thrown away by the conversion downstream.

    The result is EMPTY for 1 170 of the 7 308 analysed materials, whose whole
    analysis is {"LOI": 0}: colorants, grog, gums, CMC, gypsum, Darvan. That is
    not a parsing failure - it is the dump saying "this is a real ingredient
    with no oxide contribution", exactly like the 37 such materials of our own
    database. They stay in the catalogue with an empty formula: the solvers drop
    them in filter_materials_with_formula() and quality_metrics reports them
    under "unanalysed_materials", so keeping them costs nothing and dropping
    them would throw away the 3 001 recipes that use one.
    """
    formula = {}
    for oxide, value in (analysis or {}).items():
        if oxide in LOI_KEYS:
            continue
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        if value <= 0:
            continue
        formula[oxide] = float(value)
    return formula


def _parse_dump(path: str) -> Dict[str, Any]:
    """
    Parse the gzipped YAML dump into the slim structure

    PyYAML is imported here rather than at module level: it is a development
    dependency and only this function needs it.
    """
    try:
        import yaml
    except ImportError as exc:
        raise CorpusUnavailable(f'pyyaml is not installed (requirements-dev.txt): {exc}')

    loader = getattr(yaml, 'CSafeLoader', None) or yaml.SafeLoader
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        records = yaml.load(f, Loader=loader)

    materials: Dict[int, Dict[str, Any]] = {}
    raw_recipes: List[Dict[str, Any]] = []

    for record in records:
        kind = record.get('Type')
        if kind == 'Material':
            analysis = record.get('Percent Analysis')
            if not analysis:
                continue
            materials[int(record['ID'])] = {
                'id': int(record['ID']),
                'name': str(record.get('Name') or f"material {record['ID']}"),
                'formula': _clean_formula(analysis),
            }
        elif kind == 'Recipe':
            ingredients = record.get('Ingredients')
            if ingredients:
                raw_recipes.append(record)

    recipes = []
    unresolved = 0
    unusable = 0

    for record in raw_recipes:
        parsed = _parse_recipe(record, materials)
        if parsed is None:
            unresolved += 1
            continue
        if not parsed['ingredients']:
            unusable += 1
            continue
        recipes.append(parsed)

    return {
        'format_version': SLIM_FORMAT_VERSION,
        'materials': list(materials.values()),
        'recipes': recipes,
        'stats': {
            'records': len(records),
            'materials_with_analysis': len(materials),
            'materials_without_oxides': sum(1 for m in materials.values() if not m['formula']),
            'recipes_with_ingredients': len(raw_recipes),
            'recipes_resolvable': len(recipes),
            'recipes_unresolved': unresolved,
            'recipes_unusable': unusable,
        },
    }


def _parse_recipe(record: Dict[str, Any], materials: Dict[int, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    One dump recipe as (id, name, ingredients), or None when it cannot be used

    Ingredients are stored as [material id, percentage, is_addition]. The
    "Additional" key is present only when it is true, so a missing key is false.

    A recipe naming a material that has no analysis in the dump (private,
    deleted, or simply never analysed - about 1% of the dump) is dropped
    entirely rather than solved without one of its ingredients: the missing
    ingredient would silently move the target chemistry.
    """
    ingredients = []

    for ingredient in record.get('Ingredients') or []:
        material_id = ingredient.get('ID')
        if material_id is None or int(material_id) not in materials:
            return None
        percentage = ingredient.get('Percentage')
        if not isinstance(percentage, (int, float)) or isinstance(percentage, bool):
            return None
        if percentage < 0:
            return None
        ingredients.append([int(material_id), float(percentage), bool(ingredient.get('Additional'))])

    return {
        'id': int(record['ID']),
        'name': str(record.get('Name') or f"recipe {record['ID']}"),
        'ingredients': ingredients,
    }


def _slim_cache_path(dump_path: str) -> str:
    """Path of the slim cache belonging to a dump"""
    return os.path.join(cache_dir(), os.path.basename(dump_path) + '.slim.json')


def load_corpus(dump_path: Optional[str] = None, allow_download: bool = True,
                use_cache: bool = True) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load the corpus, building the slim cache on the first call

    Returns:
        (corpus, timings) where corpus holds "materials", "recipes" and "stats",
        and timings reports which path was taken and how long it took:
        {"source": "yaml"|"slim", "seconds": float, "dump": name, ...}
    """
    path = dump_path or resolve_dump_path(allow_download=allow_download)
    size = os.path.getsize(path)
    slim_path = _slim_cache_path(path)

    if use_cache and os.path.exists(slim_path):
        start = time.perf_counter()
        try:
            with open(slim_path, 'r', encoding='utf-8') as f:
                cached = json.load(f)
        except Exception as exc:
            logger.warning(f'slim cache {slim_path} is unreadable ({exc}), rebuilding it')
            cached = None

        if cached is not None:
            fresh = (
                cached.get('format_version') == SLIM_FORMAT_VERSION
                and cached.get('dump') == os.path.basename(path)
                and cached.get('dump_size') == size
            )
            if fresh:
                elapsed = time.perf_counter() - start
                return cached, {'source': 'slim', 'seconds': elapsed,
                                'dump': os.path.basename(path), 'dump_size': size,
                                'slim_path': slim_path}
            logger.info(f'slim cache {slim_path} belongs to another dump, rebuilding it')

    start = time.perf_counter()
    corpus = _parse_dump(path)
    elapsed = time.perf_counter() - start

    corpus['dump'] = os.path.basename(path)
    corpus['dump_size'] = size

    if use_cache:
        os.makedirs(cache_dir(), exist_ok=True)
        temporary = slim_path + '.part'
        with open(temporary, 'w', encoding='utf-8') as f:
            json.dump(corpus, f, separators=(',', ':'))
        os.replace(temporary, slim_path)

    return corpus, {'source': 'yaml', 'seconds': elapsed,
                    'dump': os.path.basename(path), 'dump_size': size,
                    'slim_path': slim_path}


# --------------------------------------------------------------------------
# cases
# --------------------------------------------------------------------------

def _material_popularity(recipes: Sequence[Dict[str, Any]]) -> Dict[int, int]:
    """
    How many recipes of the corpus use each material

    This is the only honest stand-in for database/priorities.json on foreign
    material records: priorities encode "how basic is this material", and in a
    corpus of 35 000 recipes the answer is how often everybody reaches for it.
    Crucially it is a GLOBAL property - it does not look at the recipe being
    solved, so it cannot leak the expected answer into the search.
    """
    counter: Dict[int, int] = {}
    for recipe in recipes:
        for material_id, _percentage, _addition in recipe['ingredients']:
            counter[material_id] = counter.get(material_id, 0) + 1
    return counter


def _merge_ingredients(ingredients: Sequence[Sequence[Any]]) -> Tuple[Dict[int, float], float, float]:
    """
    Merge the base and the additional parts of one recipe

    The dump's convention is that additions COUNT towards the chemistry, so both
    parts go into one weight vector, which is then renormalized to 100. A
    material appearing on both sides is summed rather than overwritten.

    Stage 6 will add the shared helpers for this model (load_recipes() returning
    base / additional / include_additions, see TZ_SOLVER_V2.md 6.2); they do not
    exist yet, so the merge is inlined here and must be replaced by them once
    they land.

    Returns:
        ({material id: percent of the merged batch}, base sum, addition sum)
    """
    merged: Dict[int, float] = {}
    base_sum = 0.0
    addition_sum = 0.0

    for material_id, percentage, is_addition in ingredients:
        merged[material_id] = merged.get(material_id, 0.0) + percentage
        if is_addition:
            addition_sum += percentage
        else:
            base_sum += percentage

    total = base_sum + addition_sum
    if total <= 0:
        return {}, base_sum, addition_sum

    normalized = {material_id: value * 100.0 / total for material_id, value in merged.items()}
    return normalized, base_sum, addition_sum


def build_cases(corpus: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Turn the corpus into solvable cases

    A case carries everything a run needs and nothing else: the merged original
    recipe by material NAME, the material records to inject, and the target UMF
    computed by our own forward calculation. Computing the target ourselves is
    what makes the flux convention and Glazy's own normalization irrelevant to
    the measurement - both sides of the comparison come out of the same code.

    Recipes are dropped when the merged batch cannot be used:
      - the base part sums to zero (nothing but additions, or all zeros);
      - two materials of the recipe share a name, so a name-keyed recipe would
        silently lose one of them;
      - the batch carries no oxide, or carries no flux at all. Without a flux
        weights_to_umf() falls back to normalizing on the smallest positive
        molar amount, which produces a target that is not a unity formula and
        means nothing as a benchmark goal.

    Returns:
        (cases, rejected) where rejected counts each reason
    """
    materials_by_id = {material['id']: material for material in corpus['materials']}
    popularity = _material_popularity(corpus['recipes'])

    fluxes = set(flux_oxides())

    cases: List[Dict[str, Any]] = []
    rejected = {'empty_batch': 0, 'no_base': 0, 'duplicate_name': 0, 'no_oxides': 0, 'no_flux': 0,
                'single_ingredient': 0}

    for recipe in corpus['recipes']:
        ingredients = recipe['ingredients']
        if len(ingredients) < 2:
            rejected['single_ingredient'] += 1
            continue

        merged, base_sum, _addition_sum = _merge_ingredients(ingredients)
        if not merged:
            rejected['empty_batch'] += 1
            continue
        if base_sum <= 0:
            rejected['no_base'] += 1
            continue

        records = [materials_by_id[material_id] for material_id in merged]
        names = [record['name'] for record in records]
        if len(set(names)) != len(names):
            rejected['duplicate_name'] += 1
            continue

        original = {materials_by_id[material_id]['name']: share
                    for material_id, share in merged.items()}

        # Explicit priority: without it every injected record collapses onto
        # DEFAULT_PRIORITY and _priority_start_set() hands the solver the whole
        # inventory in one group, which is not the search the API runs
        # (TZ_SOLVER_V2.md 10.7 item 4). Ranking by global popularity gives the
        # common materials the low numbers, exactly as priorities.json does.
        ordered = sorted(records, key=lambda record: (-popularity.get(record['id'], 0), record['name']))
        injected = [
            {'name': record['name'], 'formula': dict(record['formula']),
             'priority': index + 1, 'glazy_id': record['id']}
            for index, record in enumerate(ordered)
        ]

        composition = calculate_recipe_composition(injected, original)
        if not composition or sum(composition.values()) <= 0:
            rejected['no_oxides'] += 1
            continue
        if not any(composition.get(oxide, 0.0) > 0 for oxide in fluxes):
            rejected['no_flux'] += 1
            continue

        target_umf = weights_to_umf(composition)

        cases.append({
            'glazy_id': recipe['id'],
            'name': recipe['name'],
            'size': len(merged),
            'bucket': size_bucket(len(merged)),
            'original': original,
            'materials': injected,
            'target_umf': target_umf,
            # Ingredients the dump analyses as carrying no oxide at all. They
            # are part of the original recipe but can never be part of a
            # solution, so a case with one of them starts a component short.
            'unanalysed': sum(1 for record in injected if not record['formula']),
        })

    return cases, rejected


def size_bucket(size: int) -> Optional[str]:
    """Label of the stratification bucket a recipe of this many materials falls in"""
    for label, low, high in SIZE_BUCKETS:
        if size >= low and (high is None or size <= high):
            return label
    return None


def stratified_sample(cases: Sequence[Dict[str, Any]], size: int = DEFAULT_SAMPLE_SIZE,
                      seed: int = DEFAULT_SEED) -> List[Dict[str, Any]]:
    """
    Deterministic sample of the corpus, stratified over the size buckets

    Equal shares per bucket, the remainder going to the earlier buckets, and a
    bucket that cannot fill its share gives the rest back to the others. The
    result is sorted by glazy_id so that the order of the cases does not depend
    on the order of the corpus file.
    """
    by_bucket: Dict[str, List[Dict[str, Any]]] = {label: [] for label, _low, _high in SIZE_BUCKETS}
    for case in cases:
        if case['bucket'] in by_bucket:
            by_bucket[case['bucket']].append(case)

    for bucket in by_bucket.values():
        bucket.sort(key=lambda case: case['glazy_id'])

    labels = [label for label, _low, _high in SIZE_BUCKETS]
    quota = {label: size // len(labels) for label in labels}
    for index in range(size - sum(quota.values())):
        quota[labels[index % len(labels)]] += 1

    # Give back what a short bucket cannot supply, so the sample keeps its size
    shortfall = 0
    for label in labels:
        available = len(by_bucket[label])
        if quota[label] > available:
            shortfall += quota[label] - available
            quota[label] = available
    for label in labels:
        if shortfall <= 0:
            break
        spare = len(by_bucket[label]) - quota[label]
        take = min(spare, shortfall)
        quota[label] += take
        shortfall -= take

    rng = random.Random(seed)
    sample: List[Dict[str, Any]] = []
    for label in labels:
        sample.extend(rng.sample(by_bucket[label], quota[label]))

    sample.sort(key=lambda case: case['glazy_id'])
    return sample


def subsample(sample: Sequence[Dict[str, Any]], size: int, seed: int = DEFAULT_SEED) -> List[Dict[str, Any]]:
    """Deterministic subsample of an already drawn sample, same seed discipline"""
    if size >= len(sample):
        return list(sample)
    rng = random.Random(seed + 1)
    picked = rng.sample(list(sample), size)
    picked.sort(key=lambda case: case['glazy_id'])
    return picked


# --------------------------------------------------------------------------
# running
# --------------------------------------------------------------------------

_INVENTORY_CACHE: List[Dict[str, Any]] = []


def inventory_materials() -> List[Dict[str, Any]]:
    """
    Our own stock: the inInventory records of database/materials.json

    Loaded once and handed out as the same list every time. Scenario B needs it
    twice per case - the feasibility gate builds its LP from it and
    quality_metrics measures the answer against it - and re-reading the database
    a hundred times would be the slowest part of a benchmark that otherwise
    takes four seconds.

    Priorities are loaded with the records, so _priority_start_set() sees the
    same starting set the API sees. This is deliberately NOT passed to the
    solver: scenario B calls it with materials=None, which is the production
    path, and this list only exists for the two consumers that cannot go through
    the solver to get it.
    """
    if not _INVENTORY_CACHE:
        _INVENTORY_CACHE.extend(load_materials(only_inventory=True, priority=True))
    return _INVENTORY_CACHE


def quality_materials(case: Dict[str, Any], scenario: str) -> List[Dict[str, Any]]:
    """
    The material records quality_metrics needs to judge one case

    In scenario A the answer and the original are built from the same catalogue,
    so the case's own records are the whole of it.

    In scenario B they are built from two different catalogues - the answer from
    our stock, the original from the dump - and quality_metrics needs both: the
    solution's side for the conditioning and the rounding drift, the original's
    side for the set of oxides the chemistry is made of, which is what tells a
    junk component apart from a load bearing one.

    The two name spaces DO intersect, which is worth knowing before assuming
    they do not. Somebody uploaded part of the SegerLab catalogue our own
    database came from (DATA_NOTES.md) to Glazy, so five names appear on both
    sides: "Полевой шпат FFF", "Доломит МИДОЛ", "Улексит (Химпэк)", "Фритта 100
    (Рускерамика)" and "Каолин КЖФ-1". Our record wins, because the answer being
    scored was built from it, and calculate_recipe_composition takes the first
    match. Seven of the eight dump entries under those names carry an analysis
    identical to ours to two decimals; the eighth (Glazy id 364162, a second
    "Каолин КЖФ-1" with SiO2 67.52 against our 47.00) does not, so an original
    using THAT record is scored with our analysis instead of its own. What that
    can move is the original's side of the junk rule and its condition number,
    neither of which scenario B gates on - and no case of the pinned subsample
    uses it. The alternative, letting the dump's record win, would misprice and
    mis-analyse OUR answer, which the scenario does gate on.
    """
    if scenario == SCENARIO_A:
        return list(case['materials'])

    ours = inventory_materials()
    known = {record.get('name') for record in ours}
    return list(ours) + [record for record in case['materials']
                         if record.get('name') not in known]


def _solve(case: Dict[str, Any], engine: str,
           scenario: str = SCENARIO_A) -> Tuple[Optional[Dict[str, float]], str]:
    """
    Run one engine over one case and return (recipe, status)

    Scenario A injects the recipe's OWN materials through the materials= seam of
    both engines, so the target is reachable by construction and what is
    measured is the quality of the answer rather than whether one exists at all.

    Scenario B injects nothing. materials=None is the production call: the
    engine loads database/materials.json, resolves the default inventory out of
    the inInventory flags and works from the 19 materials the workshop has. The
    benchmark deliberately does not hand it a hand-built catalogue, because then
    it would be measuring a configuration nobody runs.
    """
    materials = case['materials'] if scenario == SCENARIO_A else None
    target = case['target_umf']

    try:
        if engine == ENGINE_ITERATIVE:
            solutions = find_best_recipe(
                None, target,
                max_solutions=MAX_SOLUTIONS,
                verbose=False,
                materials=materials,
            )
        else:
            solutions = find_multiple_solutions(
                target,
                max_solutions=MAX_SOLUTIONS,
                min_materials=True,
                logging=False,
                seed=CLASSIC_SEED,
                materials=materials,
            )
    except Exception as exc:
        return None, f'failed: {type(exc).__name__}: {exc}'

    if isinstance(solutions, dict):
        return None, f"failed: {solutions.get('error', 'unknown error')}"
    if not solutions:
        return None, 'failed: no solutions'

    recipe = solutions[0].get('recipe') or {}
    if not recipe:
        return None, 'failed: empty recipe'

    return recipe, 'solved'


def run_case(case: Dict[str, Any], engine: str = ENGINE_ITERATIVE,
             scenario: str = SCENARIO_A) -> Dict[str, Any]:
    """
    Solve one case and reduce it to the raw components the baseline stores

    Rolling those components up into a score happens at diff time, not here:
    changing the score formula must not invalidate a baseline (TZ_SOLVER_V2.md
    7.7).

    PRICES are always handed to solution_quality(), in both scenarios, and the
    same list decides both. In scenario A it covers nothing - Glazy material
    names are not in database/prices.json - so cost_abs, the cost ratio and
    assembly_score all come back None, exactly as they did when the prices were
    not passed at all. In scenario B it covers 18 of our 19 materials, so
    cost_abs and assembly_score are real numbers wherever the answer avoids the
    one unpriced material. One code path, and the abstention is a property of
    the data rather than of a branch here.

    PRIORITIES are never passed, in either scenario. The original recipe is a
    Glazy one in both, its materials have no entry in priorities.json, and
    scoring it against ours would invent a verdict out of no data.

    THE QUALITY VERDICT differs, and this is the one place the two scenarios are
    not symmetric. In scenario A the dump recipe is a genuine baseline: same
    materials, same supply reality, so "is the answer worse than the original"
    is a question with an answer, and quality_ok holds it. In scenario B there
    is no comparable original - the dump recipe is made of American frits and
    ours of Russian raw materials - so quality_ok is None: not "passed", not
    "failed", but "not a question this scenario can ask". The raw components
    (count, min_portion, junk_count, cond, cost_abs, assembly_score) are
    recorded exactly the same way in both and are what scenario B is tracked by,
    against the baseline rather than against a threshold.

    In scenario B the FEASIBILITY GATE runs first, and the solver runs whatever
    it says. Running the solver on a target the LP called unreachable is not
    wasted work: it is the only way to catch the LP being wrong about it, and
    the disagreements in both directions are what the scenario is really for.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f'unknown scenario {scenario!r}, expected one of {SCENARIOS}')

    result: Dict[str, Any] = {
        'glazy_id': case['glazy_id'],
        'name': case['name'],
        'scenario': scenario,
        'engine': engine,
        'bucket': case['bucket'],
        'size': case['size'],
        'status': 'failed: not run',
        'seconds': 0.0,
        'umf_error': None,
        'count': None,
        'cost_abs': None,
        'assembly_score': None,
        'min_portion': None,
        'junk_count': None,
        'rounding_drift': None,
        'cond': None,
        'failures': [],
        'chemistry_ok': False,
        'quality_ok': False if scenario == SCENARIO_A else None,
        # Scenario A asks nobody whether the target is reachable: it is, by
        # construction. These stay None there and are what says so.
        'feasible': None,
        'max_relative_deviation': None,
        'unreachable_oxides': None,
    }

    start = time.perf_counter()

    if scenario == SCENARIO_B:
        verdict = feasibility.check_feasibility(
            case['target_umf'], inventory_materials(), tol=FEASIBILITY_TOL)
        result.update({
            'feasible': verdict.get('feasible'),
            'max_relative_deviation': verdict.get('max_relative_deviation'),
            'unreachable_oxides': list(verdict.get('unreachable_oxides') or []),
            'feasibility_error': verdict.get('error'),
        })

    recipe, status = _solve(case, engine, scenario)
    result['status'] = status
    result['seconds'] = time.perf_counter() - start

    if recipe is None:
        return result

    materials = quality_materials(case, scenario)
    actual_umf = weights_to_umf(calculate_recipe_composition(materials, recipe))
    umf_error = float(calculate_umf_error(case['target_umf'], actual_umf))

    quality = qm.solution_quality(recipe, case['original'], materials, prices=qm.load_prices())

    result.update({
        'umf_error': umf_error,
        'count': len(recipe),
        'cost_abs': quality['cost']['cost_abs'],
        'assembly_score': quality['assembly_score']['value'],
        'min_portion': quality['min_portion']['solution'],
        'junk_count': quality['junk']['solution'],
        'rounding_drift': quality['rounding_drift']['value'],
        'cond': quality['conditioning']['cond'],
        'failures': list(quality['failures']),
        'chemistry_ok': umf_error <= MAX_UMF_ERROR,
        'quality_ok': (not quality['failures']) if scenario == SCENARIO_A else None,
        # Kept for the failure log: which side of a gated metric actually moved
        'detail': {
            'count': quality['count'],
            'junk': quality['junk'],
            'min_portion': quality['min_portion'],
            'conditioning': quality['conditioning'],
            'cost': quality['cost'],
        },
    })

    return result


def run_sample(sample: Sequence[Dict[str, Any]], engine: str = ENGINE_ITERATIVE,
               progress: Optional[Any] = None,
               scenario: str = SCENARIO_A) -> List[Dict[str, Any]]:
    """Run a whole sample, optionally reporting progress through a callable"""
    results = []
    for index, case in enumerate(sample, start=1):
        results.append(run_case(case, engine, scenario))
        if progress is not None and index % 25 == 0:
            progress(index, len(sample))
    return results


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------

def profile(values: Sequence[float]) -> Optional[Dict[str, float]]:
    """
    min / p10 / median / mean / p90 / p99 / max of a sample, plus its size

    None for an empty sample - a metric that is defined on no case has no
    distribution, and returning zeros would invent one.
    """
    if not values:
        return None
    array = np.asarray(list(values), dtype=float)
    return {
        'n': int(array.size),
        'min': float(array.min()),
        'p10': float(np.percentile(array, 10)),
        'median': float(np.percentile(array, 50)),
        'mean': float(array.mean()),
        'p90': float(np.percentile(array, 90)),
        'p99': float(np.percentile(array, 99)),
        'max': float(array.max()),
    }


def chemistry_ok(row: Dict[str, Any]) -> bool:
    """
    Whether one snapshot row passed level 1 of the two-level criterion

    run_case() records the verdict as "chemistry_ok". Snapshots written before
    that field entered the baseline carry only the error, so the verdict is
    re-derived from it with TODAY's MAX_UMF_ERROR - which is the honest thing to
    do and also the reason the threshold is written into every history record:
    move it and the re-derived shares of the old runs move with it.
    """
    recorded = row.get('chemistry_ok')
    if recorded is not None:
        return bool(recorded)
    error = row.get('umf_error')
    if error is None:
        return False
    return float(error) <= MAX_UMF_ERROR


def accounted_ok(row: Dict[str, Any]) -> bool:
    """
    Whether scenario B answered this case correctly, one way or the other

    Two answers count, and the second is the whole point of the scenario:

      * the chemistry gate was met - the target was reproduced from our stock;
      * feasibility said the target is out of reach of our stock. There is no
        recipe to be had, so "there is no recipe" is the correct answer and
        counting it as a failure would make the benchmark punish honesty.

    Undecided (feasible is None, an LP that did not converge) is NOT accounted
    for: nobody answered anything.
    """
    if chemistry_ok(row):
        return True
    return row.get('feasible') is False


def misclassification(row: Dict[str, Any]) -> Optional[str]:
    """
    Where the feasibility gate and the solver contradict each other, if they do

    Each disagreement is a bug report against one of the two, and which one has
    to be read case by case, so the row is named rather than counted:

      "reachable_unsolved"   the LP promised a recipe and the solver did not
                             find one. Either the search is missing something
                             the LP can see, or the LP is over-promising
      "unreachable_solved"   the LP said no recipe exists and the solver
                             produced one that passes the chemistry gate anyway.
                             The two verdicts are not measured on the same
                             scale - the LP bounds the worst RELATIVE deviation
                             of each oxide, the gate is an RMS of the ABSOLUTE
                             deviations over the target's oxides - so a target
                             whose only unreachable oxide is a colourant at UMF
                             0.02 is honestly out of reach and honestly inside
                             an RMS of 0.1 at the same time

    None when they agree, and None for scenario A, which never asks.
    """
    feasible = row.get('feasible')
    if feasible is None:
        return None
    if feasible and not chemistry_ok(row):
        return 'reachable_unsolved'
    if not feasible and chemistry_ok(row):
        return 'unreachable_solved'
    return None


def group_key(scenario: Optional[str], engine: str) -> str:
    """
    The name one (scenario, engine) pair is profiled and logged under

    Scenario A keeps the bare engine name, and that asymmetry is deliberate.
    Every run recorded before scenario B existed measured scenario A and is
    keyed "iterative" / "classic" in bench/history.jsonl; re-keying them would
    either break the series the log exists for or require rewriting an append
    only file. So scenario A stays the unlabelled default - it is what an
    unqualified engine name has always meant - and a new scenario opens its own
    series under its own key instead of silently redefining the old one.
    """
    if not scenario or scenario == SCENARIO_A:
        return engine
    return f'{scenario}/{engine}'


def snapshot_groups(cases: Sequence[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """(scenario, engine) pairs present in a snapshot, in a stable order"""
    pairs = {(row.get('scenario') or SCENARIO_A, row.get('engine'))
             for row in cases if row.get('engine')}
    return sorted(pairs)


def engine_profile(cases: Sequence[Dict[str, Any]], engine: str,
                   scenario: Optional[str] = None) -> Dict[str, Any]:
    """
    Roll the rows of ONE engine of one snapshot up into a single run's numbers

    This is the single-run twin of diff_baseline.compare_engine(), and the two
    differ in one way that matters when the numbers are read side by side: the
    diff profiles a metric over the cases solved by BOTH runs, because a case
    flipping between solved and failed would otherwise shift every aggregate
    silently. A single run has no counterpart to intersect with, so the profile
    here covers every case this run solved and for which the metric is defined.
    A history line and a diff line of the same run therefore need not carry the
    same median, and the per-metric "n" is what says how many cases each spoke
    for.

    The three shares are all taken over the FULL case count of the engine, not
    over the solved ones - an unsolved case is a failed case at both levels,
    exactly as tests/test_inverse_corpus.py counts it.

    "both_levels_share" is None rather than 0.0 when the rows do not carry
    quality_ok: the quality verdict is not derivable from the stored components
    (it compares the solution against the original recipe, which the snapshot
    does not keep), so a snapshot written before that field existed cannot
    answer the question and must not pretend to. Scenario B carries a None there
    for a different reason with the same shape - it has no comparable original
    at all - and the two are indistinguishable from here on purpose: both mean
    "this run cannot answer that question".

    "reachability" is None for scenario A, which never asks whether the target
    can be reached, and holds the scenario B verdict counts and the accounted
    share otherwise. "priced" is filled in for both, and for scenario A it is
    honestly zero: no Glazy material has a price.

    Args:
        scenario: keep only the rows of this scenario. None means every row of
            the engine, which is what a snapshot holding one scenario wants and
            what every caller written before scenario B existed asked for
    """
    rows = [row for row in cases
            if row.get('engine') == engine
            and (scenario is None or (row.get('scenario') or SCENARIO_A) == scenario)]
    total = len(rows)

    solved = [row for row in rows if row.get('status') == 'solved']
    chemistry = [row for row in rows if chemistry_ok(row)]
    quality_known = all(row.get('quality_ok') is not None for row in solved)
    both = [row for row in chemistry if row.get('quality_ok')] if quality_known else None

    metrics: Dict[str, Optional[Dict[str, float]]] = {}
    for name, _direction in METRICS:
        values = [float(row[name]) for row in solved if row.get(name) is not None]
        metrics[name] = profile(values)

    priced = [row for row in solved if row.get('cost_abs') is not None]

    return {
        'cases': total,
        'solved': len(solved),
        'solved_share': (len(solved) / total) if total else None,
        'chemistry': len(chemistry),
        'chemistry_share': (len(chemistry) / total) if total else None,
        'both_levels': len(both) if both is not None else None,
        'both_levels_share': (len(both) / total) if (both is not None and total) else None,
        'reachability': _reachability_profile(rows),
        'priced': len(priced),
        'priced_share': (len(priced) / len(solved)) if solved else None,
        'metrics': metrics,
    }


def _reachability_profile(rows: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    The scenario B half of a profile: the verdict counts and what came of them

    None when no row carries a verdict at all, which is what scenario A looks
    like. An empty feasibility field is not the same as "everything was
    reachable" and must not be rendered as one.
    """
    judged = [row for row in rows if 'feasible' in row and row.get('feasible') is not None]
    if not judged:
        return None

    total = len(rows)
    reachable = [row for row in judged if row.get('feasible') is True]
    unreachable = [row for row in judged if row.get('feasible') is False]
    undecided = total - len(judged)

    accounted = [row for row in rows if accounted_ok(row)]
    solved_reachable = [row for row in reachable if chemistry_ok(row)]
    misclassified = [row for row in rows if misclassification(row)]

    return {
        'reachable': len(reachable),
        'unreachable': len(unreachable),
        'undecided': undecided,
        'accounted': len(accounted),
        'accounted_share': (len(accounted) / total) if total else None,
        'solved_among_reachable': len(solved_reachable),
        'solved_among_reachable_share': (
            (len(solved_reachable) / len(reachable)) if reachable else None),
        'misclassified': len(misclassified),
        'reachable_unsolved': sum(
            1 for row in rows if misclassification(row) == 'reachable_unsolved'),
        'unreachable_solved': sum(
            1 for row in rows if misclassification(row) == 'unreachable_solved'),
        'tol': FEASIBILITY_TOL,
    }


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------

def file_hash(path: str) -> Optional[str]:
    """sha256 of a file, or None when it is not there"""
    if not os.path.exists(path):
        return None
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 16), b''):
            digest.update(chunk)
    return digest.hexdigest()


def data_hashes() -> Dict[str, Optional[str]]:
    """Hashes of the input data files the solver's answers depend on"""
    return {relative: file_hash(os.path.join(PROJECT_ROOT, relative))
            for relative in HASHED_DATA_FILES}


def git_commit() -> Optional[str]:
    """Current commit of the work tree, or None outside a repository"""
    try:
        output = subprocess.run(
            ['git', '-C', PROJECT_ROOT, 'rev-parse', 'HEAD'],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except Exception:
        return None
    if output.returncode != 0:
        return None
    return output.stdout.strip() or None


def git_dirty() -> Optional[bool]:
    """Whether the work tree has uncommitted changes; None when git is unusable"""
    try:
        output = subprocess.run(
            ['git', '-C', PROJECT_ROOT, 'status', '--porcelain'],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except Exception:
        return None
    if output.returncode != 0:
        return None
    return bool(output.stdout.strip())
