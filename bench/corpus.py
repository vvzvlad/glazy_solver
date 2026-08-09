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

    tests/test_inverse_corpus.py   the pass/fail gate of scenario A
    bench/diff_baseline.py         the baseline snapshot and its regression diff

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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import flux_oxides, weights_to_umf
from solver_classic import (
    calculate_recipe_composition,
    calculate_umf_error,
    find_multiple_solutions,
)
from solver_iterative import find_best_recipe
import quality_metrics as qm

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINE_PATH = os.path.join(BENCH_DIR, 'quality_baseline.json')

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

# The configuration the tests and the API actually run in. Numbers measured at
# max_solutions=1 do not carry over: the beam width and the number of children
# are both 1 there, which is a different search (TZ_SOLVER_V2.md 10.8).
MAX_SOLUTIONS = 5
CLASSIC_SEED = 42

# Level 1 of the two-level criterion of 7.1: the chemistry has to match.
MAX_UMF_ERROR = 0.1

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

def _solve(case: Dict[str, Any], engine: str) -> Tuple[Optional[Dict[str, float]], str]:
    """
    Run one engine over one case and return (recipe, status)

    The inventory is the recipe's OWN materials, injected through the
    materials= seam of both engines, so the target is reachable by construction
    and what is measured is the quality of the answer rather than whether one
    exists at all.
    """
    materials = case['materials']
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


def run_case(case: Dict[str, Any], engine: str = ENGINE_ITERATIVE) -> Dict[str, Any]:
    """
    Solve one case and reduce it to the raw components the baseline stores

    Rolling those components up into a score happens at diff time, not here:
    changing the score formula must not invalidate a baseline (TZ_SOLVER_V2.md
    7.7).

    Note what is deliberately NOT passed to solution_quality(): prices and
    priorities. Glazy material names do not appear in database/prices.json and
    have no entry in priorities.json, so both metrics abstain (ok=None) and,
    being None rather than False, can never enter "failures". Inventing either
    of them would manufacture a verdict out of no data.
    """
    start = time.perf_counter()
    recipe, status = _solve(case, engine)
    elapsed = time.perf_counter() - start

    result: Dict[str, Any] = {
        'glazy_id': case['glazy_id'],
        'name': case['name'],
        'scenario': 'A',
        'engine': engine,
        'bucket': case['bucket'],
        'size': case['size'],
        'status': status,
        'seconds': elapsed,
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
        'quality_ok': False,
    }

    if recipe is None:
        return result

    materials = case['materials']
    actual_umf = weights_to_umf(calculate_recipe_composition(materials, recipe))
    umf_error = float(calculate_umf_error(case['target_umf'], actual_umf))

    quality = qm.solution_quality(recipe, case['original'], materials)

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
        'quality_ok': not quality['failures'],
        # Kept for the failure log: which side of a gated metric actually moved
        'detail': {
            'count': quality['count'],
            'junk': quality['junk'],
            'min_portion': quality['min_portion'],
            'conditioning': quality['conditioning'],
        },
    })

    return result


def run_sample(sample: Sequence[Dict[str, Any]], engine: str = ENGINE_ITERATIVE,
               progress: Optional[Any] = None) -> List[Dict[str, Any]]:
    """Run a whole sample, optionally reporting progress through a callable"""
    results = []
    for index, case in enumerate(sample, start=1):
        results.append(run_case(case, engine))
        if progress is not None and index % 25 == 0:
            progress(index, len(sample))
    return results


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
