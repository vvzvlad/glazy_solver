#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# flake8: noqa
# pylint: disable=broad-exception-raised, raise-missing-from, too-many-arguments, redefined-outer-name
# pylint: disable=too-many-positional-arguments, too-many-locals, too-many-branches, too-many-statements
# pylint: disable=multiple-statements, logging-fstring-interpolation, trailing-whitespace, line-too-long
# pylint: disable=broad-exception-caught, missing-function-docstring, missing-class-docstring
# pylint: disable=f-string-without-interpolation
# pylance: disable=reportMissingImports, reportMissingModuleSource

"""
Quality baseline and regression diff of the Glazy corpus (TZ_SOLVER_V2.md 7.7)

The point of this script: the quality of a solution is not gated by numbers
pulled from the air. The corpus is run once, the raw per-case components are
committed as bench/quality_baseline.json, and every later change to the solver
is measured as a diff against that snapshot.

    python bench/diff_baseline.py                 # run and print the diff
    python bench/diff_baseline.py --check         # ... and exit 1 on a regression
    python bench/diff_baseline.py --rebaseline    # overwrite the snapshot

The snapshot stores RAW COMPONENTS per case - umf_error, count, cost_abs,
assembly_score, min_portion, junk_count, rounding_drift, conditioning.cond - and
never a rolled up score. The roll-up happens here, at diff time, so changing the
score formula does not invalidate a baseline that took ten minutes to produce.

Three rules keep the comparison honest, and each of them exists because the
naive version lies:

  * percentiles are computed over the INTERSECTION of the solved sets of the two
    runs. A case flipping between solved and failed would otherwise drop out of
    one distribution and shift every aggregate silently. Changes of the solved
    set are reported as their own line, with the ids;
  * assembly_score is None wherever the prices do not cover the recipe - which
    for Glazy materials is everywhere - so its percentiles use the same
    intersection rule on "is it defined", not just on "is it solved";
  * the input data files are hashed into the snapshot. An updated price list
    moves every cost metric without a line of solver code being touched, and
    that is not a regression. When the hashes differ the report says so loudly
    instead of quietly comparing.

The distribution profile is the whole distribution - min / p10 / median / mean /
p90 / p99 / max, before and after - because a mean hides the tail: a change can
improve the median while wrecking a handful of cases, and only p90 / p99 / max
show it. --check gates the tail alongside the centre, on the p90 as well as on
the median.

This script is deliberately NOT part of `unittest discover`: it is a manual gate
to run before merging a solver change.
"""

import argparse
import datetime
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench import corpus as bench_corpus


BASELINE_FORMAT_VERSION = 1

# A metric may worsen by this much before --check calls it a regression. Five
# percent is the band inside which a solver tweak is noise rather than a change
# of behaviour; anything larger is a decision somebody has to defend.
REGRESSION_TOLERANCE = 0.05

# The chemistry error is gated on its maximum rather than on a percentile: one
# case that stops converging is a regression even if the other 299 improve. The
# epsilon only absorbs floating point noise between platforms, nothing more.
MAX_CHEMISTRY_ERROR_EPSILON = 1e-6

# What --check gates on.
#
# 7.7 names assembly_score. It is cost_abs x len(recipe) and it is None wherever
# the prices do not cover the recipe, which for Glazy material names is every
# single case - so on this corpus the gate would have nothing to hold on to.
# 10.10 says what to do about that in as many words: "with assembly_score = None
# the baseline compares count, junk, min_portion and conditioning". Those four
# are therefore the fallback, and they are gated individually rather than being
# mashed into an invented composite - a gate nobody can name the subject of is
# not a gate.
PRIMARY_TRACKED_SCORE = 'assembly_score'
FALLBACK_TRACKED_SCORES = ('count', 'junk_count', 'min_portion', 'cond')

# The aggregates that gate. The centre and the tail together: a change that
# improves the median while wrecking a handful of cases must not pass, and a
# change that only moves the tail must not pass either.
GATED_AGGREGATES = ('median', 'p90')

# A metric defined on fewer cases than this is reported but not gated: a
# percentile over five numbers is noise, not a distribution.
MIN_TRACKED_CASES = 10

# Raw components and the direction that counts as better. Everything here is
# "smaller is better" except the smallest portion of a recipe, where a larger
# value means the recipe is easier to weigh out.
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

# Components stored per case. Kept explicit so that a field added to run_case()
# does not silently enter the snapshot and change its meaning.
CASE_FIELDS = (
    'glazy_id', 'scenario', 'engine', 'status', 'bucket', 'size',
    'umf_error', 'count', 'cost_abs', 'assembly_score', 'min_portion',
    'junk_count', 'rounding_drift', 'cond',
)

# Below this the two values are the same number and the case is "unchanged".
EQUAL_EPSILON = 1e-9


# --------------------------------------------------------------------------
# running
# --------------------------------------------------------------------------

def run_corpus(seed: int, sample_size: int, classic_size: int,
               verbose: bool = True) -> Dict[str, Any]:
    """
    Run scenario A over the sample and return a snapshot-shaped dictionary

    Both engines are run: the iterative one over the whole sample, the classic
    one over its subsample, exactly as the corpus test does.
    """
    corpus, timings = bench_corpus.load_corpus()
    cases, rejected = bench_corpus.build_cases(corpus)
    sample = bench_corpus.stratified_sample(cases, size=sample_size, seed=seed)
    classic_sample = bench_corpus.subsample(sample, classic_size, seed=seed)

    if verbose:
        print(f"corpus {timings['dump']} loaded from {timings['source']} in {timings['seconds']:.2f}s; "
              f"{len(cases)} usable cases, sample {len(sample)}, classic subsample {len(classic_sample)}")

    rows: List[Dict[str, Any]] = []
    started = time.perf_counter()

    for engine, subset in ((bench_corpus.ENGINE_ITERATIVE, sample),
                           (bench_corpus.ENGINE_CLASSIC, classic_sample)):
        engine_started = time.perf_counter()
        for case in subset:
            result = bench_corpus.run_case(case, engine)
            rows.append({field: result.get(field) for field in CASE_FIELDS})
        if verbose:
            print(f"  {engine}: {len(subset)} cases in {time.perf_counter() - engine_started:.1f}s")

    return {
        'format_version': BASELINE_FORMAT_VERSION,
        'generated': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
        'run': {
            'seed': seed,
            'sample_size': sample_size,
            'classic_subsample': classic_size,
            'scenario': 'A',
            'sampled_ids': {
                bench_corpus.ENGINE_ITERATIVE: [case['glazy_id'] for case in sample],
                bench_corpus.ENGINE_CLASSIC: [case['glazy_id'] for case in classic_sample],
            },
            'dump': timings['dump'],
            'dump_size': timings['dump_size'],
            'corpus_stats': corpus['stats'],
            'rejected': rejected,
            'git_commit': bench_corpus.git_commit(),
            'git_dirty': bench_corpus.git_dirty(),
            'data_hashes': bench_corpus.data_hashes(),
            'solver_config': {
                'max_solutions': bench_corpus.MAX_SOLUTIONS,
                'classic_seed': bench_corpus.CLASSIC_SEED,
                'candidate_search': 'exhaustive',
            },
            'seconds': time.perf_counter() - started,
        },
        'cases': rows,
    }


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------

def _profile(values: Sequence[float]) -> Optional[Dict[str, float]]:
    """min / p10 / median / mean / p90 / p99 / max of a sample"""
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


def _index(cases: Sequence[Dict[str, Any]], engine: str) -> Dict[int, Dict[str, Any]]:
    """Cases of one engine keyed by their Glazy id"""
    return {row['glazy_id']: row for row in cases if row.get('engine') == engine}


def _solved_ids(index: Dict[int, Dict[str, Any]]) -> set:
    return {glazy_id for glazy_id, row in index.items() if row.get('status') == 'solved'}


def _better(metric_direction: str, before: float, after: float) -> int:
    """-1 worsened, 0 unchanged, +1 improved"""
    if abs(after - before) <= EQUAL_EPSILON:
        return 0
    if metric_direction == 'lower':
        return 1 if after < before else -1
    return 1 if after > before else -1


def _relative_change(direction: str, before: float, after: float) -> Optional[float]:
    """
    How much worse the value got, as a positive fraction of the old one

    Negative means it improved. None when the old value is zero and the relative
    change is undefined - the caller then falls back to the absolute comparison.
    """
    if before == 0:
        return None
    if direction == 'lower':
        return (after - before) / abs(before)
    return (before - after) / abs(before)


def compare_engine(baseline_index: Dict[int, Dict[str, Any]],
                   current_index: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    """Everything the report needs about one engine"""
    baseline_solved = _solved_ids(baseline_index)
    current_solved = _solved_ids(current_index)

    shared_ids = sorted(set(baseline_index) & set(current_index))
    lost = sorted(baseline_solved - current_solved)
    gained = sorted(current_solved - baseline_solved)
    intersection = sorted(baseline_solved & current_solved)

    metrics: Dict[str, Any] = {}
    for name, direction in METRICS:
        pairs = []
        for glazy_id in intersection:
            before = baseline_index[glazy_id].get(name)
            after = current_index[glazy_id].get(name)
            # The same intersection rule as for the solved set, applied to
            # "is this metric defined at all": assembly_score and cost_abs are
            # None without prices, cond is None for a dependent material set.
            if before is None or after is None:
                continue
            pairs.append((glazy_id, float(before), float(after)))

        improved = [p for p in pairs if _better(direction, p[1], p[2]) > 0]
        worsened = [p for p in pairs if _better(direction, p[1], p[2]) < 0]
        unchanged = [p for p in pairs if _better(direction, p[1], p[2]) == 0]

        undefined_before = sum(1 for glazy_id in intersection
                               if baseline_index[glazy_id].get(name) is None)
        undefined_after = sum(1 for glazy_id in intersection
                              if current_index[glazy_id].get(name) is None)

        metrics[name] = {
            'direction': direction,
            'pairs': pairs,
            'before': _profile([p[1] for p in pairs]),
            'after': _profile([p[2] for p in pairs]),
            'improved': len(improved),
            'worsened': len(worsened),
            'unchanged': len(unchanged),
            'undefined_before': undefined_before,
            'undefined_after': undefined_after,
            'worst': sorted(
                worsened,
                key=lambda item: -abs(item[2] - item[1]),
            )[:10],
        }

    return {
        'baseline_cases': len(baseline_index),
        'current_cases': len(current_index),
        'missing_from_current': sorted(set(baseline_index) - set(current_index)),
        'new_in_current': sorted(set(current_index) - set(baseline_index)),
        'shared': shared_ids,
        'baseline_solved': len(baseline_solved),
        'current_solved': len(current_solved),
        'baseline_solved_share': len(baseline_solved) / len(baseline_index) if baseline_index else 0.0,
        'current_solved_share': len(current_solved) / len(current_index) if current_index else 0.0,
        'lost': lost,
        'gained': gained,
        'intersection': intersection,
        'metrics': metrics,
    }


def _is_gateable(metrics: Dict[str, Any], name: str) -> bool:
    """Whether a metric is defined on enough cases of the intersection to gate"""
    entry = metrics.get(name)
    return bool(entry and entry['before'] and entry['before']['n'] >= MIN_TRACKED_CASES)


def tracked_scores(metrics: Dict[str, Any]) -> List[str]:
    """
    Which metrics --check gates on for this run

    assembly_score alone when the prices define it, and the four raw components
    of 10.10 otherwise. Returning the NAMES rather than a number keeps the
    choice visible in the report.
    """
    if _is_gateable(metrics, PRIMARY_TRACKED_SCORE):
        return [PRIMARY_TRACKED_SCORE]
    return [name for name in FALLBACK_TRACKED_SCORES if _is_gateable(metrics, name)]


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def _format_number(value: Optional[float]) -> str:
    if value is None:
        return '-'
    if abs(value) >= 1000 or (value != 0 and abs(value) < 0.001):
        return f'{value:.3e}'
    return f'{value:.4f}'


def _profile_lines(name: str, entry: Dict[str, Any]) -> List[str]:
    before, after = entry['before'], entry['after']
    if before is None or after is None:
        return [f'  {name:<16} not defined on any shared case '
                f"(undefined before {entry['undefined_before']}, after {entry['undefined_after']})"]

    lines = [f"  {name:<16} n={before['n']:<4} direction={entry['direction']:<6} "
             f"improved {entry['improved']} / worsened {entry['worsened']} / unchanged {entry['unchanged']}"]
    if entry['undefined_before'] or entry['undefined_after']:
        lines.append(f"      undefined on {entry['undefined_before']} baseline / "
                     f"{entry['undefined_after']} current cases of the intersection")

    header = f"      {'':<8}" + ''.join(f'{key:>13}' for key in ('min', 'p10', 'median', 'mean', 'p90', 'p99', 'max'))
    lines.append(header)
    for label, profile in (('before', before), ('after', after)):
        row = f'      {label:<8}' + ''.join(
            f'{_format_number(profile[key]):>13}' for key in ('min', 'p10', 'median', 'mean', 'p90', 'p99', 'max'))
        lines.append(row)

    deltas = f"      {'delta':<8}" + ''.join(
        f'{_format_number(after[key] - before[key]):>13}'
        for key in ('min', 'p10', 'median', 'mean', 'p90', 'p99', 'max'))
    lines.append(deltas)

    return lines


def build_report(baseline: Dict[str, Any], current: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
    """
    Assemble the whole report and the machine readable comparison behind it

    Returns (lines, comparison) where comparison holds one entry per engine plus
    the provenance findings.
    """
    lines: List[str] = []
    comparison: Dict[str, Any] = {'engines': {}, 'provenance': {}}

    lines.append('GLAZY CORPUS QUALITY DIFF')
    lines.append('=' * 78)
    lines.append(f"baseline generated {baseline.get('generated')} "
                 f"at commit {(baseline['run'].get('git_commit') or '?')[:12]}"
                 f"{' (dirty tree)' if baseline['run'].get('git_dirty') else ''}")
    lines.append(f"current  run       {current.get('generated')} "
                 f"at commit {(current['run'].get('git_commit') or '?')[:12]}"
                 f"{' (dirty tree)' if current['run'].get('git_dirty') else ''}")
    lines.append('')

    # --- provenance -------------------------------------------------------
    findings = []

    if baseline['run'].get('seed') != current['run'].get('seed'):
        findings.append(f"SEED CHANGED: {baseline['run'].get('seed')} -> {current['run'].get('seed')}; "
                        f"the two runs are different corpora and the numbers below are not comparable")
    if baseline['run'].get('dump') != current['run'].get('dump'):
        findings.append(f"DUMP CHANGED: {baseline['run'].get('dump')} -> {current['run'].get('dump')}")

    for engine, ids in (baseline['run'].get('sampled_ids') or {}).items():
        current_ids = (current['run'].get('sampled_ids') or {}).get(engine)
        if current_ids is not None and list(ids) != list(current_ids):
            findings.append(f"SAMPLE CHANGED for {engine}: the same seed drew a different set of ids, "
                            f"so the corpus itself moved under the baseline")

    baseline_hashes = baseline['run'].get('data_hashes') or {}
    current_hashes = current['run'].get('data_hashes') or {}
    changed_data = [name for name in sorted(set(baseline_hashes) | set(current_hashes))
                    if baseline_hashes.get(name) != current_hashes.get(name)]
    comparison['provenance']['changed_data'] = changed_data

    if changed_data:
        lines.append('*' * 78)
        lines.append('*  WARNING: THE INPUT DATA CHANGED, NOT ONLY THE SOLVER')
        for name in changed_data:
            lines.append(f"*    {name}: {(baseline_hashes.get(name) or 'absent')[:12]} -> "
                         f"{(current_hashes.get(name) or 'absent')[:12]}")
        lines.append('*  Every difference below may come from the data rather than from the code.')
        lines.append('*' * 78)
        lines.append('')

    for finding in findings:
        lines.append(f'!! {finding}')
    if findings:
        lines.append('')
    comparison['provenance']['findings'] = findings

    # --- per engine -------------------------------------------------------
    engines = sorted({row.get('engine') for row in baseline['cases']}
                     | {row.get('engine') for row in current['cases']})

    for engine in engines:
        result = compare_engine(_index(baseline['cases'], engine), _index(current['cases'], engine))
        comparison['engines'][engine] = result

        lines.append(f'--- engine: {engine} ' + '-' * (78 - 13 - len(engine)))
        lines.append(f"  cases: baseline {result['baseline_cases']}, current {result['current_cases']}")
        if result['missing_from_current'] or result['new_in_current']:
            lines.append(f"  case set changed: {len(result['missing_from_current'])} gone, "
                         f"{len(result['new_in_current'])} new")
        lines.append(f"  solved: {result['baseline_solved']}/{result['baseline_cases']} "
                     f"({result['baseline_solved_share']:.2%}) -> "
                     f"{result['current_solved']}/{result['current_cases']} "
                     f"({result['current_solved_share']:.2%})")
        if result['lost']:
            lines.append(f"  SOLVED -> FAILED ({len(result['lost'])}): {result['lost']}")
        if result['gained']:
            lines.append(f"  FAILED -> SOLVED ({len(result['gained'])}): {result['gained']}")
        lines.append(f"  percentiles are computed over the {len(result['intersection'])} cases "
                     f"solved by BOTH runs")
        lines.append('')

        for name, _direction in METRICS:
            lines.extend(_profile_lines(name, result['metrics'][name]))
            lines.append('')

        chosen = tracked_scores(result['metrics'])
        if chosen == [PRIMARY_TRACKED_SCORE]:
            reason = 'the prices define it'
        elif chosen:
            reason = f'{PRIMARY_TRACKED_SCORE} is undefined without prices, so 10.10 applies'
        else:
            reason = 'nothing is defined on enough cases - the quality gate cannot run'
        lines.append(f"  gated by --check: {', '.join(chosen) or 'nothing'} ({reason}), "
                     f"on {' and '.join(GATED_AGGREGATES)}, tolerance {REGRESSION_TOLERANCE:.0%}")

        worst = []
        for name, _direction in METRICS:
            for glazy_id, before, after in result['metrics'][name]['worst']:
                worst.append((abs(after - before), name, glazy_id, before, after))
        worst.sort(key=lambda item: -item[0])
        if worst:
            lines.append('  top worsened cases:')
            for _size, name, glazy_id, before, after in worst[:10]:
                lines.append(f"    id={glazy_id:<8} {name:<16} "
                             f"{_format_number(before)} -> {_format_number(after)}")
        lines.append('')

    return lines, comparison


# --------------------------------------------------------------------------
# the gate
# --------------------------------------------------------------------------

def check_regressions(comparison: Dict[str, Any]) -> List[str]:
    """
    Everything --check considers a regression

    Three gates, per engine:
      * the median OR the p90 of a tracked score worsens by more than
        REGRESSION_TOLERANCE - the tail is gated alongside the centre, because a
        tweak that improves the median while wrecking ten cases is not an
        improvement;
      * the share of solved cases drops;
      * the maximum chemistry error grows.

    A measured warning about the reach of the first gate. Percentiles only move
    when enough cases move: a degradation confined to 4% of the sample slips
    past both the median and the p90 by construction. That is not hypothetical -
    running this corpus with PRUNE_ERROR_TOLERANCE forced to 0, which all but
    disables the backward elimination pass, added a component to 13 of 300 cases
    and left the count median and p90 exactly where they were. What caught it
    was min_portion, whose median fell 3.215 -> 2.985. Gating the four
    components of 10.10 individually rather than one rolled up number is what
    makes the gate reach that far, and even so the per-case "worsened" counts
    printed above the gate are the more sensitive instrument. Read them.
    """
    problems: List[str] = []

    for engine, result in sorted(comparison['engines'].items()):
        metrics = result['metrics']

        if result['current_solved_share'] < result['baseline_solved_share'] - EQUAL_EPSILON:
            problems.append(f"{engine}: the solved share dropped "
                            f"{result['baseline_solved_share']:.2%} -> {result['current_solved_share']:.2%} "
                            f"(lost: {result['lost']})")

        error = metrics.get('umf_error')
        if error and error['before'] and error['after']:
            before_max, after_max = error['before']['max'], error['after']['max']
            if after_max > before_max + MAX_CHEMISTRY_ERROR_EPSILON:
                problems.append(f"{engine}: the maximum chemistry error grew "
                                f"{before_max:.6f} -> {after_max:.6f}")

        names = tracked_scores(metrics)
        if not names:
            problems.append(f"{engine}: no tracked score is defined on enough cases, "
                            f"so the quality gate could not run at all")
            continue

        for name in names:
            entry = metrics[name]
            direction = entry['direction']
            for aggregate in GATED_AGGREGATES:
                before = entry['before'][aggregate]
                after = entry['after'][aggregate]
                change = _relative_change(direction, before, after)
                if change is None:
                    # Undefined relative change: fall back to the plain direction
                    if _better(direction, before, after) < 0:
                        problems.append(f"{engine}: {name} {aggregate} worsened {before} -> {after} "
                                        f"(the baseline value is zero, so the "
                                        f"{REGRESSION_TOLERANCE:.0%} band does not apply)")
                    continue
                if change > REGRESSION_TOLERANCE:
                    problems.append(f"{engine}: {name} {aggregate} worsened by {change:.1%} "
                                    f"({_format_number(before)} -> {_format_number(after)}), "
                                    f"the limit is {REGRESSION_TOLERANCE:.0%}")

    return problems


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def load_baseline(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        baseline = json.load(f)
    if baseline.get('format_version') != BASELINE_FORMAT_VERSION:
        raise SystemExit(f'{path}: baseline format {baseline.get("format_version")} is not '
                         f'{BASELINE_FORMAT_VERSION}; regenerate it with --rebaseline')
    return baseline


def write_baseline(path: str, snapshot: Dict[str, Any]) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(snapshot, f, indent=1, sort_keys=False, ensure_ascii=False)
        f.write('\n')


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Diff the current solver against the committed quality baseline')
    parser.add_argument('--baseline', default=bench_corpus.BASELINE_PATH,
                        help='path of the baseline snapshot (default: bench/quality_baseline.json)')
    parser.add_argument('--check', action='store_true',
                        help='exit 1 when the tracked score, the solved share or the maximum '
                             'chemistry error regressed')
    parser.add_argument('--rebaseline', action='store_true',
                        help='overwrite the snapshot with the current run; the only way to move it')
    parser.add_argument('--seed', type=int, default=None,
                        help='sampling seed; defaults to the one recorded in the baseline')
    parser.add_argument('--sample-size', type=int, default=None,
                        help='sample size; defaults to the one recorded in the baseline')
    parser.add_argument('--classic-subsample', type=int, default=None,
                        help='classic subsample size; defaults to the one recorded in the baseline')
    parser.add_argument('--save-current', default=None,
                        help='also write the current run to this path, for offline comparison')
    parser.add_argument('--current', default=None,
                        help='compare a previously saved run instead of solving again')
    args = parser.parse_args()

    existing = None
    if os.path.exists(args.baseline):
        existing = load_baseline(args.baseline)

    seed = args.seed
    sample_size = args.sample_size
    classic_size = args.classic_subsample

    if existing is not None:
        seed = seed if seed is not None else existing['run']['seed']
        sample_size = sample_size if sample_size is not None else existing['run']['sample_size']
        classic_size = classic_size if classic_size is not None else existing['run']['classic_subsample']
    else:
        seed = seed if seed is not None else bench_corpus.DEFAULT_SEED
        sample_size = sample_size if sample_size is not None else bench_corpus.DEFAULT_SAMPLE_SIZE
        classic_size = classic_size if classic_size is not None else bench_corpus.DEFAULT_CLASSIC_SUBSAMPLE

    if args.current:
        with open(args.current, 'r', encoding='utf-8') as f:
            current = json.load(f)
    else:
        try:
            current = run_corpus(seed, sample_size, classic_size)
        except bench_corpus.CorpusUnavailable as exc:
            print(f'Glazy corpus unavailable: {exc}')
            return 2

    if args.save_current:
        write_baseline(args.save_current, current)
        print(f'current run written to {args.save_current}')

    if args.rebaseline:
        if existing is not None:
            lines, _comparison = build_report(existing, current)
            print('\n'.join(lines))
            print('--- the snapshot above is being REPLACED ---')
        write_baseline(args.baseline, current)
        print(f'baseline written to {args.baseline}')
        print('Commit it separately, with the reason the level moved in the commit message.')
        return 0

    if existing is None:
        raise SystemExit(f'{args.baseline} does not exist; create it with --rebaseline')

    lines, comparison = build_report(existing, current)
    print('\n'.join(lines))

    problems = check_regressions(comparison)

    if problems:
        print('REGRESSIONS')
        for problem in problems:
            print(f'  - {problem}')
    else:
        print('no regression by the --check rules')

    if args.check and problems:
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
