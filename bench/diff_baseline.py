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
    python bench/diff_baseline.py --record "note" # log the run without moving it

The snapshot stores RAW COMPONENTS per case - umf_error, count, cost_abs,
assembly_score, min_portion, junk_count, rounding_drift, conditioning.cond - and
never a rolled up score. The roll-up happens here, at diff time, so changing the
score formula does not invalidate a baseline that took ten minutes to produce.

Four rules keep the comparison honest, and each of them exists because the naive
version lies:

  * the two scenarios are profiled SEPARATELY, one block per (scenario, engine)
    pair. Mixing them would average a population built from a recipe's own
    materials, where nothing has a price, with one built from our 19 material
    stock, where almost everything does - the cost aggregates of the mixture
    would describe neither;
  * percentiles are computed over the INTERSECTION of the solved sets of the two
    runs. A case flipping between solved and failed would otherwise drop out of
    one distribution and shift every aggregate silently. Changes of the solved
    set are reported as their own line, with the ids;
  * assembly_score is None wherever the prices do not cover the recipe - always
    in scenario A, and in scenario B wherever the answer uses the one inventory
    material nobody sells - so its percentiles use the same intersection rule on
    "is it defined", not just on "is it solved";
  * the input data files are hashed into the snapshot. An updated price list
    moves every cost metric without a line of solver code being touched, and
    that is not a regression. When the hashes differ the report says so loudly
    instead of quietly comparing.

The distribution profile is the whole distribution - min / p10 / median / mean /
p90 / p99 / max, before and after - because a mean hides the tail: a change can
improve the median while wrecking a handful of cases, and only p90 / p99 / max
show it. --check gates the tail alongside the centre, on the p90 as well as on
the median.

The snapshot is a single point, and moving it is a deliberate act, so it can
never answer "was the solver getting better or worse over the last month". That
question belongs to bench/history.jsonl, the append only log described in
bench/history.py: --rebaseline always appends a line to it, --record appends one
without moving the baseline, and a plain diff run appends nothing.

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

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench import corpus as bench_corpus
from bench import history as bench_history


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
# single case of scenario A - so on that half of the corpus the gate would have
# nothing to hold on to. 10.10 says what to do about that in as many words:
# "with assembly_score = None the baseline compares count, junk, min_portion and
# conditioning". Those four are therefore the fallback, and they are gated
# individually rather than being mashed into an invented composite - a gate
# nobody can name the subject of is not a gate.
#
# Scenario B is where the primary score finally has values: its answers are
# built from our own inventory, 18 of whose 19 materials are priced. The choice
# is made per (scenario, engine) block from the data rather than from the
# scenario name, and the report prints which one it made and why.
PRIMARY_TRACKED_SCORE = 'assembly_score'
FALLBACK_TRACKED_SCORES = ('count', 'junk_count', 'min_portion', 'cond')

# The aggregates that gate. The centre and the tail together: a change that
# improves the median while wrecking a handful of cases must not pass, and a
# change that only moves the tail must not pass either.
GATED_AGGREGATES = ('median', 'p90')

# A metric defined on fewer cases than this is reported but not gated: a
# percentile over five numbers is noise, not a distribution.
MIN_TRACKED_CASES = 10

# Raw components and the direction that counts as better. The list lives in
# bench/corpus.py, because bench/history.py rolls the same numbers up for one
# run and the two must not drift apart; it is re-exported here under its old
# name so that nothing importing diff_baseline.METRICS has to care.
METRICS: Tuple[Tuple[str, str], ...] = bench_corpus.METRICS

# Components stored per case. Kept explicit so that a field added to run_case()
# does not silently enter the snapshot and change its meaning.
#
# chemistry_ok and quality_ok are the two-level verdict of 7.1 and were added
# when bench/history.jsonl started recording the pass shares. quality_ok in
# particular cannot be recomputed from the components: it compares the solution
# against the ORIGINAL recipe, which the snapshot does not store. The format
# version is deliberately NOT bumped for them - they are additive, every reader
# takes fields by name, and bumping it would make the committed baseline
# unreadable and force a rebaseline, which is exactly the deliberate act this
# file exists to keep rare. Snapshots written before them simply have no
# quality verdict, and bench/corpus.engine_profile() reports that as None
# rather than as zero.
# The scenario B fields go in for the same reason and on the same terms. The
# feasibility verdict cannot be recomputed from the components either - it is
# the answer of an LP over our inventory, not a property of the recipe - and
# without it a stored run cannot say which of its cases were honestly
# unreachable, which is the one thing scenario B measures. unreachable_oxides is
# the list itself rather than a count because that list IS the bug report: "no
# lithium source" and "the flux ratio is 20% out on MgO" are different findings
# and a number cannot tell them apart.
CASE_FIELDS = (
    'glazy_id', 'scenario', 'engine', 'status', 'bucket', 'size',
    'umf_error', 'count', 'cost_abs', 'assembly_score', 'min_portion',
    'junk_count', 'rounding_drift', 'cond', 'chemistry_ok', 'quality_ok',
    'feasible', 'max_relative_deviation', 'unreachable_oxides',
)

# Below this the two values are the same number and the case is "unchanged".
EQUAL_EPSILON = 1e-9


# --------------------------------------------------------------------------
# running
# --------------------------------------------------------------------------

def run_corpus(seed: int, sample_size: int, classic_size: int,
               scenario_b_size: Optional[int] = None,
               verbose: bool = True) -> Dict[str, Any]:
    """
    Run both scenarios over the sample and return a snapshot-shaped dictionary

    Three passes, exactly the ones the corpus test makes:

        A / iterative   the whole sample
        A / classic     its subsample, because the classic engine is slow
        B / iterative   a second subsample of the same sample, against our own
                        19 material inventory

    Scenario B runs the iterative engine only. It is the engine POST /api/solve
    defaults to, it is the one with the backward elimination pass, and the
    spec asks for one B run rather than an engine comparison; adding a classic
    B pass would double the block count of every report to say something nobody
    asked.
    """
    scenario_b_size = (bench_corpus.DEFAULT_SCENARIO_B_SUBSAMPLE
                       if scenario_b_size is None else scenario_b_size)

    corpus, timings = bench_corpus.load_corpus()
    cases, rejected = bench_corpus.build_cases(corpus)
    sample = bench_corpus.stratified_sample(cases, size=sample_size, seed=seed)
    classic_sample = bench_corpus.subsample(sample, classic_size, seed=seed)
    # Drawn out of the SAME sample with the SAME seed, so every scenario B case
    # is a scenario A case with the same glazy_id and the two can be read side
    # by side, case by case.
    scenario_b_sample = bench_corpus.subsample(sample, scenario_b_size, seed=seed)

    if verbose:
        print(f"corpus {timings['dump']} loaded from {timings['source']} in {timings['seconds']:.2f}s; "
              f"{len(cases)} usable cases, sample {len(sample)}, classic subsample "
              f"{len(classic_sample)}, scenario B subsample {len(scenario_b_sample)}")

    rows: List[Dict[str, Any]] = []
    started = time.perf_counter()

    passes = (
        (bench_corpus.SCENARIO_A, bench_corpus.ENGINE_ITERATIVE, sample),
        (bench_corpus.SCENARIO_A, bench_corpus.ENGINE_CLASSIC, classic_sample),
        (bench_corpus.SCENARIO_B, bench_corpus.ENGINE_ITERATIVE, scenario_b_sample),
    )

    for scenario, engine, subset in passes:
        pass_started = time.perf_counter()
        for case in subset:
            result = bench_corpus.run_case(case, engine, scenario)
            rows.append({field: result.get(field) for field in CASE_FIELDS})
        if verbose:
            print(f"  {bench_corpus.group_key(scenario, engine)}: {len(subset)} cases "
                  f"in {time.perf_counter() - pass_started:.1f}s")

    return {
        'format_version': BASELINE_FORMAT_VERSION,
        'generated': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
        'run': {
            'seed': seed,
            'sample_size': sample_size,
            'classic_subsample': classic_size,
            'scenario_b_subsample': scenario_b_size,
            'scenario': '+'.join(bench_corpus.SCENARIOS),
            # Keyed by series, not by engine: two passes share the iterative
            # engine and draw different subsets, and one list per engine could
            # only hold one of them.
            'sampled_ids': {
                bench_corpus.group_key(scenario, engine): [case['glazy_id'] for case in subset]
                for scenario, engine, subset in passes
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
                'feasibility_tol': bench_corpus.FEASIBILITY_TOL,
            },
            'seconds': time.perf_counter() - started,
        },
        'cases': rows,
    }


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------

# min / p10 / median / mean / p90 / p99 / max of a sample. Shared with
# bench/history.py through bench/corpus.py: one implementation, so a history
# line and a diff line of the same run cannot disagree about what a median is.
_profile = bench_corpus.profile


def _index(cases: Sequence[Dict[str, Any]], engine: str,
           scenario: Optional[str] = None) -> Dict[int, Dict[str, Any]]:
    """
    Cases of one (scenario, engine) pair keyed by their Glazy id

    The scenario has to be part of the filter, not only of the report heading:
    the same glazy_id appears once per scenario, so an engine-only index would
    silently keep whichever row came last and compare a scenario B answer
    against a scenario A baseline.
    """
    return {row['glazy_id']: row for row in cases
            if row.get('engine') == engine
            and (scenario is None
                 or (row.get('scenario') or bench_corpus.SCENARIO_A) == scenario)}


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


def _reachability(index: Dict[int, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    The scenario B side of one run: the verdict counts and the two misclass sets

    None when no row carries a feasibility verdict, which is what scenario A
    looks like - an absent verdict is not "everything was reachable".
    """
    judged = [row for row in index.values() if row.get('feasible') is not None]
    if not judged:
        return None

    rows = list(index.values())
    accounted = [row for row in rows if bench_corpus.accounted_ok(row)]

    return {
        'cases': len(rows),
        'reachable': sum(1 for row in judged if row.get('feasible') is True),
        'unreachable': sum(1 for row in judged if row.get('feasible') is False),
        'undecided': len(rows) - len(judged),
        'accounted': len(accounted),
        'accounted_share': len(accounted) / len(rows) if rows else 0.0,
        'reachable_unsolved': sorted(
            row['glazy_id'] for row in rows
            if bench_corpus.misclassification(row) == 'reachable_unsolved'),
        'unreachable_solved': sorted(
            row['glazy_id'] for row in rows
            if bench_corpus.misclassification(row) == 'unreachable_solved'),
    }


def compare_engine(baseline_index: Dict[int, Dict[str, Any]],
                   current_index: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    """Everything the report needs about one (scenario, engine) pair"""
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
        'baseline_reachability': _reachability(baseline_index),
        'current_reachability': _reachability(current_index),
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

    header = f"      {'':<8}" + ''.join(f'{key:>13}' for key in bench_corpus.PROFILE_KEYS)
    lines.append(header)
    for label, profile in (('before', before), ('after', after)):
        row = f'      {label:<8}' + ''.join(
            f'{_format_number(profile[key]):>13}' for key in bench_corpus.PROFILE_KEYS)
        lines.append(row)

    deltas = f"      {'delta':<8}" + ''.join(
        f'{_format_number(after[key] - before[key]):>13}'
        for key in bench_corpus.PROFILE_KEYS)
    lines.append(deltas)

    return lines


def _reachability_lines(result: Dict[str, Any]) -> List[str]:
    """
    The scenario B block: the verdict split, the accounted share, the disputes

    Silent for a series that carries no feasibility verdict. The two
    misclassification sets are printed as ID LISTS and diffed against the
    baseline's, because they are the output of the scenario, not a statistic
    about it: a case joining or leaving either set is a bug appearing or being
    fixed, and a count would hide which.
    """
    before = result.get('baseline_reachability')
    after = result.get('current_reachability')
    if not before and not after:
        return []

    lines: List[str] = []
    empty: Dict[str, Any] = {'reachable': 0, 'unreachable': 0, 'undecided': 0,
                             'accounted': 0, 'accounted_share': 0.0, 'cases': 0,
                             'reachable_unsolved': [], 'unreachable_solved': []}
    before = before or empty
    after = after or empty

    lines.append(f"  feasibility: reachable {before['reachable']} -> {after['reachable']}, "
                 f"unreachable {before['unreachable']} -> {after['unreachable']}, "
                 f"undecided {before['undecided']} -> {after['undecided']}")
    lines.append(f"  accounted (solved OR honestly unreachable): "
                 f"{before['accounted']}/{before['cases']} ({before['accounted_share']:.2%}) -> "
                 f"{after['accounted']}/{after['cases']} ({after['accounted_share']:.2%})")

    for key, label in (('reachable_unsolved', 'LP said reachable, solver missed'),
                       ('unreachable_solved', 'LP said unreachable, solver hit it')):
        was, now = set(before[key]), set(after[key])
        lines.append(f"  {label}: {len(was)} -> {len(now)}  {sorted(now)}")
        if was - now:
            lines.append(f"      no longer disputed: {sorted(was - now)}")
        if now - was:
            lines.append(f"      newly disputed:     {sorted(now - was)}")

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

    for series, ids in (baseline['run'].get('sampled_ids') or {}).items():
        current_ids = (current['run'].get('sampled_ids') or {}).get(series)
        if current_ids is not None and list(ids) != list(current_ids):
            findings.append(f"SAMPLE CHANGED for {series}: the same seed drew a different set of ids, "
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

    # --- per (scenario, engine) -------------------------------------------
    # One block per series, never a merged one: scenario A's population is
    # unpriced and reproduces a recipe from its own materials, scenario B's is
    # priced and works from a 19 material stock. An average over both would
    # describe neither of them.
    groups = sorted(set(bench_corpus.snapshot_groups(baseline['cases']))
                    | set(bench_corpus.snapshot_groups(current['cases'])))

    for scenario, engine in groups:
        series = bench_corpus.group_key(scenario, engine)
        result = compare_engine(_index(baseline['cases'], engine, scenario),
                                _index(current['cases'], engine, scenario))
        comparison['engines'][series] = result

        lines.append(f'--- engine: {series} ' + '-' * max(0, 78 - 13 - len(series)))
        lines.append(f"  scenario {scenario}, engine {engine}")
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
        lines.extend(_reachability_lines(result))
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

    Four gates, per series:
      * the median OR the p90 of a tracked score worsens by more than
        REGRESSION_TOLERANCE - the tail is gated alongside the centre, because a
        tweak that improves the median while wrecking ten cases is not an
        improvement;
      * the share of solved cases drops;
      * the maximum chemistry error grows;
      * scenario B only: the accounted share drops. "Solved OR honestly
        unreachable" is that scenario's own criterion, and it can fall while the
        solved share does not - a case can keep returning a recipe and stop
        passing the chemistry gate. A rising unreachable count does not trip
        this gate by itself, and must not: declining a target our stock cannot
        reach is a correct answer, not a lost case.

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

        before_reach = result.get('baseline_reachability')
        after_reach = result.get('current_reachability')
        if before_reach and after_reach:
            if after_reach['accounted_share'] < before_reach['accounted_share'] - EQUAL_EPSILON:
                problems.append(
                    f"{engine}: the accounted share dropped "
                    f"{before_reach['accounted_share']:.2%} -> {after_reach['accounted_share']:.2%} "
                    f"(newly disputed: "
                    f"{sorted(set(after_reach['reachable_unsolved']) - set(before_reach['reachable_unsolved']))})")

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


def record_run(log_path: str, snapshot: Dict[str, Any], recorded_at: str,
               kind: str, note: str) -> None:
    """
    Append one line to bench/history.jsonl and say so

    The timestamp is an argument all the way down: main() reads the clock once,
    at the CLI boundary, and nothing below it ever asks what time it is.
    """
    record = bench_history.build_record(snapshot, recorded_at=recorded_at, kind=kind, note=note)
    sealed = bench_history.append_record(log_path, record)
    if sealed:
        print(f'{log_path}: the previous append had been left unterminated; '
              f'sealed that line before appending')
    print(f'run recorded in {log_path} as a "{kind}" line'
          f"{' with the note ' + repr(note) if note else ' with no note'}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description='Diff the current solver against the committed quality baseline')
    parser.add_argument('--baseline', default=bench_corpus.BASELINE_PATH,
                        help='path of the baseline snapshot (default: bench/quality_baseline.json)')
    parser.add_argument('--check', action='store_true',
                        help='exit 1 when the tracked score, the solved share, the maximum '
                             'chemistry error or the scenario B accounted share regressed')
    parser.add_argument('--rebaseline', action='store_true',
                        help='overwrite the snapshot with the current run; the only way to move it')
    parser.add_argument('--record', nargs='?', const='', default=None, metavar='NOTE',
                        help='append this run to bench/history.jsonl with an optional note '
                             'saying what it was testing. --rebaseline always appends a line; '
                             'this flag appends one without moving the baseline, and supplies '
                             'the note in both cases. A plain diff run appends nothing')
    parser.add_argument('--history-log', default=bench_corpus.HISTORY_PATH,
                        help='path of the run history (default: bench/history.jsonl)')
    parser.add_argument('--seed', type=int, default=None,
                        help='sampling seed; defaults to the one recorded in the baseline')
    parser.add_argument('--sample-size', type=int, default=None,
                        help='sample size; defaults to the one recorded in the baseline')
    parser.add_argument('--classic-subsample', type=int, default=None,
                        help='classic subsample size; defaults to the one recorded in the baseline')
    parser.add_argument('--scenario-b-subsample', type=int, default=None,
                        help='scenario B subsample size; defaults to the one recorded in the '
                             'baseline, or to bench/corpus.DEFAULT_SCENARIO_B_SUBSAMPLE for a '
                             'baseline written before scenario B existed')
    parser.add_argument('--save-current', default=None,
                        help='also write the current run to this path, for offline comparison')
    parser.add_argument('--current', default=None,
                        help='compare a previously saved run instead of solving again')
    args = parser.parse_args(argv)

    existing = None
    if os.path.exists(args.baseline):
        existing = load_baseline(args.baseline)

    seed = args.seed
    sample_size = args.sample_size
    classic_size = args.classic_subsample
    scenario_b_size = args.scenario_b_subsample

    # A baseline written before scenario B existed has no size for it recorded,
    # so the default fills in rather than the run refusing to start: the point
    # of a diff against an older baseline is to see what changed, and "the
    # baseline has no scenario B" is one of the things that changed.
    recorded = (existing or {}).get('run', {})
    seed = seed if seed is not None else recorded.get('seed', bench_corpus.DEFAULT_SEED)
    sample_size = (sample_size if sample_size is not None
                   else recorded.get('sample_size', bench_corpus.DEFAULT_SAMPLE_SIZE))
    classic_size = (classic_size if classic_size is not None
                    else recorded.get('classic_subsample', bench_corpus.DEFAULT_CLASSIC_SUBSAMPLE))
    scenario_b_size = (scenario_b_size if scenario_b_size is not None
                       else recorded.get('scenario_b_subsample',
                                         bench_corpus.DEFAULT_SCENARIO_B_SUBSAMPLE))

    if args.current:
        with open(args.current, 'r', encoding='utf-8') as f:
            current = json.load(f)
    else:
        try:
            current = run_corpus(seed, sample_size, classic_size, scenario_b_size)
        except bench_corpus.CorpusUnavailable as exc:
            print(f'Glazy corpus unavailable: {exc}')
            return 2

    if args.save_current:
        write_baseline(args.save_current, current)
        print(f'current run written to {args.save_current}')

    # The clock is read exactly here, at the CLI boundary, and handed down as a
    # value: nothing under bench/ asks the time by itself, so a test can pin the
    # whole record by pinning this one string.
    recorded_at = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')

    if args.rebaseline:
        if existing is not None:
            lines, _comparison = build_report(existing, current)
            print('\n'.join(lines))
            print('--- the snapshot above is being REPLACED ---')
        write_baseline(args.baseline, current)
        print(f'baseline written to {args.baseline}')
        print('Commit it separately, with the reason the level moved in the commit message.')
        # A new reference point is by definition a run worth keeping, so the
        # history line is not optional here - only its note is.
        record_run(args.history_log, current, recorded_at,
                   bench_history.KIND_REBASELINE, args.record or '')
        return 0

    if args.record is not None:
        record_run(args.history_log, current, recorded_at,
                   bench_history.KIND_RECORD, args.record)

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
