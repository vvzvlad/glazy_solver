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
Append only history of the corpus runs worth remembering (bench/history.jsonl)

bench/quality_baseline.json holds ONE snapshot and --rebaseline overwrites it.
That is right for a baseline - it is the reference point, and moving it should
be a deliberate act with its own commit. But it leaves the project without a
SERIES: "did the solver get better or worse over the last month" can then only
be answered by walking the git history of that file and re-deriving the
aggregates from every version, and everything measured along the way - engine
comparisons, threshold experiments, prototypes - is written nowhere at all.

This module is that series. One JSON object per line, appended, never rewritten,
never sorted and never pruned by the code. A line holds enough to be read years
later without this file: when the run happened, on which commit and whether the
tree was dirty, what was run, the hashes of the input data, the whole
distribution profile of every metric and the three pass shares, plus a free text
note saying what the run was testing.

    python bench/history.py                     # print the series
    python bench/history.py --engine iterative  # one series only ("B/iterative"
                                                # is the scenario B one)
    python bench/history.py --last 5            # the tail, deltas still vs the
                                                # previous line of the log

Lines are written by bench/diff_baseline.py, on --rebaseline (always: a new
reference point is by definition worth keeping) and on --record [note] (a run to
be remembered without moving the baseline). A plain diff run writes nothing -
the log is for deliberate measurements, not for every invocation.

Why a separate script rather than a --history mode of diff_baseline.py. Reading
the log must not depend on being able to run the corpus: diff_baseline.py is
built around solving 450 recipes and needs the 8 MB dump, pyyaml and a warm
cache, while answering "what happened over the last month" has to work on a
laptop on a train. Importing this module costs nothing beyond the standard
library and numpy, and its CLI never touches the dump.

WHAT A LINE HOLDS

    schema          version of this record shape, bumped on any change of meaning
    recorded_at     when the LINE was appended (ISO 8601, UTC)
    ran_at          when the RUN happened - the snapshot's own "generated"
    kind            "rebaseline" | "record"
    note            free text: what this run was testing
    seeded_from     null for a run recorded by the tool itself; otherwise where
                    the numbers were reconstructed from, for the lines that were
                    back-filled from older snapshots
    git_commit      commit of the tree the run happened on
    git_dirty       whether that tree had uncommitted changes
    thresholds      the constants the shares depend on, so that moving one later
                    does not silently reinterpret the old lines
    run             seed, sample size, scenario, dump and solver configuration
    sampled_ids     per series: how many ids were drawn and a DIGEST of the list
    data_hashes     bench/corpus.py's hashes of materials / prices / oxide
                    classification, so "the data changed" is distinguishable
                    from "the solver changed"
    engines         per series: case count, the pass shares, the reachability
                    verdict counts of scenario B and the min / p10 / median /
                    mean / p90 / p99 / max profile of every metric, computed by
                    bench/corpus.engine_profile()

ONE SERIES PER SCENARIO AND ENGINE

A run measures more than one thing: scenario A over two engines and scenario B
over the iterative one. They must never be averaged together - scenario A's
population is unpriced and reproduces a recipe from its own materials, scenario
B's is priced and works from a 19 material stock - so the log keys a separate
series per (scenario, engine) pair through bench/corpus.group_key().

Scenario A keeps the bare engine name as its key. That is not tidiness, it is
the only way the series survives: the lines written before scenario B existed
all measured scenario A under the keys "iterative" and "classic", and an append
only log cannot be re-keyed after the fact. Scenario B opens "B/iterative"
alongside them, starting its own series from zero rather than pretending to
continue one.

The schema version is deliberately NOT bumped for this. No field changed
meaning: "iterative" still means what it meant in every earlier line, and the
new keys are additive, exactly like chemistry_ok and quality_ok before them.
What did change is that run.scenario left COMPARABLE_RUN_KEYS - see the comment
there for why keeping it would have raised a false alarm on every scenario A
series the moment a run also measured B.

The sampled ids are stored as a DIGEST rather than in full, and this is the one
place where the record is deliberately not self-contained. The full list is 300
integers per engine; written out it is 2 kB of a 3 kB line and makes the log
unreadable to the human being it exists for, while answering exactly one
question - "is this the same sample as the previous run" - that a hash answers
just as well. The full lists stay in the baseline snapshot, which is where a
reader who needs the actual ids should look. The digest is reproducible without
this code: it is the sha256 of the decimal ids joined by commas, in the order
the sample was drawn, encoded as UTF-8.

APPEND SAFETY

A record is serialized to a single line - json.dumps escapes newlines inside
strings, so no note can break the one-object-per-line rule - and written with
one write() to a file opened in append mode, followed by flush and fsync.

That is not an atomicity proof, so the two ends are built for a torn write
instead of against it:

  * the writer checks the last byte of the file first. Every complete record
    ends with a newline, so a file that does not end with one was left torn by
    an interrupted append; the writer seals it with a newline before adding its
    own record. The damage then stays confined to the one broken line instead of
    swallowing the new record too;
  * the reader splits on newlines, parses each line on its own and skips the
    ones that do not parse, reporting them by line number. Bytes after the last
    newline are reported as a truncated final line. A torn write therefore costs
    exactly the record that was being written, never the log.
"""

import argparse
import hashlib
import json
import os
import sys
import textwrap
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench import corpus as bench_corpus


# Bump when the meaning of a field changes. A reader must refuse to interpret a
# line it does not understand rather than guess - but it must still print the
# line's own metadata, because "there is a run here I cannot read" is itself
# information the series must not lose.
SCHEMA_VERSION = 1

KIND_REBASELINE = 'rebaseline'
KIND_RECORD = 'record'

# What the summary table shows per run, as (metric, aggregate). The aggregate is
# named in the header because it is not the same one everywhere: junk_count has
# a median of 0 on this corpus - most recipes carry no junk at all - so its
# median would hide the very changes the metric exists to catch. Its mean does
# not: the load-bearing-component rule moved it 0.53 -> 0.20 while the median
# sat at 0 on both sides. Everything else is shown at the median.
HEADLINE = (
    ('umf_error', 'median'),
    ('count', 'median'),
    ('junk_count', 'mean'),
    ('min_portion', 'median'),
    ('cond', 'median'),
)

# Short labels for the table header, kept apart from the metric names so the
# names in the record stay the long explicit ones.
HEADLINE_LABELS = {
    'umf_error': 'umf',
    'count': 'count',
    'junk_count': 'junk',
    'min_portion': 'minp',
    'cond': 'cond',
}
AGGREGATE_LABELS = {'median': 'med', 'mean': 'mean'}

# Width of the fixed columns of the table, before the metric columns. The whole
# row is assembled from this one list so that the header and the two kinds of
# data row cannot drift out of alignment.
FIXED_COLUMNS = (
    ('#', 3, '<'),
    ('ran (UTC)', 16, '<'),
    ('commit', 8, '<'),
    ('data', 6, '<'),
    ('solved', 7, '>'),
    ('chem', 7, '>'),
    ('both', 7, '>'),
    # Scenario B's own gate: "solved OR honestly unreachable". A dash in
    # scenario A, which never asks the question.
    ('acct', 7, '>'),
)
METRIC_WIDTH = 10

# Notes are free text and can be long; the table above them is not.
NOTE_WIDTH = 100

# Run parameters that must match before two lines can be compared at all. The
# data hashes are checked separately and reported in their own words, because a
# changed price list is the one incomparability that looks like a solver change.
#
# "scenario" is NOT here, and used to be. It is now part of the series key, so
# two lines of one series are the same scenario by construction and the check
# would be dead weight - worse than that, it would fire on every scenario A
# series the first time a run also measured scenario B, calling two runs of the
# same 300 recipes "different corpora" because a second scenario was added
# beside them.
COMPARABLE_RUN_KEYS = ('seed', 'sample_size', 'classic_subsample', 'dump')


# --------------------------------------------------------------------------
# building a record
# --------------------------------------------------------------------------

def sampled_ids_digest(ids: Sequence[Any]) -> str:
    """
    sha256 of a list of sampled ids, in the order they were drawn

    The canonical form is the decimal ids joined by commas and encoded as UTF-8,
    documented in the module docstring so that the digest can be re-derived in
    ten years by anyone holding the list and no code of ours.
    """
    payload = ','.join(str(int(value)) for value in ids)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def build_record(snapshot: Dict[str, Any], recorded_at: str, kind: str = KIND_RECORD,
                 note: str = '', seeded_from: Optional[str] = None) -> Dict[str, Any]:
    """
    Turn a corpus run into the line that will be appended to the log

    Args:
        snapshot: a run in the shape diff_baseline.run_corpus() produces, which
            is also the shape of bench/quality_baseline.json - so an old
            baseline can be back-filled into the log by loading it and passing
            it here
        recorded_at: ISO 8601 timestamp of the append. Passed IN rather than
            read from the clock here on purpose: bench/corpus.py keeps the
            workflow-run path free of hidden non-determinism, so the clock is
            read once at the CLI boundary and tests can pin it
        kind: KIND_REBASELINE when the baseline moved with this run, KIND_RECORD
            when the run was only measured
        note: what this run was testing, in the words of whoever ran it
        seeded_from: where the numbers came from when the line was reconstructed
            from an older snapshot rather than written by the run itself

    Returns:
        A JSON serializable dict. Nothing is read from disk and no clock is
        consulted, so the same snapshot and timestamp always give the same line.
    """
    run = snapshot.get('run') or {}
    cases = snapshot.get('cases') or []

    sampled = {}
    for series, ids in sorted((run.get('sampled_ids') or {}).items()):
        sampled[series] = {'count': len(ids), 'sha256': sampled_ids_digest(ids)}

    engines = {bench_corpus.group_key(scenario, engine):
               bench_corpus.engine_profile(cases, engine, scenario)
               for scenario, engine in bench_corpus.snapshot_groups(cases)}

    return {
        'schema': SCHEMA_VERSION,
        'recorded_at': recorded_at,
        'ran_at': snapshot.get('generated'),
        'kind': kind,
        'note': note or '',
        'seeded_from': seeded_from,
        'git_commit': run.get('git_commit'),
        'git_dirty': run.get('git_dirty'),
        'thresholds': {
            'max_umf_error': bench_corpus.MAX_UMF_ERROR,
        },
        'run': {
            'seed': run.get('seed'),
            'sample_size': run.get('sample_size'),
            'classic_subsample': run.get('classic_subsample'),
            'scenario': run.get('scenario'),
            'dump': run.get('dump'),
            'dump_size': run.get('dump_size'),
            'solver_config': run.get('solver_config'),
            'seconds': run.get('seconds'),
        },
        'sampled_ids': sampled,
        'data_hashes': run.get('data_hashes') or {},
        'engines': engines,
    }


# --------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------

def _seal_torn_tail(path: str) -> bool:
    """
    Terminate a file whose last append was interrupted, and say whether it was

    Every complete record ends with a newline, so a non-empty file that does not
    is one an earlier append tore in half. Appending the missing newline keeps
    the damage inside that one line: without it the next record would be glued
    onto the broken tail and both would be lost.
    """
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False

    with open(path, 'rb') as f:
        f.seek(-1, os.SEEK_END)
        last = f.read(1)

    if last == b'\n':
        return False

    with open(path, 'ab') as f:
        f.write(b'\n')
    return True


def append_record(path: str, record: Dict[str, Any]) -> bool:
    """
    Append one record to the log and return whether a torn tail had to be sealed

    The record is serialized first and written in one call: a line that cannot
    be serialized must not leave half of itself in the log.
    """
    line = json.dumps(record, ensure_ascii=False, sort_keys=False)
    if '\n' in line:
        raise ValueError('a record must serialize to a single line')

    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)

    sealed = _seal_torn_tail(path)

    with open(path, 'a', encoding='utf-8') as f:
        f.write(line + '\n')
        f.flush()
        os.fsync(f.fileno())

    return sealed


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------

def read_log(path: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Read the log, tolerating a torn last line

    The file is read as bytes and split on newlines rather than iterated as
    text: an interrupted append can cut a UTF-8 sequence in half, and a decode
    error on one line must not take the other hundred with it.

    Returns:
        (records, problems) - the records in FILE ORDER, never sorted, each with
        a "_line" key holding its 1-based line number, and one message per line
        that could not be read.
    """
    if not os.path.exists(path):
        return [], []

    with open(path, 'rb') as f:
        payload = f.read()

    if not payload:
        return [], []

    chunks = payload.split(b'\n')
    # Bytes after the last newline are an append that never finished. An empty
    # tail is the normal case: a well formed log ends with a newline.
    tail = chunks.pop()

    records: List[Dict[str, Any]] = []
    problems: List[str] = []

    for index, chunk in enumerate(chunks, start=1):
        text = chunk.decode('utf-8', errors='replace').strip()
        if not text:
            continue
        try:
            record = json.loads(text)
        except ValueError as exc:
            # Either a torn line that a later append sealed with a newline, or
            # real corruption. The two are indistinguishable from here, so the
            # message says so instead of picking one.
            problems.append(f'line {index}: not readable as JSON, skipped '
                            f'(a sealed torn write, or corruption): {exc}')
            continue
        if not isinstance(record, dict):
            problems.append(f'line {index}: not a JSON object, skipped')
            continue
        record['_line'] = index
        records.append(record)

    if tail.strip():
        problems.append(f'line {len(chunks) + 1}: truncated final line '
                        f'({len(tail)} bytes with no newline), skipped - '
                        f'an append that did not finish')

    return records, problems


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def _format_number(value: Optional[float], signed: bool = False) -> str:
    """One metric value, in the same shape the regression diff prints"""
    if value is None:
        return '-'
    if abs(value) >= 1000 or (value != 0 and abs(value) < 0.001):
        return f'{value:+.2e}' if signed else f'{value:.2e}'
    return f'{value:+.4f}' if signed else f'{value:.4f}'


def _format_share(value: Optional[float]) -> str:
    return '-' if value is None else f'{value:.2%}'


def _format_share_delta(before: Optional[float], after: Optional[float]) -> str:
    if before is None or after is None:
        return '-'
    return f'{(after - before) * 100:+.2f}pp'


def _short(value: Optional[str], size: int = 7) -> str:
    return (value or '?')[:size]


def data_digest(record: Dict[str, Any]) -> str:
    """
    A short stand-in for the whole set of input data hashes

    One column that says "these two runs saw the same materials, prices and
    oxide classification" is worth more in a table than three columns of
    sha256 nobody reads.
    """
    hashes = record.get('data_hashes') or {}
    payload = json.dumps(hashes, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:6]


def changed_data(previous: Dict[str, Any], current: Dict[str, Any]) -> List[str]:
    """Names of the input data files whose hash differs between two records"""
    before = previous.get('data_hashes') or {}
    after = current.get('data_hashes') or {}
    return [name for name in sorted(set(before) | set(after))
            if before.get(name) != after.get(name)]


def changed_run(previous: Dict[str, Any], current: Dict[str, Any]) -> List[str]:
    """
    Run parameters that differ between two records

    The sample itself is included: the same seed can draw a different set of
    ids once the corpus underneath it moves, and then nothing below the header
    is comparable either.
    """
    before, after = previous.get('run') or {}, current.get('run') or {}
    differences = [key for key in COMPARABLE_RUN_KEYS if before.get(key) != after.get(key)]

    before_ids, after_ids = previous.get('sampled_ids') or {}, current.get('sampled_ids') or {}
    for engine in sorted(set(before_ids) & set(after_ids)):
        if before_ids[engine].get('sha256') != after_ids[engine].get('sha256'):
            differences.append(f'sampled ids of {engine}')

    return differences


def _ran_at(record: Dict[str, Any]) -> str:
    """The run's own timestamp, trimmed to the minute for the table"""
    value = record.get('ran_at') or record.get('recorded_at') or '?'
    return value.replace('T', ' ')[:16]


def _headline_values(entry: Optional[Dict[str, Any]]) -> List[Optional[float]]:
    """The headline aggregate of every metric of one engine, or None"""
    values: List[Optional[float]] = []
    metrics = (entry or {}).get('metrics') or {}
    for name, aggregate in HEADLINE:
        block = metrics.get(name)
        values.append(None if not block else block.get(aggregate))
    return values


def _assemble(fixed: Sequence[str], metrics: Sequence[str]) -> str:
    """One table row from its cells, padded to the one column list there is"""
    cells = [f'{value:{align}{width}}'
             for value, (_label, width, align) in zip(fixed, FIXED_COLUMNS)]
    cells.extend(value.rjust(METRIC_WIDTH) for value in metrics)
    return '  ' + ' '.join(cells)


def _table_header() -> Tuple[str, str]:
    """The header line of the per-engine table and the ruler under it"""
    header = _assemble(
        [label for label, _width, _align in FIXED_COLUMNS],
        [f'{HEADLINE_LABELS[name]}({AGGREGATE_LABELS.get(aggregate, aggregate)})'
         for name, aggregate in HEADLINE],
    )
    return header, '  ' + '-' * (len(header) - 2)


def _accounted_share(entry: Dict[str, Any]) -> Optional[float]:
    """Scenario B's accounted share, or None for a series that never asks"""
    reachability = (entry or {}).get('reachability')
    return None if not reachability else reachability.get('accounted_share')


def _row(index: int, record: Dict[str, Any], entry: Dict[str, Any]) -> str:
    """One run of one series"""
    commit = _short(record.get('git_commit')) + ('*' if record.get('git_dirty') else '')
    return _assemble(
        [
            str(index),
            _ran_at(record),
            commit,
            data_digest(record),
            f"{entry.get('solved', 0)}/{entry.get('cases', 0)}",
            _format_share(entry.get('chemistry_share')),
            _format_share(entry.get('both_levels_share')),
            _format_share(_accounted_share(entry)),
        ],
        [_format_number(value) for value in _headline_values(entry)],
    )


def _delta_row(label: str, before: Dict[str, Any], after: Dict[str, Any]) -> str:
    """The change between two runs of the same series, in the same columns"""
    return _assemble(
        [
            '',
            label,
            '',
            '',
            f"{after.get('solved', 0) - before.get('solved', 0):+d}",
            _format_share_delta(before.get('chemistry_share'), after.get('chemistry_share')),
            _format_share_delta(before.get('both_levels_share'), after.get('both_levels_share')),
            _format_share_delta(_accounted_share(before), _accounted_share(after)),
        ],
        [('-' if old is None or new is None else _format_number(new - old, signed=True))
         for old, new in zip(_headline_values(before), _headline_values(after))],
    )


def _annotation_lines(entry: Dict[str, Any]) -> List[str]:
    """
    What the fixed columns cannot hold: the reachability split and the pricing

    Both are printed only where they mean something. A scenario A series has no
    feasibility verdict and no priced case, and two lines of dashes under every
    row would say nothing at the cost of making the table unreadable.
    """
    lines: List[str] = []

    reachability = entry.get('reachability')
    if reachability:
        lines.append(
            f"      feasibility(tol={reachability.get('tol')}): "
            f"reachable {reachability.get('reachable')}, "
            f"unreachable {reachability.get('unreachable')}, "
            f"undecided {reachability.get('undecided')}; "
            f"misclassified {reachability.get('misclassified')} "
            f"(LP said reachable, solver missed: {reachability.get('reachable_unsolved')}; "
            f"LP said unreachable, solver hit it: {reachability.get('unreachable_solved')})")

    if entry.get('priced'):
        lines.append(
            f"      fully priced {entry.get('priced')}/{entry.get('solved')} solved "
            f"({_format_share(entry.get('priced_share'))}) - cost_abs and assembly_score "
            f"are real numbers on those and None on the rest")

    return lines


def _note_lines(record: Dict[str, Any]) -> List[str]:
    """The note and the provenance of one run, wrapped under its row"""
    note = (record.get('note') or '').strip() or 'no note'
    head = f"      [{record.get('kind') or '?'}] "
    lines = textwrap.wrap(note, width=NOTE_WIDTH, initial_indent=head,
                          subsequent_indent=' ' * len(head)) or [head]

    seeded = record.get('seeded_from')
    if seeded:
        lines.extend(textwrap.wrap(f'reconstructed from {seeded}', width=NOTE_WIDTH,
                                   initial_indent=' ' * len(head),
                                   subsequent_indent=' ' * len(head)))
    return lines


def render(records: Sequence[Dict[str, Any]], problems: Sequence[str],
           path: str, engines: Optional[Sequence[str]] = None,
           last: Optional[int] = None) -> List[str]:
    """
    The whole report: one block per series, one row per run, deltas underneath

    Deltas are always taken against the PREVIOUS run of the same series in the
    log, even when --last hides that run: a regression that crept in over three
    runs has to read as a trend down the column, and a delta measured against
    whatever happens to be on screen would not be one.

    A "series" is one (scenario, engine) pair - "iterative" for scenario A,
    "B/iterative" for scenario B - and the two are never merged: their
    populations differ in what is on the shelf and in whether it has a price.
    """
    lines: List[str] = []
    lines.append('GLAZY CORPUS RUN HISTORY')
    lines.append('=' * 78)
    lines.append(f'log {path}')
    lines.append(f'{len(records)} runs, in file order - this log is append only and is never sorted')

    if problems:
        lines.append('')
        for problem in problems:
            lines.append(f'!! {problem}')

    if not records:
        lines.append('')
        lines.append('no readable runs yet; bench/diff_baseline.py --record writes the first one')
        return lines

    unknown = sorted({record.get('schema') for record in records} - {SCHEMA_VERSION})
    if unknown:
        lines.append(f'!! records of schema {unknown} are present; this reader knows '
                     f'schema {SCHEMA_VERSION} and may be reading them wrong')

    present = sorted({engine for record in records for engine in (record.get('engines') or {})})
    wanted = [engine for engine in present if engines is None or engine in engines]
    if engines is not None:
        missing = [engine for engine in engines if engine not in present]
        for engine in missing:
            lines.append(f'!! no run in the log carries a series called {engine!r}; '
                         f'the log holds {present}')

    for engine in wanted:
        series = [record for record in records if engine in (record.get('engines') or {})]
        if not series:
            continue

        lines.append('')
        lines.append(f'--- engine: {engine} ' + '-' * max(0, 78 - 13 - len(engine)))
        header, ruler = _table_header()
        lines.append(header)
        lines.append(ruler)

        shown = series if not last else series[-last:]
        first_shown = len(series) - len(shown)

        for offset, record in enumerate(shown):
            index = first_shown + offset + 1
            entry = record['engines'][engine]

            if offset == 0 and first_shown:
                lines.append(f'  ... {first_shown} earlier run(s) hidden by --last; '
                             f'the first delta below is still measured against run '
                             f'#{first_shown}')

            lines.append(_row(index, record, entry))
            lines.extend(_annotation_lines(entry))

            if index > 1:
                previous = series[index - 2]
                broken = changed_data(previous, record)
                if broken:
                    lines.append(f'      !! INPUT DATA CHANGED vs #{index - 1} ({", ".join(broken)}) '
                                 f'- the delta below is not a solver measurement')
                moved = changed_run(previous, record)
                if moved:
                    lines.append(f'      !! RUN CHANGED vs #{index - 1} ({", ".join(moved)}) '
                                 f'- the two runs are different corpora')
                lines.append(_delta_row(f'delta vs #{index - 1}',
                                        previous['engines'][engine], entry))

            lines.extend(_note_lines(record))

        # Three runs are enough for a drift nobody noticed step by step, so the
        # sum of the steps is printed as well as the steps.
        if len(series) >= 3:
            first, current = series[0], series[-1]
            lines.append('')
            lines.append(_delta_row(f'total #1 -> #{len(series)}',
                                    first['engines'][engine], current['engines'][engine]))
            if changed_data(first, current):
                lines.append(f'      !! the input data of run #1 and run #{len(series)} differ; '
                             f'the total above is not a solver measurement')

    return lines


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description='Print the append only history of the corpus runs')
    parser.add_argument('--log', default=bench_corpus.HISTORY_PATH,
                        help='path of the log (default: bench/history.jsonl)')
    parser.add_argument('--engine', action='append', default=None,
                        help='show only this series; repeatable. A series is a '
                             '(scenario, engine) pair: "iterative" and "classic" are '
                             'scenario A, "B/iterative" is scenario B')
    parser.add_argument('--last', type=int, default=None,
                        help='show only the last N runs of each series; deltas are '
                             'still measured against the previous run of the log')
    args = parser.parse_args(argv)

    records, problems = read_log(args.log)
    print('\n'.join(render(records, problems, args.log,
                           engines=args.engine, last=args.last)))

    # A log nobody can read is worth saying out loud, but it is not a failure of
    # this script: the readable runs were printed and are still true.
    return 0


if __name__ == '__main__':
    sys.exit(main())
