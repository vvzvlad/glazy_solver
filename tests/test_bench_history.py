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
The append only run history of the corpus benchmark (bench/history.py)

What is pinned here is what the log promises: a record round-trips through the
file unchanged, a torn last line costs exactly that line and not the log, a run
that saw different input data is marked as incomparable, and a plain diff run -
the common case, run dozens of times while a solver is being tuned - writes
nothing at all.

Nothing here needs the Glazy dump: the snapshots are synthetic and small, which
is the point of the reader being a separate script in the first place.
"""

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest

# Fix imports by adding parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bench import corpus as bench_corpus
from bench import diff_baseline
from bench import history


# A timestamp is never read from the clock inside the record path, so every test
# pins it and compares the whole record byte for byte.
PINNED = '2026-08-09T12:00:00+00:00'

DATA_HASHES = {
    'database/materials.json': 'a' * 64,
    'database/prices.json': 'b' * 64,
    'database/oxide_classification.json': 'c' * 64,
}


def case(glazy_id, engine='iterative', status='solved', umf_error=0.01,
         count=5, min_portion=3.0, junk_count=0, cond=10.0, quality_ok=True):
    """One row in the shape diff_baseline.CASE_FIELDS stores"""
    return {
        'glazy_id': glazy_id,
        'scenario': 'A',
        'engine': engine,
        'status': status,
        'bucket': '5-7',
        'size': count,
        'umf_error': umf_error,
        'count': count,
        'cost_abs': None,
        'assembly_score': None,
        'min_portion': min_portion,
        'junk_count': junk_count,
        'rounding_drift': 0.01,
        'cond': cond,
        'chemistry_ok': umf_error is not None and umf_error <= bench_corpus.MAX_UMF_ERROR,
        'quality_ok': quality_ok,
    }


def snapshot(cases=None, generated='2026-08-09T10:00:00+00:00', commit='0' * 40,
             dirty=False, data_hashes=None, seed=20260531, sample_size=3):
    """A minimal run in the shape diff_baseline.run_corpus() produces"""
    cases = cases if cases is not None else [case(1), case(2), case(3)]
    ids = [row['glazy_id'] for row in cases if row['engine'] == 'iterative']
    return {
        'format_version': diff_baseline.BASELINE_FORMAT_VERSION,
        'generated': generated,
        'run': {
            'seed': seed,
            'sample_size': sample_size,
            'classic_subsample': 0,
            'scenario': 'A',
            'sampled_ids': {'iterative': ids},
            'dump': 'glazy_20260531.yaml.gz',
            'dump_size': 8023778,
            'git_commit': commit,
            'git_dirty': dirty,
            'data_hashes': dict(data_hashes if data_hashes is not None else DATA_HASHES),
            'solver_config': {'max_solutions': 5, 'classic_seed': 42},
            'seconds': 1.0,
        },
        'cases': cases,
    }


class RecordRoundTrip(unittest.TestCase):
    """A line written into the log is the line that comes back out"""

    def setUp(self):
        self.directory = tempfile.mkdtemp(prefix='glazy_history_')
        self.log = os.path.join(self.directory, 'history.jsonl')

    def tearDown(self):
        shutil.rmtree(self.directory, ignore_errors=True)

    def test_record_round_trips_through_the_log(self):
        record = history.build_record(snapshot(), recorded_at=PINNED,
                                      kind=history.KIND_RECORD, note='a note with "quotes"')
        history.append_record(self.log, record)

        back, problems = history.read_log(self.log)
        self.assertEqual([], problems)
        self.assertEqual(1, len(back))

        # _line is the reader's own bookkeeping, not part of the record
        self.assertEqual(1, back[0].pop('_line'))
        self.assertEqual(record, back[0])

    def test_the_timestamp_is_an_argument_not_a_clock(self):
        """
        Two builds of the same snapshot with the same timestamp are identical

        bench/corpus.py keeps the workflow-run path free of hidden clocks, and a
        record that read the time by itself could never be compared like this.
        """
        first = history.build_record(snapshot(), recorded_at=PINNED, note='x')
        second = history.build_record(snapshot(), recorded_at=PINNED, note='x')
        self.assertEqual(first, second)
        self.assertEqual(PINNED, first['recorded_at'])
        # ... and the run's own moment is kept apart from the moment of the append
        self.assertEqual('2026-08-09T10:00:00+00:00', first['ran_at'])

    def test_the_record_carries_the_shares_and_the_whole_profile(self):
        rows = [case(1, umf_error=0.01, junk_count=0, quality_ok=True),
                case(2, umf_error=0.5, junk_count=2, quality_ok=True),
                case(3, status='failed: no solutions', umf_error=None,
                     count=None, min_portion=None, junk_count=None, cond=None,
                     quality_ok=False)]
        record = history.build_record(snapshot(rows), recorded_at=PINNED)
        entry = record['engines']['iterative']

        self.assertEqual(3, entry['cases'])
        self.assertEqual(2, entry['solved'])
        self.assertAlmostEqual(2 / 3, entry['solved_share'])
        # Case 2 is solved but its chemistry misses by 0.5, and the unsolved case
        # counts against every share: the denominator is the whole engine.
        self.assertAlmostEqual(1 / 3, entry['chemistry_share'])
        self.assertAlmostEqual(1 / 3, entry['both_levels_share'])

        profile = entry['metrics']['junk_count']
        self.assertEqual(2, profile['n'])
        self.assertEqual(1.0, profile['mean'])
        self.assertEqual(2.0, profile['max'])
        for key in bench_corpus.PROFILE_KEYS:
            self.assertIn(key, profile)

        # Nothing prices these cases, so the cost metrics have no distribution
        self.assertIsNone(entry['metrics']['assembly_score'])

    def test_a_snapshot_without_the_quality_flag_says_so_instead_of_guessing(self):
        """
        both_levels_share is None, never 0.0, when the snapshot cannot answer

        The quality verdict compares the solution against the original recipe,
        which the snapshot does not store, so a snapshot written before the flag
        existed - and the two seeded lines of the real log are exactly that -
        has no way back to it.
        """
        rows = [case(1), case(2)]
        for row in rows:
            row.pop('quality_ok')
        record = history.build_record(snapshot(rows), recorded_at=PINNED)
        entry = record['engines']['iterative']

        self.assertIsNone(entry['both_levels_share'])
        self.assertIsNone(entry['both_levels'])
        # The chemistry verdict, unlike the quality one, IS re-derivable
        self.assertEqual(1.0, entry['chemistry_share'])

    def test_the_sampled_ids_are_a_documented_digest(self):
        """The digest is reproducible from the module docstring alone"""
        import hashlib
        expected = hashlib.sha256(b'7,1,4').hexdigest()
        self.assertEqual(expected, history.sampled_ids_digest([7, 1, 4]))
        # ... and the order is part of it: the sample is not a set
        self.assertNotEqual(expected, history.sampled_ids_digest([1, 4, 7]))

        record = history.build_record(snapshot(), recorded_at=PINNED)
        self.assertEqual(3, record['sampled_ids']['iterative']['count'])
        self.assertEqual(history.sampled_ids_digest([1, 2, 3]),
                         record['sampled_ids']['iterative']['sha256'])

    def test_a_note_with_a_newline_cannot_break_the_one_line_rule(self):
        record = history.build_record(snapshot(), recorded_at=PINNED,
                                      note='two\nlines\tand a tab')
        history.append_record(self.log, record)

        with open(self.log, 'rb') as f:
            payload = f.read()
        self.assertEqual(1, payload.count(b'\n'))

        back, problems = history.read_log(self.log)
        self.assertEqual([], problems)
        self.assertEqual('two\nlines\tand a tab', back[0]['note'])


class TruncatedWrites(unittest.TestCase):
    """A torn append costs the record it tore, never the log"""

    def setUp(self):
        self.directory = tempfile.mkdtemp(prefix='glazy_history_')
        self.log = os.path.join(self.directory, 'history.jsonl')
        for index in (1, 2):
            history.append_record(self.log, history.build_record(
                snapshot(commit=f'{index}' * 40), recorded_at=PINNED, note=f'run {index}'))

    def tearDown(self):
        shutil.rmtree(self.directory, ignore_errors=True)

    def _tear(self, fragment):
        with open(self.log, 'ab') as f:
            f.write(fragment)

    def test_a_truncated_final_line_is_skipped_and_reported(self):
        self._tear(b'{"schema": 1, "recorded_at": "2026')

        records, problems = history.read_log(self.log)
        self.assertEqual(2, len(records))
        self.assertEqual(['run 1', 'run 2'], [record['note'] for record in records])
        self.assertEqual(1, len(problems))
        self.assertIn('truncated final line', problems[0])
        self.assertIn('line 3', problems[0])

    def test_a_truncated_final_line_cut_inside_a_utf8_sequence(self):
        """
        The reader decodes per line, so a half written character is one bad line

        Reading the file as text would raise before a single record came back.
        """
        self._tear('{"note": "по'.encode('utf-8')[:-1])

        records, problems = history.read_log(self.log)
        self.assertEqual(2, len(records))
        self.assertEqual(1, len(problems))
        self.assertIn('truncated final line', problems[0])

    def test_the_next_append_seals_the_torn_line_instead_of_gluing_onto_it(self):
        self._tear(b'{"schema": 1, "recorded')

        sealed = history.append_record(self.log, history.build_record(
            snapshot(commit='3' * 40), recorded_at=PINNED, note='run 3'))
        self.assertTrue(sealed)

        records, problems = history.read_log(self.log)
        self.assertEqual(['run 1', 'run 2', 'run 3'],
                         [record['note'] for record in records])
        self.assertEqual(1, len(problems))
        self.assertIn('line 3', problems[0])
        self.assertIn('not readable as JSON', problems[0])

    def test_an_unreadable_line_in_the_middle_does_not_stop_the_reader(self):
        with open(self.log, 'r', encoding='utf-8') as f:
            good = f.readlines()
        with open(self.log, 'w', encoding='utf-8') as f:
            f.write(good[0])
            f.write('{ this was never JSON }\n')
            f.write(good[1])

        records, problems = history.read_log(self.log)
        self.assertEqual(['run 1', 'run 2'], [record['note'] for record in records])
        self.assertEqual([1, 3], [record['_line'] for record in records])
        self.assertEqual(1, len(problems))

    def test_a_torn_log_still_renders(self):
        self._tear(b'{"schema": 1, "reco')
        records, problems = history.read_log(self.log)
        text = '\n'.join(history.render(records, problems, self.log))
        self.assertIn('truncated final line', text)
        self.assertIn('run 2', text)

    def test_an_empty_or_missing_log_is_not_an_error(self):
        missing = os.path.join(self.directory, 'nothing.jsonl')
        self.assertEqual(([], []), history.read_log(missing))

        empty = os.path.join(self.directory, 'empty.jsonl')
        open(empty, 'wb').close()
        self.assertEqual(([], []), history.read_log(empty))

        text = '\n'.join(history.render([], [], empty))
        self.assertIn('no readable runs yet', text)


class Incomparability(unittest.TestCase):
    """Runs that cannot be compared on the solver axis have to say so"""

    def test_changed_data_hashes_are_flagged(self):
        first = history.build_record(snapshot(commit='1' * 40), recorded_at=PINNED,
                                     note='before')
        moved = dict(DATA_HASHES, **{'database/prices.json': 'd' * 64})
        second = history.build_record(snapshot(commit='2' * 40, data_hashes=moved),
                                      recorded_at=PINNED, note='after the price update')

        self.assertEqual(['database/prices.json'], history.changed_data(first, second))
        self.assertNotEqual(history.data_digest(first), history.data_digest(second))

        text = '\n'.join(history.render([first, second], [], 'log'))
        self.assertIn('INPUT DATA CHANGED vs #1', text)
        self.assertIn('database/prices.json', text)
        self.assertIn('not a solver measurement', text)

    def test_identical_data_hashes_are_not_flagged(self):
        first = history.build_record(snapshot(commit='1' * 40), recorded_at=PINNED)
        second = history.build_record(snapshot(commit='2' * 40), recorded_at=PINNED)

        self.assertEqual([], history.changed_data(first, second))
        text = '\n'.join(history.render([first, second], [], 'log'))
        self.assertNotIn('INPUT DATA CHANGED', text)
        self.assertIn('delta vs #1', text)

    def test_a_changed_sample_is_flagged_too(self):
        first = history.build_record(snapshot(commit='1' * 40), recorded_at=PINNED)
        second = history.build_record(
            snapshot([case(9), case(8), case(7)], commit='2' * 40, seed=7),
            recorded_at=PINNED)

        self.assertIn('seed', history.changed_run(first, second))
        self.assertIn('sampled ids of iterative', history.changed_run(first, second))
        text = '\n'.join(history.render([first, second], [], 'log'))
        self.assertIn('RUN CHANGED vs #1', text)

    def test_the_series_is_printed_in_file_order_with_a_total(self):
        records = [history.build_record(snapshot([case(1, junk_count=junk),
                                                  case(2, junk_count=junk),
                                                  case(3, junk_count=junk)],
                                                 commit=str(index) * 40),
                                        recorded_at=PINNED, note=f'run {index}')
                   for index, junk in enumerate((3, 2, 1), start=1)]

        text = '\n'.join(history.render(records, [], 'log'))
        self.assertLess(text.index('run 1'), text.index('run 2'))
        self.assertLess(text.index('run 2'), text.index('run 3'))
        # Each step is -1.0 and the three-run drift is -2.0: the trend is the
        # point of the log, so it is printed rather than left to be added up
        self.assertIn('total #1 -> #3', text)
        self.assertIn('-2.0000', text)

    def test_last_hides_rows_but_not_the_delta_they_are_measured_against(self):
        records = [history.build_record(snapshot(commit=str(index) * 40),
                                        recorded_at=PINNED, note=f'run {index}')
                   for index in (1, 2, 3)]
        text = '\n'.join(history.render(records, [], 'log', last=1))

        self.assertNotIn('run 1', text)
        self.assertIn('run 3', text)
        self.assertIn('delta vs #2', text)
        self.assertIn('2 earlier run(s) hidden', text)


class DiffBaselineWritesTheLog(unittest.TestCase):
    """Where a line comes from: --rebaseline always, --record on request, a plain diff never"""

    def setUp(self):
        self.directory = tempfile.mkdtemp(prefix='glazy_history_')
        self.log = os.path.join(self.directory, 'history.jsonl')
        self.baseline = os.path.join(self.directory, 'quality_baseline.json')
        self.current = os.path.join(self.directory, 'current.json')

        diff_baseline.write_baseline(self.baseline, snapshot(commit='1' * 40))
        diff_baseline.write_baseline(
            self.current, snapshot([case(1, junk_count=1), case(2), case(3)], commit='2' * 40))

    def tearDown(self):
        shutil.rmtree(self.directory, ignore_errors=True)

    def _run(self, *extra):
        """diff_baseline with the corpus replaced by the saved run, output swallowed"""
        argv = ['--baseline', self.baseline, '--current', self.current,
                '--history-log', self.log] + list(extra)
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = diff_baseline.main(argv)
        return code, buffer.getvalue()

    def _lines(self):
        if not os.path.exists(self.log):
            return []
        with open(self.log, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f if line.strip()]

    def test_a_plain_diff_run_writes_nothing(self):
        code, _output = self._run()
        self.assertEqual(0, code)
        self.assertFalse(os.path.exists(self.log),
                         'a plain diff run must not touch the history log')

        # ... and neither does --check, which is the same run with a gate on it
        self._run('--check')
        self.assertFalse(os.path.exists(self.log))

    def test_the_log_is_appended_to_not_rewritten(self):
        self._run('--record', 'first measurement')
        before = os.path.getsize(self.log)
        self._run('--record', 'second measurement')

        records = self._lines()
        self.assertEqual(['first measurement', 'second measurement'],
                         [record['note'] for record in records])
        self.assertGreater(os.path.getsize(self.log), before)
        self.assertEqual([history.KIND_RECORD, history.KIND_RECORD],
                         [record['kind'] for record in records])

    def test_record_without_a_note_is_still_a_record(self):
        self._run('--record')
        records = self._lines()
        self.assertEqual(1, len(records))
        self.assertEqual('', records[0]['note'])

    def test_record_does_not_move_the_baseline(self):
        with open(self.baseline, 'rb') as f:
            before = f.read()
        self._run('--record', 'just measuring')
        with open(self.baseline, 'rb') as f:
            self.assertEqual(before, f.read())

    def test_rebaseline_always_writes_a_line(self):
        code, _output = self._run('--rebaseline')
        self.assertEqual(0, code)

        records = self._lines()
        self.assertEqual(1, len(records))
        self.assertEqual(history.KIND_REBASELINE, records[0]['kind'])
        self.assertEqual('2' * 40, records[0]['git_commit'])

        # The baseline moved as well, and the line describes the run that moved it
        with open(self.baseline, 'r', encoding='utf-8') as f:
            self.assertEqual('2' * 40, json.load(f)['run']['git_commit'])

    def test_rebaseline_takes_its_note_from_record(self):
        self._run('--rebaseline', '--record', 'after the sole-carrier rule')
        records = self._lines()
        self.assertEqual(1, len(records))
        self.assertEqual(history.KIND_REBASELINE, records[0]['kind'])
        self.assertEqual('after the sole-carrier rule', records[0]['note'])


class SharedAggregation(unittest.TestCase):
    """The diff and the history must not compute "the median" two ways"""

    def test_the_profile_is_one_implementation(self):
        self.assertIs(diff_baseline._profile, bench_corpus.profile)
        self.assertIs(diff_baseline.METRICS, bench_corpus.METRICS)

    def test_the_snapshot_keeps_the_two_level_verdict(self):
        """
        Without these fields the log could not report the both-levels share

        The field list is explicit precisely so that this is a decision rather
        than an accident, so the decision is pinned.
        """
        self.assertIn('chemistry_ok', diff_baseline.CASE_FIELDS)
        self.assertIn('quality_ok', diff_baseline.CASE_FIELDS)
        self.assertEqual(1, diff_baseline.BASELINE_FORMAT_VERSION,
                         'the two flags are additive; bumping the format would force a rebaseline')


class TheCommittedLog(unittest.TestCase):
    """The real bench/history.jsonl has to stay readable"""

    def test_it_parses_and_renders(self):
        path = bench_corpus.HISTORY_PATH
        if not os.path.exists(path):
            self.skipTest('bench/history.jsonl has not been created yet')

        records, problems = history.read_log(path)
        self.assertEqual([], problems, 'the committed log has unreadable lines')
        self.assertGreaterEqual(len(records), 1)

        for record in records:
            self.assertEqual(history.SCHEMA_VERSION, record['schema'])
            self.assertTrue(record['ran_at'])
            self.assertTrue(record['git_commit'])
            self.assertIn('iterative', record['engines'])

        # File order is chronological and stays that way: the log is appended to
        moments = [record['ran_at'] for record in records]
        self.assertEqual(sorted(moments), moments)

        text = '\n'.join(history.render(records, problems, path))
        self.assertIn('engine: iterative', text)


if __name__ == '__main__':
    unittest.main()
