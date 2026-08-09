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
import feasibility


# A timestamp is never read from the clock inside the record path, so every test
# pins it and compares the whole record byte for byte.
PINNED = '2026-08-09T12:00:00+00:00'

DATA_HASHES = {
    'database/materials.json': 'a' * 64,
    'database/prices.json': 'b' * 64,
    'database/oxide_classification.json': 'c' * 64,
}


def case(glazy_id, engine='iterative', status='solved', umf_error=0.01,
         count=5, min_portion=3.0, junk_count=0, cond=10.0, quality_ok=True,
         max_relative=None):
    """
    One row in the shape diff_baseline.CASE_FIELDS stores

    max_relative mirrors umf_error unless a test says otherwise. The two are
    different measurements on real data, but every call site here only needs a
    row that is either comfortably inside the chemistry gate or clearly outside
    it, and mirroring keeps the fixtures saying what they said before 10.18
    moved the gate from the absolute norm onto the relative one.
    """
    max_relative = umf_error if max_relative is None else max_relative
    return {
        'glazy_id': glazy_id,
        'scenario': 'A',
        'engine': engine,
        'status': status,
        'bucket': '5-7',
        'size': count,
        'umf_error': umf_error,
        'max_relative': max_relative,
        'worst_oxide': None if max_relative is None else 'SiO2',
        'dropped_oxides': [],
        'count': count,
        'cost_abs': None,
        'assembly_score': None,
        'min_portion': min_portion,
        'junk_count': junk_count,
        'rounding_drift': 0.01,
        'cond': cond,
        'chemistry_ok': max_relative is not None and max_relative <= bench_corpus.CHEMISTRY_TOL,
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
            # A run produced by today's diff_baseline carries the marker; the
            # back-fill tests below strip it to make a pre-10.18 one.
            'chemistry_gate': dict(bench_corpus.CHEMISTRY_GATE),
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

    def test_the_snapshot_keeps_the_metric_the_verdict_is_drawn_with(self):
        """
        max_relative is the gate since 10.18, so the snapshot has to store it

        Without it a stored run cannot be re-scored at all: umf_error is a
        different measurement and re-deriving the verdict from it would answer
        a question nobody asked.
        """
        self.assertIn('max_relative', diff_baseline.CASE_FIELDS)
        self.assertIn('worst_oxide', diff_baseline.CASE_FIELDS)
        self.assertIn('dropped_oxides', diff_baseline.CASE_FIELDS)
        self.assertIn(('max_relative', 'lower'), bench_corpus.METRICS)
        self.assertEqual(1, diff_baseline.BASELINE_FORMAT_VERSION,
                         'the fields are additive; chemistry_ok changed meaning and is '
                         'handled by run.chemistry_gate instead of a format bump')


class TheChemistryGate(unittest.TestCase):
    """
    Which rule a stored row is scored by, and how a reader knows (10.18)

    The log is append only, so a line written under the retired absolute gate
    can never be corrected. It has to be readable as what it is instead.
    """

    def test_a_recorded_verdict_wins_over_any_re_derivation(self):
        row = {'chemistry_ok': True, 'max_relative': 0.9, 'umf_error': 9.0}
        self.assertTrue(bench_corpus.chemistry_ok(row))
        row = {'chemistry_ok': False, 'max_relative': 0.0, 'umf_error': 0.0}
        self.assertFalse(bench_corpus.chemistry_ok(row))

    def test_without_a_verdict_the_relative_gate_decides(self):
        tol = bench_corpus.CHEMISTRY_TOL
        self.assertTrue(bench_corpus.chemistry_ok({'max_relative': tol, 'umf_error': 5.0}))
        self.assertFalse(bench_corpus.chemistry_ok({'max_relative': tol * 2, 'umf_error': 0.0}))

    def test_a_snapshot_older_than_the_change_falls_back_to_the_retired_gate(self):
        """A row with only umf_error is scored by the rule it was written under"""
        self.assertTrue(bench_corpus.chemistry_ok({'umf_error': bench_corpus.MAX_UMF_ERROR}))
        self.assertFalse(bench_corpus.chemistry_ok({'umf_error': bench_corpus.MAX_UMF_ERROR * 2}))
        self.assertFalse(bench_corpus.chemistry_ok({}))

    def test_the_gate_is_the_feasibility_verdict_and_not_a_second_opinion(self):
        self.assertEqual(feasibility.DEFAULT_FEASIBILITY_TOL, bench_corpus.CHEMISTRY_TOL)

    def test_the_record_says_which_gate_scored_it(self):
        record = history.build_record(snapshot(), recorded_at=PINNED)
        self.assertEqual({'metric': 'max_relative', 'tol': bench_corpus.CHEMISTRY_TOL},
                         record['thresholds']['chemistry_gate'])
        # The retired threshold stays: chemistry_ok() still falls back to it
        self.assertEqual(bench_corpus.MAX_UMF_ERROR, record['thresholds']['max_umf_error'])

    def test_an_oxide_left_out_of_the_comparison_cannot_pass(self):
        """
        Fail closed on a formula that was never fully compared

        The retired metric did this by accident: calculate_umf_error returned
        NaN and "nan <= 0.1" is False. umf_deviation reports the bad oxide in
        "dropped" instead, so the gate has to look, or a non-finite analysis
        turns into a pass.
        """
        row = {'max_relative': 0.0, 'umf_error': 0.0, 'dropped_oxides': ['SiO2']}
        self.assertFalse(bench_corpus.chemistry_ok(row))
        # ... and an empty list is not a dropped oxide
        self.assertTrue(bench_corpus.chemistry_ok(
            {'max_relative': 0.0, 'umf_error': 0.0, 'dropped_oxides': []}))

    def test_a_snapshot_carries_the_gate_it_was_scored_by(self):
        """
        chemistry_ok changed meaning in 10.18, so the snapshot has to say which

        Without the marker a pre-10.18 baseline and a post-10.18 run compare as
        one measurement, and --check reports the re-scoring as a regression.
        """
        self.assertEqual(('max_relative', bench_corpus.CHEMISTRY_TOL),
                         diff_baseline.snapshot_chemistry_gate(snapshot()))

        # A snapshot with no marker is the retired gate, which is the only
        # thing it can have been
        unmarked = snapshot()
        del unmarked['run']['chemistry_gate']
        self.assertEqual(('umf_error', bench_corpus.MAX_UMF_ERROR),
                         diff_baseline.snapshot_chemistry_gate(unmarked))

    @staticmethod
    def _coverage_comparison(gate_changed):
        """
        One series that lost a reachable target, with the gate flag set

        Ten reachable targets on both sides and one of the six the search used
        to reach gone. The reachable SETS are what the gate divides by, so the
        fixture carries them rather than a ready-made share.
        """
        reachable = list(range(1, 11))
        return {
            'provenance': {'chemistry_gate': {'changed': gate_changed}},
            'engines': {'B/iterative': {
                'metrics': {},
                'baseline_solved_share': 1.0, 'current_solved_share': 1.0, 'lost': [],
                'baseline_dropped': [], 'current_dropped': [],
                'baseline_reachability': {
                    'accounted_share': 1.0, 'reachable_ids': list(reachable),
                    'solved_among_reachable_ids': [1, 2, 3, 4, 5, 6],
                    'reachable_unsolved_ids': [7, 8, 9, 10]},
                'current_reachability': {
                    'accounted_share': 0.99, 'reachable_ids': list(reachable),
                    'solved_among_reachable_ids': [2, 3, 4, 5, 6],
                    'reachable_unsolved_ids': [1, 7, 8, 9, 10]},
            }},
        }

    def test_the_coverage_gate_stands_down_across_a_gate_change(self):
        """
        Re-scoring is not a regression, and --check must not call it one

        The coverage shares are derived from chemistry_ok, so they move by
        themselves when the gate moves. The raw component gates are untouched.
        """
        comparison = self._coverage_comparison(gate_changed=True)
        problems = diff_baseline.check_regressions(comparison)
        self.assertEqual([], [p for p in problems if 'reachable' in p])
        ungated = comparison['provenance']['ungated']
        self.assertTrue(any('solved-among-reachable' in note for note in ungated), ungated)

        # ... and it does fire when the gate did not move
        comparison = self._coverage_comparison(gate_changed=False)
        problems = diff_baseline.check_regressions(comparison)
        self.assertTrue(any('proved reachable' in p for p in problems), problems)

    def test_the_coverage_gate_is_the_sensitive_one(self):
        """
        Solved-among-reachable, not the accounted share: ten times the signal

        One lost reachable target out of ten moves the accounted share by a
        point (90 of the 100 cases are in the unreachable bucket and do not
        care) and this one by ten. The fixture above encodes exactly that
        event, and the message has to name the sensitive number.
        """
        comparison = self._coverage_comparison(gate_changed=False)
        problems = diff_baseline.check_regressions(comparison)
        coverage = [p for p in problems if 'proved reachable' in p]
        self.assertEqual(1, len(coverage), problems)
        # Counted, not only shared: "6/10 -> 5/10" is a sentence a reader can
        # check against the block above, "60.00% -> 50.00%" is not
        self.assertIn('6/10 (60.00%) -> 5/10 (50.00%)', coverage[0])
        self.assertIn('[1]', coverage[0])

    @staticmethod
    def _scenario_b_snapshot(passing_ids, reachable_ids=(11, 12, 13, 14)):
        """
        Five scenario B targets, as a SNAPSHOT rather than as a comparison

        The fixtures above hand check_regressions a ready-made
        solved_among_reachable_share, so they cannot see the field going missing
        from the snapshot side. This one goes through _reachability(), which is
        where the number is built and where it was absent for as long as the
        gate existed: every run defined it as None, so the gate could only ever
        reach its "not gated" branch.
        """
        rows = []
        for glazy_id in (11, 12, 13, 14, 15):
            feasible = glazy_id in reachable_ids
            row = case(glazy_id, max_relative=0.0 if glazy_id in passing_ids else 0.5)
            row.update({'scenario': bench_corpus.SCENARIO_B, 'feasible': feasible,
                        'max_relative_deviation': 0.0 if feasible else 0.4,
                        'unreachable_oxides': [] if feasible else ['Li2O']})
            rows.append(row)
        return snapshot(cases=rows)

    def test_losing_a_reachable_target_fires_the_gate_from_the_snapshot(self):
        """
        The share the gate reads has to exist in what the diff builds

        Three of the four reachable targets were reproduced and now two are, so
        the search covers less of what the LP proved possible. Everything the
        gate needs comes out of the rows here, exactly as it does on a real run.
        """
        baseline = self._scenario_b_snapshot({11, 12, 13})
        current = self._scenario_b_snapshot({11, 12})

        _lines, comparison = diff_baseline.build_report(baseline, current)
        reach = comparison['engines']['B/iterative']['current_reachability']
        self.assertEqual(4, reach['reachable'])
        self.assertEqual(2, reach['solved_among_reachable'])
        self.assertEqual(0.5, reach['solved_among_reachable_share'])

        problems = diff_baseline.check_regressions(comparison)
        coverage = [p for p in problems if 'proved reachable' in p]
        self.assertEqual(1, len(coverage), problems)
        self.assertIn('3/4 (75.00%) -> 2/4 (50.00%)', coverage[0])
        self.assertIn('[13]', coverage[0])
        # ... and the gate ran, rather than standing down and saying so
        self.assertEqual([], [note for note in comparison['provenance']['ungated']
                              if 'solved-among-reachable' in note])

    def test_the_LP_reclassifying_a_target_is_not_a_search_regression(self):
        """
        The denominator belongs to feasibility, so it cannot judge the search

        Identical search on both sides - the same recipes, the same chemistry -
        and the LP alone changed its mind about target 13. Per-run shares read
        3/4 = 75% against 2/3 = 66.67% and would fire, naming a search that did
        not move a single case. Over the three targets both runs call reachable
        the two runs are level, which is the only honest reading, and what the
        LP did is reported instead of being scored.
        """
        baseline = self._scenario_b_snapshot({11, 12, 13})
        current = self._scenario_b_snapshot({11, 12, 13}, reachable_ids=(11, 12, 14))

        lines, comparison = diff_baseline.build_report(baseline, current)
        before = comparison['engines']['B/iterative']['baseline_reachability']
        after = comparison['engines']['B/iterative']['current_reachability']
        # The raw per-run shares are exactly the trap: they do differ
        self.assertAlmostEqual(0.75, before['solved_among_reachable_share'])
        self.assertAlmostEqual(2 / 3, after['solved_among_reachable_share'])

        problems = diff_baseline.check_regressions(comparison)
        self.assertEqual([], [p for p in problems if 'proved reachable' in p], problems)
        # ... and the reader is told the denominator moved, rather than left to
        # wonder why the gate said nothing
        moved = [line for line in lines if 'LP MOVED' in line]
        self.assertEqual(1, len(moved), lines)
        self.assertIn('reachable 4 -> 3', moved[0])
        self.assertIn('[13]', moved[0])

    def test_a_target_lost_inside_the_shared_set_still_fires(self):
        """
        The intersection narrows the question, it does not disarm the gate

        Same reclassification as above and, on top of it, target 11 stops being
        reproduced. 11 is in the set both runs call reachable, so it counts.
        """
        baseline = self._scenario_b_snapshot({11, 12, 13})
        current = self._scenario_b_snapshot({12, 13}, reachable_ids=(11, 12, 14))

        _lines, comparison = diff_baseline.build_report(baseline, current)
        problems = diff_baseline.check_regressions(comparison)
        coverage = [p for p in problems if 'proved reachable' in p]
        self.assertEqual(1, len(coverage), problems)
        # Both runs call 11, 12 and 14 reachable; 13 left the reachable set and
        # is out of the comparison entirely. Of those three the baseline reached
        # 11 and 12, the current run only 12 - 14 was never reached by either
        self.assertIn('2/3 (66.67%) -> 1/3 (33.33%)', coverage[0])
        self.assertIn('[11]', coverage[0])

    def test_a_run_with_nothing_reachable_names_the_run_and_the_cause(self):
        """
        No denominator is not a failure, and the note says which run and why

        A share of 0.0 would be a different statement - "the search reached none
        of them" - so a sample whose shelf can reach nothing leaves the gate
        without a denominator instead of failing it. The note must not assert
        that cause blindly either: two runs whose reachable sets merely fail to
        overlap land in the same branch and are told apart below.
        """
        baseline = self._scenario_b_snapshot({11, 12, 13})
        current = self._scenario_b_snapshot({11, 12, 13}, reachable_ids=())

        _lines, comparison = diff_baseline.build_report(baseline, current)
        after = comparison['engines']['B/iterative']['current_reachability']
        self.assertEqual(0, after['reachable'])
        self.assertIsNone(after['solved_among_reachable_share'])

        problems = diff_baseline.check_regressions(comparison)
        self.assertEqual([], [p for p in problems if 'proved reachable' in p])
        note = [n for n in comparison['provenance']['ungated']
                if 'solved-among-reachable' in n]
        self.assertEqual(1, len(note), comparison['provenance']['ungated'])
        self.assertIn('the current run has no reachable target at all', note[0])

        # Both runs reach something, and nothing in common: a different cause,
        # and the note is not allowed to report it as the one above
        disjoint = diff_baseline.build_report(
            self._scenario_b_snapshot({11, 12}, reachable_ids=(11, 12)),
            self._scenario_b_snapshot({13, 14}, reachable_ids=(13, 14)))[1]
        diff_baseline.check_regressions(disjoint)
        note = [n for n in disjoint['provenance']['ungated']
                if 'solved-among-reachable' in n]
        self.assertEqual(1, len(note), disjoint['provenance']['ungated'])
        self.assertIn('do not overlap', note[0])
        self.assertNotIn('no reachable target at all', note[0])

    def test_a_verdict_on_one_side_only_is_reported_rather_than_skipped(self):
        """
        A gate that quietly does not run reads exactly like one that passed

        A baseline predating scenario B, or a run whose LP answered nothing at
        all, leaves one side without a reachability block. Silence there is
        legitimate only for scenario A, which never asks the question.
        """
        rows = [row for row in self._scenario_b_snapshot({11, 12, 13})['cases']]
        without = json.loads(json.dumps(rows))
        for row in without:
            row['feasible'] = None

        lines, comparison = diff_baseline.build_report(snapshot(cases=without),
                                                       snapshot(cases=rows))
        series = comparison['engines']['B/iterative']
        self.assertIsNone(series['baseline_reachability'])
        self.assertIsNotNone(series['current_reachability'])

        # ... and the report says the baseline never asked, rather than
        # reporting the placeholder's empty set as a reclassification of every
        # target ("LP MOVED: reachable 0 -> 4, newly reachable [11, 12, 13, 14]")
        self.assertEqual([], [line for line in lines if 'LP MOVED' in line], lines)
        self.assertTrue(any('the baseline carries no feasibility verdict' in line
                            for line in lines), lines)

        diff_baseline.check_regressions(comparison)
        note = [n for n in comparison['provenance']['ungated']
                if 'solved-among-reachable' in n]
        self.assertEqual(1, len(note), comparison['provenance']['ungated'])
        self.assertIn('only the current run carries a feasibility verdict', note[0])

        # Scenario A asks nothing and says nothing: no note at all
        plain = diff_baseline.build_report(snapshot(), snapshot())[1]
        diff_baseline.check_regressions(plain)
        self.assertEqual([], [n for n in plain['provenance']['ungated']
                              if 'solved-among-reachable' in n])

    def test_the_report_prints_the_number_the_gate_will_quote(self):
        """
        The gated share has to appear in the block, not only in the problem line

        A reader who sees the accounted share go 4/5 -> 4/5 and then reads a
        problem about "75.00% -> 50.00%" cannot find that number anywhere.
        """
        lines, _comparison = diff_baseline.build_report(
            self._scenario_b_snapshot({11, 12, 13}),
            self._scenario_b_snapshot({11, 12}))

        gated = [line for line in lines if 'solved among reachable' in line]
        self.assertEqual(1, len(gated), lines)
        self.assertIn('3/4 (75.00%) -> 2/4 (50.00%)', gated[0])

    def test_the_log_annotation_prints_the_case_count_it_promises(self):
        """
        "accounted 4/5", not "accounted 4/None", on a line of the OLD shape

        The annotation read the denominator out of the reachability block,
        which did not carry one - it lives on the profile a level up - so every
        scenario B line of the log printed None where the case count belongs.
        The renderer not raising is not the same as the renderer being right,
        so the assertion is on the text.

        The block is stripped of the "cases" key it now gets from the shared
        counts, because that is the shape of every line written before the two
        shapes were unified - the six scenario B lines of the committed log
        among them. Leaving the key in would let the annotation go back to
        reading it from the block and still pass, which is the failure this
        test exists to prevent.
        """
        record = history.build_record(self._scenario_b_snapshot({11, 12, 13}),
                                      recorded_at=PINNED)
        del record['engines']['B/iterative']['reachability']['cases']
        text = '\n'.join(history.render([record], [], 'bench/history.jsonl'))

        accounted = [line for line in text.splitlines()
                     if 'accounted (solved OR unreachable)' in line]
        self.assertEqual(1, len(accounted), text)
        # 11, 12 and 13 pass the chemistry gate and 15 is honestly unreachable
        self.assertIn('4/5 (80.00%)', accounted[0])

    def test_a_new_uncomparable_case_is_a_regression(self):
        """
        dropped_oxides is read by something, not just written to the snapshot

        A case whose formula carries a non-finite value cannot be scored at
        all: its metrics are None, so it leaves no trace in any distribution.
        The count is the only place it shows up.
        """
        comparison = self._coverage_comparison(gate_changed=False)
        comparison['engines']['B/iterative']['current_dropped'] = [7]
        problems = diff_baseline.check_regressions(comparison)
        self.assertTrue(any('dropped from the comparison' in p for p in problems), problems)

        # An unchanged count is not a regression, even when it is not zero
        comparison = self._coverage_comparison(gate_changed=False)
        comparison['engines']['B/iterative']['baseline_dropped'] = [7]
        comparison['engines']['B/iterative']['current_dropped'] = [7]
        problems = diff_baseline.check_regressions(comparison)
        self.assertEqual([], [p for p in problems if 'dropped from the comparison' in p])

    def test_a_line_from_before_the_change_is_flagged_as_incomparable(self):
        """A run scored by one gate and a run scored by another are not a delta"""
        new = history.build_record(snapshot(), recorded_at=PINNED)
        old = json.loads(json.dumps(new))
        del old['thresholds']['chemistry_gate']

        self.assertIsNone(history.changed_gate(new, new))
        self.assertEqual(('umf_error', bench_corpus.MAX_UMF_ERROR), history.chemistry_gate(old))
        flagged = history.changed_gate(old, new)
        self.assertIsNotNone(flagged)
        self.assertIn('max_relative', flagged)

    def test_back_filling_an_old_snapshot_records_the_OLD_gate(self):
        """
        The marker comes out of the snapshot, never out of today's constant

        build_record's own docstring advertises back-filling an old baseline
        into the log - lines #1 and #2 of the real log were made that way - and
        that is the one path where reading the constant instead of the snapshot
        goes wrong: the line would claim the relative gate over a chemistry
        share that was computed with the absolute one, changed_gate() would see
        nothing, and a reader would compare the two as one measurement. A
        missing marker is honest; a wrong one is not.
        """
        old_snapshot = snapshot()
        del old_snapshot['run']['chemistry_gate']

        record = history.build_record(old_snapshot, recorded_at=PINNED,
                                      seeded_from='bench/quality_baseline.json@096a12d')
        self.assertEqual({'metric': 'umf_error', 'tol': bench_corpus.MAX_UMF_ERROR},
                         record['thresholds']['chemistry_gate'])
        self.assertEqual(('umf_error', bench_corpus.MAX_UMF_ERROR),
                         history.chemistry_gate(record))

        # ... and the back-filled line is therefore flagged against a current one
        current = history.build_record(snapshot(), recorded_at=PINNED)
        self.assertIsNotNone(history.changed_gate(record, current))

    def test_a_marker_that_cannot_be_read_is_not_read_as_the_old_gate(self):
        """
        Present but unreadable means something different from absent

        Absent says "written before the marker existed". Unreadable says "the
        writer meant to name a gate and failed", and answering the retired gate
        there would be inventing knowledge. "unknown" compares unequal to
        everything, including itself, so the comparability check warns.
        """
        self.assertEqual(('unknown', None), bench_corpus.read_chemistry_gate({'tol': 0.05}))
        self.assertEqual(('unknown', None), bench_corpus.read_chemistry_gate('garbage'))
        self.assertEqual(('umf_error', bench_corpus.MAX_UMF_ERROR),
                         bench_corpus.read_chemistry_gate(None))

        broken = snapshot()
        broken['run']['chemistry_gate'] = {'tol': 0.05}
        self.assertIsNotNone(history.changed_gate(
            history.build_record(broken, recorded_at=PINNED),
            history.build_record(snapshot(), recorded_at=PINNED)))


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

        # Every annotation is filled in. This is the line the bug was found on:
        # the case count was read from the reachability block, which does not
        # carry one, and all six scenario B lines of THIS file rendered
        # "accounted (solved OR unreachable) 100/None".
        annotations = [line for line in text.splitlines()
                       if 'accounted (solved OR unreachable)' in line]
        self.assertTrue(annotations, 'the committed log has no scenario B line left')
        for line in annotations:
            self.assertNotIn('/None', line)


if __name__ == '__main__':
    unittest.main()
