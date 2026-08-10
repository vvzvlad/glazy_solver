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
Corpus benchmark on the Glazy dump - scenarios A and B (TZ_SOLVER_V2.md 7.6)

Scenario A reproduces a real recipe FROM ITS OWN MATERIALS. The target is
reachable by construction, so nothing there measures whether a solution exists:
it measures how good the one that comes back is. Every part of the setup exists
to remove an excuse:

  * the inventory is the recipe's own materials, injected through the
    materials= seam of both engines, so a miss cannot be blamed on the stock;
  * the target is OUR forward calculation of the merged recipe, so neither the
    flux convention nor Glazy's own normalization can influence the score;
  * the original recipe of the dump is the baseline of the quality half - an
    existence proof, not the right answer (7.1).

Its verdict is two-level:

  1. chemistry - no oxide is off by more than bench_corpus.CHEMISTRY_TOL of its
     own scale, max(target, common.OXIDE_SCALE_FLOOR). That is the feasibility
     LP's own measurement at the LP's own tolerance (TZ_SOLVER_V2.md 10.18),
     not a second opinion invented for the benchmark;
  2. quality   - quality_metrics.solution_quality(...).failures == [].

Both shares are reported, and so is the GAP between them: that gap is the share
of answers that are chemically correct and still bad.

Scenario B is the real use case, and it is a different question. You saw a
recipe made of American frits and you want that chemistry out of what is on your
shelf in Russia, so the inventory is OUR 19 inInventory materials and the target
is the same forward calculation of the same dump recipe - the two scenarios are
drawn from one sample with one seed and can be read case by case. The answer may
not exist, so feasibility.check_feasibility runs FIRST and a target our stock
cannot reach goes into an "honestly unreachable" bucket, which is the correct
answer and not a failure.

Its verdict is single-level and deliberately so:

  * chemistry only - the same relative gate as scenario A - and the gated
    number is the share of the LP-REACHABLE targets the search reached, not the
    "solved OR honestly unreachable" share it replaced. That one is still
    printed, and it is explicitly NOT GATED: 90 of the 100 targets are in the
    unreachable bucket, so it cannot fall below 90% however badly the search
    does - see MIN_SOLVED_AMONG_REACHABLE below for the numbers;
  * quality is NOT gated. There is no comparable original: the dump recipe lives
    in another supply reality, so "is our answer worse than it" is not a
    question with an answer. The quality components are recorded in
    bench/quality_baseline.json and tracked against it instead;
  * every disagreement between the feasibility gate and the solver is logged
    with its id. Those are the interesting output - each one is a bug report
    against one of the two - and they are printed in full rather than counted.

Scenario B is also the only place the cost metrics have values. No Glazy
material appears in database/prices.json, so cost_abs and assembly_score are
None on every scenario A case; 18 of our 19 inventory materials are priced -
all but wood ash, which nobody sells - so in scenario B they are real numbers.
What is never computed is a cost RATIO against the dump recipe: that would
measure the distance between two countries' supply chains, not the quality of
the solver.

The tests are slow and need an 8 MB download, so they are skipped unless
GLAZY_CORPUS is set, and they skip themselves - never fail - when the dump is
neither on disk nor reachable. The dump is CC BY-NC-SA data: it lives in
~/.cache/glazy_solver/ and never inside the repository.

    GLAZY_CORPUS=1 python -m unittest tests.test_inverse_corpus -v
    GLAZY_CORPUS=1 GLAZY_DUMP_PATH=/some/glazy.yaml.gz python -m unittest ...
"""

import os
import sys
import time
import unittest

# Fix imports by adding parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import quality_metrics as qm
from bench import corpus as bench_corpus


# The benchmark is opt-in. Without this the module contributes nothing but skips
# to `python -m unittest discover tests`, and in particular it never imports
# pyyaml, downloads anything or reads the dump.
CORPUS_ENV = 'GLAZY_CORPUS'

# Share of the sample that has to pass BOTH levels. TZ_SOLVER_V2.md 7.6 names
# 90% as a starting point "to be fixed after the first run"; the first run gave
# 93.00% for the iterative engine over 300 cases and 96% for the classic engine
# over its 50 case subsample, so 90% was kept, with a margin of 3.00 points
# (9 cases) and 6 points (3 cases) respectively. By the baseline of 3101b1f the
# shares had moved with the solver to 93.33% and 98.00%.
#
# THE THRESHOLD MOVED 0.90 -> 0.78 IN 10.18, AND THE SOLVER DID NOT MOVE WITH
# IT. What changed is the instrument: level 1 stopped being "the L2 norm of the
# absolute deviations over the target's oxides is at most 0.1" and became "no
# oxide is off by more than 5% of its own scale", which is the feasibility LP's
# measurement. The same run, re-scored, gives 82.67% both-levels on the
# iterative engine (94.67% -> 83.67% on chemistry alone, 251 of 300) and an
# unchanged 98.00% on the classic one - the classic subsample never leaves 0.02
# relative deviation, so the new gate does not touch it. Nothing in the solver
# was touched: bench/diff_baseline.py reports 0 cases moved on every metric of
# every series across the change.
#
# THE THRESHOLD MOVED 0.78 -> 0.93 IN 10.19, AND THIS TIME THE SOLVER DID MOVE.
# The NNLS residual and the search objective went onto the same relative scale
# the gate has measured on since 10.18, so the search is now optimizing what it
# is graded by: chemistry alone went 83.67% -> 98.67% (251 -> 296 of 300) and
# both levels 82.67% -> 98.00%. A gate at 0.78 would have to lose 60 cases
# before it noticed, which is not a gate.
#
# The margin still has to absorb sampling luck, and it was re-measured rather
# than assumed. Over the same five seeds (20260531, 1, 42, 999, 20260101) the
# iterative both-levels share now ranges 97.33 - 99.00%, the pinned seed sitting
# at 98.00% in the middle of it instead of at the bottom. 0.93 clears the whole
# spread and clears the pinned seed by 5.00 points, which on 300 cases is 15
# cases - the same margin discipline as the 4.67 points it replaced, over a
# spread that is a third as wide.
#
# ONE CONSTANT GATES TWO SERIES OF DIFFERENT SIZE, and the margin in CASES is
# what matters, not the margin in points. Re-measured on the classic series,
# which is 50 cases and therefore quantized at 2 points:
#
#     seed       classic chemistry   both levels
#     20260531       100.00%           98.00%   <- pinned
#     1              100.00%          100.00%
#     42             100.00%           98.00%
#     999            100.00%          100.00%
#     20260101       100.00%          100.00%
#
# so 0.93 leaves the classic series 2 cases (49 of 50 today, and 47 of 50 is the
# last passing value) against the iterative series' 15. The previous 0.78 left
# classic 10. Two is thin but it is not fragile: both series are deterministic -
# the sample seed is hard-wired and the classic engine runs on the pinned
# CLASSIC_SEED - so the margin absorbs deliberate change rather than noise, and
# this change does not touch the classic engine at all (bench/diff_baseline.py
# reports 0 cases moved on every classic metric). Splitting the constant per
# series would buy nothing but a second number to keep in step. If a future
# change does move classic, split it then, with the measurement in hand.
#
# It does NOT have to absorb solver changes: those are what
# bench/diff_baseline.py measures, case by case, against a committed snapshot.
MIN_BOTH_LEVELS_SHARE = 0.93

# Scenario B's gate is the share of the REACHABLE targets the search actually
# reached. It replaced the accounted share in 10.18, and the reason is
# sensitivity, not that the old gate was fake. "Solved OR honestly unreachable"
# is dominated by the unreachable bucket - 90 of the 100 targets on this corpus
# - so losing a reachable case moves it by ONE point while it moves this one by
# TEN. On the pinned seed the search reaches 6 of 10.
#
# What the old gate actually was. Its floor is |unreachable| / 100, not a
# constant 0.90: an UNDECIDED case, one where the LP failed to answer at all,
# leaves the unreachable bucket and takes a point of the accounted share with
# it. At 7 undecided the share is 83 + 6 = 89% and the old 0.90 fires. So the
# accounted share was a detector of the LP falling over en masse, and a poor
# detector of anything else. That job is now done directly and unambiguously,
# by the assertion on the undecided count in the test below - which is strictly
# sharper: it fires on the FIRST unanswered case, not on the seventh.
#
# THE THRESHOLD IS SET FROM THE PINNED SEED, and cross-seed spread is not the
# argument. The seed is hard-wired: tests/test_inverse_corpus._shared_corpus
# builds the sample with bench_corpus.DEFAULT_SEED and there is no override, so
# the only run this assertion will ever judge is the one with 10 reachable
# targets and TEN reached. At n=10 the metric is quantized at 10 points, so the
# threshold is set two cases below the measured value: 0.80 demands 8 of 10.
# One or two lost cases is what a real regression looks like, and 0.80 sees the
# third.
#
# THE PREVIOUS COMMENT ASKED FOR THIS, IN THESE WORDS: "6 of 10 is not a target
# anybody should be content with - the LP has PROVEN a recipe exists for all
# ten. Stage 3 (the relative NNLS residual) is expected to take this to 10/10,
# and on that day 0.40 becomes exactly the catastrophe gate this comment was
# written to complain about. Raise it in the same commit." Stage 3 is 10.19 and
# it did take it to 10/10, so the number is raised here.
#
# For reference only, the same five seeds re-measured under 10.19:
#
#     seed       reachable   solved among them        was
#     20260531       10         10   (100.00%)   <- pinned, 6 (60.00%)
#     1              10          8    (80.00%)              5 (50.00%)
#     42             11         11   (100.00%)              4 (36.36%)
#     999            15          9    (60.00%)              3 (20.00%)
#     20260101       11         10    (90.91%)              3 (27.27%)
#
# Seed 999 is the honest footnote: 9 of 15 there, so "the search reaches every
# target the LP calls reachable" is a statement about the pinned corpus and not
# a law. What is a law is the direction - every one of the five improved - and
# the pinned seed is the only one this assertion judges.
#
# READ THE MISCLASSIFICATION LOG, NOT ONLY THIS NUMBER. What the run is actually
# worth is in the two dispute lists printed below it, and since 10.18 those
# lists mean something sharper: "reachable_unsolved" is a bug report against the
# search (0 of 100 on the pinned seed since 10.19, 4 before it), and
# "unreachable_solved" is a soundness check on the LP, modulo the rounding
# quantum - see bench/corpus.py.
MIN_SOLVED_AMONG_REACHABLE = 0.80

# How many failing cases are printed in full. All of them are counted; the log
# would otherwise be unreadable on a bad run.
MAX_LOGGED_FAILURES = 25


def _corpus_enabled():
    """Whether the corpus benchmark was explicitly asked for"""
    value = os.environ.get(CORPUS_ENV, '')
    return value.strip().lower() not in ('', '0', 'false', 'no')


_CORPUS_CACHE = {}


def _shared_corpus():
    """
    Load, parse and sample the corpus once for both scenario classes

    Two TestCase classes need the same corpus, the same 300 case sample and the
    same subsamples drawn from it - scenario B's subsample is drawn from
    scenario A's sample with the same seed, which is what makes the two
    comparable case by case. Building it twice would cost a second parse and,
    worse, would let the two drift apart if one of them ever changed a seed.

    Raises:
        CorpusUnavailable: no dump and no way to fetch one. The callers turn
            that into a skip, never into a failure.
    """
    if not _CORPUS_CACHE:
        corpus, timings = bench_corpus.load_corpus()

        start = time.perf_counter()
        cases, rejected = bench_corpus.build_cases(corpus)
        build_seconds = time.perf_counter() - start

        sample = bench_corpus.stratified_sample(
            cases, size=bench_corpus.DEFAULT_SAMPLE_SIZE, seed=bench_corpus.DEFAULT_SEED)

        _CORPUS_CACHE.update({
            'corpus': corpus,
            'timings': timings,
            'cases': cases,
            'rejected': rejected,
            'build_seconds': build_seconds,
            'sample': sample,
            'classic_sample': bench_corpus.subsample(
                sample, bench_corpus.DEFAULT_CLASSIC_SUBSAMPLE, seed=bench_corpus.DEFAULT_SEED),
            'scenario_b_sample': bench_corpus.subsample(
                sample, bench_corpus.DEFAULT_SCENARIO_B_SUBSAMPLE, seed=bench_corpus.DEFAULT_SEED),
        })

    return _CORPUS_CACHE


def _profile_line(label, values):
    """One metric's whole distribution on one line, or a note that it has none"""
    if not values:
        return f"  {label:<16} defined on no case"
    array = np.asarray(list(values), dtype=float)
    cells = ' '.join(
        f'{key}={value:.2f}' for key, value in (
            ('min', array.min()),
            ('p10', float(np.percentile(array, 10))),
            ('median', float(np.percentile(array, 50))),
            ('mean', float(array.mean())),
            ('p90', float(np.percentile(array, 90))),
            ('p99', float(np.percentile(array, 99))),
            ('max', array.max()),
        ))
    return f"  {label:<16} n={array.size:<4} {cells}"


@unittest.skipUnless(_corpus_enabled(), f'set {CORPUS_ENV}=1 to run the Glazy corpus benchmark')
class GlazyCorpusScenarioA(unittest.TestCase):
    """Scenario A of 7.6: reproduce a dump recipe from its own materials"""

    # Filled in by setUpClass; the empty containers only keep a static checker
    # from treating them as None everywhere below
    corpus: dict = {}
    cases: list = []
    sample: list = []
    classic_sample: list = []
    load_timings: dict = {}
    rejected: dict = {}

    @classmethod
    def setUpClass(cls):
        try:
            shared = _shared_corpus()
        except bench_corpus.CorpusUnavailable as exc:
            # No dump and no network is a skip, never a failure: a benchmark
            # that needs an 8 MB download must not redden an offline suite.
            raise unittest.SkipTest(f'Glazy corpus unavailable: {exc}')

        corpus, timings = shared['corpus'], shared['timings']
        cls.corpus = corpus
        cls.load_timings = timings
        cls.cases, cls.rejected = shared['cases'], shared['rejected']
        cls.sample = shared['sample']
        cls.classic_sample = shared['classic_sample']

        buckets = {}
        for case in cls.sample:
            buckets[case['bucket']] = buckets.get(case['bucket'], 0) + 1

        print(f"\nGlazy corpus: dump {timings['dump']} ({timings['dump_size']} bytes), "
              f"loaded from {timings['source']} in {timings['seconds']:.2f}s, "
              f"cases built in {shared['build_seconds']:.2f}s")
        print(f"  corpus: {corpus['stats']}")
        print(f"  rejected while building cases: {cls.rejected}")
        print(f"  sample: {len(cls.sample)} cases, seed {bench_corpus.DEFAULT_SEED}, "
              f"buckets {buckets}; classic subsample {len(cls.classic_sample)}, "
              f"scenario B subsample {len(shared['scenario_b_sample'])}")
        print(f"  sampled ids: {[case['glazy_id'] for case in cls.sample]}")

    def _run_scenario(self, engine, sample):
        """Run one engine over a sample and assert the two-level criterion"""
        start = time.perf_counter()
        results = bench_corpus.run_sample(sample, engine)
        elapsed = time.perf_counter() - start

        total = len(results)
        self.assertGreater(total, 0, 'the sample is empty')

        solved = [row for row in results if row['status'] == 'solved']
        chemistry = [row for row in results if row['chemistry_ok']]
        both = [row for row in chemistry if row['quality_ok']]

        chemistry_share = len(chemistry) / total
        both_share = len(both) / total

        print(f"\n[{engine}] {total} cases in {elapsed:.1f}s "
              f"({elapsed / total * 1000:.0f} ms per case)")
        print(f"  solved:          {len(solved)}/{total}")
        print(f"  chemistry only:  {chemistry_share:.2%}")
        print(f"  chemistry+quality: {both_share:.2%}")
        print(f"  GAP (chemically correct but bad): {chemistry_share - both_share:.2%} "
              f"({len(chemistry) - len(both)} cases)")

        by_bucket = {}
        for row in results:
            entry = by_bucket.setdefault(row['bucket'], [0, 0, 0])
            entry[0] += 1
            entry[1] += 1 if row['chemistry_ok'] else 0
            entry[2] += 1 if row['chemistry_ok'] and row['quality_ok'] else 0
        for bucket in sorted(by_bucket):
            count, chem, good = by_bucket[bucket]
            print(f"  bucket {bucket:<4} n={count:<4} chemistry {chem:<4} both {good}")

        self._log_failures(engine, results)

        # A raised exception is a bug, not a hard case: the engines are supposed
        # to report "no solution" rather than blow up on real data.
        crashed = [row for row in results if row['status'].startswith('failed: ')
                   and 'Error' in row['status']]
        self.assertEqual(
            [], [(row['glazy_id'], row['status']) for row in crashed],
            f'{engine}: the engine raised on some cases')

        self.assertGreaterEqual(
            both_share, MIN_BOTH_LEVELS_SHARE,
            f'{engine}: only {both_share:.2%} of {total} cases pass both levels, '
            f'the threshold is {MIN_BOTH_LEVELS_SHARE:.0%}')

        # Level 2 is measured on top of level 1, so this can only break if the
        # bookkeeping above is wrong.
        self.assertGreaterEqual(chemistry_share, both_share)

    @staticmethod
    def _log_failures(engine, results):
        """Print every failing case with the metric and the value that failed"""
        failures = [row for row in results if not (row['chemistry_ok'] and row['quality_ok'])]
        if not failures:
            print(f"  no failures for {engine}")
            return

        # Worst chemistry first, on the metric the gate is drawn with; a case
        # that produced no recipe at all leads
        failures.sort(key=lambda row: -(row['max_relative']
                                        if row['max_relative'] is not None else 1e9))

        print(f"  {len(failures)} failing cases for {engine}:")
        for row in failures[:MAX_LOGGED_FAILURES]:
            if row['status'] != 'solved':
                print(f"    id={row['glazy_id']:<8} level=solve   {row['status']:<28} "
                      f"size={row['size']:<3} {row['name'][:48]}")
                continue

            if not row['chemistry_ok']:
                # max_relative is None when the formula could not be compared
                # in full - a non-finite oxide - and then dropped_oxides is the
                # whole story, so print that instead of a number nobody has.
                if row['max_relative'] is None:
                    print(f"    id={row['glazy_id']:<8} level=chemistry metric=max_relative "
                          f"value=UNCOMPARABLE dropped={row.get('dropped_oxides')} "
                          f"size={row['size']} used={row['count']} {row['name'][:48]}")
                else:
                    print(f"    id={row['glazy_id']:<8} level=chemistry metric=max_relative "
                          f"value={row['max_relative']:.4f} limit={bench_corpus.CHEMISTRY_TOL} "
                          f"on={row['worst_oxide']} (l2={row['umf_error']:.4f}) "
                          f"size={row['size']} used={row['count']} {row['name'][:48]}")

            for metric in row['failures']:
                detail = row.get('detail', {}).get(metric, {})
                if metric == 'conditioning':
                    value = detail.get('cond')
                    original = detail.get('original')
                    extra = f" rank={detail.get('rank')} redundancy={detail.get('redundancy')}"
                else:
                    value = detail.get('solution')
                    original = detail.get('original')
                    extra = ''
                print(f"    id={row['glazy_id']:<8} level=quality  metric={metric} "
                      f"value={value} original={original}{extra} "
                      f"size={row['size']} {row['name'][:48]}")

        if len(failures) > MAX_LOGGED_FAILURES:
            print(f"    ... and {len(failures) - MAX_LOGGED_FAILURES} more")

    def test_scenario_a_iterative(self):
        """The iterative engine over the whole sample"""
        self._run_scenario(bench_corpus.ENGINE_ITERATIVE, self.sample)

    def test_scenario_a_classic(self):
        """The classic engine over the 50 case subsample"""
        self._run_scenario(bench_corpus.ENGINE_CLASSIC, self.classic_sample)

    def test_abstaining_metrics_cannot_fail_a_case(self):
        """
        Cost and priority abstain on Glazy materials, and abstaining is not failing

        There are no prices and no priorities for foreign material names, so both
        metrics must report ok=None and stay out of "failures". If either ever
        started returning a verdict here, the whole quality share above would be
        measuring an invented number, so the property is pinned rather than
        assumed.
        """
        checked = 0
        for case in self.sample[:20]:
            quality = qm.solution_quality(case['original'], case['original'], case['materials'])
            self.assertIsNone(quality['cost']['ok'], f"case {case['glazy_id']}: cost did not abstain")
            self.assertIsNone(quality['priority']['ok'], f"case {case['glazy_id']}: priority did not abstain")
            self.assertNotIn('cost', quality['failures'])
            self.assertNotIn('priority', quality['failures'])
            self.assertIsNone(quality['assembly_score']['value'],
                              f"case {case['glazy_id']}: assembly_score must be None without prices")
            checked += 1

        self.assertEqual(20, checked)


@unittest.skipUnless(_corpus_enabled(), f'set {CORPUS_ENV}=1 to run the Glazy corpus benchmark')
class GlazyCorpusScenarioB(unittest.TestCase):
    """Scenario B of 7.6: a foreign chemistry out of our own 19 material stock"""

    sample: list = []
    inventory: list = []

    @classmethod
    def setUpClass(cls):
        try:
            shared = _shared_corpus()
        except bench_corpus.CorpusUnavailable as exc:
            raise unittest.SkipTest(f'Glazy corpus unavailable: {exc}')

        cls.sample = shared['scenario_b_sample']
        cls.inventory = bench_corpus.inventory_materials()

        buckets = {}
        for case in cls.sample:
            buckets[case['bucket']] = buckets.get(case['bucket'], 0) + 1

        print(f"\nScenario B: {len(cls.sample)} targets drawn from the "
              f"{len(shared['sample'])} case scenario A sample with the same seed "
              f"({bench_corpus.DEFAULT_SEED}), buckets {buckets}")
        print(f"  inventory: {len(cls.inventory)} inInventory materials, "
              f"feasibility tol {bench_corpus.FEASIBILITY_TOL}")
        print(f"  subsample ids: {[case['glazy_id'] for case in cls.sample]}")

    def test_scenario_b_iterative(self):
        """
        Every target the LP says we CAN make is made, or the search has a bug

        The gate is the solved share among the reachable targets and nothing
        else. The accounted share it replaced is still printed, but it cannot
        fall below the unreachable bucket's own 90 points and so could never
        fail - see MIN_SOLVED_AMONG_REACHABLE. Quality has no threshold here -
        see the module docstring - and the misclassifications are logged rather
        than gated, because which of the two sides is wrong has to be read case
        by case and a red test would only say that they disagree, which is
        already known.
        """
        engine = bench_corpus.ENGINE_ITERATIVE
        start = time.perf_counter()
        results = bench_corpus.run_sample(self.sample, engine,
                                          scenario=bench_corpus.SCENARIO_B)
        elapsed = time.perf_counter() - start

        total = len(results)
        self.assertGreater(total, 0, 'the scenario B subsample is empty')

        solved = [row for row in results if row['status'] == 'solved']
        chemistry = [row for row in results if bench_corpus.chemistry_ok(row)]
        # The gate's two sets come from bench/corpus.py, which is also where
        # the diff and the history log get them. This test used to build them
        # itself off the raw "chemistry_ok" field - the same rule spelled a
        # fourth way, next to three others that had already drifted apart.
        reachable = bench_corpus.reachable_rows(results)
        unreachable = [row for row in results if row['feasible'] is False]
        undecided = [row for row in results if row['feasible'] is None]
        accounted = [row for row in results if bench_corpus.accounted_ok(row)]
        solved_reachable = bench_corpus.solved_reachable_rows(results)

        accounted_share = len(accounted) / total

        print(f"\n[{engine}, scenario B] {total} cases in {elapsed:.1f}s "
              f"({elapsed / total * 1000:.0f} ms per case, feasibility included)")
        print(f"  returned a recipe:   {len(solved)}/{total}")
        print(f"  feasibility says reachable {len(reachable)}, unreachable "
              f"{len(unreachable)}, undecided {len(undecided)}")
        print(f"  chemistry ok:        {len(chemistry)}/{total} "
              f"({len(chemistry) / total:.2%})")
        print(f"  SOLVED AMONG REACHABLE: {len(solved_reachable)}/{len(reachable)}"
              + (f" ({len(solved_reachable) / len(reachable):.2%})" if reachable else "")
              + f", threshold {MIN_SOLVED_AMONG_REACHABLE:.0%}")
        print(f"  accounted (solved OR honestly unreachable): {len(accounted)}/{total} "
              f"({accounted_share:.2%}), NOT GATED - {len(unreachable)} of it is the "
              f"unreachable bucket, so it barely moves when the search does; it only "
              f"falls when the LP stops answering, which the undecided assertion "
              f"catches directly")

        # An LP that could not answer is neither a reachable nor an unreachable
        # target: it is a case nobody judged, and it costs the accounted share
        # exactly like an unsolved one, so it is named rather than folded in.
        for row in undecided:
            print(f"    UNDECIDED id={row['glazy_id']:<8} "
                  f"feasibility said {row.get('feasibility_error') or 'nothing at all'} "
                  f"{row['name'][:48]}")

        by_bucket = {}
        for row in results:
            entry = by_bucket.setdefault(row['bucket'], [0, 0, 0, 0])
            entry[0] += 1
            entry[1] += 1 if row['feasible'] else 0
            entry[2] += 1 if bench_corpus.chemistry_ok(row) else 0
            entry[3] += 1 if bench_corpus.accounted_ok(row) else 0
        for bucket in sorted(by_bucket):
            count, reach, chem, acct = by_bucket[bucket]
            print(f"  bucket {bucket:<4} n={count:<4} reachable {reach:<4} chemistry {chem:<4} "
                  f"accounted {acct}")

        self._log_misclassifications(results)
        self._log_cost(results)

        crashed = [row for row in results if row['status'].startswith('failed: ')
                   and 'Error' in row['status']]
        self.assertEqual(
            [], [(row['glazy_id'], row['status']) for row in crashed],
            f'{engine}: the engine raised on some cases')

        # An LP that could not answer judged nothing, so the case is neither
        # reachable nor unreachable and simply leaves the denominator of the
        # gate below - which means the gate cannot see it. The accounted share
        # used to catch this indirectly and only in bulk (see
        # MIN_SOLVED_AMONG_REACHABLE); this catches the first one. Measured 0
        # of 100 on the pinned seed and on four others, so any nonzero value is
        # a change of behaviour and worth a red test.
        self.assertEqual(
            [], [(row['glazy_id'], row.get('feasibility_error')) for row in undecided],
            f'{engine}: feasibility failed to answer on some targets, so nothing '
            f'judged whether our stock can reach them')

        # The gate is the reachable half. A target the LP declares out of reach
        # of our stock is correctly declined and says nothing about the search;
        # a target the LP proves reachable and the search does not reach is the
        # only thing in this scenario that can be a regression.
        self.assertTrue(reachable, 'the LP declared no target reachable at all, '
                                   'so nothing here measures the search')
        solved_reachable_share = len(solved_reachable) / len(reachable)
        self.assertGreaterEqual(
            solved_reachable_share, MIN_SOLVED_AMONG_REACHABLE,
            f'{engine}: the search reached only {len(solved_reachable)} of the '
            f'{len(reachable)} targets the LP proved reachable '
            f'({solved_reachable_share:.2%}), the threshold is '
            f'{MIN_SOLVED_AMONG_REACHABLE:.0%}')

    @staticmethod
    def _log_misclassifications(results):
        """
        Print every case where the feasibility gate and the solver disagree

        In full, with ids, and never truncated: this is the output of the
        scenario, not a statistic about it. Each line is a bug report against
        one of the two, and which one it is has to be read from the numbers on
        the line - the deviation the LP measured, the error the solver reached
        and the oxides the LP named.
        """
        disputes = {'reachable_unsolved': [], 'unreachable_solved': []}
        for row in results:
            verdict = bench_corpus.misclassification(row)
            if verdict:
                disputes[verdict].append(row)

        headings = {
            'reachable_unsolved':
                'LP said REACHABLE and the solver did not get there. Since 10.18 '
                'this is a bug report against the SEARCH and nothing else: the LP '
                'proved a point inside the tolerance on every oxide exists',
            'unreachable_solved':
                'LP said UNREACHABLE and the solver passed the chemistry gate anyway. '
                'Since 10.18 the two are the same measurement at the same tolerance, '
                'so a case here is either an unsound LP or a target sitting on the '
                'boundary within the rounding quantum - weights_to_umf rounds to 0.001, '
                'which is 20% of the tolerance at the scale floor - and which one it is '
                'has to be read off the numbers on the line',
        }

        for key, rows in disputes.items():
            print(f"\n  {len(rows)} misclassified - {headings[key]}:")
            if not rows:
                print('    none')
                continue
            rows.sort(key=lambda row: -(row['max_relative']
                                        if row['max_relative'] is not None else 1e9))
            for row in rows:
                achieved = '-' if row['max_relative'] is None else f"{row['max_relative']:.4f}"
                error = '-' if row['umf_error'] is None else f"{row['umf_error']:.4f}"
                print(f"    id={row['glazy_id']:<8} LP={row['max_relative_deviation']:.4f} "
                      f"achieved={achieved:<8} l2={error:<8} status={row['status']:<20} "
                      f"size={row['size']:<3} "
                      f"unreachable={row['unreachable_oxides']} {row['name'][:40]}")

    @staticmethod
    def _log_cost(results):
        """
        The cost half, which exists only in this scenario

        Reported as a share and a whole distribution rather than a mean: this is
        the first benchmark run in which assembly_score has values at all, so
        the shape of it is the finding.
        """
        solved = [row for row in results if row['status'] == 'solved']
        priced = [row for row in solved if row['cost_abs'] is not None]
        unpriced = [row for row in solved if row['cost_abs'] is None]

        print(f"\n  fully priced: {len(priced)}/{len(solved)} solved answers "
              + (f"({len(priced) / len(solved):.2%})" if solved else ""))
        print(_profile_line('cost_abs', [row['cost_abs'] for row in priced]))
        print(_profile_line('assembly_score', [row['assembly_score'] for row in priced]))
        print(_profile_line('count', [row['count'] for row in solved if row['count'] is not None]))
        print(_profile_line('min_portion',
                            [row['min_portion'] for row in solved if row['min_portion'] is not None]))
        print(_profile_line('junk_count',
                            [row['junk_count'] for row in solved if row['junk_count'] is not None]))
        print(_profile_line('rounding_drift',
                            [row['rounding_drift'] for row in solved if row['rounding_drift'] is not None]))
        print(_profile_line('cond', [row['cond'] for row in solved if row['cond'] is not None]))

        if unpriced:
            print(f"    {len(unpriced)} answers have no cost at all: "
                  f"{[row['glazy_id'] for row in unpriced]}")

    def test_the_price_list_is_what_makes_this_scenario_the_priced_one(self):
        """
        18 of our 19 materials are priced, and no scenario A answer can be

        The whole difference between "assembly_score is None everywhere" in
        scenario A and "assembly_score is a number" in scenario B rests on one
        property of the data, and nothing in either scenario states it - the
        cost metric simply abstains when a material has no price. Pinned here,
        because a price appearing for wood ash would silently move every cost
        aggregate of the baseline with no solver change behind it.

        The second half is not the tautology it looks like. The two name spaces
        are NOT disjoint: five Russian names are in the Glazy dump as well,
        because part of the SegerLab catalogue our own database came from was
        uploaded there. What keeps scenario A unpriced is therefore not "no
        Glazy material has a price" - five of them do - but the weaker and
        actually true statement that no scenario A recipe is made ENTIRELY of
        them, since cost_abs needs every component priced. Two cases of the 300
        contain one such material each. If a dump recipe ever appeared that was
        built only from those five, scenario A would start reporting costs and
        its "assembly_score is None" invariant would break where nobody was
        looking.
        """
        prices = qm.load_prices()
        names = [material['name'] for material in self.inventory]

        unpriced = sorted(name for name in names if name not in prices)
        self.assertEqual(
            1, len(unpriced),
            f'exactly one inventory material is expected to have no price, got {unpriced}')
        print(f"\n  the one unpriced inventory material: {unpriced[0]}")

        shared = _shared_corpus()
        glazy_names = {record['name'] for case in shared['sample'] for record in case['materials']}
        collisions = sorted(glazy_names & set(prices))
        print(f"  Glazy material names that carry one of our prices: {collisions}")

        fully_priced = [case['glazy_id'] for case in shared['sample']
                        if all(name in prices for name in case['original'])]
        self.assertEqual(
            [], fully_priced,
            'a scenario A case is fully priced, so its cost_abs and assembly_score are '
            'no longer None and the two scenarios no longer differ in the way the '
            'baseline assumes')

    def test_the_cost_ratio_against_the_dump_recipe_never_materializes(self):
        """
        Absolute cost only: a ratio here would compare two countries, not two recipes

        The original is American and frit based, ours is Russian and raw
        material based, so a cost ratio between them measures supply chains. The
        benchmark does not compute one, and the reason it cannot is structural
        rather than a rule anybody has to remember: no material of the dump has
        a price, so the original side of the ratio is undefined. That is worth a
        test precisely because it is invisible - nothing would fail if it broke,
        a number would simply appear.
        """
        prices = qm.load_prices()
        # Every priced material of our stock in equal shares - a recipe nobody
        # would fire, but a fully priced one, which is what this is about
        priced_names = [material['name'] for material in self.inventory
                        if material['name'] in prices]
        recipe = {name: 100.0 / len(priced_names) for name in priced_names}

        checked = 0
        for case in self.sample[:20]:
            materials = bench_corpus.quality_materials(case, bench_corpus.SCENARIO_B)
            quality = qm.solution_quality(recipe, case['original'], materials, prices=prices)

            self.assertIsNone(quality['cost']['original'],
                              f"case {case['glazy_id']}: the dump recipe must have no cost")
            self.assertIsNone(quality['cost']['ratio'],
                              f"case {case['glazy_id']}: a cost ratio must not be computable")
            self.assertIsNone(quality['cost']['ok'])
            self.assertNotIn('cost', quality['failures'])

            # ... while OUR side of it is a real number, which is the point
            self.assertIsNotNone(quality['cost']['cost_abs'],
                                 f"case {case['glazy_id']}: our own recipe must be priced")
            checked += 1

        self.assertEqual(20, checked)


if __name__ == '__main__':
    unittest.main()
