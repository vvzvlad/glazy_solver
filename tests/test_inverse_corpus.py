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
Corpus benchmark on the Glazy dump - scenario A (TZ_SOLVER_V2.md 7.6)

Scenario A reproduces a real recipe FROM ITS OWN MATERIALS. The target is
reachable by construction, so nothing here measures whether a solution exists:
it measures how good the one that comes back is. Every part of the setup exists
to remove an excuse:

  * the inventory is the recipe's own materials, injected through the
    materials= seam of both engines, so a miss cannot be blamed on the stock;
  * the target is OUR forward calculation of the merged recipe, so neither the
    flux convention nor Glazy's own normalization can influence the score;
  * the original recipe of the dump is the baseline of the quality half - an
    existence proof, not the right answer (7.1).

The verdict is two-level:

  1. chemistry - calculate_umf_error <= 0.1;
  2. quality   - quality_metrics.solution_quality(...).failures == [].

Both shares are reported, and so is the GAP between them: that gap is the share
of answers that are chemically correct and still bad, which is the one number
this benchmark exists to produce.

The test is slow and needs an 8 MB download, so it is skipped unless
GLAZY_CORPUS is set, and it skips itself - never fails - when the dump is
neither on disk nor reachable. The dump is CC BY-NC-SA data: it lives in
~/.cache/glazy_solver/ and never inside the repository.

    GLAZY_CORPUS=1 python -m unittest tests.test_inverse_corpus -v
    GLAZY_CORPUS=1 GLAZY_DUMP_PATH=/some/glazy.yaml.gz python -m unittest ...

Scenario B (our inventory against a foreign chemistry) is not implemented here.
"""

import os
import sys
import time
import unittest

# Fix imports by adding parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import quality_metrics as qm
from bench import corpus as bench_corpus


# The benchmark is opt-in. Without this the module contributes nothing but skips
# to `python -m unittest discover tests`, and in particular it never imports
# pyyaml, downloads anything or reads the dump.
CORPUS_ENV = 'GLAZY_CORPUS'

# Share of the sample that has to pass BOTH levels. TZ_SOLVER_V2.md 7.6 names
# 90% as a starting point "to be fixed after the first run"; the first run gave
# 93.00% for the iterative engine over 300 cases and 96% for the classic engine
# over its 50 case subsample, so 90% is kept, with a margin of 3.00 points
# (9 cases) and 6 points (3 cases) respectively.
#
# The margin has to absorb sampling luck, and it does: over five seeds the
# iterative share ranged 93.00 - 95.67% and the classic one 94 - 100%, the
# pinned seed being the least lucky of the five. It does NOT have to absorb
# solver changes - those are what bench/diff_baseline.py measures, case by case,
# against a committed snapshot.
MIN_BOTH_LEVELS_SHARE = 0.90

# How many failing cases are printed in full. All of them are counted; the log
# would otherwise be unreadable on a bad run.
MAX_LOGGED_FAILURES = 25


def _corpus_enabled():
    """Whether the corpus benchmark was explicitly asked for"""
    value = os.environ.get(CORPUS_ENV, '')
    return value.strip().lower() not in ('', '0', 'false', 'no')


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
            corpus, timings = bench_corpus.load_corpus()
        except bench_corpus.CorpusUnavailable as exc:
            # No dump and no network is a skip, never a failure: a benchmark
            # that needs an 8 MB download must not redden an offline suite.
            raise unittest.SkipTest(f'Glazy corpus unavailable: {exc}')

        cls.corpus = corpus
        cls.load_timings = timings

        start = time.perf_counter()
        cls.cases, cls.rejected = bench_corpus.build_cases(corpus)
        build_seconds = time.perf_counter() - start

        cls.sample = bench_corpus.stratified_sample(
            cls.cases, size=bench_corpus.DEFAULT_SAMPLE_SIZE, seed=bench_corpus.DEFAULT_SEED)
        cls.classic_sample = bench_corpus.subsample(
            cls.sample, bench_corpus.DEFAULT_CLASSIC_SUBSAMPLE, seed=bench_corpus.DEFAULT_SEED)

        buckets = {}
        for case in cls.sample:
            buckets[case['bucket']] = buckets.get(case['bucket'], 0) + 1

        print(f"\nGlazy corpus: dump {timings['dump']} ({timings['dump_size']} bytes), "
              f"loaded from {timings['source']} in {timings['seconds']:.2f}s, "
              f"cases built in {build_seconds:.2f}s")
        print(f"  corpus: {corpus['stats']}")
        print(f"  rejected while building cases: {cls.rejected}")
        print(f"  sample: {len(cls.sample)} cases, seed {bench_corpus.DEFAULT_SEED}, "
              f"buckets {buckets}; classic subsample {len(cls.classic_sample)}")
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

        # Worst chemistry first; a case that produced no recipe at all leads
        failures.sort(key=lambda row: -(row['umf_error'] if row['umf_error'] is not None else 1e9))

        print(f"  {len(failures)} failing cases for {engine}:")
        for row in failures[:MAX_LOGGED_FAILURES]:
            if row['status'] != 'solved':
                print(f"    id={row['glazy_id']:<8} level=solve   {row['status']:<28} "
                      f"size={row['size']:<3} {row['name'][:48]}")
                continue

            if not row['chemistry_ok']:
                print(f"    id={row['glazy_id']:<8} level=chemistry metric=calculate_umf_error "
                      f"value={row['umf_error']:.4f} limit={bench_corpus.MAX_UMF_ERROR} "
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


if __name__ == '__main__':
    unittest.main()
