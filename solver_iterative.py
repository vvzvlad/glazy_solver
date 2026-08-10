#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# flake8: noqa
# pylint: disable=broad-exception-raised, raise-missing-from, too-many-arguments, redefined-outer-name
# pylint: disable=multiple-statements, logging-fstring-interpolation, trailing-whitespace, line-too-long
# pylint: disable=broad-exception-caught, missing-function-docstring, missing-class-docstring
# pylint: disable=f-string-without-interpolation
# pylance: disable=reportMissingImports, reportMissingModuleSource
# type: ignore

"""
Iterative glaze recipe solver.

The algorithm mimics the way a human ceramist works:

1. Start from the highest priority materials of the inventory (whole priority
   groups are taken one by one until the starting set is big enough).
2. Solve the mix with NNLS in WEIGHT space (the same math the classic solver
   uses: target UMF -> weights, NNLS over the material formulas, result back
   to UMF), drop materials weighing less than MIN_MATERIAL_WEIGHT and re-solve.
   The rows of that fit are WEIGHTED so that the residual it minimizes is the
   relative per-oxide deviation the rest of the system judges by, rather than
   the weight percentage it is posed in - see _build_problem for the derivation
   and OBJECTIVE_DEADBAND for what the numbers mean.
3. Look at the per-oxide residual, find the oxide that is the furthest away
   from the target - the focus oxide of this step.
4. Rank the materials that are not in the set yet by a score that is built from
   three separate terms: the gain on the focus oxide, a smaller gain on the
   remaining deficits and a penalty for contaminating the oxides that already
   match or are already in excess. Material priority is blended into that score,
   so a high priority material wins unless a low priority one is clearly better.
5. Actually solve the top candidates (see candidate_search) and keep the ones
   that really improve the recipe, then go back to step 2.

How much of step 4 survives depends on candidate_search, and the difference is
worth stating plainly:

* 'heuristic' solves only the TOP_CANDIDATES best ranked materials, so the focus
   oxide and the priority genuinely decide what is tried. This is the human
   procedure, and it costs a constant number of NNLS runs per step.
* 'exhaustive' (the default) solves every remaining material of the inventory,
   which makes the ranking a tie break rather than a filter: it decides between
   material sets whose error agrees to four decimals, and nothing more. The
   focus oxide and the priority therefore do not steer this mode.

The default is 'exhaustive' because on the reference set it is measurably more
accurate - the heuristic only matches it once K grows to about HALF the
inventory (see find_best_recipe for the measured numbers), at which point it has
stopped being a shortcut and costs about the same as the exhaustive pass anyway.
The honest summary is that greedy forward selection beats the human shortcut
here; the shortcut is kept, and named, for the cases where the inventory is
large enough that O(inventory) NNLS runs per step hurt.

A branch stops when its error drops below the threshold and the pool already
holds as many acceptable recipes as the caller asked for, when the material
limit is reached, when the iteration limit is reached, or when the error stops
improving (less than 1% per iteration). Nothing is ever lost: every solved set
goes into a pool and the best states are picked from it at the end, which is
the rollback to the best state found.

When more than one solution is requested the candidate step feeds several
children into the beam and the search turns into a beam search over several
branches.

The search itself only ever ADDS materials, so the last thing that happens to a
recipe is _prune_solution(): greedy backward elimination that drops every
material whose removal does not make the fit worse. That is where noise
components go - not into a weight threshold, which cannot tell a rounding
artefact apart from half a percent of cobalt carbonate.
"""

import argparse
import json
import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from common import (
    DEFAULT_PRIORITY,
    NON_OXIDE_KEYS,
    OXIDE_SCALE_FLOOR,
    filter_materials_by_inventory,
    filter_materials_with_formula,
    flux_oxides,
    load_molar_masses,
    resolve_inventory,
    resolve_material_pool,
    umf_to_weights,
    weights_to_umf,
)
from solver_classic import (
    calculate_recipe_composition,
    calculate_umf_error,
    create_oxide_matrix,
    solve_recipe,
)

logger = logging.getLogger(__name__)

# Numerical floor of the NNLS solution, NOT a way to keep the recipe short.
# A weight this small is the solver's own noise - a column that got a sliver of
# mass while fitting the fourth decimal of an oxide - and 0.1 is where
# solver_classic.solve_recipe already cuts, so the two agree.
#
# THIS NUMBER MUST NOT BE RAISED. It was 1.0 for one commit, on the argument
# that one gram of a 100 g batch is what a studio scale can still weigh, and it
# was reverted because a weight threshold cannot tell a numerical artefact apart
# from an ingredient that is genuinely tiny. Measured on a target built from
# feldspar 39.5 / quartz 30 / chalk 20 / kaolin 10 / cobalt carbonate 0.5, with
# the cobalt in the inventory:
#
#   MIN=0.1  err 0.0030  {kaolin 9.99, quartz 30.01, chalk 20.02,
#                         feldspar 39.48, CoCO3 0.5}
#   MIN=1.0  err 0.0575  {kaolin 10.04, quartz 30.16, chalk 20.12, feldspar 39.68}
#
# 0.0575 is the ABSOLUTE norm, and it is below the default error_threshold of
# 0.1, so the second answer used to come back as acceptable with nothing in it
# saying that an ingredient was dropped: a blue glaze silently becomes a clear
# one. Colourants are exactly the class of material that weighs the least and
# matters the most. The cut was fuzzy on top of that - a component genuinely
# needed at 1.00% went too, because the first NNLS estimate landed just below
# the floor and _solve_material_set then removed the material from `active` for
# good.
#
# The same answer scores 0.1300 on the relative objective this module minimizes
# now, because the CoO it dropped is missed by 0.15 of its own scale, so it no
# longer passes the threshold and the branch is no longer declared converged.
# That closes the "returned as acceptable" half of the defect and nothing else:
# a weight floor of 1.0 would still throw the cobalt away, it would just be
# reported honestly. The floor stays at 0.1.
#
# Keeping a recipe free of pointless components is the job of _prune_solution(),
# which asks whether a material is the only source of something the target asked
# for, and failing that whether removing it makes the fit worse - two questions
# that have nothing to do with how much it weighs.
MIN_MATERIAL_WEIGHT = 0.1

# --- the scale everything below is measured on -------------------------------
#
# _objective_error returns the L2 norm of the RELATIVE per-oxide deviations,
# |target - actual| / max(target, common.OXIDE_SCALE_FLOOR), which is the metric
# the feasibility LP minimizes and the benchmark gate accepts at 0.05. Every
# threshold compared against that number - the error_threshold argument,
# PRUNE_OBJECTIVE_TOLERANCE, SOLUTION_ERROR_TIE_ABS - is therefore in RELATIVE
# units and was re-derived by measurement when the scale changed, not converted
# by a factor. PRUNE_ERROR_TOLERANCE is the odd one out: it bounds `error`, the
# retired absolute L2, whose scale did not move.
#
# STALL_IMPROVEMENT and SOLUTION_ERROR_TIE_REL are ratios of the objective to
# itself, so the change of scale cannot touch them.
#
# --- the deadband ------------------------------------------------------------
#
# How much relative deviation on one oxide is free. TUNABLE POLICY, NOT PHYSICS.
#
# What it buys is weighability. Driving an oxide that is already inside its
# tolerance the rest of the way to zero costs a material, and 27 of the 300
# scenario A cases paid one for exactly that - a deviation moved from 0.04 to
# 0.02, an improvement no threshold and no potter can see. This is
# UI_DECISIONS.md 2.2's passenger idea generalised from the unrequested oxides
# to all of them: an oxide already inside its tolerance is not an error to be
# minimized.
#
# EVERY SWEEP TABLE IN THIS MODULE NAMES ITS WHOLE PINNED SET, because a table
# that does not cannot be reproduced: the first version of this one was taken
# while SOLUTION_ERROR_TIE_ABS was still 0.01, said only "at the old pruning
# budget", and a reader pinning the shipped 0.02 instead got median min_portion
# 2.70 where the table says 2.35. "Everything else at its shipped value" is the
# default below and the deviations are spelled out.
#
# Swept twice on the 300 case scenario A corpus at max_solutions=5, exhaustive,
# paired case by case against the answers of commit 1f521c9. First with
# PRUNE_OBJECTIVE_TOLERANCE = PRUNE_ERROR_TOLERANCE = 0.005 and
# SOLUTION_ERROR_TIE_ABS = 0.01 - the pre-10.19 budget, which isolates the
# deadband from the widening of the pruning pass:
#
#   deadband   rel<=0.05   mean count   median min_portion   count worse/better
#     none        296         5.960          2.350                 76 / 0
#     0.01        296         5.887          2.905                 61 / 1
#     0.02        296         5.833          2.905                 53 / 4
#     0.03        295         5.817          2.905                 50 / 5
#     0.04        295         5.787          2.915                 48 / 9
#     0.05        287         5.757          2.920                 45 / 12
#
# and again with everything else at its shipped value, where the last column
# also carries the scenario B score:
#
#   deadband   rel<=0.05   mean count   median min_portion   worse/better   B
#     none        296         5.797          2.960             52 / 9      10/10
#     0.01        296         5.760          2.995             49 / 16      9/10
#     0.02        296         5.730          3.050             47 / 22     10/10
#     0.03        291         5.697          3.075             45 / 25      9/10
#     0.04        281         5.653          3.215             40 / 32      5/10
#     0.05        273         5.623          3.295             35 / 33      5/10
#
# Two things the second table says that the first cannot:
#
# * the deadband MUST be narrower than the acceptance tolerance the answer is
#   graded by (0.05, common to the LP and the benchmark gate), because the
#   pruning pass spends its own budget ON TOP of the band. Inside the band every
#   removal already reads as free; give the pass 0.03 more and a deadband of
#   0.03 already walks 5 of 300 cases out the far side of the gate, 0.04 walks
#   15 and takes half of scenario B with it. The first table, measured at a
#   pruning budget of 0.005, put that cliff at 0.05 - so the width that is safe
#   depends on what the pass is allowed to spend afterwards, and the two have to
#   be read together;
# * 0.02 is the knee. It keeps the full chemistry of the deadband-free version
#   (296 of 300, and all ten reachable scenario B targets) and recovers most of
#   the weighability the relative weights cost: against today's answers the
#   paired count regression is 47 worse / 22 better, where the relative weights
#   with no deadband and the old pruning budget cost 76 worse / 0 better.
#
# The tempting rule "deadband + pruning budget <= 0.05" is NOT a law, and it was
# checked rather than assumed. Five splits that all sum to 0.05, everything else
# shipped, score on the gate and on scenario B:
#
#   deadband + PRUNE_OBJECTIVE_TOLERANCE   rel<=0.05   B solved/reachable
#     0.00 + 0.05                             294            7/10
#     0.01 + 0.04                             295            8/10
#     0.02 + 0.03                             296           10/10
#     0.03 + 0.02                             295           10/10
#     0.04 + 0.01                             290           10/10
#
# The sum is a ceiling worth knowing; where the budget sits inside it is a
# measurement, and 0.02 + 0.03 is the best point on that line.
OBJECTIVE_DEADBAND = 0.02

# How much a removal may cost the SEARCH OBJECTIVE, per removal, against the
# state it starts from. RELATIVE UNITS, so this number is not comparable with
# the absolute one below and was measured rather than converted.
#
# Swept on the 300 case scenario A corpus at max_solutions=5, exhaustive,
# everything else at its shipped value EXCEPT the reported-error gate below,
# pinned at its old 0.005 so that this axis is read on its own:
#
#   tolerance   rel<=0.05   mean count   median min_portion   B solved/reachable
#     0.005        296         5.817         2.905                 10/10
#     0.01         296         5.817         2.905                 10/10
#     0.02         296         5.817         2.905                 10/10
#     0.03         296         5.813         2.905                 10/10
#     0.04         295         5.807         2.910                 10/10
#     0.05         293         5.800         2.910                 10/10
#     0.1          293         5.800         2.910                 10/10
#
# and the same sweep with the two gates moved TOGETHER, everything else shipped,
# which is what a single constant would have done and is where the cliff is:
#
#     0.005        296         5.817         2.905                 10/10
#     0.02         296         5.753         2.995                 10/10
#     0.03         296         5.730         3.050                 10/10
#     0.04         295         5.710         3.105                  9/10
#     0.05         285         5.663         3.250                  8/10
#     0.1          263         5.567         3.375                  7/10
#
# 0.03 is the knee on both readings: the last value that keeps the whole
# chemistry (296 of 300, and all ten reachable scenario B targets). Past it the
# pass starts spending the acceptance tolerance itself - the deadband already
# gives every oxide 0.02 for free, so 0.05 on top of that is a licence to walk a
# recipe out of the gate one removal at a time.
#
# The first table is also the reason the two constants exist. Moving this axis
# alone buys almost nothing (mean count 5.817 -> 5.813) because the absolute
# gate below is the binding one, while moving it alone past 0.03 is what breaks
# the chemistry. The two gates have different safe ranges, and one numeral would
# hide that: somebody raising "the pruning tolerance" to 0.05 would lose eleven
# cases and two reachable scenario B targets without being told which half of it
# did the damage.
#
# THIS is the gate that protects a trace ingredient, and it does it far better
# than the absolute one ever could, which is the reason the absolute one below
# could be relaxed. Same base and same colourants as the table under
# PRUNE_ERROR_TOLERANCE, measured again on both numbers:
#
#   material                       %   d error   d objective
#   Карбонат кобальта, CoCO3     1.0    0.1196        0.2815
#   Карбонат кобальта, CoCO3     0.2    0.0242        0.0400
#   Оксид никеля зеленый, NiO    0.3    0.0545        0.1300
#   Оксид хрома, Cr2O3           1.0    0.0230        0.2200
#   Оксид хрома, Cr2O3           0.3    0.0063        0.0500
#   Хромат железа, FeCr2O4       0.5    0.0064        0.0424
#   Оксид железа красный, Fe2O3  0.3    0.0034        0.0400
#
# The last row is the point: 0.3% of red iron oxide costs 0.0034 on the absolute
# norm, under the 0.005 the pass used to allow, and was thrown away; on the
# relative objective it costs 0.0400, over this tolerance, and is kept. Every
# row of the table is blocked here at 0.03 and several of them were not blocked
# at 0.005 there. It is still not a colourant rule - see PRUNE_ERROR_TOLERANCE
# for why no error threshold can be one - but it is no longer the wrong quantity
# by an order of magnitude.
PRUNE_OBJECTIVE_TOLERANCE = 0.03

# How much a removal may cost the REPORTED error, per removal, against the state
# it starts from. ABSOLUTE units: `error` is calculate_umf_error, the L2 of the
# absolute UMF deviations, and it kept its scale when the objective changed.
# One numeral for both gates would be one ruler for two scales, which is why
# there are two constants where there used to be one.
#
# What this tolerance is for is recognising FIT NOISE: a material the search
# added to shave a fourth decimal off one oxide, which by construction is not
# the only thing in the recipe carrying that oxide. It is now the BACKSTOP of
# the pass rather than its main gate - the relative tolerance above is the one
# that decides - and it was re-measured in that role. Swept with everything else
# at its shipped value, so the objective gate sits at 0.03 throughout:
#
#   tolerance   rel<=0.05   mean count   median min_portion   B solved/reachable
#     0.005        296         5.813         2.905                 10/10
#     0.01         296         5.777         2.925                 10/10
#     0.02         296         5.737         3.030                 10/10
#     0.03         296         5.730         3.050                 10/10
#     0.05         296         5.717         3.075                 10/10
#     0.1          296         5.713         3.075                 10/10
#
# Nothing on this axis touches the chemistry - the relative gate is already
# holding that line - and the weighability saturates at 0.03, so that is where
# it sits. At 0.03 the cap is at most 1% of a SiO2 of 3.0, a fifth of the 0.05
# the whole system accepts, so it is still the conservative of the two.
#
# The old value was 0.005, argued from "0.005 of UMF error is invisible in a
# fired glaze, smaller than the 0.0033 that the published percentages of a
# textbook recipe already cost by being rounded to one decimal". That argument
# is sound and is exactly why this gate cannot be the main one: 0.005 of
# absolute L2 is 0.17% of a SiO2 of 3.0 and 10% of an MgO of 0.05, one numeral
# meaning two completely different amounts of chemistry.
#
# What this tolerance is NOT for, at any calibration, is protecting a colourant,
# and it is worth being explicit about that because the first version of this
# pass claimed otherwise. Removing 0.5% of cobalt carbonate costs 0.0545, ten
# times the old tolerance, which looked like proof that the test protects
# colourants. It is not: cobalt is a FLUX, so losing it drags the unity
# denominator of the whole formula and inflates the measured cost. Colourants
# that are not fluxes get no such amplification, and the same measurement on the
# same base says so (re-measured under the relative row weights, which change
# the fit itself and moved a few of these rows in the third decimal):
#
#   material              oxide          1.0%    0.5%    0.3%    0.2%
#   Карбонат кобальта     CoO (flux)   0.1196  0.0545  0.0351  0.0242
#   Оксид никеля          NiO (flux)   0.1824  0.0964  0.0545  0.0302
#   Оксид хрома           Cr2O3        0.0230  0.0110  0.0063  0.0040
#   Хромат железа         Cr2O3/Fe2O3  0.0157  0.0064  0.0027  0.0020
#   Оксид железа красный  Fe2O3        0.0196  0.0102  0.0034  0.0022
#
# Everything at or below the tolerance in that table is a colourant this test
# throws away. Raising the number to catch them would only move the line to some
# other colourant, because the quantity being measured is the wrong one. A
# colourant works OPTICALLY and contributes almost nothing to the chemistry;
# that is what makes it a colourant. 0.5% of cobalt is the difference between a
# blue glaze and a clear one and 0.15% of chrome oxide between a green one and a
# clear one, and no threshold on UMF error can see that difference at any
# setting.
#
# So colourants are not protected by this number at all. They are protected by
# the sole carrier rule in _prune_solution, which asks a different question
# entirely and has no quantity in it.
PRUNE_ERROR_TOLERANCE = 0.03

# How many candidate recipes the pruning pass may work through, as a multiple of
# max_solutions. The pass must run before the sort and before the max_solutions
# cut (see find_best_recipe), which in principle means pruning every distinct
# recipe the pool holds - neither the objective nor the material count moves
# monotonically under pruning, so there is no admissible way to prove in advance
# that a candidate cannot reach the top max_solutions.
#
# In practice that is unaffordable. The pool holds one state per material set
# TRIED, and while many of those collapse onto the same recipe, what is left
# still scales with the catalogue: over the eleven reference recipes the pool
# holds 2 to 48 distinct recipes on the 19 material inventory and 14 to 267 on
# the whole 216 material one. Measured on recipe 03 over the full catalogue at
# max_solutions=5, pruning every distinct recipe costs 3959 scipy nnls calls
# against the 1369 of the unpruned search, and takes a POST /api/solve from
# 236 ms to 619 ms.
#
# The margin is the compromise: candidates are pruned in the order the UNPRUNED
# sort key gives, and the pass stops after max_solutions * this many of them
# (with one exception, see find_best_recipe). Three leaves room for a recipe to
# shrink past two others and still be returned, which a plain cut at
# max_solutions could not do. Same measurement with the margin: 1453 calls and
# 250 ms, so the pass costs about 6% over not pruning at all instead of 3x.
PRUNE_CANDIDATE_MARGIN = 3

# Whole priority groups are added to the starting set until it holds at least
# this many materials
DEFAULT_MIN_START_MATERIALS = 3

# A human converges in 5-6 steps, the default limit keeps some headroom
DEFAULT_MAX_ITERATIONS = 8

# A branch is considered stalled when one iteration improves the error by less
# than this relative amount
STALL_IMPROVEMENT = 0.01

# How many candidate materials are really solved per iteration in the
# 'heuristic' candidate search: the heuristic proposes, NNLS disposes
TOP_CANDIDATES = 3

# Objective difference below which two candidates of one expansion step count as
# equally good, so that the heuristic order - and through it the material
# priority - decides between them instead of the last decimal.
#
# It is named rather than written as a literal round(objective, 4) because it is
# a tolerance in the OBJECTIVE's own units, which makes it one of the numbers the
# move to the relative scale silently changed the meaning of. Written out: a
# relative deviation of 1e-4 spread over the oxides, two orders of magnitude
# under the deadband and three under the acceptance tolerance.
#
# Swept from 1e-4 to 2e-3 over the 300 case scenario A corpus, everything else
# shipped: not one case moves, on any metric. It stays at the old granularity
# because the measurement
# gives no reason to move it, and a number that was left alone should look like
# one. What the tie between two chemically interchangeable materials really
# turns on is _solution_sort_key, not this step - recipe 09 of the reference set
# is the worked example there.
CANDIDATE_TIE_STEP = 1e-4

# Maximum number of branches kept alive by the beam search
MAX_BEAM_WIDTH = 4

# Solutions whose objective is within this distance from the best one are
# treated as equally good, so that the material count decides between them.
#
# SOLUTION_ERROR_TIE_REL is a ratio of the objective to itself and is unaffected
# by the change of scale. SOLUTION_ERROR_TIE_ABS is in RELATIVE units like the
# objective and was re-measured. It is the floor of the band, and the deadband
# made it matter more than it used to: many states now score exactly 0, where
# best * SOLUTION_ERROR_TIE_REL is 0 too and this constant is the whole band.
#
# Swept with everything else at its shipped value, scenario A, 300 cases:
#
#   tie_abs   rel<=0.05   mean count   median min_portion   B solved/reachable
#     0.0        296         5.743         3.030                10/10
#     0.01       296         5.733         3.050                10/10
#     0.02       296         5.730         3.050                10/10
#     0.05       286         5.670         3.160                 8/10
#     0.1        259         5.563         3.375                 7/10
#
# Everything from 0 to 0.02 is one flat region - the choice inside it is not
# measurable, and saying otherwise would be inventing a result. What IS measured
# is the cliff at 0.05, which is the acceptance tolerance itself: a band that
# wide calls a recipe missing an oxide by the whole tolerance "as good as" one
# that hits it, and then prefers it for being shorter.
#
# 0.02 is picked inside the flat region because it is the deadband: an objective
# difference smaller than the amount of relative deviation this module has
# already declared free is not a difference. 0.01 measures the same; it is not a
# better number, it is the old numeral meaning something new.
SOLUTION_ERROR_TIE_REL = 0.2
SOLUTION_ERROR_TIE_ABS = 0.02

# --- unity basis ------------------------------------------------------------
#
# A UMF is a formula normalized so that the fluxes (R2O + RO) sum to 1, and
# weights_to_umf always produces such a vector for a recipe. A target does not
# have to be one: a target typed by hand can list no flux at all (SiO2 + Al2O3
# only), or list fluxes that add up to something other than 1. In that case the
# target and the recipe are normalized by two different quantities and comparing
# them oxide by oxide measures the difference of the two conventions, not the
# chemistry - the recipe for {SiO2: 3.0, Al2O3: 0.35} used to report SiO2 = 8.57
# and an error of 5.6 while having exactly the requested SiO2:Al2O3 ratio.
#
# The fix is to bring the recipe onto the basis of the target with a single
# scalar before comparing, and the two decisions behind it are:
#
# * WHEN. Only when the target is not a unity formula itself. If its fluxes do
#   sum to 1 the length of the target vector is meaningful and must not be
#   fitted away: a glaze carrying 1.5x the silica per unit of flux is a
#   different glaze, and scaling that difference out would hide it instead of
#   reporting it. So a proper UMF target is compared as is (scale 1.0), and only
#   a target that carries no basis of its own is fitted.
# * HOW. By least squares over the listed oxides: k = sum(target*result) /
#   sum(result^2) is the scalar that minimizes ||target - k*result||, and it
#   weights every oxide by its own magnitude. The alternative of pinning one
#   chosen oxide of the target puts the whole scale on that single component,
#   which is why it is not used: pinning a trace oxide such as Fe2O3 = 0.002
#   that the recipe misses by a factor of two would rescale the entire formula
#   by two. The classic solver used to do exactly that and no longer does.
UNITY_BASIS_TOLERANCE = 0.01

# --- candidate scoring ------------------------------------------------------
#
# The heuristic lives entirely in WEIGHT space, and so does the residual it
# works on. That is not an arbitrary choice:
#   * the NNLS problem itself is posed on weight percentages (the target UMF is
#     converted with umf_to_weights and the material formulas are weight
#     percent), so the residual of that very problem is a weight-space vector;
#   * a material formula answers "how many grams of oxide X does one gram of
#     this material carry", which is a weight-space quantity, while UMF is
#     renormalized by the flux sum and is therefore non-linear with respect to
#     mixing - "UMF gain per gram" is not even well defined;
#   * mixing the two spaces (focus picked in UMF, residual measured in weights)
#     was exactly the inconsistency the review found in the previous version.
# The reported error stays in UMF space, because that is the metric the callers
# and the acceptance tests speak; the heuristic only proposes candidates, the
# real NNLS solve decides.

# Deficits on oxides other than the focus one are worth less than the focus
# deficit: this step is about the focus oxide, the others get their own step
SECONDARY_GAIN_WEIGHT = 0.35

# Overshooting is asymmetrically bad: a deficit can be filled by adding another
# material later, an excess can never be subtracted, so contaminating an oxide
# that is already over the target costs more than filling a deficit gains
CONTAMINATION_WEIGHT = 1.5

# An oxide whose weight-percent residual is inside this band counts as matched
MATCHED_OXIDE_TOLERANCE = 0.5

# Bringing anything into an already matched oxide is a disturbance. Its residual
# is ~0 by definition, so without an explicit term the score would ignore it
# completely; MATCHED_OXIDE_TOLERANCE is used as the residual scale to keep the
# term dimensionally comparable with the gain terms
MATCHED_DISTURBANCE_WEIGHT = 0.5

# How strongly the material priority bends the chemical ranking. The chemical
# score is divided by the size of the focus gap, which is the score a material
# made purely of the focus oxide would get, so 0.25 means "a higher priority
# material wins unless the other candidate closes more than a quarter of the
# focus gap on top of what this one closes". That scale is absolute: it does not
# drift with the number of candidates the way a min-max normalization does.
PRIORITY_WEIGHT = 0.25

# Candidate search modes
SEARCH_HEURISTIC = 'heuristic'
SEARCH_EXHAUSTIVE = 'exhaustive'
CANDIDATE_SEARCH_MODES = (SEARCH_HEURISTIC, SEARCH_EXHAUSTIVE)


def _known_oxide(oxide: Any, molar_masses: Dict[str, float]) -> bool:
    """
    THE rule that decides whether a key can take part in the fit at all.

    A key is usable when molar_masses.json gives it a POSITIVE mass, and the two
    halves of that sentence are not the same test. The presence half is what
    keeps a key the system cannot express in UMF - Loi, an unanalysed "Carbon" -
    out of the oxide set. The positivity half is what keeps the row weight of
    _build_problem, penalty / (mass * scale), from dividing by zero.

    IT LIVES HERE BECAUSE IT IS APPLIED TWICE: usable_target cleans the
    requested target with it, _expand_target applies it to the target's keys and
    to the materials'. The two used to spell it differently - membership in one,
    a positive mass in the other - and they agreed only because all 64 entries
    of the shipped table happen to be positive. A zero or a NaN in that table
    would have split them: the target key would pass the clean, be refused by
    the expansion, and _build_problem narrows target_umf to whatever the
    expansion kept - so the reported error and the returned target would quietly
    have been about a smaller problem than the one that was asked for.

    NaN needs no branch of its own: NaN > 0.0 is False.
    """
    mass = molar_masses.get(oxide)
    return mass is not None and mass > 0.0


def usable_target(target_umf: Dict[str, Any]) -> Tuple[Dict[str, float], List[str]]:
    """
    Keep only the target entries the math can actually work with: a key
    _known_oxide accepts, carrying a finite, non negative number.

    An oxide asked for as an explicit ZERO is kept, and that is deliberate:
    'Fe2O3': 0.0 is a legal input to this system and it means "give me no iron",
    which is a constraint rather than the absence of an opinion. Dropping those
    entries used to move them into the unlisted group, where penalize_unlisted
    decides their fate - so with a soft weight "no iron please" silently turned
    into "iron is fine". A zero stays in the target, gets its NNLS row with a
    zero right hand side and is penalized like any other requested value, which
    is what the classic solver does too.

    Anything else is dropped: an unknown oxide, a non numeric value, a negative
    value, NaN and infinity.

    WHAT WAS DROPPED IS RETURNED, and that is the difference between a clean and
    a silent one. An oxide the caller asked for and did not get is the single
    thing it cannot deduce from the answer: a recipe comes back, the target it
    was graded against no longer holds the key, and the response used to say
    nothing at all - while /api/feasibility, given the very same target, has
    always answered "оксиды цели не распознаны и не учтены".

    PUBLIC, AND THAT IS THE POINT OF THE FUNCTION. /api/solve calls it ONCE, at
    the boundary, and hands the CLEANED dictionary to whichever engine was
    asked for. The alternative - each engine cleaning to its own taste - is what
    the endpoint used to do, and the two do not agree: the classic engine drops
    an unknown NAME (in umf_to_weights) but feeds a NEGATIVE VALUE straight into
    its NNLS, so a warning drawn from this rule sat next to a recipe fitted to
    'Fe2O3': -0.5 and changed every material in it. Cleaning above the fork
    makes the warning, the returned target_composition and the vector actually
    fitted the same object by construction rather than by coincidence.

    Loss on ignition is dropped as well but NOT reported: a target carrying Loi
    is bookkeeping that leaked in from a material analysis rather than an oxide
    somebody asked for, and feasibility._usable_target passes over it in silence
    for the same reason (pinned there by its own test, so the two cannot drift
    apart quietly).

    A ZERO SURVIVING THIS FUNCTION DOES NOT MAKE THE TARGET SOLVABLE. A formula
    of nothing BUT zeros is not a UMF - the unity it would be normalized by is
    the sum of its own oxides, which is why umf_to_weights divides by zero on
    it - and this function is not where that is caught, because it is not a
    property of any single entry. find_best_recipe answers such a target with an
    empty list; the interfaces refuse it outright and say why, /api/solve with
    422 "zero_target" and solver_classic.main() with a message.

    Returns:
        (usable, dropped) - the cleaned target, and the names refused, SORTED.
        Sorted rather than in the order they arrived, because the same refusal
        has to read the same way whichever endpoint reports it, and the order a
        JSON object's keys arrive in is a property of the client's serializer.
        feasibility._usable_target sorts for the same reason.
    """
    if not target_umf:
        return {}, []

    molar_masses = load_molar_masses()
    usable: Dict[str, float] = {}
    dropped: List[str] = []

    for oxide, value in target_umf.items():
        if oxide in NON_OXIDE_KEYS:
            continue
        if not _known_oxide(oxide, molar_masses):
            dropped.append(str(oxide))
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            dropped.append(str(oxide))
            continue
        if math.isfinite(number) and number >= 0.0:
            usable[oxide] = number
        else:
            dropped.append(str(oxide))

    return usable, sorted(dropped)


# The one wording for "the target asked for something that cannot be fitted",
# shared with the API layer. One HTTP request must produce ONE line about it,
# and grepping a log for that line must give the same count whichever engine ran
# and whichever layer noticed - two phrasings of one fact are two facts to
# anybody counting.
DROPPED_TARGET_OXIDES_LOG = "target oxides not recognized and not fitted: {oxides}"


def _flux_sum(umf: Dict[str, float]) -> float:
    """Sum of the fluxes (R2O + RO) of a formula - the UMF unity denominator"""
    fluxes = set(flux_oxides())
    return sum(float(value) for oxide, value in umf.items() if oxide in fluxes)


def _unity_scale(target_umf: Dict[str, float], result_umf: Dict[str, float]) -> float:
    """
    Scalar that brings the UMF of a recipe onto the normalization basis of the
    target, so that the two vectors can be compared oxide by oxide.

    Returns exactly 1.0 when both formulas are already unity formulas (their
    fluxes sum to 1 within UNITY_BASIS_TOLERANCE), which is the normal case: the
    target of a real recipe and every recipe the solver builds are both proper
    UMFs, so nothing is scaled and nothing is hidden.

    Otherwise - a target with no flux at all, a target whose fluxes do not add
    up to 1, or the rare recipe that carries no flux and is therefore normalized
    by weights_to_umf against its smallest oxide - only the direction of the
    target vector is meaningful, and the length is fitted by least squares:
    k = sum(target*result) / sum(result^2) minimizes ||target - k*result|| over
    the listed oxides. See the UNITY_BASIS_TOLERANCE block above for why the
    gate exists and why the least squares fit is preferred to pinning a single
    chosen oxide of the target.

    Falls back to 1.0 when the fit is degenerate (an empty or all zero result,
    or a result that shares nothing with the target): a non positive scale is
    not a formula.
    """
    target_fluxes = _flux_sum(target_umf)
    result_fluxes = _flux_sum(result_umf)

    if (abs(target_fluxes - 1.0) <= UNITY_BASIS_TOLERANCE
            and abs(result_fluxes - 1.0) <= UNITY_BASIS_TOLERANCE):
        return 1.0

    numerator = 0.0
    denominator = 0.0

    for oxide, expected in target_umf.items():
        actual = result_umf.get(oxide, 0.0)
        numerator += expected * actual
        denominator += actual * actual

    if denominator <= 0.0 or numerator <= 0.0:
        return 1.0

    return numerator / denominator


def _expand_target(target_umf: Dict[str, float],
                   materials: Sequence[Dict]) -> Tuple[Dict[str, float], List[str]]:
    """
    Build the oxide set of the fit: the requested target plus an explicit zero
    for every oxide the available materials can bring in.

    Whether those zeros are actually enforced is decided by the caller through
    penalize_unlisted: the expansion only builds the list of oxides, the weight
    attached to them says how much an unlisted oxide is allowed to appear.

    ONE RULE DECIDES WHAT GETS A ROW, and it is _known_oxide: a key with a
    positive molar mass gets a row, a key without one does not. It is applied to
    both sides here - the target's keys and the materials' - so that the oxide
    set, the row weights and the objective cannot disagree about what is being
    fitted. usable_target calls the same predicate on the target upstream;
    repeating it costs nothing and makes the guarantee local to this function.
    The predicate is shared rather than restated because the two spellings of it
    used to differ, and the divergence was invisible on the shipped table - see
    _known_oxide.

    WHY THE RULE IS THE MOLAR MASS AND NOT A LIST OF NAMES. The residual of row
    i is turned into a relative UMF deviation by dividing it by M_i * s_i (see
    _build_problem). A key with no molar mass cannot be expressed in UMF at all,
    so there is no relative deviation to compute and no principled weight to
    give it - in weight percent its weight is arbitrary. Under the relative
    weights an arbitrary weight is not a small mistake: a chemical row is
    1 / (M_i * s_i), which over this database runs from 0.005548 (SiO2 at a
    target of 3.0) to at most 0.5264 (fluorine at the scale floor, the lightest
    key in molar_masses.json), so a row left on the bare penalty of 1.0 is
    heavier than ANY chemical row - by 1.9x in the most favourable case and by
    180x in the common one (1 / 0.005548 = 180.2; the 182 this line used to
    carry was the reciprocal of the rounded 0.0055 rather than of the row).
    Measured on a wood-ash fixture whose only difference is an
    unanalysed "Carbon" key on one material, where the chemical rows run 0.0055
    to 0.0705 and the refused row would sit at 1.0:

        clean  {Ash 40.57, Kaolin 31.16, Silica 28.27}  objective 0.0117
        dirty  {Kaolin 33.51, Silica 43.49, Whiting 23.0} objective 0.4800

    The only carrier of the requested P2O5 is thrown out whole and the objective
    grows 41-fold, because minimizing an unnamed quantity outweighed the
    chemistry. Refusing the row instead makes the two runs identical.

    Loss on ignition is the case that occurs in practice, and it falls out of
    the same rule rather than needing a name list: Loi is bookkeeping - the mass
    a material loses in the kiln - and every verdict in the system already
    ignores it (weights_to_umf leaves it out of the formula, so do umf_deviation
    and the feasibility LP). Giving it a row with a zero right hand side was an
    instruction to MINIMIZE loss on ignition, a preference for oxides over
    carbonates that nobody asked for and nothing grades.

    MEASURED, because the first version of this comment guessed and was wrong by
    two orders of magnitude: 3 of the 216 materials of database/materials.json
    carry a Loi key - Дисульфид Молибдена, Метакаолин BMK-45 and Нефелин-сиенит
    А-270 - and all three are inInventory: false. That is exactly why neither
    measurement rig sees the row and a run over the full catalogue does: the
    default 19 material inventory excludes all three, and bench/corpus strips
    LOI from every Glazy formula before it builds a case. Only an explicit
    inventory naming one of the three, or an injected catalogue, reaches it.

    Returns:
        (full_target, refused) - the oxides that get a row, and the sorted names
        that were refused for having no molar mass. _build_problem logs them: a
        key silently dropped from the fit is a key nobody can debug.

        ON THE PRODUCTION PATH THAT LIST IS ABOUT THE MATERIALS, NOT ABOUT THE
        TARGET, and this docstring used to promise otherwise. find_best_recipe
        runs usable_target first, so by the time a target reaches this function
        its unknown keys are already gone and nothing here can report them; they
        are reported by usable_target instead, which is where the caller's own
        keys are actually refused. The target half of the rule still runs here,
        because the function is also called directly with a raw target and the
        oxide set may not be allowed to disagree with the row weights.
    """
    molar_masses = load_molar_masses()

    full_target = {oxide: value for oxide, value in target_umf.items()
                   if _known_oxide(oxide, molar_masses)}
    refused = {oxide for oxide in target_umf if oxide not in full_target}

    for material in materials:
        for key in material.get('formula', {}):
            if key in full_target:
                continue
            if not _known_oxide(key, molar_masses):
                refused.add(key)
                continue
            full_target[key] = 0.0

    return full_target, sorted(refused)


def _normalize_unlisted_weight(penalize_unlisted: Any) -> float:
    """
    Turn the penalize_unlisted argument into a weight in [0, 1].

    True/False are accepted as the hard 1.0 / 0.0 ends of the same scale, so
    that a boolean flag and a soft weight can be used interchangeably.

    Raises ValueError for anything that is not a finite number: a JSON null or a
    typo used to be silently turned into 1.0, which meant the caller got a
    completely different search than the one it asked for and never learned it.
    A finite number outside [0, 1] is clamped and logged - the ends of the scale
    are hard bounds ("must be zero" / "do not care"), there is nothing to
    extrapolate beyond them.
    """
    if penalize_unlisted is True:
        return 1.0
    if penalize_unlisted is False:
        return 0.0

    try:
        weight = float(penalize_unlisted)
    except (TypeError, ValueError):
        raise ValueError(f"penalize_unlisted must be a number in [0, 1] or a boolean, "
                         f"got {penalize_unlisted!r}")

    if not math.isfinite(weight):
        raise ValueError(f"penalize_unlisted must be a finite number in [0, 1], "
                         f"got {penalize_unlisted!r}")

    if weight < 0.0 or weight > 1.0:
        logger.warning(f"penalize_unlisted={weight} is outside [0, 1], clamped to the nearest bound")
        return min(max(weight, 0.0), 1.0)

    return weight


def _int_argument(value: Any, name: str) -> int:
    """
    Coerce one of the integer arguments, refusing what cannot be one.

    min_materials=None (a JSON null that made it through a caller) used to raise
    a bare TypeError from a comparison in the middle of the search; the caller
    now gets a ValueError that names the argument. A float is truncated, which
    is what int() has always done to max_materials here.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an integer, got {value!r}")


def _oxide_scale(target_value: float) -> float:
    """
    The scale one oxide's deviation is measured against: max(target, floor).

    common.OXIDE_SCALE_FLOOR is the single definition of that floor, and it is
    deliberately the same one the feasibility LP and the benchmark's chemistry
    gate use: the search, the verdict and the gate have to be measuring the same
    thing or the search optimizes something nobody is grading.
    """
    return max(float(target_value), OXIDE_SCALE_FLOOR)


def _build_problem(target_umf: Dict[str, float], materials: Sequence[Dict],
                   unlisted_weight: float) -> Dict[str, Any]:
    """
    Pack everything the search needs to know about the target into one context.

    THE ROW WEIGHTS ARE THE POINT OF THIS FUNCTION, so here is where they come
    from. The NNLS problem is posed in WEIGHT PERCENT of oxides, while every
    verdict in this system - the feasibility LP, the benchmark gate,
    common.umf_deviation - is the worst RELATIVE deviation of a UMF value,
    |actual - target| / max(target, OXIDE_SCALE_FLOOR). Unweighted, the fit
    therefore spends its effort where the weight percentages are large (SiO2,
    Al2O3) and cannot see an MgO of 0.05 being missed entirely. Three steps
    connect the two:

      * a deviation of d weight percent on oxide i is d / (M_i * S) in UMF
        units, where M_i is the molar mass and S is the flux molar sum the UMF
        is normalized by (weights_to_umf: umf_i = (w_i / M_i) / S);
      * S is the same for every row of one recipe, so it is a constant factor
        of the whole residual and drops out of the argmin. Only 1 / M_i has to
        be carried;
      * dividing by s_i = max(target_i, OXIDE_SCALE_FLOOR) turns the UMF
        deviation into the relative one the verdict is drawn on.

    which gives

        w_i = penalty_i / (M_i * s_i)

    with penalty_i = 1.0 for an oxide the target names and unlisted_weight for
    one it does not. THE PENALTY IS A FACTOR OF THIS, not a replacement for it:
    penalize_unlisted keeps exactly the meaning it had, it now rides on top of
    a residual that is already on the relative scale.

    THERE IS NO BRANCH FOR AN OXIDE WITHOUT A MOLAR MASS, and that is the
    point: 1 / (M_i * s_i) has no value to take, and the previous version left
    such a row on the bare penalty of 1.0 - heavier than every chemical row this
    database can produce, the heaviest of which is 0.5264. _expand_target
    refuses the key instead, so every oxide reaching this loop has a mass by
    construction and molar_masses[oxide] is indexed rather than probed - a
    KeyError here would be a broken invariant, not a data problem to paper over.

    The weights are handed to solve_recipe(row_weights=...), which scales the
    matrix and the right hand side together. That is not a detail: the unlisted
    rows have b_i = 0 and could be scaled matrix-only, but the listed ones
    cannot, and mixing the two would silently solve a different problem.
    """
    full_target, refused = _expand_target(target_umf, materials)
    if refused:
        logger.warning(f"no molar mass for {', '.join(refused)} - those keys get no NNLS row "
                       f"and no term of the objective, because a quantity the system cannot "
                       f"express in UMF has no relative deviation to minimize")

    # The requested target is narrowed to the same set, so that target_umf,
    # oxides and row_weights cannot describe three different problems
    target_umf = {oxide: value for oxide, value in target_umf.items() if oxide in full_target}

    oxides = list(full_target.keys())
    unlisted = tuple(oxide for oxide in oxides if oxide not in target_umf)

    molar_masses = load_molar_masses()
    row_weights = np.empty(len(oxides))
    for index, oxide in enumerate(oxides):
        penalty = 1.0 if oxide in target_umf else unlisted_weight
        row_weights[index] = penalty / (molar_masses[oxide] * _oxide_scale(full_target[oxide]))

    return {
        'target_umf': dict(target_umf),
        'full_target': full_target,
        'oxides': oxides,
        # Membership test of _weight_residual, which runs once per search step;
        # the list above is kept because the row order of the matrix is its order
        'oxide_set': frozenset(oxides),
        'unlisted': unlisted,
        'unlisted_weight': unlisted_weight,
        'row_weights': row_weights,
        'target_weights': umf_to_weights(full_target),
    }


def _objective_error(problem: Dict[str, Any], result_umf: Dict[str, float]) -> float:
    """
    The quantity the search minimizes: the L2 norm of the RELATIVE per-oxide
    deviations, each one first given OBJECTIVE_DEADBAND for free.

        residual_i = max(0, |target_i - actual_i| / s_i - OBJECTIVE_DEADBAND)
        objective  = sqrt(sum residual_i^2)

    with s_i = max(target_i, OXIDE_SCALE_FLOOR), the same scale _build_problem
    weights the NNLS rows by and the same one common.umf_deviation draws the
    verdict on. The unlisted oxides deviate from zero and carry unlisted_weight,
    exactly as before - only the scale of the term changed.

    THE NUMBER IS IN RELATIVE UNITS, and it is NOT the reported `error` by any
    other name - on a well fitted target of large oxides it comes out below it,
    on a target of trace oxides far above. Every threshold compared against it
    was re-derived by measurement rather than carried over: the error_threshold
    argument, PRUNE_OBJECTIVE_TOLERANCE, SOLUTION_ERROR_TIE_ABS.
    PRUNE_ERROR_TOLERANCE bounds `error` instead and stayed absolute.

    A consequence worth stating before somebody rediscovers it: with the
    deadband many states score exactly 0, they all land in one tie band, and
    _solution_sort_key then ranks them by material count. Among recipes that
    meet the spec the SHORTEST one wins - not because a preference was coded,
    but because every other difference between them became zero.
    """
    squared = 0.0
    target_umf = problem['target_umf']

    for oxide, expected in target_umf.items():
        deviation = abs(expected - result_umf.get(oxide, 0.0)) / _oxide_scale(expected)
        residual = deviation - OBJECTIVE_DEADBAND
        if residual > 0.0:
            squared += residual * residual

    weight = problem['unlisted_weight']
    if weight > 0.0:
        # An unlisted oxide is wanted at zero, so its scale is the floor
        for oxide in problem['unlisted']:
            deviation = abs(result_umf.get(oxide, 0.0)) / OXIDE_SCALE_FLOOR
            residual = deviation - OBJECTIVE_DEADBAND
            if residual > 0.0:
                squared += (weight * residual) ** 2

    return math.sqrt(squared)


def _normalize_to_100(composition: Dict[str, float]) -> Dict[str, float]:
    """Scale an oxide composition so that its parts sum up to 100"""
    total = sum(composition.values())
    if total <= 0:
        return {oxide: 0.0 for oxide in composition}
    return {oxide: value * 100.0 / total for oxide, value in composition.items()}


def _recipe_to_exactly_100(recipe: Dict[str, float]) -> Optional[Dict[str, float]]:
    """
    Scale a recipe to 100% and round it to two decimals so that the parts add up
    to exactly 100.

    Rounding every part on its own leaves a drift of up to half a hundredth per
    material (99.99 / 100.01 in practice); the drift is poured into the heaviest
    component, where it is relatively the least significant.
    """
    total = float(sum(recipe.values()))
    if total <= 0:
        return None

    scaled = {name: round(float(weight) * 100.0 / total, 2) for name, weight in recipe.items()}

    drift = round(100.0 - sum(scaled.values()), 2)
    if drift:
        # sorted() first, so that equal weights always pick the same material
        heaviest = max(sorted(scaled), key=lambda name: scaled[name])
        scaled[heaviest] = round(scaled[heaviest] + drift, 2)

    return scaled


def _priority_start_set(materials: Sequence[Dict], min_count: int) -> List[Dict]:
    """
    Build the starting material set out of whole priority groups.

    Groups are taken in order of increasing priority number (lower number =
    higher priority) until the set holds at least min_count materials.
    """
    ordered = sorted(materials, key=lambda m: (m.get('priority', DEFAULT_PRIORITY), m.get('name', '')))

    start_set: List[Dict] = []
    index = 0

    while index < len(ordered):
        current_priority = ordered[index].get('priority', DEFAULT_PRIORITY)

        # Take the whole priority group, never a part of it
        while index < len(ordered) and ordered[index].get('priority', DEFAULT_PRIORITY) == current_priority:
            start_set.append(ordered[index])
            index += 1

        if len(start_set) >= min_count:
            break

    return start_set


def _solve_material_set(material_set: Sequence[Dict], problem: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Solve one material set with NNLS, dropping the materials weighing less than
    MIN_MATERIAL_WEIGHT and re-solving until the recipe is stable.

    The whole material set is kept in the state even when the solver does not
    use all of it: a material that is useless now may become useful once
    another one joins the set on a later iteration.

    Returns a state dictionary with the recipe, the resulting UMF and both error
    numbers, or None when no recipe could be built.
    """
    full_target = problem['full_target']
    oxides = problem['oxides']
    row_weights = problem['row_weights']

    active = list(material_set)
    recipe: Dict[str, float] = {}

    try:
        # Dropping a material changes the optimum, so re-solve until the set settles
        for _ in range(len(material_set)):
            oxide_matrix, material_names = create_oxide_matrix(active, oxides)

            # The weighting is done INSIDE solve_recipe, which scales the matrix
            # and the right hand side together. It used to be done here, on the
            # matrix alone, which was exact only while every weighted row had a
            # zero right hand side (see _build_problem and solve_recipe)
            solution = solve_recipe(oxide_matrix, full_target, material_names, active,
                                    row_weights=row_weights)

            recipe = solution.get('recipe') or {}
            recipe = {name: weight for name, weight in recipe.items() if weight >= MIN_MATERIAL_WEIGHT}
            if not recipe:
                return None

            used = [material for material in active if material['name'] in recipe]
            if len(used) == len(active):
                break
            active = used

        recipe = _recipe_to_exactly_100(recipe)
        if not recipe:
            return None

        composition = calculate_recipe_composition(active, recipe)
        recipe_umf = {oxide: float(value) for oxide, value in weights_to_umf(composition).items()}

        # Both errors below compare the target with the recipe oxide by oxide,
        # so the recipe first has to be put on the normalization basis of the
        # target. Normally the two already agree and the scale is exactly 1.0
        unity_scale = _unity_scale(problem['target_umf'], recipe_umf)
        if unity_scale == 1.0:
            result_umf = recipe_umf
        else:
            result_umf = {oxide: value * unity_scale for oxide, value in recipe_umf.items()}
    except (ValueError, ZeroDivisionError, ArithmeticError) as exc:
        # A degenerate material set (nothing convertible, zero total weight) is
        # not a server error, it simply produces no recipe
        logger.debug(f"material set produced no recipe: {exc}")
        return None

    return {
        'materials': list(material_set),
        'recipe': recipe,
        'result_umf': result_umf,
        # Scale that was applied to the UMF of the recipe to make it comparable
        # with the target; 1.0 means the two were already on the same basis
        'unity_scale': float(unity_scale),
        # Reported error: reproducible by the caller from target_umf/result_umf
        'error': float(calculate_umf_error(problem['target_umf'], result_umf)),
        # Search objective, and NOT the reported error by another name: the L2
        # of the RELATIVE per-oxide deviations with a deadband, plus the damped
        # contamination. See _objective_error for why the two can order either
        # way round
        'objective_error': float(_objective_error(problem, result_umf)),
        'weight_composition': composition,
        'materials_count': len(recipe),
    }


def _shrink_to_limit(state: Dict[str, Any], problem: Dict[str, Any],
                     max_materials: int) -> Dict[str, Any]:
    """
    Bring a state down to the material limit by dropping the lightest material
    of the recipe and solving again.

    Needed when whole priority groups make the starting set larger than the
    caller allows, and when max_materials is smaller than the starting set the
    priority rule produces (a two component recipe is a legitimate request).
    """
    while state['materials_count'] > max_materials:
        lightest = min(sorted(state['recipe']), key=lambda name: state['recipe'][name])
        reduced = [material for material in state['materials'] if material['name'] != lightest]
        if not reduced:
            break

        smaller = _solve_material_set(reduced, problem)
        if smaller is None:
            break
        smaller['iterations'] = state['iterations']
        state = smaller

    return state


def _sole_carriers(used: Sequence[Dict], target_umf: Dict[str, float]) -> set:
    """
    Names of the materials that are the ONLY source of a requested oxide.

    "Requested" means the target asks for a positive amount of it. An oxide the
    target lists as an explicit zero is deliberately excluded: there "no other
    material carries it" is a reason to remove the carrier, not to keep it. So
    is an oxide the target never mentions, which is contamination by definition.

    Args:
        used: the material records the recipe actually uses
        target_umf: the requested target, as cleaned by usable_target

    Returns:
        set of material names that must not be pruned away
    """
    carriers: Dict[str, List[str]] = {}

    for material in used:
        name = material.get('name', '')
        for oxide, amount in material.get('formula', {}).items():
            if amount > 0.0 and target_umf.get(oxide, 0.0) > 0.0:
                carriers.setdefault(oxide, []).append(name)

    return {names[0] for names in carriers.values() if len(names) == 1}


def _prune_solution(state: Dict[str, Any], problem: Dict[str, Any],
                    min_materials: int) -> Dict[str, Any]:
    """
    Drop from a solved state every material the recipe turns out not to need.

    Greedy backward elimination: each round re-solves the recipe once per
    removable material with that material taken out, keeps the removals that
    pass the test below, applies the one that ends up with the lowest objective
    and starts again. It stops when no single removal qualifies any more.

    A removal has to clear TWO gates, and they answer different questions.

    1. THE SOLE CARRIER RULE, which is structural and has no quantity in it. A
       material that is the only thing in the recipe carrying an oxide the
       target asked for is never removed. Not because removing it would score
       badly - because the result would be a recipe that does not answer the
       request. The target said CoO, one material carries CoO, so that material
       stays.

       This is what protects colourants, opacifiers and every other ingredient
       that is present in a small amount for a reason, and it is the only thing
       that can. A colourant works optically and contributes almost nothing to
       the chemistry - that is what makes it a colourant - so no threshold on
       chemical error can tell one from fit noise at any calibration. The table
       in PRUNE_ERROR_TOLERANCE is the demonstration: cobalt survived the
       error test only because it happens to be a flux and its removal drags the
       unity denominator, while chrome oxide and iron chromate are just as much
       colourants, are not fluxes, and were being thrown away.

       The relative objective narrowed that hole without closing it - the second
       table under PRUNE_OBJECTIVE_TOLERANCE measures by how much, and 0.3% of
       red iron oxide is the row that used to fall through and no longer does -
       but "narrower" is not "closed", and the rule above is still the only
       thing here that answers the right question.

    2. THE ERROR TOLERANCES, for everything else. Fit noise - a material the
       search added to shave a fourth decimal off one oxide - is by construction
       not the sole carrier of anything requested, so gate 1 lets it through and
       this one measures it.

       BOTH error numbers are checked, not just the objective the search
       minimizes, and each has its OWN tolerance because the two are no longer
       in the same units: the objective is the L2 of the RELATIVE deviations and
       is bounded by PRUNE_OBJECTIVE_TOLERANCE, while `error` is the absolute L2
       the caller receives and is bounded by PRUNE_ERROR_TOLERANCE. One numeral
       for both would be one ruler for two scales.

       Why the second gate exists at all: with penalize_unlisted > 0 - the
       default, and what the API sends - the objective folds in the
       contamination of the unlisted oxides, so removing a material that brought
       unrequested oxides shrinks that term and can pay for a rise in `error`,
       which is the number the caller actually receives. Checking the objective
       alone therefore has no bound on `error` at all.
       tests/test_solver_inverse.py TestPruningChecksBothErrors builds the trade
       explicitly and pins that the removal is refused.

    Only the materials the recipe actually USES are candidates and only they are
    re-solved: the state carries the whole material set it was built from, and
    putting a material that NNLS already zeroed back into the matrix would turn
    a removal into a swap.

    The floor is max(min_materials, 1): the caller's minimum is never broken
    from this side, and the last material is never taken away even when the
    caller asked for a minimum of zero.

    Args:
        state: a state as returned by _solve_material_set
        problem: the problem context the state was solved against
        min_materials: the caller's min_materials

    Returns:
        the pruned state, or the very same object when nothing could go
    """
    current = state
    floor = max(int(min_materials), 1)
    target_umf = problem['target_umf']

    while current['materials_count'] > floor:
        used = [material for material in current['materials']
                if material['name'] in current['recipe']]
        protected = _sole_carriers(used, target_umf)

        objective_limit = current['objective_error'] + PRUNE_OBJECTIVE_TOLERANCE
        error_limit = current['error'] + PRUNE_ERROR_TOLERANCE
        best: Optional[Dict[str, Any]] = None

        # sorted() so that two equally good removals always resolve the same
        # way, whatever order the material set happens to be in
        for dropped in sorted(used, key=lambda material: material.get('name', '')):
            if dropped['name'] in protected:
                continue

            reduced = [material for material in used if material['name'] != dropped['name']]
            if not reduced:
                continue

            candidate = _solve_material_set(reduced, problem)
            if candidate is None or candidate['materials_count'] < floor:
                continue
            if candidate['objective_error'] > objective_limit or candidate['error'] > error_limit:
                continue

            if best is None:
                best = candidate
            elif candidate['objective_error'] < best['objective_error']:
                best = candidate

        if best is None:
            break

        # _solve_material_set knows nothing about the search, so the bookkeeping
        # of the state being replaced is carried over by hand
        best['iterations'] = current.get('iterations', 1)
        best['set_names'] = frozenset(material['name'] for material in best['materials'])
        current = best

    return current


def _weight_residual(problem: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, float]:
    """
    Residual of the current recipe in weight percent: target minus actual.

    A positive value is a deficit (the recipe delivers too little of that oxide)
    and a negative one an excess. Both sides are normalized to 100, so the two
    vectors are directly comparable.

    THE OXIDE SET IS problem['oxides'] AND NOTHING ELSE. calculate_recipe_
    composition keeps every key of every material formula, loss on ignition
    included, so the actual side arrives carrying keys that _expand_target has
    already refused a row and a molar mass. Taking the union of the two sides -
    which this function used to do - let them back in through the side door, and
    with real consequences: on a recipe of whiting and kaolin the union produced
    residuals of Loi -10.1 and LOI -4.4 against a worst real oxide of +10.0, so
    _focus_oxide picked "Loi" as the oxide of the step and _score_candidate then
    scored every material by how little it burns off. One rule, applied in both
    places, is what keeps that from coming back.

    Normalizing comes AFTER the filter, and the order is load bearing. Scaling
    the actual composition to 100 with the loss on ignition still in it divides
    every real oxide by a total that is 10-15% too large, which biases every
    residual of every oxide upwards at once - a systematic deficit the search
    then chases.
    """
    target_weights = problem['target_weights']
    oxides = problem['oxides']

    fitted = {oxide: value for oxide, value in state['weight_composition'].items()
              if oxide in problem['oxide_set']}
    actual_weights = _normalize_to_100(fitted)

    residual: Dict[str, float] = {}
    for oxide in sorted(oxides):
        residual[oxide] = target_weights.get(oxide, 0.0) - actual_weights.get(oxide, 0.0)

    return residual


def _focus_oxide(residual: Dict[str, float]) -> Optional[str]:
    """
    Return the oxide the current recipe is the furthest away from, measured in
    weight percent - the oxide this iteration is about.

    THE ONE PLACE OF THE SEARCH LEFT ON THE ABSOLUTE WEIGHT SCALE, and that is a
    decision rather than an oversight. Everything that RANKS an answer went
    relative in 10.19; this picks which oxide one step is about, and under
    candidate_search='exhaustive' - the default - every candidate is solved
    anyway, so the focus oxide only orders a tie between sets whose objective
    agrees to CANDIDATE_TIE_STEP. Under 'heuristic' it really does filter, and
    the absolute scale costs something there, but less than the relative
    residual costs the exhaustive mode nothing: measured over the 300 case
    scenario A corpus, 'heuristic' goes 253 -> 294 inside the chemistry gate
    across 10.19 while 'exhaustive' goes 251 -> 296. Converting this too is a
    separate change with its own measurement, not a free tidy-up.

    The oxides are walked in sorted order so that an exact tie between two
    oxides always resolves the same way; iterating a set here used to make the
    choice depend on the hash seed.
    """
    worst_oxide = None
    worst_gap = 0.0

    for oxide in sorted(residual):
        gap = abs(residual[oxide])
        if gap > worst_gap:
            worst_gap = gap
            worst_oxide = oxide

    return worst_oxide


def _score_candidate(material: Dict, residual: Dict[str, float], focus: Optional[str]) -> Optional[float]:
    """
    Score one candidate material against the current residual.

    The score is built from four disjoint terms, every oxide of the material
    falling into exactly one of them:

      + focus gain          residual on the focus oxide times the share of that
                            oxide in the material (signed: a material rich in an
                            oxide that is already in excess scores negative)
      + secondary gain      the same product on the other oxides that are still
                            short, damped by SECONDARY_GAIN_WEIGHT
      - contamination       what the material adds to the oxides that are
                            already over the target, scaled up by
                            CONTAMINATION_WEIGHT because an excess cannot be
                            subtracted later
      - disturbance         what the material adds to the oxides that already
                            match; their residual is ~0, so this term needs its
                            own scale (MATCHED_OXIDE_TOLERANCE) to exist at all

    A formula key the residual does not know - one _expand_target refused for
    having no molar mass - scores in NONE of the four terms. It used to score in
    the fourth by accident: `residual.get(key, 0.0)` returns 0.0, |0.0| is
    inside MATCHED_OXIDE_TOLERANCE, so the key landed in "disturbance" and every
    material was penalized in proportion to how much of it burns off. That is
    the same "minimize loss on ignition" instruction _weight_residual removes,
    arriving by a second route.

    The DENOMINATOR is 100, the basis a material analysis is stated on, and NOT
    the sum of the analysis. 100 g of whiting as weighed out yields 56.1 g of
    CaO, so the CaO fraction of whiting is 0.561; dividing by the sum of its
    analysis would make it 1.0 and claim that whiting is pure calcium oxide.

    This is the convention the rest of the system already reads material
    analyses by - feasibility.build_molar_matrix calls a column "moles per 100 g
    as weighed out", and calculate_recipe_composition multiplies recipe percents
    straight into the analysis values - so using the sum here was this function
    disagreeing with everything around it.

    THIS IS A CONSISTENCY FIX AND NOT A QUALITY ONE, and the comment says so
    because the numbers do. What the old denominator did wrong is exact: only 65
    of the 179 analysed materials of the shipped database have an analysis that
    adds up to 100, so for the other 114 dividing by the sum inflated every
    fraction by 1 / (oxide sum) - whiting by 1.783, kaolin by 1.173, quartz by
    1.000. Two materials with the SAME REAL CHEMISTRY therefore scored
    differently depending only on whether their analysis happens to list what
    burns off, which is a property of how the row was typed and of nothing else.
    That alone is reason enough to change it.

    What it did NOT do is cost answers. A/B over the 300 case scenario A corpus,
    old denominator against new, both candidate_search modes:

        heuristic   old  rel<=0.05 294/300  mean count 5.70  mean rel 0.0144
                    new  rel<=0.05 294/300  mean count 5.71  mean rel 0.0143
                    1 case of the 300 came back different at all
        exhaustive  old  rel<=0.05 296/300  mean count 5.73  mean rel 0.0094
                    new  rel<=0.05 296/300  mean count 5.73  mean rel 0.0094
                    0 cases changed

    One case in the only mode where this ranking is more than a tie break. The
    bias is a single multiplicative factor per material, and it is dominated by
    the differences between the residual gaps it multiplies into: a material is
    picked here for closing a shortfall, not for the second decimal of its
    fraction. Under 'exhaustive' - the default - every candidate is solved
    anyway, so the ranking only orders a tie and nothing moves at all.

    A fraction above 1.0 is possible and correct: cryolite analyses to 122.9 and
    zircon to 135.22, because firing them gains oxygen rather than losing mass.
    validate_database already accepts both as sound analyses.

    Returns None for a material whose analysis carries nothing.
    """
    formula = material.get('formula', {})
    if sum(formula.values()) <= 0:
        return None

    # Fractions of 100 g of the material AS WEIGHED OUT; the loop below is what
    # skips the keys that carry no residual
    fractions = {oxide: value / 100.0 for oxide, value in formula.items()}

    focus_gain = 0.0
    secondary_gain = 0.0
    contamination = 0.0
    disturbance = 0.0

    for oxide in sorted(fractions):
        if oxide not in residual:
            continue

        fraction = fractions[oxide]
        gap = residual[oxide]

        if oxide == focus:
            focus_gain = gap * fraction
        elif abs(gap) <= MATCHED_OXIDE_TOLERANCE:
            disturbance += MATCHED_OXIDE_TOLERANCE * fraction
        elif gap > 0.0:
            secondary_gain += gap * fraction
        else:
            contamination += -gap * fraction

    return (focus_gain
            + SECONDARY_GAIN_WEIGHT * secondary_gain
            - CONTAMINATION_WEIGHT * contamination
            - MATCHED_DISTURBANCE_WEIGHT * disturbance)


def _rank_candidates(candidates: Sequence[Dict], residual: Dict[str, float],
                     focus: Optional[str]) -> List[Tuple[float, Dict]]:
    """
    Rank the materials that are not in the set yet, best first.

    Two things decide the order and both of them really move it:

    * the chemical score of _score_candidate, divided by the size of the focus
      gap so that 1.0 means "closes the whole gap this step is about";
    * the material priority, folded in as a penalty of PRIORITY_WEIGHT times the
      normalized position of its priority group. This is not a tie break: a top
      priority material overtakes a chemically better one whenever the other one
      is ahead by less than PRIORITY_WEIGHT of the focus gap.

    Returns (chemical score, material) pairs in the blended order.
    """
    scored: List[Tuple[float, Dict]] = []

    for material in candidates:
        score = _score_candidate(material, residual, focus)
        if score is None:
            continue
        scored.append((score, material))

    if not scored:
        return []

    # A material made purely of the focus oxide would score exactly the focus
    # gap, which makes that gap the natural unit of this step. The fallbacks
    # only matter for an already converged recipe, where the order is moot
    reference = abs(residual.get(focus, 0.0)) if focus is not None else 0.0
    if reference <= 0.0:
        reference = max((abs(score) for score, _ in scored), default=0.0)
    if reference <= 0.0:
        reference = 1.0

    # Priority groups present in this step, mapped onto [0, 1]
    priorities = sorted({material.get('priority', DEFAULT_PRIORITY) for _, material in scored})
    priority_rank = {value: (index / (len(priorities) - 1) if len(priorities) > 1 else 0.0)
                     for index, value in enumerate(priorities)}

    def blended(item: Tuple[float, Dict]) -> Tuple[float, float, float, str]:
        score, material = item
        normalized = score / reference
        penalty = PRIORITY_WEIGHT * priority_rank[material.get('priority', DEFAULT_PRIORITY)]
        # Sorted ascending, hence the negated values; the name closes the last
        # possible tie so that the order never depends on the input order
        return (-(normalized - penalty),
                material.get('priority', DEFAULT_PRIORITY),
                -score,
                material.get('name', ''))

    scored.sort(key=blended)

    return scored


def _expand_state(state: Dict[str, Any], available_materials: Sequence[Dict],
                  problem: Dict[str, Any], seen_sets: set, candidate_limit: Optional[int],
                  verbose: bool) -> List[Dict[str, Any]]:
    """
    Try to add one material to the set of a state.

    The candidates are ranked by the residual heuristic first and only the best
    candidate_limit of them are really solved (None means "all of them", the
    exhaustive mode). Solving is what decides in the end: the heuristic cannot
    tell that wollastonite can be replaced by chalk plus quartz without any
    visible change in the weight space residual.

    Material sets that some other branch already solved are skipped without
    spending one of the candidate_limit slots on them, so a step always costs at
    most candidate_limit NNLS runs of new work.

    Returns the resulting states ordered from best to worst.
    """
    used_names = {material['name'] for material in state['materials']}
    candidates = [material for material in available_materials if material['name'] not in used_names]
    if not candidates:
        return []

    residual = _weight_residual(problem, state)
    focus = _focus_oxide(residual)
    ranked = _rank_candidates(candidates, residual, focus)

    if verbose:
        logger.info(f"worst oxide {focus}, best candidates by heuristic: "
                    f"{[material['name'] for _, material in ranked[:TOP_CANDIDATES]]}")

    trials: List[Tuple[float, int, Dict[str, Any]]] = []
    solved = 0

    for rank, (_score, candidate) in enumerate(ranked):
        if candidate_limit is not None and solved >= candidate_limit:
            break

        new_set = list(state['materials']) + [candidate]
        set_names = frozenset(material['name'] for material in new_set)
        if set_names in seen_sets:
            continue
        seen_sets.add(set_names)

        solved += 1
        new_state = _solve_material_set(new_set, problem)
        if new_state is None:
            continue

        new_state['set_names'] = set_names
        new_state['added'] = candidate['name']

        # Quantizing the objective keeps equally good candidates together, so
        # that the heuristic order (and through it the priority) decides between
        # them - see CANDIDATE_TIE_STEP
        trials.append((round(new_state['objective_error'] / CANDIDATE_TIE_STEP),
                       rank, new_state))

    trials.sort(key=lambda item: (item[0], item[1]))

    return [item[2] for item in trials]


def _recipe_key(recipe: Dict[str, float]) -> Tuple:
    """Composition based identity of a recipe, used to drop duplicates"""
    return tuple(sorted((name, round(weight, 1)) for name, weight in recipe.items()))


def _is_stalled(previous_error: float, next_error: float) -> bool:
    """
    Whether one step improved the objective by less than STALL_IMPROVEMENT of
    what there was to improve - the branch is then abandoned.

    The threshold is relative on purpose: an absolute one would keep polishing a
    recipe that is already at 0.001 and would give up on one that is at 10.
    """
    improvement = previous_error - next_error
    return improvement <= previous_error * STALL_IMPROVEMENT


def _solution_tie_limit(best_error: float) -> float:
    """
    Objective up to which a solution counts as "as good as the best one", so
    that the material count decides between them instead of the fourth decimal.
    """
    return best_error + max(best_error * SOLUTION_ERROR_TIE_REL, SOLUTION_ERROR_TIE_ABS)


def _recipe_priority(solution: Dict[str, Any]) -> float:
    """
    Weight-weighted material priority of a recipe, lower being more basic.

    The same quantity quality_metrics._priority_metric scores a solution by:
    sum(priority * share / 100) over the materials the recipe uses, with
    DEFAULT_PRIORITY for a material that carries none. A catalogue without
    priorities collapses to DEFAULT_PRIORITY everywhere and the number is
    constant, which is exactly what a tie break should do with no data.
    """
    by_name = {material.get('name'): material for material in solution['materials']}

    total = 0.0
    for name, share in solution['recipe'].items():
        material = by_name.get(name) or {}
        total += float(material.get('priority', DEFAULT_PRIORITY)) * float(share) / 100.0

    return total


def _solution_sort_key(solution: Dict[str, Any], best_error: float):
    """
    Order the solutions the way the caller is promised they are ordered:

      1. everything within the tie band of the best objective comes first, and
         inside that band FEWER MATERIALS WINS (the objective only breaks a tie
         between two recipes of the same size);
      2. everything outside the band follows, ordered by the objective;
      3. an exact tie on all of the above is broken by the weighted material
         PRIORITY, lower first.

    So the first solution is not necessarily the one with the smallest
    objective - by design. Two recipes whose errors agree to the third decimal
    are indistinguishable in the glaze bucket, and the shorter one is the better
    answer; find_best_recipe documents this and the tests pin it.

    Rung 3 was added when the objective went relative and gained a deadband,
    which made the exact tie the common case instead of a curiosity: several
    recipes now score EXACTLY 0 on the same number of materials, and until this
    rung existed the winner among them was whichever the pool happened to hold
    first. Recipe 09 of the reference set is what that costs - zinc oxide and
    zinc carbonate carry the same oxide, both recipes fit the target exactly,
    they differ by 0.001 of recipe-rounding noise, and the arbitrary order
    returned the carbonate that the workshop keeps at priority 100 against the
    oxide's 10. Priority is the module's own stated preference between materials
    that are chemically interchangeable (see PRIORITY_WEIGHT and
    _priority_start_set); once the chemistry is a tie there is nothing else left
    to prefer with.

    It is deliberately the LAST rung and not an earlier one: a recipe is never
    made worse chemically, nor longer, to reach a preferred material.

    MEASURED WHERE THE PRIORITIES ARE REAL, which is scenario B of the corpus -
    our own 19 material shelf with database/priorities.json behind it. Scenario
    A is the wrong place to judge this rung and the first version of this note
    used it anyway: there the priorities are synthesised from Glazy popularity
    and the rung moves 1 top-1 answer of 300. On scenario B it moves 9 of 100,
    and all nine are the behaviour described above rather than a trade:
    objective_error, materials_count and max_relative are identical to the last
    bit on every one of them, while the weighted priority strictly improves. The
    improvements, every one of the nine:

        241639  39.94 ->  9.50      433712  12.14 ->  3.90
        281365  26.88 ->  3.45        8900   9.19 ->  2.76
        557542  36.00 -> 24.27      427954   7.28 ->  3.88
        459155  52.26 -> 42.87      530155  23.51 -> 23.12
                                      2982   8.04 ->  7.84

    a range of 0.20 to 30.44 points; 281365 is zinc oxide at 18.09% replacing
    zinc carbonate at 25.39%. Nothing gets worse on any of the three chemistry
    numbers, which is what "the last rung" is supposed to guarantee and now has
    evidence for.
    """
    tie_limit = _solution_tie_limit(best_error)

    if solution['objective_error'] <= tie_limit:
        return (0, solution['materials_count'], solution['objective_error'],
                _recipe_priority(solution))
    return (1, solution['objective_error'], solution['materials_count'],
            _recipe_priority(solution))


def find_best_recipe(inventory, target_umf, min_materials=1, max_materials=10,
                     max_solutions=5, verbose=False, error_threshold=0.1,
                     penalize_unlisted=1.0,
                     candidate_search=SEARCH_EXHAUSTIVE,
                     materials=None) -> List[Dict[str, Any]]:
    """
    Find glaze recipes for a target UMF by adding materials one at a time.

    Every recipe is put through _prune_solution() before it is ranked and
    returned: the search only ever adds materials, and the pruning pass is what
    takes back the ones that turned out not to be needed. A material is dropped
    only when it is not the sole source of a requested oxide AND its removal
    costs at most PRUNE_OBJECTIVE_TOLERANCE on the objective and
    PRUNE_ERROR_TOLERANCE on the reported error, so an ingredient the recipe
    genuinely answers the request with survives however little it weighs - see
    MIN_MATERIAL_WEIGHT for what happens when smallness is used as the criterion
    instead.

    Two consequences of the pass are worth knowing before reading the numbers
    below, because neither is visible in the returned shape:

    * error_threshold is NOT rechecked after pruning. A branch stops when its
      objective reaches the threshold, and a later removal can push the returned
      objective past it by up to one tolerance per removal. Re-measured over the
      300 recipe Glazy corpus under the relative objective: NO top-1 answer of
      the 300 crosses the default 0.1 any more, and the worst growth the pass
      inflicts on a returned answer is 0.0346. The number in "error" is always
      the true error of the recipe returned; what still does not hold exactly is
      the sentence "the search stopped because this recipe was under the
      threshold".
    * pruning can produce a one material recipe, and on an UNREACHABLE target
      the relative tie band of _solution_sort_key can rank it first, because it
      is the shortest of a set of equally hopeless answers. Measured over the
      same 300 targets: solved against their own materials, where the answer is
      reachable by construction, the share of single material top-1 answers is
      0.3%. Solved against the 19 material inventory, where most of them are
      unreachable, it is 1.0% - it was 1.7% under the absolute objective and
      4.7% before the sole carrier rule, which blocks most of these collapses
      because a hopeless target usually has exactly one carrier left for
      something it asked for. A caller showing a headline recipe may still want
      to read materials_count together with the error.

    Args:
        inventory: list of available material names
        target_umf: target UMF formula as {oxide: value}. An oxide listed as an
            explicit zero is a constraint ("none of this"), not an omission:
            unknown oxides, negative and non numeric values are dropped, zeros
            are kept and penalized like any other requested value. What was
            dropped is logged once per call; a caller that has to SHOW it calls
            usable_target() itself and passes the cleaned half in here, which is
            what /api/solve does - the answer cannot carry the list, since a
            target of nothing but unknown oxides returns no solutions to carry
            anything
        min_materials: minimum number of materials in a returned recipe; when no
            recipe reaches it the result is an empty list, the constraint is
            never silently broken. The pruning pass respects it too and never
            takes the last material away even when it is 0
        max_materials: maximum number of materials in a recipe; the starting set
            built from whole priority groups is shrunk down to this limit, so
            values below DEFAULT_MIN_START_MATERIALS are honoured too
        max_solutions: upper bound on how many solutions to return; 0 or less
            returns []. Fewer can come back than asked for, and the pruning pass
            made that more common: several recipes of the search can prune onto
            the same answer, and they are merged rather than back-filled. The
            merged_variants field of each solution says how many.
        verbose: log the search process
        error_threshold: objective error below which a recipe counts as
            acceptable. A branch that reached it is dropped only once the pool
            already holds max_solutions different acceptable recipes; until then
            the branch keeps being refined, because the extra recipes have to
            come from somewhere.
            IN RELATIVE UNITS: the objective is the L2 of the per-oxide relative
            deviations (see OBJECTIVE_DEADBAND), several times larger than the
            absolute norm this argument used to be compared against. The default
            was re-measured rather than carried over, and 0.1 is where it landed
            anyway. Swept over the 300 case scenario A corpus at
            max_solutions=5: 0.02, 0.05 and 0.1 give the identical run (296 of
            300 inside the chemistry gate, mean material count 5.73), because
            under the relative objective almost no branch gets that low and the
            early-convergence rule hardly ever fires; 0.2 costs three cases and
            one reachable scenario B target, 0.3 costs seven, 0.5 costs
            nineteen and takes scenario B back to where it started. So 0.1 is
            the largest value that costs nothing, and anything smaller buys
            nothing.
        penalize_unlisted: how hard an oxide that the target does not mention is
            pushed towards zero. 1.0 / True means "not listed = must be zero",
            0.0 / False means "not listed = do not care", anything in between is
            a soft weight applied both to the NNLS rows of those oxides and to
            their share of the search objective. Targets derived from a real
            recipe list every oxide it brings, so 1.0 is right for them; a
            target typed by hand in the UI lists only what the user cares about,
            and a hard 1.0 there makes the solver sacrifice the requested oxides
            to zero out the unmentioned ones.
        candidate_search: 'exhaustive' (default) solves every candidate material
            of every step. It costs O(len(inventory)) NNLS runs per step, it is
            the more accurate of the two, and it reduces the candidate ranking
            to a tie break between sets whose error agrees to four decimals -
            the focus oxide and the priority do not steer it.
            'heuristic' solves only the TOP_CANDIDATES best ranked candidates,
            which is what a human does: there the ranking, the focus oxide and
            the priority really pick what gets tried, and a step costs a
            constant number of NNLS runs.
            Measured by calling find_best_recipe once per reference recipe (11
            of them, the 19 material inventory of database/materials.json,
            max_solutions=5 and every other argument at its default) and
            counting the calls to scipy.optimize.nnls, the pruning pass
            included: exhaustive 1680 runs, recovering the original material set
            exactly on 9 of the 11 recipes; heuristic 624 runs and 7 of 11. The
            heuristic gets to 8 of 11 at TOP_CANDIDATES = 9, and there it costs
            1448 runs - about half the inventory is tried per step, so it is no
            longer a shortcut and no longer cheaper, and it still does not catch
            up. The 11th recipe is out of reach for every mode: it needs MnO2
            and no material of the inventory carries any.
            The configuration matters to these numbers and used to be left out
            of this paragraph, which is a good way to mislead yourself: the run
            count scales with max_solutions through the beam width AND through
            the pruning budget, and the pass costs 787 -> 1680 runs on this
            sweep. On the full 216 material catalogue the worst single call of
            the eleven is 1851 runs against 1706 unpruned.
        materials: optional material records to use as the database, same shape
            as database/materials.json entries. Meant for the tests and for
            callers carrying their own catalogue; when it is given together with
            inventory=None it bypasses the inventory resolution and every
            injected material is available. See common.resolve_material_pool().
            "priority" is optional in those records and defaults to
            DEFAULT_PRIORITY, which puts every injected material in one group -
            so _priority_start_set() starts from the whole catalogue at once
            unless the records carry explicit priorities.

    Raises:
        ValueError: candidate_search is not one of CANDIDATE_SEARCH_MODES,
            penalize_unlisted is not a number or a boolean, or one of
            min_materials / max_materials / max_solutions is not an integer.

    Returns:
        list of solutions, best first, where "best" is the order documented in
        _solution_sort_key: the recipes whose objective is within the tie band
        of the best one come first and among those the SHORTEST one leads, the
        rest follow by increasing objective. The first solution therefore has an
        objective inside the tie band, not necessarily the smallest one in the
        list. That order is established AFTER the pruning pass, so it holds on
        the recipes actually returned rather than on the ones the search built.
        Every solution holds:
            recipe          {material: weight percent}, adds up to exactly 100
            error           calculate_umf_error(target_umf, result_umf); it can
                            be recomputed from the two dictionaries below
            objective_error what the search minimized, and NOT the same quantity
                            as "error": the L2 of the per-oxide RELATIVE
                            deviations, each given OBJECTIVE_DEADBAND for free,
                            plus the damped contamination of the unlisted
                            oxides. It is on the scale the feasibility LP and
                            the benchmark gate use, so it is comparable with
                            their tolerances and not with "error"
            result_umf      UMF of the recipe, brought onto the normalization
                            basis of the target (see unity_scale)
            unity_scale     the scale that was applied to get there; 1.0 in the
                            normal case, where the target is a unity formula and
                            nothing needs scaling. The untouched UMF of the
                            recipe is result_umf divided by this
            target_umf      the requested target, cleaned of unusable entries
            effective_target_umf  target_umf plus a zero for every oxide the
                            inventory can bring, the oxides penalize_unlisted
                            talks about
            unlisted_weight the penalize_unlisted value actually applied
            materials_count number of materials in the recipe, after pruning
            merged_variants how many OTHER distinct recipes of the search pruned
                            onto this same one and are therefore not listed
                            separately; 0 when nothing collapsed onto it. Counted
                            over the candidates the pruning pass actually looked
                            at, which is PRUNE_CANDIDATE_MARGIN * max_solutions
                            of them at most, so it is a floor rather than an
                            exhaustive census of the pool
            iterations      how many search steps the recipe took; the pruning
                            pass is not a step and does not raise it
    """
    solution_limit = _int_argument(max_solutions, 'max_solutions')
    material_limit = _int_argument(max_materials, 'max_materials')
    material_floor = _int_argument(min_materials, 'min_materials')

    if solution_limit <= 0:
        return []

    if candidate_search not in CANDIDATE_SEARCH_MODES:
        raise ValueError(f"unknown candidate_search '{candidate_search}', "
                         f"expected one of: {', '.join(CANDIDATE_SEARCH_MODES)}")

    unlisted_weight = _normalize_unlisted_weight(penalize_unlisted)

    # A target of unknown oxides or of nothing but zeros cannot be converted
    # into weights at all (umf_to_weights would divide by a zero total weight);
    # catching it here keeps ZeroDivisionError out of the callers
    clean_target, dropped_oxides = usable_target(target_umf)
    # ONCE PER CALL, not once per search step, and in the ONE wording the API
    # layer uses too - so that a log grepped for this line counts requests
    # rather than layers. A caller that cleaned the target itself (/api/solve
    # does, above the choice of engine) hands us a clean one and this never
    # fires; a caller that did not gets told here.
    if dropped_oxides:
        logger.warning(DROPPED_TARGET_OXIDES_LOG.format(oxides=', '.join(dropped_oxides)))

    if not any(value > 0.0 for value in clean_target.values()):
        if verbose:
            logger.info("the target holds no usable oxide")
        return []

    if material_limit < 1:
        logger.warning(f"max_materials={max_materials} leaves no room for a recipe")
        return []

    if material_floor > material_limit:
        logger.warning(f"min_materials={material_floor} is above max_materials={material_limit}, "
                       f"no recipe can satisfy both")
        return []

    all_materials, available_names = resolve_material_pool(materials, inventory)
    # A material with an empty formula can never move the UMF. _rank_candidates
    # already skips it, but _priority_start_set does not, so it is dropped here
    available_materials = filter_materials_with_formula(
        filter_materials_by_inventory(all_materials, available_names))

    if not available_materials:
        if verbose:
            logger.info("no materials available in the inventory")
        return []

    problem = _build_problem(clean_target, available_materials, unlisted_weight)

    candidate_limit = None if candidate_search == SEARCH_EXHAUSTIVE else TOP_CANDIDATES
    beam_width = 1 if solution_limit <= 1 else min(MAX_BEAM_WIDTH, solution_limit)
    # How many children of one state are allowed to stay in the beam
    beam_children = 1 if solution_limit <= 1 else TOP_CANDIDATES

    min_start_materials = max(material_floor, DEFAULT_MIN_START_MATERIALS)
    start_set = _priority_start_set(available_materials, min_start_materials)
    if not start_set:
        return []

    if verbose:
        logger.info(f"starting set ({len(start_set)} materials): {[m['name'] for m in start_set]}")

    start_state = _solve_material_set(start_set, problem)
    if start_state is None:
        if verbose:
            logger.info("the starting set produced no recipe")
        return []

    start_state['iterations'] = 1
    start_state = _shrink_to_limit(start_state, problem, material_limit)
    start_state['set_names'] = frozenset(m['name'] for m in start_state['materials'])

    pool: List[Dict[str, Any]] = [start_state]
    beam: List[Dict[str, Any]] = [start_state]
    seen_sets = {start_state['set_names']}
    # Different recipes that already meet the requested quality
    found_recipes = set()
    if start_state['objective_error'] <= error_threshold:
        found_recipes.add(_recipe_key(start_state['recipe']))

    for iteration in range(2, DEFAULT_MAX_ITERATIONS + 1):
        next_beam: List[Dict[str, Any]] = []

        for state in beam:
            # A branch that is good enough is dropped, but only once the pool
            # holds as many different recipes as the caller asked for
            if state['objective_error'] <= error_threshold and len(found_recipes) >= solution_limit:
                if verbose:
                    logger.info(f"branch converged with error {state['objective_error']:.4f}")
                continue

            if state['materials_count'] >= material_limit:
                continue

            children = _expand_state(state, available_materials, problem, seen_sets,
                                     candidate_limit, verbose)

            for child in children:
                child['iterations'] = iteration
                pool.append(child)
                if child['objective_error'] <= error_threshold:
                    found_recipes.add(_recipe_key(child['recipe']))

            # Only the best branches are kept alive; a branch that stops
            # improving is abandoned and the pool keeps whatever it already found
            for child in children[:beam_children]:
                if _is_stalled(state['objective_error'], child['objective_error']):
                    if verbose:
                        logger.info(f"branch stalled on {child['added']}: "
                                    f"{state['objective_error']:.4f} -> {child['objective_error']:.4f}")
                    continue

                if verbose:
                    logger.info(f"iteration {iteration}: added {child['added']}, "
                                f"error {state['objective_error']:.4f} -> {child['objective_error']:.4f}")

                next_beam.append(child)

        if not next_beam:
            break

        next_beam.sort(key=lambda s: (s['objective_error'], s['materials_count']))
        beam = next_beam[:beam_width]

    # Keep only recipes that respect the material limits. An empty result here
    # means the limits are unreachable with this inventory - the caller is told
    # so instead of being handed a recipe that breaks them
    solutions = [s for s in pool if material_floor <= s['materials_count'] <= material_limit]
    if not solutions:
        logger.warning(f"no recipe with {material_floor}..{material_limit} materials was reachable, "
                       f"the pool holds {len(pool)} states")
        return []

    # Backward elimination, and it happens HERE - after the search, before the
    # sort and before the max_solutions cut. Three decisions are packed into
    # that placement:
    #
    # * NOT inside the beam loop. Pruning one state costs O(materials) NNLS runs
    #   per round, the loop solves tens to hundreds of states per call, and the
    #   beam is going to add materials on top of whatever was pruned anyway.
    # * BEFORE the sort, because pruning changes materials_count, and
    #   _solution_sort_key ranks by materials_count inside the tie band. Sorting
    #   first would order the recipes by a size they no longer have.
    # * BEFORE the max_solutions cut, for the same reason: a recipe that prunes
    #   from seven materials down to four has to be able to overtake one that
    #   started at five and stayed there, and one that prunes onto a recipe
    #   already in the list has to lose its slot to the next distinct answer.
    #
    # HOW MANY candidates are pruned is a cost decision, and the honest version
    # of it is that pruning every distinct recipe of the pool is what the
    # placement above really wants and what nobody can afford. Neither the
    # objective nor the material count moves monotonically under pruning (a
    # removal may grow the objective, and on the reference set it sometimes
    # shrinks it - recipe 06 went from 5.8311 on three materials to 3.2224 on
    # one), so no candidate can be proved irrelevant in advance. But the pool
    # holds 2 to 48 distinct recipes on the 19 material inventory and 14 to 267
    # on the whole 216 material catalogue, and pruning all of them took one
    # /api/solve request from 236 ms to 619 ms. So the candidates are taken in
    # the order the UNPRUNED sort key gives and the pass stops after
    # PRUNE_CANDIDATE_MARGIN * max_solutions of them - a bounded overshoot
    # rather than an exact cut at max_solutions, so that a recipe still has room
    # to shrink past two others and be returned.
    #
    # The budget yields to one thing: having max_solutions DISTINCT answers to
    # return. Several candidates can prune onto the same recipe - that is the
    # point of the pass - and stopping at the budget while the list is still
    # short would silently under-deliver alternatives that do exist further down
    # the pool. Measured over the eleven references on the full catalogue, a
    # hard stop at the budget returns 35 alternatives against the 55 of the
    # unpruned search; letting it run on until the list is full returns 44, and
    # the extra work is only ever done when the collapse actually happened.
    pre_prune_best = min(s['objective_error'] for s in solutions)
    ordered = sorted(solutions, key=lambda s: _solution_sort_key(s, pre_prune_best))
    prune_budget = max(solution_limit * PRUNE_CANDIDATE_MARGIN, 1)

    seen_before_pruning = set()
    distinct_pruned = set()
    pruned: List[Dict[str, Any]] = []

    for solution in ordered:
        if len(pruned) >= prune_budget and len(distinct_pruned) >= solution_limit:
            break
        key = _recipe_key(solution['recipe'])
        if key in seen_before_pruning:
            continue
        seen_before_pruning.add(key)
        pruned_state = _prune_solution(solution, problem, material_floor)
        pruned.append(pruned_state)
        distinct_pruned.add(_recipe_key(pruned_state['recipe']))

    best_error = min(s['objective_error'] for s in pruned)
    pruned.sort(key=lambda s: _solution_sort_key(s, best_error))

    # Two different recipes can prune onto the same one - that is exactly what
    # happens when both of them carry the same answer plus one redundant
    # material each - so the list is deduplicated again here.
    #
    # The collapse is counted rather than back-filled. Topping the list up with
    # the UNPRUNED states would restore the very recipes the pass just judged to
    # be this same answer plus noise - the junk the pass exists to remove,
    # dressed up as an alternative. Measured over the eleven references on the
    # full catalogue, pruning takes 55 alternatives down to 44, and recipe 08
    # collapses 5 -> 1 because all five were one four component core plus one or
    # two percent of kaolin or alum. Handing those five back would be a worse
    # answer honestly counted, so each returned solution reports merged_variants
    # instead and a caller can show "4 near-identical variants were merged".
    merged_counts: Dict[Tuple, int] = {}
    for solution in pruned:
        key = _recipe_key(solution['recipe'])
        merged_counts[key] = merged_counts.get(key, 0) + 1

    unique: List[Dict[str, Any]] = []
    seen_recipes = set()

    for solution in pruned:
        if len(unique) >= solution_limit:
            break

        key = _recipe_key(solution['recipe'])
        if key in seen_recipes:
            continue
        seen_recipes.add(key)

        unique.append({
            'recipe': solution['recipe'],
            'error': solution['error'],
            'objective_error': solution['objective_error'],
            'result_umf': solution['result_umf'],
            'unity_scale': solution['unity_scale'],
            'target_umf': dict(problem['target_umf']),
            'effective_target_umf': dict(problem['full_target']),
            'unlisted_weight': unlisted_weight,
            'materials_count': solution['materials_count'],
            'merged_variants': merged_counts[key] - 1,
            'iterations': solution['iterations'],
        })

    if verbose and unique:
        logger.info(f"returning {len(unique)} solutions, best error {unique[0]['error']:.4f}")

    return unique


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description='Iterative Glaze Recipe Solver')
    parser.add_argument('--umf', type=str, help='Target UMF composition as JSON string')
    parser.add_argument('--solutions', type=int, default=5, help='Number of solutions to find (default: 5)')
    parser.add_argument('--max-materials', type=int, default=10, help='Maximum number of materials (default: 10)')
    parser.add_argument('--error-threshold', type=float, default=0.1,
                        help='Objective error at which the search stops, in RELATIVE units - '
                             'the L2 of the per-oxide deviations divided by their own scale '
                             '(default: 0.1)')
    parser.add_argument('--penalize-unlisted', type=float, default=1.0,
                        help='How hard an oxide missing from the target is pushed to zero, 0.0..1.0 (default: 1.0)')
    parser.add_argument('--candidate-search', choices=CANDIDATE_SEARCH_MODES, default=SEARCH_EXHAUSTIVE,
                        help=f'Candidate search mode (default: {SEARCH_EXHAUSTIVE})')
    parser.add_argument('--quiet', action='store_true', help='Do not log the search process')
    args = parser.parse_args()

    # Smoke run target: the transparent glaze made of the five base materials
    target_umf = {
        "Al2O3": 0.379,
        "B2O3": 0.266,
        "CaO": 0.718,
        "Fe2O3": 0.002,
        "K2O": 0.086,
        "MgO": 0.048,
        "Na2O": 0.143,
        "SiO2": 3.151,
        "SrO": 0.005,
        "TiO2": 0.003
    }

    original_recipe = {
        "Волластонит МИВОЛЛ": 20,
        "Каолин КЖФ-1": 15,
        "Кварцевая мука Кварцверке W12": 20,
        "Нефелин-сиенит VR13": 30,
        "Улексит (Химпэк)": 15
    }

    if args.umf:
        target_umf = json.loads(args.umf)
        original_recipe = None

    print("Target UMF:")
    for oxide, value in sorted(target_umf.items()):
        print(f"  {oxide}: {value}")

    if original_recipe:
        print("\nOriginal recipe:")
        for material, weight in original_recipe.items():
            print(f"  {material}: {weight}%")

    inventory = resolve_inventory()

    print("\nSearching for solutions...")
    solutions = find_best_recipe(
        inventory,
        target_umf,
        max_materials=args.max_materials,
        max_solutions=args.solutions,
        verbose=not args.quiet,
        error_threshold=args.error_threshold,
        penalize_unlisted=args.penalize_unlisted,
        candidate_search=args.candidate_search,
    )

    if not solutions:
        print("\nNo solutions found for the target UMF!")
        return

    print(f"\nFound {len(solutions)} solutions!")
    for index, solution in enumerate(solutions):
        print(f"\nSolution {index + 1}")
        print(f"Error: {solution['error']:.4f} | objective: {solution['objective_error']:.4f} "
              f"| materials: {solution['materials_count']} | iterations: {solution['iterations']}")
        print("Recipe:")
        for material, weight in sorted(solution['recipe'].items(), key=lambda item: -item[1]):
            print(f"  {material}: {weight:.2f}%")

        print("Resulting UMF:")
        for oxide in sorted(set(solution['target_umf']) | set(solution['result_umf'])):
            actual = solution['result_umf'].get(oxide, 0.0)
            expected = solution['target_umf'].get(oxide, 0.0)
            if actual > 0.0005 or expected > 0.0005:
                print(f"  {oxide}: {actual:.3f} (target: {expected:.3f})")


if __name__ == "__main__":
    main()
