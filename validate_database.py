#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# flake8: noqa
# pylint: disable=broad-exception-raised, raise-missing-from, too-many-arguments, redefined-outer-name
# pylint: disable=multiple-statements, logging-fstring-interpolation, trailing-whitespace, line-too-long
# pylint: disable=broad-exception-caught, missing-function-docstring, missing-class-docstring
# pylint: disable=f-string-without-interpolation
# pylance: disable=reportMissingImports, reportMissingModuleSource
# type: ignore

# Checks on database/materials.json that no amount of testing the solvers can
# make: a material can be perfectly well formed and still be unusable, and the
# failure shows up much later as a wrong number rather than as an exception.
#
#   python validate_database.py        # report, exit 1 if there are errors
#   python validate_database.py --quiet
#
# tests/test_database.py runs the same checks and fails on errors only.
#
# THE LEVELS ARE THE WHOLE DESIGN HERE. An error is a record the code cannot
# use; a warning is a record worth a human look; a note is a fact about the
# database that the reader should know and that is not a defect at all. The
# distinction is not cosmetic - three of the five checks of TZ_SOLVER_V2.md 2.3
# were specified as errors and would have failed on the shipped database from
# the first commit, because what they describe is legal:
#
#   * 37 of the 216 materials have an analysis that sums to zero - water, CMC,
#     gypsum, silicon carbide in six grades, 21 pigments. They are real stock
#     entries that simply carry no oxide, and 6.5 of the same specification
#     calls the empty formula of SiC legal outright. They are a NOTE, under
#     their own category, and the solvers already drop them by
#     common.filter_materials_with_formula().
#   * Two analyses sum above the [15, 105] band: Cryolite at 122.90 and Zircon
#     at 135.22. Both are correct - the LOI model allows a sum above 100 - so
#     the band stays and they stay warnings.
#   * Three materials carry a "Loi" key, which is loss on ignition and not a
#     lost oxide. It is a NOTE, and it is excluded from the formula sum.

import argparse
import json
import logging
import os
import sys

from common import NON_OXIDE_KEYS

logger = logging.getLogger(__name__)

DATABASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'database')

LEVEL_ERROR = 'error'
LEVEL_WARNING = 'warning'
LEVEL_NOTE = 'note'
LEVELS = (LEVEL_ERROR, LEVEL_WARNING, LEVEL_NOTE)

# Categories, so that a caller can count what it cares about without matching on
# message text
CATEGORY_NON_OXIDE_MATERIAL = 'non-oxide material'
CATEGORY_FORMULA_SUM = 'formula sum out of range'
CATEGORY_UNKNOWN_OXIDE = 'oxide without a molar mass'
CATEGORY_LOI_KEY = 'loss on ignition key'
CATEGORY_DUPLICATE_NAME = 'duplicate material name'
CATEGORY_UNKNOWN_PRIORITY = 'priority for a material that does not exist'
CATEGORY_BAD_RECORD = 'malformed record'

# Plausible band for the sum of an oxide analysis, in weight percent. The lower
# end catches a truncated record without condemning a carbonate with a large
# LOI; the upper end is above 100 because an analysis reported on the ignited
# basis legitimately sums higher.
FORMULA_SUM_MIN = 15.0
FORMULA_SUM_MAX = 105.0


def issue(level, category, message, material=None):
    return {"level": level, "category": category, "message": message, "material": material}


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_database(database_dir=DATABASE_DIR):
    """
    Read the three files the checks need, straight from disk

    Deliberately not through common.load_materials(): that one merges priorities
    into the records and filters by inventory, and a validator has to see the
    file as it is written.

    Returns:
        (materials, priorities, molar_masses)
    """
    return (load_json(os.path.join(database_dir, 'materials.json')),
            load_json(os.path.join(database_dir, 'priorities.json')),
            load_json(os.path.join(database_dir, 'molar_masses.json')))


def validate_materials(materials, molar_masses):
    """
    Every check that looks at one material record

    Args:
        materials: the list as stored in database/materials.json
        molar_masses: the {oxide: mass} table

    Returns:
        list of issues
    """
    issues = []
    seen_names = {}

    for position, material in enumerate(materials):
        if not isinstance(material, dict):
            issues.append(issue(LEVEL_ERROR, CATEGORY_BAD_RECORD,
                                f"record {position} is not an object"))
            continue

        name = material.get('name')
        if not isinstance(name, str) or not name.strip():
            issues.append(issue(LEVEL_ERROR, CATEGORY_BAD_RECORD,
                                f"record {position} has no usable name"))
            continue

        seen_names[name] = seen_names.get(name, 0) + 1

        formula = material.get('formula')
        if formula is None:
            formula = {}
        if not isinstance(formula, dict):
            issues.append(issue(LEVEL_ERROR, CATEGORY_BAD_RECORD,
                                f"formula is not an object", name))
            continue

        total = 0.0
        for oxide, value in formula.items():
            try:
                number = float(value)
            except (TypeError, ValueError):
                issues.append(issue(LEVEL_ERROR, CATEGORY_BAD_RECORD,
                                    f"{oxide} is not a number: {value!r}", name))
                continue

            if oxide in NON_OXIDE_KEYS:
                # Loss on ignition: bookkeeping, not a lost oxide. Out of the
                # sum, out of the unknown oxide check, and said out loud
                issues.append(issue(LEVEL_NOTE, CATEGORY_LOI_KEY,
                                    f"carries the non-oxide key {oxide} = {number:g}, "
                                    f"excluded from the analysis sum", name))
                continue

            total += number

            if oxide not in molar_masses:
                issues.append(issue(LEVEL_ERROR, CATEGORY_UNKNOWN_OXIDE,
                                    f"{oxide} has no entry in molar_masses.json, so it is "
                                    f"silently lost in every UMF conversion", name))

        if total == 0.0:
            issues.append(issue(LEVEL_NOTE, CATEGORY_NON_OXIDE_MATERIAL,
                                f"analysis carries no oxide - a pigment, SiC, CMC, water "
                                f"or the like; the solvers drop it", name))
        elif not FORMULA_SUM_MIN <= total <= FORMULA_SUM_MAX:
            issues.append(issue(LEVEL_WARNING, CATEGORY_FORMULA_SUM,
                                f"analysis sums to {total:.2f}, outside "
                                f"[{FORMULA_SUM_MIN:g}, {FORMULA_SUM_MAX:g}]", name))

    for name, count in seen_names.items():
        if count > 1:
            issues.append(issue(LEVEL_ERROR, CATEGORY_DUPLICATE_NAME,
                                f"appears {count} times; a recipe naming it is ambiguous",
                                name))

    return issues


def validate_priorities(priorities, materials):
    """
    Priorities pointing at materials that are not in the database

    A warning and not an error: an entry nobody can reach changes no number, it
    just means the file has drifted - a renamed or removed material.
    """
    known = {material.get('name') for material in materials if isinstance(material, dict)}
    issues = []

    for name in priorities:
        if name not in known:
            issues.append(issue(LEVEL_WARNING, CATEGORY_UNKNOWN_PRIORITY,
                                f"priorities.json sets a priority for a material that is "
                                f"not in materials.json", name))

    return issues


def validate_database(materials=None, priorities=None, molar_masses=None,
                      database_dir=DATABASE_DIR):
    """
    Run every check

    Args:
        materials, priorities, molar_masses: injected data, for the tests. Any
            of them left as None is read from database_dir

    Returns:
        list of issues, each {"level", "category", "message", "material"}
    """
    if materials is None or priorities is None or molar_masses is None:
        file_materials, file_priorities, file_molar = load_database(database_dir)
        materials = file_materials if materials is None else materials
        priorities = file_priorities if priorities is None else priorities
        molar_masses = file_molar if molar_masses is None else molar_masses

    return validate_materials(materials, molar_masses) + validate_priorities(priorities, materials)


def by_level(issues, level):
    return [item for item in issues if item['level'] == level]


def count_by_category(issues, level=None):
    counts = {}
    for item in issues:
        if level is not None and item['level'] != level:
            continue
        counts[item['category']] = counts.get(item['category'], 0) + 1
    return counts


def format_report(issues):
    """
    The report, errors first, with the per category counts at the end

    Materials of the same category are listed on one line each: a category with
    37 members is a fact about the database, and 37 paragraphs about it would
    hide the one line that is an actual defect.
    """
    lines = []

    for level in LEVELS:
        found = by_level(issues, level)
        if not found:
            continue
        lines.append(f"{level.upper()}S: {len(found)}")
        for item in found:
            where = f"{item['material']}: " if item['material'] else ""
            lines.append(f"  [{item['category']}] {where}{item['message']}")
        lines.append("")

    lines.append("summary")
    for level in LEVELS:
        counts = count_by_category(issues, level)
        total = sum(counts.values())
        detail = ', '.join(f"{category} x{count}" for category, count in sorted(counts.items()))
        lines.append(f"  {level}: {total}{' (' + detail + ')' if detail else ''}")

    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate database/materials.json against molar_masses.json and priorities.json")
    parser.add_argument('--database', default=DATABASE_DIR,
                        help="directory holding the json files (default: ./database)")
    parser.add_argument('--quiet', action='store_true',
                        help="print only the summary and the errors")
    args = parser.parse_args(argv)

    issues = validate_database(database_dir=args.database)
    errors = by_level(issues, LEVEL_ERROR)

    if args.quiet:
        for item in errors:
            where = f"{item['material']}: " if item['material'] else ""
            print(f"  [{item['category']}] {where}{item['message']}")
        counts = count_by_category(issues)
        print(f"errors {len(errors)}, warnings {len(by_level(issues, LEVEL_WARNING))}, "
              f"notes {len(by_level(issues, LEVEL_NOTE))}, categories {len(counts)}")
    else:
        print(format_report(issues))

    # Only errors decide the exit code: a warning is for a human to look at, and
    # a CI job that fails on the 37 legal pigments teaches everyone to ignore it
    return 1 if errors else 0


if __name__ == '__main__':
    sys.exit(main())
