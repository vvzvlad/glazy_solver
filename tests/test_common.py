#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# flake8: noqa
# pylint: disable=broad-exception-raised, raise-missing-from, too-many-arguments, redefined-outer-name
# pylint: disable=multiple-statements, logging-fstring-interpolation, trailing-whitespace, line-too-long
# pylint: disable=broad-exception-caught, missing-function-docstring, missing-class-docstring
# pylint: disable=f-string-without-interpolation
# pylance: disable=reportMissingImports, reportMissingModuleSource

import json
import math
import unittest
import sys
import os
from unittest import mock

# Fix imports by adding parent directory to path
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_DIR)
import common
from common import (
    OXIDE_SCALE_FLOOR,
    weights_to_umf,
    umf_to_weights,
    umf_deviation,
    calc_ratios_umf,
    calculate_umf_from_recipe,
    flux_oxides,
    make_json_safe,
    load_materials,
    load_molar_masses,
    oxides_classification,
    resolve_inventory,
    filter_materials_by_inventory,
    filter_materials_with_formula,
)
from solver_classic import calculate_recipe_composition, calculate_umf_error
from solver_iterative import _flux_sum

# The classification the code is supposed to be reading; the tests below load it
# straight from disk so that a hardcoded copy sneaking back into common.py fails
CLASSIFICATION_PATH = os.path.join(PROJECT_DIR, 'database', 'oxide_classification.json')


def read_classification_file():
    """Read database/oxide_classification.json without going through common.py"""
    with open(CLASSIFICATION_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

# Relative tolerance for round-trip checks, in percent. Both conversions round
# their output (2 decimals for weights, 3 for UMF) and molar masses themselves
# are rounded in the database, so an exact round-trip is not expected.
TOLERANCE_PERCENT = 1.0

# Absolute floor of the tolerance, for trace oxides whose 1% band is narrower
# than the rounding step of the result. UMF values are rounded to 3 decimals, so
# the floor is exactly one such step: enough to absorb a rounding difference, but
# small enough that dropping even the smallest expected oxide (0.002) fails the
# check. Measured deviations of the checked compositions are 0.000-0.017, and
# every non-zero one of them is already covered by the 1% relative band.
TOLERANCE_ABSOLUTE = 0.001


def assert_umf_close(test_case, expected_umf, actual_umf):
    """Compare UMF dictionaries oxide by oxide within the shared tolerance"""
    for oxide, expected_value in expected_umf.items():
        actual_value = actual_umf.get(oxide, 0)
        allowed = max(TOLERANCE_ABSOLUTE, abs(expected_value) * TOLERANCE_PERCENT / 100)
        test_case.assertLessEqual(
            abs(actual_value - expected_value), allowed,
            f"{oxide}: expected {expected_value}, got {actual_value}")


class TestOxideClassification(unittest.TestCase):
    """Классификация оксидов должна приходить из файла, а не из кода"""

    def test_classification_comes_from_the_json_file(self):
        """oxides_classification() отдаёт ровно содержимое файла классификации"""
        classification = read_classification_file()

        expected = {group: classification[group] for group in ('r2o', 'ro', 'r2o3', 'ro2')}

        self.assertEqual(oxides_classification(), expected)

    def test_classification_does_not_leak_the_unity_key(self):
        """unity — это имена групп, а не группа оксидов: наружу она не выходит"""
        self.assertNotIn('unity', oxides_classification())

    def test_classification_lists_are_copies(self):
        """Вызывающий не может испортить кэш, изменив полученные списки"""
        first = oxides_classification()
        first['ro'].append('НетТакогоОксида')

        self.assertNotIn('НетТакогоОксида', oxides_classification()['ro'])

    def test_flux_oxides_are_the_unity_groups_of_the_file(self):
        """flux_oxides() — это r2o + ro файла, в порядке групп, и PbO среди них"""
        classification = read_classification_file()

        self.assertEqual(classification['unity'], ['r2o', 'ro'])
        self.assertEqual(flux_oxides(), classification['r2o'] + classification['ro'])
        self.assertIn('PbO', flux_oxides())

    def test_lead_oxide_participates_in_the_normalization(self):
        """Состав со свинцом нормируется с участием PbO: сумма флюсов == 1"""
        umf = weights_to_umf({"PbO": 40.0, "SiO2": 50.0, "Al2O3": 10.0})

        self.assertAlmostEqual(umf['PbO'], 1.0, delta=1e-6)
        self.assertAlmostEqual(sum(umf.get(oxide, 0) for oxide in flux_oxides()), 1.0, delta=1e-6)

    def test_ratios_and_normalization_share_the_flux_set(self):
        """calc_ratios_umf и weights_to_umf считают флюсы по одним и тем же группам"""
        classification = read_classification_file()

        # MnO и FeO попадают в RO по классификации, но не попадали в него в
        # старом инлайн-списке calc_ratios_umf — на этом составе расхождение видно
        umf = weights_to_umf({
            "SiO2": 60.0, "Al2O3": 10.0, "CaO": 10.0, "Na2O": 5.0, "MnO": 8.0, "FeO": 7.0,
        })

        self.assertAlmostEqual(sum(umf.get(oxide, 0) for oxide in flux_oxides()), 1.0, delta=1e-6)

        r2o_sum = sum(umf.get(oxide, 0) for oxide in classification['r2o'])
        ro_sum = sum(umf.get(oxide, 0) for oxide in classification['ro'])

        ratios = calc_ratios_umf(umf)

        self.assertEqual(ratios['R2O:RO'], round(r2o_sum / ro_sum, 2))
        self.assertEqual(ratios['RO:R2O'], round(ro_sum / r2o_sum, 2))


class TestOxideClassificationIntegrity(unittest.TestCase):
    """Битый файл классификации обязан падать, а не тихо портить всю математику"""

    def setUp(self):
        # These tests replace the module level cache, so the real one is put
        # back afterwards whatever happens
        self.saved_cache = common._OXIDE_CLASSIFICATION_CACHE
        self.addCleanup(self.restore_cache)

    def restore_cache(self):
        common._OXIDE_CLASSIFICATION_CACHE = self.saved_cache

    def test_the_shipped_file_passes_validation(self):
        common._validate_oxide_classification(read_classification_file(), CLASSIFICATION_PATH)

    def test_a_missing_group_is_rejected(self):
        broken = read_classification_file()
        del broken['ro']

        with self.assertRaises(common.ClassificationError):
            common._validate_oxide_classification(broken, 'broken.json')

    def test_a_missing_unity_key_is_rejected(self):
        """Без unity список флюсов пуст, и вся нормировка UMF уезжает молча"""
        broken = read_classification_file()
        del broken['unity']

        with self.assertRaises(common.ClassificationError):
            common._validate_oxide_classification(broken, 'broken.json')

    def test_an_empty_unity_is_rejected(self):
        broken = read_classification_file()
        broken['unity'] = []

        with self.assertRaises(common.ClassificationError):
            common._validate_oxide_classification(broken, 'broken.json')

    def test_unity_naming_an_undefined_group_is_rejected(self):
        """Опечатка в имени группы — та же пустая нормировка"""
        broken = read_classification_file()
        broken['unity'] = ['r2o', 'r0']

        with self.assertRaises(common.ClassificationError):
            common._validate_oxide_classification(broken, 'broken.json')

    def test_a_group_that_is_a_bare_string_is_rejected(self):
        """Строка вместо списка развернулась бы в список отдельных букв"""
        broken = read_classification_file()
        broken['r2o'] = 'Na2O'

        with self.assertRaises(common.ClassificationError):
            common._validate_oxide_classification(broken, 'broken.json')

    def test_a_unity_that_is_not_a_list_is_rejected(self):
        broken = read_classification_file()
        broken['unity'] = 'r2o'

        with self.assertRaises(common.ClassificationError):
            common._validate_oxide_classification(broken, 'broken.json')

    def test_empty_unity_groups_are_rejected(self):
        """Пустые группы базиса — тот же пустой список флюсов, только через данные.

        Файл при этом валиден по структуре: все ключи на месте, все значения —
        списки. Но flux_oxides() вернёт [], weights_to_umf уйдёт в ветку «нет
        флюсов» и пронормирует состав по наименьшему оксиду.
        """
        broken = read_classification_file()
        broken['r2o'] = []
        broken['ro'] = []

        with self.assertRaises(common.ClassificationError):
            common._validate_oxide_classification(broken, 'broken.json')

    def test_one_empty_unity_group_is_rejected(self):
        """Достаточно потерять одну группу базиса, чтобы конвенция поехала"""
        broken = read_classification_file()
        broken['r2o'] = []

        with self.assertRaises(common.ClassificationError):
            common._validate_oxide_classification(broken, 'broken.json')

    def test_non_string_oxide_names_are_rejected(self):
        """Числа вместо имён оксидов не совпадут ни с чем и тихо обнулят базис"""
        broken = read_classification_file()
        broken['r2o'] = [1, 2]

        with self.assertRaises(common.ClassificationError):
            common._validate_oxide_classification(broken, 'broken.json')

    def test_unhashable_oxide_entries_are_rejected(self):
        """Иначе flux_oxides() кидает TypeError, а solver_iterative ловит его как «нет решений»"""
        broken = read_classification_file()
        broken['ro'] = [{'name': 'CaO'}]

        with self.assertRaises(common.ClassificationError):
            common._validate_oxide_classification(broken, 'broken.json')

    def test_unity_naming_itself_is_rejected(self):
        """unity: ['unity'] развернулось бы в имена групп вместо оксидов"""
        broken = read_classification_file()
        broken['unity'] = ['unity']

        with self.assertRaises(common.ClassificationError):
            common._validate_oxide_classification(broken, 'broken.json')

    def test_non_string_unity_entries_are_rejected(self):
        broken = read_classification_file()
        broken['unity'] = [{'group': 'r2o'}]

        with self.assertRaises(common.ClassificationError):
            common._validate_oxide_classification(broken, 'broken.json')

    def test_every_rejected_file_would_have_broken_the_flux_list(self):
        """Каждый отвергнутый случай действительно ломает flux_oxides(), а не придирка.

        Без валидации каждый из этих файлов даёт либо пустой список флюсов
        (нормировка молча уезжает на наименьший оксид), либо мусор вместо имён
        оксидов, либо TypeError.
        """
        def unvalidated_flux_oxides(classification):
            fluxes = []
            for group in classification['unity']:
                fluxes.extend(classification[group])
            return list(dict.fromkeys(fluxes))

        cases = {
            'empty unity groups': {'r2o': [], 'ro': []},
            'non string oxides': {'r2o': [1, 2]},
            'unity naming itself': {'unity': ['unity']},
        }

        for name, patch in cases.items():
            with self.subTest(case=name):
                broken = read_classification_file()
                broken.update(patch)

                fluxes = unvalidated_flux_oxides(broken)

                self.assertFalse(
                    fluxes and all(isinstance(oxide, str) and oxide in load_molar_masses()
                                   for oxide in fluxes),
                    f"{name}: expected a broken flux list, got {fluxes}")

                with self.assertRaises(common.ClassificationError):
                    common._validate_oxide_classification(broken, 'broken.json')

        # The unhashable case does not even survive the concatenation
        broken = read_classification_file()
        broken['ro'] = [{'name': 'CaO'}]
        with self.assertRaises(TypeError):
            unvalidated_flux_oxides(broken)

    def test_the_loader_validates_and_the_error_reaches_a_consumer(self):
        """Проверка должна стоять в загрузчике, а не только существовать.

        Тест бьёт по flux_oxides(), а не по _validate_oxide_classification:
        если вызов валидатора выкинуть из _oxide_classification(), тесты выше
        останутся зелёными, а этот упадёт.
        """
        broken = read_classification_file()
        del broken['unity']

        common._OXIDE_CLASSIFICATION_CACHE = None
        with mock.patch('builtins.open', mock.mock_open(read_data=json.dumps(broken))):
            with self.assertRaises(common.ClassificationError):
                flux_oxides()

        # A rejected file must not be cached as if it had been accepted
        self.assertIsNone(common._OXIDE_CLASSIFICATION_CACHE)

    def test_a_classification_error_is_not_a_value_error(self):
        """Иначе solver_iterative._solve_material_set поймает её и вернёт «нет решений»"""
        self.assertFalse(issubclass(common.ClassificationError, ValueError))

    def test_overlapping_unity_groups_are_not_double_counted(self):
        """Оксид из двух unity-групп попадает в список флюсов один раз.

        Иначе weights_to_umf делит на удвоенную сумму, а
        solver_iterative._flux_sum, который сворачивает список в set, — нет,
        и два движка начинают считать UMF в разных нормировках.
        """
        common._OXIDE_CLASSIFICATION_CACHE = {
            'r2o': ['Na2O', 'K2O', 'Li2O'],
            'ro': ['CaO'],
            'ro_extra': ['CaO'],
            'r2o3': ['Al2O3'],
            'ro2': ['SiO2'],
            'unity': ['r2o', 'ro', 'ro_extra'],
        }

        self.assertEqual(flux_oxides(), ['Na2O', 'K2O', 'Li2O', 'CaO'])

        umf = weights_to_umf({"SiO2": 60.0, "Al2O3": 10.0, "CaO": 30.0})

        self.assertAlmostEqual(umf['CaO'], 1.0, delta=1e-6)
        self.assertAlmostEqual(sum(umf.get(oxide, 0) for oxide in flux_oxides()), 1.0, delta=1e-6)
        self.assertAlmostEqual(_flux_sum(umf), 1.0, delta=1e-6)

    def test_presets_that_are_not_an_object_are_rejected(self):
        broken = read_classification_file()
        broken['unity_presets'] = ['Na2O', 'CaO']

        with self.assertRaises(common.ClassificationError):
            common._validate_oxide_classification(broken, 'broken.json')

    def test_an_empty_preset_is_rejected(self):
        """An empty basis is the "no fluxes" branch of weights_to_umf, silently"""
        broken = read_classification_file()
        broken['unity_presets'] = {'glazy': []}

        with self.assertRaises(common.ClassificationError):
            common._validate_oxide_classification(broken, 'broken.json')

    def test_a_preset_that_is_a_bare_string_is_rejected(self):
        """A string would expand into a list of single characters"""
        broken = read_classification_file()
        broken['unity_presets'] = {'glazy': 'Na2O'}

        with self.assertRaises(common.ClassificationError):
            common._validate_oxide_classification(broken, 'broken.json')

    def test_non_string_oxide_names_in_a_preset_are_rejected(self):
        broken = read_classification_file()
        broken['unity_presets'] = {'glazy': ['Na2O', 7]}

        with self.assertRaises(common.ClassificationError):
            common._validate_oxide_classification(broken, 'broken.json')

    def test_a_file_without_presets_is_still_valid(self):
        """The key is optional: a file without it simply has no named conventions"""
        without_presets = read_classification_file()
        without_presets.pop('unity_presets', None)

        common._validate_oxide_classification(without_presets, 'no_presets.json')


class TestFluxConventions(unittest.TestCase):
    """Named flux conventions: which oxides land in the unity denominator"""

    def test_the_shipped_file_defines_every_convention(self):
        classification = read_classification_file()

        self.assertIn('unity_presets', classification)
        self.assertEqual(sorted(classification['unity_presets']),
                         ['glazy', 'legacy', 'segerlab'])

    def test_the_segerlab_preset_is_the_upstream_flux_basis(self):
        """The preset is a transcription of segerlab.ru's Alcali + AEarth roles.

        Pinned as a literal list rather than derived: the point of the preset is
        that it does NOT follow our groups, so anything that recomputes it from
        our classification would defeat it. If upstream ever regroups an oxide,
        this test is where the difference has to be entered by hand.
        """
        self.assertEqual(
            sorted(flux_oxides('segerlab')),
            sorted(['Na2O', 'K2O', 'Li2O', 'CuO', 'Cu2O', 'SnO2',
                    'MgO', 'CaO', 'SrO', 'BaO', 'ZnO', 'PbO', 'CdO',
                    'MnO', 'MnO2', 'FeO', 'Fe2O3', 'CoO', 'V2O5']))

        # The one difference that moves every number: Fe2O3 is a flux there and
        # a stabilizer here, which is the whole 1.0022 factor between the two
        self.assertIn('Fe2O3', flux_oxides('segerlab'))
        self.assertNotIn('Fe2O3', flux_oxides())

    def test_manganese_dioxide_is_a_flux(self):
        """MnO2 belongs to the unity basis, not to RO2.

        Two materials of the database are almost pure MnO2. With MnO2 outside
        the basis the flux sum of a manganese glaze collapses to the traces its
        clay brings, and the UMF inflates by two orders of magnitude - so this
        is a numeric guarantee, not a taxonomy preference. See flux_oxides().
        """
        classification = read_classification_file()

        self.assertIn('MnO2', classification['ro'])
        self.assertNotIn('MnO2', classification['ro2'])
        self.assertIn('MnO2', flux_oxides())

        # A manganese glaze: the flux sum must be carried by the MnO2 itself
        composition = {"MnO2": 43.45, "SiO2": 33.4, "Al2O3": 10.8, "K2O": 0.3}
        umf = weights_to_umf(composition)
        self.assertAlmostEqual(umf['MnO2'], 1.0, delta=0.02)

    def test_unity_cannot_name_the_presets_block_as_a_group(self):
        """Otherwise the basis would expand into convention names, not oxides"""
        self.assertIn('unity_presets', common.CLASSIFICATION_META_KEYS)

        broken = read_classification_file()
        broken['unity'] = ['unity_presets']

        with self.assertRaises(common.ClassificationError):
            common._validate_oxide_classification(broken, 'broken.json')

    def test_classification_does_not_leak_the_presets_key(self):
        self.assertNotIn('unity_presets', oxides_classification())

    def test_the_legacy_preset_matches_the_group_definition(self):
        """The preset must not drift away from the "unity" groups it copies.

        Compared as sets on purpose: the groups list K2O before Na2O and the
        preset the other way round, and the order of the flux list changes
        nothing - every consumer sums over it.
        """
        self.assertEqual(set(flux_oxides()), set(flux_oxides('legacy')))

    def test_the_default_is_the_legacy_convention(self):
        """The default keeps every existing caller byte-identical"""
        composition = {"SiO2": 60.0, "Al2O3": 10.0, "CaO": 12.0, "Na2O": 4.0, "CuO": 4.0}

        self.assertEqual(weights_to_umf(composition),
                         weights_to_umf(composition, convention='legacy'))
        self.assertEqual(calculate_umf_from_recipe(composition),
                         calculate_umf_from_recipe(composition, convention='legacy'))

    def test_an_unknown_convention_is_rejected_and_lists_the_known_ones(self):
        with self.assertRaises(common.ClassificationError) as caught:
            flux_oxides('ceramicscalc')

        message = str(caught.exception)
        self.assertIn('ceramicscalc', message)
        self.assertIn('glazy', message)
        self.assertIn('legacy', message)

    def test_copper_moves_the_whole_umf_between_the_conventions(self):
        """A colorant is a flux in one convention and not in the other.

        CuO counts towards unity for us (ceramicscalc-2018) and does not for
        Glazy, so on a copper glaze the two conventions divide by different
        denominators and EVERY oxide of the UMF moves at once - which is why a
        target copied from Glazy cannot be compared with ours oxide by oxide.

        The factor is derived here from the molar masses rather than pinned as a
        literal: it is exactly the ratio of the two flux sums.
        """
        # A copper glaze: 4.77% CuO of a 100% analysis
        composition = {"SiO2": 69.23, "Al2O3": 9.50, "CaO": 10.00, "K2O": 3.00,
                       "Na2O": 2.30, "MgO": 1.00, "Fe2O3": 0.20, "CuO": 4.77}
        self.assertAlmostEqual(sum(composition.values()), 100.0, delta=1e-9)

        molar_masses = load_molar_masses()

        def flux_sum(convention):
            return sum(composition[oxide] / molar_masses[oxide]
                       for oxide in flux_oxides(convention) if oxide in composition)

        # CuO is the whole difference on this composition
        self.assertIn('CuO', flux_oxides())
        self.assertNotIn('CuO', flux_oxides('glazy'))
        self.assertAlmostEqual(flux_sum(None) - flux_sum('glazy'),
                               composition['CuO'] / molar_masses['CuO'], delta=1e-12)

        expected_factor = flux_sum(None) / flux_sum('glazy')

        # Direction and rough magnitude: dropping CuO shrinks the denominator,
        # so every UMF value grows, by about 22% on this composition
        self.assertGreater(expected_factor, 1.0)
        self.assertAlmostEqual(expected_factor, 1.22, delta=0.05)

        default_umf = weights_to_umf(composition)
        glazy_umf = weights_to_umf(composition, convention='glazy')

        self.assertNotEqual(default_umf, glazy_umf)

        for oxide, default_value in default_umf.items():
            # The UMF values are rounded to 3 decimals, so the comparison is
            # made against the rescaled default rather than value by ratio
            self.assertAlmostEqual(glazy_umf[oxide], default_value * expected_factor,
                                   delta=0.002,
                                   msg=f"{oxide} did not scale with the convention")


class TestUmfConversions(unittest.TestCase):

    def assert_umf_close(self, expected_umf, actual_umf):
        assert_umf_close(self, expected_umf, actual_umf)

    def test_round_trip_transparent_glaze(self):
        """UMF -> weights -> UMF keeps the transparent glaze formula"""
        umf = {
            "SiO2": 3.151,
            "Al2O3": 0.379,
            "B2O3": 0.266,
            "Na2O": 0.143,
            "K2O": 0.086,
            "CaO": 0.718,
            "MgO": 0.048,
        }

        self.assert_umf_close(umf, weights_to_umf(umf_to_weights(umf)))

    def test_round_trip_simple_feldspathic(self):
        """UMF -> weights -> UMF keeps a simple feldspathic formula"""
        umf = {
            "SiO2": 4.0,
            "Al2O3": 1.0,
            "Na2O": 0.5,
            "K2O": 0.5,
        }

        self.assert_umf_close(umf, weights_to_umf(umf_to_weights(umf)))

    def test_round_trip_with_zinc(self):
        """UMF -> weights -> UMF keeps a formula with several RO fluxes"""
        umf = {
            "SiO2": 2.5,
            "Al2O3": 0.4,
            "CaO": 0.6,
            "MgO": 0.2,
            "ZnO": 0.2,
        }

        self.assert_umf_close(umf, weights_to_umf(umf_to_weights(umf)))

    def test_weights_sum_to_100(self):
        """umf_to_weights returns weight percentages that add up to 100%"""
        umf = {"SiO2": 3.151, "Al2O3": 0.379, "CaO": 0.718, "Na2O": 0.143}

        weights = umf_to_weights(umf)

        self.assertAlmostEqual(sum(weights.values()), 100.0, delta=0.05)


class TestCalculateUmfFromRecipe(unittest.TestCase):

    def weight_composition_of(self, recipe):
        """Sum up the oxide formulas of a {material: percent} recipe"""
        materials = {m['name']: m for m in load_materials(only_inventory=False, priority=False)}

        composition = {}
        for name, percentage in recipe.items():
            self.assertIn(name, materials, f"material {name} is missing from the database")
            for oxide, content in materials[name].get('formula', {}).items():
                composition[oxide] = composition.get(oxide, 0.0) + content * (percentage / 100.0)
        return composition

    def test_umf_of_reference_transparent_recipe(self):
        """UMF of the reference transparent glaze recipe matches the known values"""
        recipe = {
            "Нефелин-сиенит VR13": 30,
            "Кварцевая мука Кварцверке W12": 20,
            "Волластонит МИВОЛЛ": 20,
            "Улексит (Химпэк)": 15,
            "Каолин КЖФ-1": 15,
        }

        expected_umf = {
            "SiO2": 3.144,
            "Al2O3": 0.378,
            "B2O3": 0.265,
            "Na2O": 0.143,
            "K2O": 0.086,
            "CaO": 0.717,
            "MgO": 0.048,
            "SrO": 0.005,
            "Fe2O3": 0.002,
            "TiO2": 0.003,
        }

        umf, raw_umf = calculate_umf_from_recipe(self.weight_composition_of(recipe))

        assert_umf_close(self, expected_umf, umf)

        # The raw values are the same numbers before rounding
        for oxide, value in umf.items():
            self.assertAlmostEqual(value, raw_umf[oxide], places=3)

    def test_matches_weights_to_umf(self):
        """calculate_umf_from_recipe agrees with weights_to_umf on the same input"""
        weight_composition = umf_to_weights({"SiO2": 3.0, "Al2O3": 0.4, "CaO": 0.7, "Na2O": 0.3})

        umf, _raw_umf = calculate_umf_from_recipe(weight_composition)

        self.assertEqual(umf, weights_to_umf(weight_composition))

    def test_fluxless_composition_normalizes_to_smallest_oxide(self):
        """Without fluxes the smallest molar amount is used as unity"""
        umf, _raw_umf = calculate_umf_from_recipe({"SiO2": 90.0, "Al2O3": 10.0})

        self.assertEqual(umf["Al2O3"], 1.0)
        self.assertGreater(umf["SiO2"], 1.0)


class TestMakeJsonSafe(unittest.TestCase):

    def test_infinities_and_nan_become_strings(self):
        self.assertEqual(make_json_safe(float('inf')), "Infinity")
        self.assertEqual(make_json_safe(float('-inf')), "-Infinity")
        self.assertEqual(make_json_safe(float('nan')), "NaN")

    def test_finite_values_are_untouched(self):
        self.assertEqual(make_json_safe(1.5), 1.5)
        self.assertEqual(make_json_safe(0), 0)
        self.assertEqual(make_json_safe("text"), "text")
        self.assertIsNone(make_json_safe(None))

    def test_nested_structures(self):
        obj = {
            "ratios": {"SiO2:Al2O3": float('inf'), "R2O:RO": 0.25},
            "errors": [float('nan'), 1.0, [float('-inf'), {"x": float('inf')}]],
            "name": "solution",
        }

        expected = {
            "ratios": {"SiO2:Al2O3": "Infinity", "R2O:RO": 0.25},
            "errors": ["NaN", 1.0, ["-Infinity", {"x": "Infinity"}]],
            "name": "solution",
        }

        self.assertEqual(make_json_safe(obj), expected)

    def test_result_is_json_serializable(self):
        import json

        payload = make_json_safe({"a": float('inf'), "b": [float('nan')]})

        self.assertEqual(json.loads(json.dumps(payload)), {"a": "Infinity", "b": ["NaN"]})


class TestInventoryHelpers(unittest.TestCase):

    def test_resolve_inventory_returns_explicit_data(self):
        explicit = ["Каолин КЖФ-1", "Мел, CaCO3"]

        self.assertEqual(resolve_inventory(explicit), explicit)

    def test_resolve_inventory_reads_in_inventory_flag(self):
        inventory = resolve_inventory()
        flagged = [m['name'] for m in load_materials(only_inventory=False, priority=False)
                   if m.get('inInventory') is True]

        self.assertGreater(len(inventory), 0)
        self.assertEqual(sorted(inventory), sorted(flagged))

    def test_resolve_inventory_none_falls_back_to_the_database(self):
        """None means "no explicit inventory": the inInventory materials are used"""
        flagged = [m['name'] for m in load_materials(only_inventory=False, priority=False)
                   if m.get('inInventory') is True]

        self.assertEqual(len(resolve_inventory(None)), len(flagged))

    def test_resolve_inventory_keeps_an_empty_list_empty(self):
        """An explicitly empty inventory must not fall back to the database.

        The distinction is load-bearing: the callers turn an empty inventory into
        a "no materials available" error, so resolving [] to the full database
        would silently solve against materials the caller does not have.
        """
        self.assertEqual(resolve_inventory([]), [])

    def test_filter_materials_by_inventory(self):
        materials = load_materials(only_inventory=False, priority=False)

        filtered = filter_materials_by_inventory(materials, ["Каолин КЖФ-1", "нет такого материала"])

        self.assertEqual([m['name'] for m in filtered], ["Каолин КЖФ-1"])

    def test_filter_materials_with_formula(self):
        """Материалы с нулевой формулой отсеиваются, остальные не трогаются"""
        materials = [
            {'name': 'с оксидами', 'formula': {'SiO2': 60.0, 'Al2O3': 40.0}},
            {'name': 'только ППП', 'formula': {'Loi': 100.0}},
            {'name': 'пустая формула', 'formula': {}},
            {'name': 'без формулы'},
        ]

        filtered = filter_materials_with_formula(materials)

        self.assertEqual([m['name'] for m in filtered], ['с оксидами'])

    def test_filter_materials_with_formula_keeps_the_whole_inventory(self):
        """Ни один материал инвентаря не имеет пустой формулы: поведение не меняется"""
        inventory_materials = load_materials(only_inventory=True, priority=False)

        self.assertEqual(len(filter_materials_with_formula(inventory_materials)),
                         len(inventory_materials))


class TestWaterSolubleFlag(unittest.TestCase):
    """The isWaterSoluble flag of database/materials.json

    Carried over from the upstream library the database is a dump of. It is
    data only for now - nothing in the solvers reads it yet - which is exactly
    why it needs a test: a field no code touches is a field the next dump
    regeneration drops without anything going red.
    """

    def test_every_material_carries_the_flag_as_a_boolean(self):
        materials = load_materials(only_inventory=False, priority=False)

        self.assertEqual(len(materials), 216)
        missing = [m['name'] for m in materials if 'isWaterSoluble' not in m]
        self.assertEqual(missing, [])

        not_boolean = [m['name'] for m in materials
                       if not isinstance(m['isWaterSoluble'], bool)]
        self.assertEqual(not_boolean, [])

    def test_exactly_twenty_seven_materials_are_water_soluble(self):
        """The count upstream publishes; a drift means the dump was regenerated badly"""
        materials = load_materials(only_inventory=False, priority=False)

        soluble = [m['name'] for m in materials if m['isWaterSoluble']]
        self.assertEqual(len(soluble), 27)

        # Spot checks in both directions: the classic soluble raw materials are
        # flagged and an ordinary insoluble one is not
        for name in ["Бура, Na2O 2 B2O3 10 H2O", "Сода кольцинированная, Na2CO3",
                     "Поташ (Карбонат калия)", "Силикат натрия", "Вода"]:
            self.assertIn(name, soluble)
        self.assertNotIn("Каолин КЖФ-1", soluble)

    def test_the_flag_survives_the_inventory_load_path(self):
        """load_materials() adds priority and filters; it must not drop fields"""
        materials = load_materials(only_inventory=True, priority=True)

        self.assertGreater(len(materials), 0)
        self.assertTrue(all('isWaterSoluble' in m for m in materials))

        # Borax is both in the inventory and water soluble - the one record that
        # makes this path meaningful
        borax = next(m for m in materials if m['name'] == "Бура, Na2O 2 B2O3 10 H2O")
        self.assertIs(borax['isWaterSoluble'], True)


class TestMaterialPriorities(unittest.TestCase):

    def test_base_materials_outrank_unlisted_ones(self):
        """Materials missing from priorities.json must not outrank the base ones"""
        materials = {m['name']: m['priority']
                     for m in load_materials(only_inventory=True, priority=True)}

        self.assertEqual(materials["Нефелин-сиенит VR13"], 2)
        self.assertGreater(materials["Бентонит"], materials["Нефелин-сиенит VR13"])
        self.assertGreater(materials["Карбонат цинка, ZnCO3"], materials["Нефелин-сиенит VR13"])


class TestUmfDeviation(unittest.TestCase):
    """
    The one scale the whole system judges a formula on (TZ_SOLVER_V2.md 10.18)

    The cases below are the ones that decide the definition, not a sample of it:
    each pins a property that the retired absolute norm got wrong or that a
    plausible simplification of the code would break.
    """

    def test_a_dropped_colourant_is_a_fifth_of_its_scale_and_a_fiftieth_of_the_norm(self):
        """
        The case the whole change exists for: a trace oxide the answer forgot

        A target asking for CoO 0.02 answered by a recipe with no cobalt in it
        is 20% out on the one oxide that decides what the glaze looks like, and
        0.02 in a norm whose retired limit was 0.1. The relative reading fails
        it and the absolute one waves it through.
        """
        deviation = umf_deviation({'SiO2': 3.0, 'Al2O3': 0.4, 'CoO': 0.02},
                                  {'SiO2': 3.0, 'Al2O3': 0.4})

        self.assertAlmostEqual(0.2, deviation['max_relative'], places=12)
        self.assertEqual('CoO', deviation['worst_oxide'])
        self.assertAlmostEqual(0.02, deviation['l2_absolute'], places=12)
        self.assertLess(deviation['l2_absolute'], 0.1)

    def test_contamination_the_target_never_asked_for_is_a_deviation(self):
        """An oxide only the answer carries is scored against 0.0, at the floor"""
        deviation = umf_deviation({'SiO2': 3.0}, {'SiO2': 3.0, 'BaO': 0.03})

        self.assertEqual('BaO', deviation['worst_oxide'])
        self.assertAlmostEqual(0.03 / OXIDE_SCALE_FLOOR, deviation['max_relative'], places=12)
        self.assertEqual({'target': 0.0, 'actual': 0.03, 'delta': 0.03,
                          'relative': 0.03 / OXIDE_SCALE_FLOOR},
                         deviation['per_oxide']['BaO'])

        # ... and it stays out of the retired norm, which only ever looked at
        # the target's own oxides. That asymmetry is deliberate: it is what
        # keeps a stored umf_error comparable with the runs before 10.18.
        self.assertEqual(0.0, deviation['l2_absolute'])

    def test_the_delta_is_signed_and_says_which_way_the_answer_missed(self):
        """
        Same field name as feasibility's per_oxide row, so the same convention

        delta = actual - target: positive is too much of the oxide, negative is
        too little. Which of the two it is is the one thing a reader wants out
        of this record, and a magnitude cannot say. check_feasibility() fills a
        row of the same shape under the same name with closest - target, and two
        blocks that disagree about the direction while agreeing about everything
        else are worse than either of them alone. The headline numbers do not
        move with the sign - "relative" is the magnitude.
        """
        under = umf_deviation({'SiO2': 3.0}, {'SiO2': 2.9})
        over = umf_deviation({'SiO2': 3.0}, {'SiO2': 3.1})

        self.assertAlmostEqual(-0.1, under['per_oxide']['SiO2']['delta'], places=12)
        self.assertAlmostEqual(0.1, over['per_oxide']['SiO2']['delta'], places=12)

        # ... and the two miss by the same amount, on every number that gates
        self.assertAlmostEqual(under['per_oxide']['SiO2']['relative'],
                               over['per_oxide']['SiO2']['relative'], places=12)
        self.assertAlmostEqual(under['max_relative'], over['max_relative'], places=12)
        self.assertAlmostEqual(under['l2_absolute'], over['l2_absolute'], places=12)

    def test_the_floor_binds_below_it_and_gets_out_of_the_way_above_it(self):
        """max(target, floor) - the same absolute miss, two different readings"""
        below = umf_deviation({'MgO': 0.05}, {'MgO': 0.09})
        above = umf_deviation({'SiO2': 3.0}, {'SiO2': 3.04})

        # 0.04 against the floor, not against the 0.05 that was asked for
        self.assertAlmostEqual(0.4, below['max_relative'], places=12)
        # 0.04 against 3.0, because the target is well clear of the floor
        self.assertAlmostEqual(0.04 / 3.0, above['max_relative'], places=12)

        # And the floor is a parameter, not a constant baked into the formula
        self.assertAlmostEqual(0.8, umf_deviation({'MgO': 0.05}, {'MgO': 0.09},
                                                  floor=0.05)['max_relative'],
                               places=12)

    def test_loss_on_ignition_never_enters_a_formula_comparison(self):
        """Loi is bookkeeping; a UMF that carries one must not be judged on it"""
        deviation = umf_deviation({'SiO2': 3.0, 'Loi': 8.0},
                                  {'SiO2': 3.0, 'LOI': 0.0})

        self.assertEqual(0.0, deviation['max_relative'])
        self.assertEqual(0.0, deviation['l2_absolute'])
        self.assertEqual(['SiO2'], sorted(deviation['per_oxide']))
        self.assertEqual([], deviation['dropped'])

    def test_a_non_finite_value_is_reported_rather_than_swallowed(self):
        """
        A NaN must not read as "everything is fine"

        Every comparison against a NaN is false, so an oxide carrying one would
        silently never be the worst. Naming it in "dropped" is the only honest
        answer - and the numbers go to None with it, because a maximum taken
        over the oxides that survived is a lower bound, and a lower bound in a
        distribution of honest values reads as the best case in it.
        """
        deviation = umf_deviation(
            {'SiO2': 3.0, 'Al2O3': 0.4, 'CaO': 0.7, 'MgO': 0.1},
            {'SiO2': float('nan'), 'Al2O3': float('inf'), 'CaO': None, 'MgO': 0.12})

        self.assertEqual(['Al2O3', 'CaO', 'SiO2'], sorted(deviation['dropped']))
        self.assertIsNone(deviation['max_relative'])
        self.assertIsNone(deviation['l2_absolute'])
        self.assertIsNone(deviation['worst_oxide'])

        # ... and what WAS computed is still there for the reader
        self.assertEqual(['MgO'], sorted(deviation['per_oxide']))
        self.assertAlmostEqual(0.2, deviation['per_oxide']['MgO']['relative'], places=12)

    def test_a_formula_that_could_not_be_read_is_not_a_perfect_score(self):
        """
        This is the regression that motivates the None: 0.0 is the BEST value

        max_relative and l2_absolute are both "lower is better" metrics whose
        distributions the benchmark tracks and whose maxima --check gates. A
        case that could not be compared at all, scored 0.0, would enter every
        one of those as the best case in the sample and would pull the gated
        maximum down - it would read as an improvement.
        """
        unreadable = umf_deviation({'SiO2': 3.0}, {'SiO2': float('nan')})
        self.assertIsNone(unreadable['max_relative'])
        self.assertIsNone(unreadable['l2_absolute'])
        self.assertEqual(['SiO2'], unreadable['dropped'])

        # A perfect answer, for contrast: same 0.0-shaped result, but real
        perfect = umf_deviation({'SiO2': 3.0}, {'SiO2': 3.0})
        self.assertEqual(0.0, perfect['max_relative'])
        self.assertEqual(0.0, perfect['l2_absolute'])
        self.assertEqual([], perfect['dropped'])
        # Nothing deviates, so there is no worst oxide to name
        self.assertIsNone(perfect['worst_oxide'])

    def test_nothing_to_compare_is_none_and_no_worst_oxide(self):
        """
        Two empty formulas are not a match, they are an absence of a comparison

        Same None as the unreadable case above, and "dropped" is what tells the
        two apart: empty here, populated there.
        """
        deviation = umf_deviation({}, {})

        self.assertIsNone(deviation['max_relative'])
        self.assertIsNone(deviation['worst_oxide'])
        self.assertIsNone(deviation['l2_absolute'])
        self.assertEqual({}, deviation['per_oxide'])
        self.assertEqual([], deviation['dropped'])

    # A target and an answer taken verbatim off the corpus (glazy_id 423584),
    # kept here because they are the cheapest proof that the SUMMATION ORDER of
    # l2_absolute is load bearing. The keys are in the order the forward
    # calculation produced them, which is not alphabetical, and summing the
    # squares alphabetically instead lands one ulp away:
    #   target key order -> 0.005385164807134405
    #   sorted           -> 0.005385164807134406
    # 27 of the 300 corpus cases behave like this, and they rewrote 55 of the
    # 450 rows of bench/quality_baseline.json when the order was wrong.
    ORDER_SENSITIVE_TARGET = {
        'SiO2': 2.733, 'Al2O3': 0.599, 'Na2O': 0.096, 'K2O': 0.204, 'CaO': 0.403,
        'Fe2O3': 0.009, 'MgO': 0.297, 'P2O5': 0.001, 'TiO2': 0.003,
    }
    ORDER_SENSITIVE_ACTUAL = {
        'CaO': 0.402, 'SiO2': 2.728, 'Al2O3': 0.598, 'Na2O': 0.096, 'K2O': 0.205,
        'MgO': 0.296, 'P2O5': 0.001, 'TiO2': 0.003, 'Fe2O3': 0.009,
    }

    def test_l2_absolute_keeps_the_retired_metric_summation_order(self):
        """
        The diagnostic field is the OLD number, to the last bit

        bench/quality_baseline.json and every line of bench/history.jsonl store
        umf_error, and it only stays comparable across 10.18 if the new code
        computes exactly what solver_classic.calculate_umf_error computed -
        which sums over the target's keys in the TARGET's own order. Float
        addition is not associative, so this test is written on a case where
        the alphabetical order gives a different double; the fixtures of
        reference_recipes.json cannot catch it, because their UMF keys happen
        to be sorted already and there is nothing left to reorder.
        """
        target, actual = self.ORDER_SENSITIVE_TARGET, self.ORDER_SENSITIVE_ACTUAL
        self.assertNotEqual(list(target), sorted(target),
                            'the fixture stopped being order sensitive')

        expected = float(calculate_umf_error(target, actual))
        self.assertEqual(expected, umf_deviation(target, actual)['l2_absolute'])

        # ... and the sorted order really is a different number, so the
        # assertion above is not passing by luck
        alphabetical = math.sqrt(sum((target[ox] - actual.get(ox, 0.0)) ** 2
                                     for ox in sorted(target)))
        self.assertNotEqual(alphabetical, expected)

    def test_l2_absolute_reproduces_the_retired_metric_on_real_recipes(self):
        """The same equality over the 11 committed fixtures, end to end"""
        with open(os.path.join(PROJECT_DIR, 'tests', 'fixtures',
                               'reference_recipes.json'), encoding='utf-8') as f:
            fixtures = json.load(f)

        materials = load_materials(only_inventory=True, priority=True)

        checked = 0
        for entry in fixtures:
            target = entry['umf']
            actual = weights_to_umf(calculate_recipe_composition(materials, entry['recipe']))

            self.assertEqual(float(calculate_umf_error(target, actual)),
                             umf_deviation(target, actual)['l2_absolute'],
                             f"{entry['id']}: the diagnostic drifted from the retired metric")
            # The same equality on a reordered target. This does NOT
            # discriminate between the two summation orders - on these eleven
            # fixtures the reference itself gives an identical double either
            # way, and eight of them have an error of exactly 0.0 - so it is a
            # consistency check and nothing more; the test above is the guard.
            reversed_target = dict(reversed(list(target.items())))
            self.assertEqual(float(calculate_umf_error(reversed_target, actual)),
                             umf_deviation(reversed_target, actual)['l2_absolute'],
                             f"{entry['id']}: the diagnostic drifted on a reordered target")
            checked += 1

        self.assertGreaterEqual(checked, 10)


if __name__ == "__main__":
    unittest.main()
