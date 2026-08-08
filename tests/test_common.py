#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# flake8: noqa
# pylint: disable=broad-exception-raised, raise-missing-from, too-many-arguments, redefined-outer-name
# pylint: disable=multiple-statements, logging-fstring-interpolation, trailing-whitespace, line-too-long
# pylint: disable=broad-exception-caught, missing-function-docstring, missing-class-docstring
# pylint: disable=f-string-without-interpolation
# pylance: disable=reportMissingImports, reportMissingModuleSource

import json
import unittest
import sys
import os
from unittest import mock

# Fix imports by adding parent directory to path
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_DIR)
import common
from common import (
    weights_to_umf,
    umf_to_weights,
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


class TestMaterialPriorities(unittest.TestCase):

    def test_base_materials_outrank_unlisted_ones(self):
        """Materials missing from priorities.json must not outrank the base ones"""
        materials = {m['name']: m['priority']
                     for m in load_materials(only_inventory=True, priority=True)}

        self.assertEqual(materials["Нефелин-сиенит VR13"], 2)
        self.assertGreater(materials["Бентонит"], materials["Нефелин-сиенит VR13"])
        self.assertGreater(materials["Карбонат цинка, ZnCO3"], materials["Нефелин-сиенит VR13"])


if __name__ == "__main__":
    unittest.main()
