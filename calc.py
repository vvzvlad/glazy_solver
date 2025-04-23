molar_masses = {
    "SiO2": 60.084,
    "Al2O3": 101.961,
    "B2O3": 69.620,
    "Li2O": 29.881,
    "Na2O": 61.979,
    "K2O": 94.196,
    "BeO": 25.012,
    "MgO": 40.304,
    "CaO": 56.077,
    "SrO": 103.620,
    "BaO": 153.326,
    "P2O5": 141.944,
    "TiO2": 79.866,
    "ZrO": 123.223,
    "ZrO2": 123.223,
    "V2O5": 181.880,
    "Cr2O3": 151.990,
    "MnO": 70.937,
    "MnO2": 86.936,
    "FeO": 71.844,
    "Fe2O3": 159.688,
    "CoO": 74.932,
    "NiO": 74.693,
    "CuO": 79.545,
    "Cu2O": 143.091,
    "CdO": 128.410,
    "ZnO": 81.380,
    "PbO": 223.200,
    "SnO2": 150.709,
    "HfO2": 210.490,
    "Nb2O5": 265.810,
    "Ta2O5": 441.890,
    "MoO3": 143.940,
    "WO3": 231.840,
    "OsO2": 223.220,
    "IrO2": 224.220,
    "PtO2": 227.080,
    "Ag2O": 231.735,
    "Au2O3": 441.930,
    "GeO2": 104.640,
    "As2O3": 197.840,
    "Sb2O3": 291.520,
    "Bi2O3": 465.960,
    "SeO2": 110.960,
    "La2O3": 325.810,
    "CeO2": 172.115,
    "PrO2": 171.908,
    "Pr2O3": 329.810,
    "Nd2O3": 336.480,
    "U3O8": 842.000,
    "Sm2O3": 348.720,
    "Eu2O3": 351.930,
    "Tb2O3": 365.930,
    "Dy2O3": 372.000,
    "Ho2O3": 377.860,
    "Er2O3": 382.520,
    "Tm2O3": 384.840,
    "Yb2O3": 394.080,
    "Lu2O3": 397.930,
    "Gd2O3": 362.500,
    "Y2O3": 225.810,
    "Tl2O3": 456.760,
    "Ga2O3": 187.440,
    "F": 18.998,
    "Loi": 0.000  # Потери при прокаливании, не имеет молярной массы
}



def weights_to_umf(weight_dict):
    """Переводит весовые доли оксидов в UMF (мольные соотношения)"""
    moles = {
        oxide: weight / molar_masses[oxide]
        for oxide, weight in weight_dict.items()
        if oxide in molar_masses
    }
    min_moles = min(moles.values())
    umf = {oxide: round(amount / min_moles, 3) for oxide, amount in moles.items()}
    return umf

def umf_to_weights(umf_dict):
    """Переводит UMF (мольные количества) в весовые доли оксидов"""
    weights = {
        oxide: amount * molar_masses[oxide]
        for oxide, amount in umf_dict.items()
        if oxide in molar_masses
    }
    total = sum(weights.values())
    weight_fractions = {oxide: round(100 * w / total, 3) for oxide, w in weights.items()}
    return weight_fractions

# 🔍 Пример использования:

# Вариант 1: из весов в UMF
weights = {"Na2O": 19.2, "P2O5": 22.0}
umf_result = weights_to_umf(weights)
print("UMF:", umf_result)

# Вариант 2: из UMF обратно в весовые доли
weights_back = umf_to_weights(umf_result)
print("Весовые доли:", weights_back)