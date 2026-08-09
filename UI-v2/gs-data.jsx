// Mock dataset for the Glaze Solver UI prototype.
// Materials, recipes and solutions come from real runs of the project solvers
// on database/materials.json. The sensitivity figures come from the real
// sensitivity.py run on the reference recipe, so the ranking shown here is the
// one the backend actually produces.

const OXIDE_GROUPS = {
  r2o: ['Na2O', 'K2O', 'Li2O'],
  ro: ['MgO', 'CaO', 'SrO', 'BaO', 'ZnO', 'PbO', 'MnO', 'MnO2', 'FeO', 'CoO', 'NiO', 'CuO'],
  r2o3: ['Al2O3', 'B2O3', 'Fe2O3', 'Cr2O3', 'Mn2O3'],
  ro2: ['SiO2', 'TiO2', 'ZrO2', 'SnO2', 'GeO2'],
  unity: ['r2o', 'ro'],
};

const TARGET = { K2O: 0.086, Na2O: 0.143, MgO: 0.048, CaO: 0.717, Al2O3: 0.378, B2O3: 0.265, SiO2: 3.144 };

// Oxides that no ordinary base material carries as an impurity: they only ever
// appear because somebody added them on purpose. Computed from materials.json -
// each of these occurs solely in materials that are essentially pure oxide.
const NEVER_ACCIDENTAL = ['CoO', 'CuO', 'Cr2O3', 'SnO2', 'ZrO2', 'NiO', 'Li2O'];

// The level at which an unrequested oxide starts to change the fired glaze.
// PROVISIONAL: literature anchors only. The corpus pass over the Glazy dump
// (spec stage 7.6) replaces these with the lower quartile of deliberate use.
const OXIDE_THRESHOLD = {
  TiO2: { notable: 0.02, strong: 0.08, effect: 'глушит' },
  Fe2O3: { notable: 0.02, strong: 0.06, effect: 'красит' },
  P2O5: { notable: 0.03, strong: 0.10, effect: 'глушит' },
  ZrO2: { notable: 0.02, strong: 0.06, effect: 'глушит' },
  SnO2: { notable: 0.01, strong: 0.04, effect: 'глушит' },
  CoO: { notable: 0.004, strong: 0.02, effect: 'синий' },
  CuO: { notable: 0.01, strong: 0.04, effect: 'зелёный' },
  MnO2: { notable: 0.02, strong: 0.08, effect: 'красит' },
};
// Fluxes are judged against the unity budget, not an absolute level: an
// unrequested flux moves the denominator and therefore the whole formula.
const FLUX_SHARE_NOTABLE = 0.01;

const MATERIALS = [
  { name: 'Нефелин-сиенит VR13', group: 'Шпаты и фритты', inv: true, price: 144, tol: 0.02, loi: 'residual',
    formula: { SiO2: 62.93, Al2O3: 20.33, Na2O: 7.09, K2O: 7.57 } },
  { name: 'Полевой шпат FFF', group: 'Шпаты и фритты', inv: true, price: 130, tol: 0.02, loi: 'residual',
    formula: { SiO2: 68.2, Al2O3: 18.1, K2O: 10.4, Na2O: 2.9 } },
  { name: 'Фритта 100 (Рускерамика)', group: 'Шпаты и фритты', inv: true, price: 1760, tol: 0.03, loi: 'unknown',
    formula: { SiO2: 60.5, B2O3: 12, Al2O3: 9.5, K2O: 7, CaO: 5, Na2O: 5 } },
  { name: 'Каолин КЖФ-1', group: 'Глины', inv: true, price: 78, tol: 0.05, loi: 'residual', clay: true,
    formula: { SiO2: 47, Al2O3: 36, K2O: 1, Fe2O3: 0.6, TiO2: 0.4, CaO: 0.15, Na2O: 0.07 } },
  { name: 'Бентонит', group: 'Глины', inv: true, price: 265, tol: 0.05, loi: 'residual', clay: true,
    formula: { SiO2: 56.1, Al2O3: 18.2, Fe2O3: 3.4, MgO: 2.3, CaO: 1.4 } },
  { name: 'Кварцевая мука Кварцверке W12', group: 'Кремнезём', inv: true, price: 126, tol: 0.02, loi: 'passport',
    formula: { SiO2: 100 } },
  { name: 'Волластонит МИВОЛЛ', group: 'Карбонаты и плавни', inv: true, price: 343, tol: 0.05, loi: 'residual',
    formula: { SiO2: 51, CaO: 46, MgO: 1, Al2O3: 0.2, TiO2: 0.01 } },
  { name: 'Улексит (Химпэк)', group: 'Карбонаты и плавни', inv: true, price: 402, tol: 0.10, loi: 'unknown', soluble: true,
    formula: { B2O3: 37, CaO: 19, SiO2: 4, Na2O: 3.5, MgO: 2.5, SrO: 1, Al2O3: 0.25, Fe2O3: 0.04 } },
  { name: 'Мел, CaCO3', group: 'Карбонаты и плавни', inv: true, price: 79, tol: 0.01, loi: 'residual',
    formula: { CaO: 55.6, MgO: 0.4 } },
  { name: 'Доломит МИДОЛ', group: 'Карбонаты и плавни', inv: true, price: 95, tol: 0.01, loi: 'residual',
    formula: { CaO: 30.32, MgO: 21.23, SiO2: 0.3, Fe2O3: 0.05, Al2O3: 0.02 } },
  { name: 'Тальк Онотский', group: 'Карбонаты и плавни', inv: true, price: 77, tol: 0.05, loi: 'residual',
    formula: { SiO2: 60, MgO: 31, Al2O3: 5, Fe2O3: 1, CaO: 0.2 } },
  { name: 'Бура, Na2O 2 B2O3 10 H2O', group: 'Карбонаты и плавни', inv: true, price: 570, tol: 0.10, loi: 'unknown', soluble: true,
    formula: { B2O3: 36.5, Na2O: 16.2 } },
  { name: 'Костная зола', group: 'Карбонаты и плавни', inv: true, price: 344, tol: 0.20, loi: 'unknown',
    formula: { CaO: 51.2, P2O5: 40.1, SrO: 0.3 } },
  { name: 'Древесная зола', group: 'Карбонаты и плавни', inv: true, price: null, tol: 0.20, loi: 'unknown',
    formula: { CaO: 41.2, K2O: 8.4, SiO2: 12.6, P2O5: 3.9, MgO: 4.1, Fe2O3: 1.2 } },
  { name: 'Глинозем, Al203', group: 'Оксиды и колоранты', inv: true, price: 228, tol: 0.01, loi: 'passport',
    formula: { Al2O3: 99.2 } },
  { name: 'Оксид цинка, ZnO', group: 'Оксиды и колоранты', inv: true, price: 838, tol: 0.01, loi: 'passport',
    formula: { ZnO: 99.5 } },
  { name: 'Карбонат цинка, ZnCO3', group: 'Оксиды и колоранты', inv: true, price: 580, tol: 0.01, loi: 'residual',
    formula: { ZnO: 71.8 } },
  { name: 'Оксид титана, TiO2', group: 'Оксиды и колоранты', inv: true, price: 640, tol: 0.01, loi: 'passport',
    formula: { TiO2: 99.3 } },
  { name: 'Оксид железа красный, Fe2O3', group: 'Оксиды и колоранты', inv: false, price: 260, tol: 0.01, loi: 'passport',
    formula: { Fe2O3: 98.4 } },
  { name: 'Карбонат бария, BaCO3', group: 'Карбонаты и плавни', inv: false, price: 310, tol: 0.01, loi: 'residual',
    formula: { BaO: 77.7 } },
  { name: 'Карбонат лития (литий углекислый)', group: 'Карбонаты и плавни', inv: false, price: 2400, tol: 0.01, loi: 'residual',
    formula: { Li2O: 40.4 } },
  { name: 'Карбонат меди, CuCO3', group: 'Оксиды и колоранты', inv: false, price: 1450, tol: 0.01, loi: 'residual',
    formula: { CuO: 64.2 } },
];

const DEFAULT_INVENTORY = MATERIALS.filter((m) => m.inv).map((m) => m.name);

// --- solutions ------------------------------------------------------------
// Near-duplicates are already collapsed: two solutions count as the same when
// the material set matches (up to a junk component under 2%) and every share
// differs by less than 1 absolute percent - the threshold Glazy itself uses to
// call two recipes the same. What survives differs for a reason, and the reason
// is what the card is labelled with.
const SOLUTIONS = [
  {
    id: 's1', engine: 'iterative', error: 0.0033, recommended: true,
    label: 'рекомендовано', labelWhy: 'меньше всего компонентов и нет мусорных долей',
    collapsed: 1,
    recipe: { 'Нефелин-сиенит VR13': 30.2, 'Волластонит МИВОЛЛ': 20.0, 'Кварцевая мука Кварцверке W12': 19.9, 'Каолин КЖФ-1': 14.9, 'Улексит (Химпэк)': 15.0 },
    umf: { CaO: 0.718, MgO: 0.048, SiO2: 3.147, TiO2: 0.003, Al2O3: 0.378, K2O: 0.086, Na2O: 0.144, Fe2O3: 0.002, SrO: 0.005, B2O3: 0.265 },
    quality: { junk: 0, minPortion: 14.9, drift: 0.004, cost: 178, clay: 14.9 },
    // real output of sensitivity.py on this recipe
    sensitivity: [
      { material: 'Улексит (Химпэк)', share: 0.700, via: 'B2O3', sigma: 0.10, flux: false, affects: ['B2O3', 'MgO'] },
      { material: 'Волластонит МИВОЛЛ', share: 0.226, via: 'CaO', sigma: 0.05, flux: true, affects: [] },
      { material: 'Нефелин-сиенит VR13', share: 0.036, via: 'K2O', sigma: 0.02, flux: true, affects: [] },
      { material: 'Каолин КЖФ-1', share: 0.035, via: 'Al2O3', sigma: 0.05, flux: false, affects: ['Al2O3', 'SiO2'] },
      { material: 'Кварцевая мука Кварцверке W12', share: 0.003, via: 'SiO2', sigma: 0.02, flux: false, affects: ['SiO2'] },
    ],
    // per-oxide spread, requirements only - a passenger's sigma answers a
    // question nobody asked
    spread: { B2O3: 0.0278, MgO: 0.0029, Al2O3: 0.0150, Na2O: 0.0051, K2O: 0.0030, SiO2: 0.1056, CaO: 0.0090 },
  },
  {
    id: 's2', engine: 'iterative', error: 0.0039,
    label: 'глинозём вместо части каолина', labelWhy: 'на один компонент больше, глины вдвое меньше',
    recipe: { 'Нефелин-сиенит VR13': 31.2, 'Кварцевая мука Кварцверке W12': 23.8, 'Волластонит МИВОЛЛ': 20.3, 'Улексит (Химпэк)': 15.2, 'Каолин КЖФ-1': 6.6, 'Глинозем, Al203': 2.9 },
    umf: { CaO: 0.717, MgO: 0.047, SiO2: 3.146, Al2O3: 0.378, K2O: 0.085, Na2O: 0.146, SrO: 0.005, B2O3: 0.265 },
    quality: { junk: 0, minPortion: 2.9, drift: 0.011, cost: 186, clay: 6.6 },
    sensitivity: [
      { material: 'Улексит (Химпэк)', share: 0.712, via: 'B2O3', sigma: 0.10, flux: false, affects: ['B2O3'] },
      { material: 'Волластонит МИВОЛЛ', share: 0.214, via: 'CaO', sigma: 0.05, flux: true, affects: [] },
      { material: 'Нефелин-сиенит VR13', share: 0.041, via: 'K2O', sigma: 0.02, flux: true, affects: [] },
      { material: 'Каолин КЖФ-1', share: 0.028, via: 'Al2O3', sigma: 0.05, flux: false, affects: ['Al2O3'] },
      { material: 'Глинозем, Al203', share: 0.002, via: 'Al2O3', sigma: 0.01, flux: false, affects: ['Al2O3'] },
      { material: 'Кварцевая мука Кварцверке W12', share: 0.003, via: 'SiO2', sigma: 0.02, flux: false, affects: ['SiO2'] },
    ],
    spread: { B2O3: 0.0281, MgO: 0.0031, Al2O3: 0.0141, Na2O: 0.0053, K2O: 0.0029, SiO2: 0.1012, CaO: 0.0094 },
  },
  {
    id: 's3', engine: 'classic', error: 0.0044,
    label: 'на костной золе', labelWhy: 'приносит P₂O₅ 0.164 — это уже другая глазурь',
    recipe: { 'Нефелин-сиенит VR13': 28.2, 'Кварцевая мука Кварцверке W12': 27.8, 'Костная зола': 15.4, 'Каолин КЖФ-1': 14.0, 'Улексит (Химпэк)': 14.0, 'Тальк Онотский': 0.6 },
    umf: { CaO: 0.718, K2O: 0.086, Na2O: 0.144, SiO2: 3.148, TiO2: 0.002, Al2O3: 0.379, Fe2O3: 0.002, MgO: 0.048, SrO: 0.005, B2O3: 0.265, P2O5: 0.164 },
    quality: { junk: 1, minPortion: 0.6, drift: 0.031, cost: 279, clay: 14.0 },
    sensitivity: [
      { material: 'Костная зола', share: 0.522, via: 'P2O5', sigma: 0.20, flux: false, affects: ['P2O5'] },
      { material: 'Улексит (Химпэк)', share: 0.361, via: 'B2O3', sigma: 0.10, flux: false, affects: ['B2O3'] },
      { material: 'Нефелин-сиенит VR13', share: 0.062, via: 'K2O', sigma: 0.02, flux: true, affects: [] },
      { material: 'Каолин КЖФ-1', share: 0.049, via: 'Al2O3', sigma: 0.05, flux: false, affects: ['Al2O3'] },
      { material: 'Тальк Онотский', share: 0.004, via: 'MgO', sigma: 0.05, flux: true, affects: [] },
      { material: 'Кварцевая мука Кварцверке W12', share: 0.002, via: 'SiO2', sigma: 0.02, flux: false, affects: ['SiO2'] },
    ],
    spread: { B2O3: 0.0276, MgO: 0.0032, Al2O3: 0.0154, Na2O: 0.0050, K2O: 0.0031, SiO2: 0.1094, CaO: 0.0142 },
  },
  {
    id: 's5', engine: 'classic', error: 0.0177,
    label: 'дешевле на 26 ₽/кг', labelWhy: 'но семь компонентов и ни одного глинистого',
    recipe: { 'Кварцевая мука Кварцверке W12': 46.1, 'Мел, CaCO3': 16.0, 'Глинозем, Al203': 9.9, 'Фритта 100 (Рускерамика)': 9.6, 'Бура, Na2O 2 B2O3 10 H2O': 8.9, 'Древесная зола': 7.4, 'Улексит (Химпэк)': 2.2 },
    umf: { B2O3: 0.266, Na2O: 0.144, Al2O3: 0.38, CaO: 0.721, K2O: 0.087, MgO: 0.048, P2O5: 0.025, SiO2: 3.161, Fe2O3: 0.014 },
    quality: { junk: 0, minPortion: 2.2, drift: 0.046, cost: 152, clay: 0 },
    sensitivity: [
      { material: 'Бура, Na2O 2 B2O3 10 H2O', share: 0.481, via: 'B2O3', sigma: 0.10, flux: false, affects: ['B2O3', 'Na2O'] },
      { material: 'Древесная зола', share: 0.288, via: 'CaO', sigma: 0.20, flux: true, affects: [] },
      { material: 'Фритта 100 (Рускерамика)', share: 0.121, via: 'B2O3', sigma: 0.03, flux: false, affects: ['B2O3', 'SiO2'] },
      { material: 'Улексит (Химпэк)', share: 0.077, via: 'B2O3', sigma: 0.10, flux: false, affects: ['B2O3'] },
      { material: 'Мел, CaCO3', share: 0.028, via: 'CaO', sigma: 0.01, flux: true, affects: [] },
      { material: 'Глинозем, Al203', share: 0.004, via: 'Al2O3', sigma: 0.01, flux: false, affects: ['Al2O3'] },
      { material: 'Кварцевая мука Кварцверке W12', share: 0.001, via: 'SiO2', sigma: 0.02, flux: false, affects: ['SiO2'] },
    ],
    spread: { B2O3: 0.0322, MgO: 0.0089, Al2O3: 0.0121, Na2O: 0.0148, K2O: 0.0121, SiO2: 0.1210, CaO: 0.0384 },
  },
];

// --- feasibility ----------------------------------------------------------
const FEASIBILITY_OK = { feasible: true, maxRelativeDeviation: 0.011, perOxide: [], unreachable: [] };

const FEASIBILITY_BAD = {
  feasible: false,
  maxRelativeDeviation: 0.86,
  perOxide: [
    { oxide: 'Li2O', target: 0.4, closest: 0.0, reachable: false, why: 'нет литиевого сырья в наборе' },
    { oxide: 'BaO', target: 0.25, closest: 0.035, reachable: false, why: 'только следы из золы' },
  ],
  unreachable: ['Li2O', 'BaO'],
};

// --- glazy import ---------------------------------------------------------
// Only a formula is imported, never a recipe: the Seger formula is the
// interchange format, and a foreign recipe would name materials we do not have.
const GLAZY_IMPORT = {
  url: 'https://glazy.org/recipes/34125',
  name: 'Leach 4321 Clear',
  author: 'linda_a',
  cone: '△10 R',
  source: 'glazy',
  umf: { K2O: 0.264, CaO: 0.736, Al2O3: 0.409, SiO2: 3.711 },
};

window.GS = {
  OXIDE_GROUPS, TARGET, MATERIALS, DEFAULT_INVENTORY, SOLUTIONS,
  FEASIBILITY_OK, FEASIBILITY_BAD, GLAZY_IMPORT,
  NEVER_ACCIDENTAL, OXIDE_THRESHOLD, FLUX_SHARE_NOTABLE,
};

function GSData() {
  return null;
}

Object.assign(window, { GSData });
