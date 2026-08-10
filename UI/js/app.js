// API Configuration
const API_URL = '/api';

// API Client
const api = {
    async request(endpoint, options = {}) {
        try {
            const url = `${API_URL}/${endpoint}`;
            const response = await fetch(url, {
                headers: {
                    'Content-Type': 'application/json',
                },
                ...options,
            });
            
            if (!response.ok) {
                const error_data = await response.json().catch(() => null);
                throw new Error(error_data?.message || `API error: ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error(`API request failed: ${error.message}`);
            throw error;
        }
    },
    
    async solve_recipe(umf, options = {}) {
        const { max_solutions = 10, min_materials = true, error_tolerance = 0.1 } = options;
        
        return this.request('solve', {
            method: 'POST',
            body: JSON.stringify({
                umf,
                max_solutions,
                min_materials,
                error_tolerance,
            }),
        });
    },
    
    async check_health() {
        return this.request('health');
    },
    
    async get_molar_masses() {
        return this.request('molar_masses');
    },

    async get_oxide_groups() {
        return this.request('oxide_groups');
    }
};

// App State
let current_umf = {
    'K2O': 0.086,
    'Na2O': 0.143,
    'MgO': 0.048,
    'CaO': 0.717,
    'SrO': 0.000,
    'Al2O3': 0.378,
    'B2O3': 0.265,
    'Fe2O3': 0.000,
    'SiO2': 3.144,
    'TiO2': 0.000
};

let current_solutions = [];
// Warnings of the last /api/solve request. Reset at the start of every request,
// so a banner can never outlive the answer it belongs to.
let current_solve_warnings = [];
let calculate_timer = null;
let all_oxides = {}; // Будет содержать все доступные оксиды из molar_masses.json
let is_calculating = false;
let use_min_materials = true; // Добавляем переменную для хранения значения min_materials
let is_importing_from_glazy = false; // A Glazy import is in flight, see import_from_glazy

// Solver engines. The iterative one is the default here and on the server: it is
// more accurate, faster and deterministic, while the classic one draws random
// subsets of materials and answers the same request differently every time.
const SOLVER_ITERATIVE = 'iterative';
const SOLVER_CLASSIC = 'classic';
const DEFAULT_SOLVER = SOLVER_ITERATIVE;
// The UMF lives in the URL hash because that is what a shared link has to carry.
// The engine is a preference of this browser rather than a part of the formula,
// so it goes to localStorage instead and keeps the shared links unchanged.
const SOLVER_STORAGE_KEY = 'glazy_solver_engine';
let current_solver = DEFAULT_SOLVER;

// Определение групп оксидов
// Filled from GET /api/oxide_groups on init(); the literal below is the
// fallback used when that endpoint is unreachable.
let oxide_groups = {
    'r2o_ro': ['K2O', 'Na2O', 'Li2O', 'MgO', 'CaO', 'SrO', 'BaO', 'ZnO', 'PbO', 'CdO', 'CuO', 'FeO', 'MnO', 'MnO2'],
    'r2o3': ['Al2O3', 'B2O3', 'Fe2O3', 'Cr2O3', 'Bi2O3', 'La2O3', 'Y2O3', 'P2O5', 'V2O5'],
    'ro2': ['SiO2', 'TiO2', 'ZrO2', 'SnO2', 'GeO2']
};

// DOM Elements
const elements = {
    recipe_name: document.getElementById('recipe_name'),
    solutions_container: document.getElementById('solutions_container'),
    r2o_ro_table: document.getElementById('r2o_ro_table'),
    r2o3_table: document.getElementById('r2o3_table'),
    ro2_table: document.getElementById('ro2_table'),
    add_oxide_buttons: document.querySelectorAll('.add-oxide-btn'),
    calculation_status: document.getElementById('calculation_status'),
    min_materials_toggle: document.getElementById('min_materials_toggle'),
    glazy_import_input: document.getElementById('glazy_import_input'),
    glazy_import_btn: document.getElementById('glazy_import_btn'),
    glazy_import_status: document.getElementById('glazy_import_status'),
    glazy_source: document.getElementById('glazy_source'),
    solver_radios: document.querySelectorAll('input[name="solver_engine"]')
};

// How the source card reads umf_basis_diff of the server (the largest per-oxide
// difference between our target and the UMF of Glazy). Up to NOTE the two flux
// bases agree for practical purposes and the card says nothing (0.01 is the step
// at which the UI already stops calling a difference negligible, see diff-low in
// create_umf_element); between NOTE and WARNING the card explains the small
// discrepancy; above WARNING the numbers are on a different scale entirely (a
// colorant recipe, where CuO is a flux for us and not for Glazy - 6% CuO alone
// already puts the two bases 0.709 apart) and the card warns prominently.
// Mirrors UMF_BASIS_DIFF_NOTE / UMF_BASIS_DIFF_WARNING of glazy_import.py.
const GLAZY_UMF_DIFF_NOTE_THRESHOLD = 0.01;
const GLAZY_UMF_DIFF_WARNING_THRESHOLD = 0.5;

// The engine currently selected in the interface
function get_current_solver() {
    return current_solver;
}

// Load the selected engine from localStorage
function load_solver_from_storage() {
    try {
        const stored = window.localStorage.getItem(SOLVER_STORAGE_KEY);
        if (stored === SOLVER_ITERATIVE || stored === SOLVER_CLASSIC) {
            current_solver = stored;
            console.log('loaded_solver_from_storage:', current_solver);
        }
    } catch (error) {
        // localStorage can be unavailable (private mode, storage disabled by
        // policy). That is no reason to break the page: the default just stays.
        console.error('failed_to_read_solver_from_storage:', error);
    }
}

// Save the selected engine to localStorage
function save_solver_to_storage(solver) {
    try {
        window.localStorage.setItem(SOLVER_STORAGE_KEY, solver);
        console.log('saved_solver_to_storage:', solver);
    } catch (error) {
        console.error('failed_to_save_solver_to_storage:', error);
    }
}

// min_materials (and error_tolerance) are understood by the classic engine
// only; the iterative one ignores them, so the checkbox is greyed out while
// the iterative engine is selected
function sync_solver_dependent_controls() {
    const checkbox = elements.min_materials_toggle;
    if (!checkbox) {
        return;
    }

    const is_classic = current_solver === SOLVER_CLASSIC;
    checkbox.disabled = !is_classic;

    const item = checkbox.closest('.material-item');
    if (item) {
        item.classList.toggle('disabled', !is_classic);
        item.title = is_classic ? '' : 'Работает только с классическим движком';
    }
}

// Bring the engine switch in the interface to the current state
function apply_solver_to_controls() {
    elements.solver_radios.forEach(radio => {
        radio.checked = radio.value === current_solver;
    });
    sync_solver_dependent_controls();
}

// Загрузить UMF из URL
function load_umf_from_storage() {
    try {
        const hash = window.location.hash.substring(1);
        if (hash) {
            current_umf = JSON.parse(decodeURIComponent(hash));
            console.log('loaded_umf_from_url:', current_umf);
        }
    } catch (error) {
        console.error('failed_to_parse_url_hash:', error);
    }
}

// Сохранить UMF в URL
function save_umf_to_storage(umf) {
    try {
        const umf_json = JSON.stringify(umf);
        const new_hash = encodeURIComponent(umf_json);
        
        // Сохраняем текущий активный элемент
        const active_element = document.activeElement;
        const active_element_id = active_element ? active_element.id : null;
        const active_element_selection_start = active_element && active_element.selectionStart ? active_element.selectionStart : null;
        const active_element_selection_end = active_element && active_element.selectionEnd ? active_element.selectionEnd : null;
        
        // Меняем хеш URL только если он отличается от текущего
        if (window.location.hash !== '#' + new_hash) {
            // Устанавливаем флаг, что изменение хеша - внутреннее
            if (typeof window.is_internal_hash_change !== 'undefined') {
                window.is_internal_hash_change = true;
            }
            
            // Устанавливаем новый хеш URL
            window.location.hash = new_hash;
            
            // Восстанавливаем фокус на активном элементе
            if (active_element_id) {
                setTimeout(() => {
                    const element = document.getElementById(active_element_id);
                    if (element) {
                        element.focus();
                        // Если был выделен текст, восстанавливаем выделение
                        if (active_element_selection_start !== null && active_element_selection_end !== null) {
                            element.selectionStart = active_element_selection_start;
                            element.selectionEnd = active_element_selection_end;
                        }
                    }
                }, 0);
            }
        }
        
        console.log('saved_umf_to_url:', umf);
    } catch (error) {
        console.error('failed_to_save_umf_to_url:', error);
    }
}

// Initialize the app
async function init() {
    // Загружаем доступные оксиды из API
    try {
        all_oxides = await api.get_molar_masses();
        console.log('loaded_molar_masses:', all_oxides);
    } catch (error) {
        console.error('failed_to_load_molar_masses_using_defaults:', error);
        // Используем дефолтное значение, если не удалось загрузить
        all_oxides = {
            'SiO2': 60.084, 'Al2O3': 101.961, 'B2O3': 69.620, 'Na2O': 61.979,
            'K2O': 94.196, 'MgO': 40.304, 'CaO': 56.077, 'SrO': 103.620,
            'BaO': 153.326, 'ZnO': 81.380, 'TiO2': 79.866, 'Fe2O3': 159.688
        };
    }

    // The oxide classification lives in database/oxide_classification.json and
    // is served by the API; the hardcoded groups above stay as the fallback
    try {
        const groups = await api.get_oxide_groups();
        oxide_groups = {
            'r2o_ro': [...(groups.r2o || []), ...(groups.ro || [])],
            'r2o3': [...(groups.r2o3 || [])],
            'ro2': [...(groups.ro2 || [])]
        };
        console.log('loaded_oxide_groups:', oxide_groups);
    } catch (error) {
        console.warn('failed_to_load_oxide_groups_using_defaults:', error);
    }

    // Check API health
    check_api_health();
    
    // Загружаем формулу из URL
    load_umf_from_storage();

    // Load the selected solver engine
    load_solver_from_storage();

    // Добавляем начальные оксиды
    add_initial_oxides();
    
    // Setup event listeners
    setup_event_listeners();
    
    // Автоматически рассчитываем решение при загрузке
    solve_recipe();
}

// Check if API server is running
async function check_api_health() {
    try {
        await api.check_health();
        console.log('api_server_is_running');
    } catch (error) {
        console.error('api_server_is_not_available');
        show_error_message('API сервер недоступен. Некоторые функции могут не работать корректно.');
    }
}

// Добавление начальных оксидов
function add_initial_oxides() {
    // Группируем оксиды по категориям
    const r2o_oxides = [];
    const ro_oxides = [];
    const r2o3_oxides = [];
    const ro2_oxides = [];
    
    for (const [oxide, value] of Object.entries(current_umf)) {
        if (value !== undefined) {
            const group = get_oxide_group(oxide);
            if (group === 'r2o_ro') {
                // R2O оксиды
                if (['K2O', 'Na2O', 'Li2O'].includes(oxide)) {
                    r2o_oxides.push([oxide, value]);
                } else {
                    // RO оксиды
                    ro_oxides.push([oxide, value]);
                }
            } else if (group === 'r2o3') {
                r2o3_oxides.push([oxide, value]);
            } else if (group === 'ro2') {
                ro2_oxides.push([oxide, value]);
            }
        }
    }
    
    // Сортируем R2O оксиды в нужном порядке
    r2o_oxides.sort((a, b) => {
        const order = ['K2O', 'Na2O', 'Li2O'];
        return order.indexOf(a[0]) - order.indexOf(b[0]);
    });
    
    // Добавляем R2O оксиды
    for (const [oxide, value] of r2o_oxides) {
        add_oxide_to_table('r2o_ro', oxide, value, true);
    }
    
    // Добавляем разделитель между R2O и RO
    if (r2o_oxides.length > 0 && ro_oxides.length > 0) {
        // Находим последний добавленный R2O оксид
        const lastR2ORow = elements.r2o_ro_table.querySelector('tr:last-child');
        if (lastR2ORow) {
            lastR2ORow.classList.add('r2o-ro-divider');
        }
    }
    
    // Добавляем RO оксиды
    for (const [oxide, value] of ro_oxides) {
        add_oxide_to_table('r2o_ro', oxide, value);
    }
    
    // Добавляем R2O3 оксиды
    for (const [oxide, value] of r2o3_oxides) {
        add_oxide_to_table('r2o3', oxide, value);
    }
    
    // Добавляем RO2 оксиды
    for (const [oxide, value] of ro2_oxides) {
        add_oxide_to_table('ro2', oxide, value);
    }
}

// Определяем группу оксида
function get_oxide_group(oxide) {
    for (const [group, oxides] of Object.entries(oxide_groups)) {
        if (oxides.includes(oxide)) {
            return group;
        }
    }
    
    // Если не найдено, определяем по химической формуле
    if (oxide.includes('2O3') || oxide.includes('2O5')) {
        return 'r2o3';
    } else if (oxide.includes('O2')) {
        return 'ro2';
    } else if (oxide.includes('2O') || oxide.includes('O')) {
        return 'r2o_ro';
    }
    
    return 'r2o_ro'; // По умолчанию
}

// Добавляем оксид в таблицу
function add_oxide_to_table(group, selected_oxide = null, value = 0, is_r2o = false) {
    const table = elements[`${group}_table`];
    if (!table) return;
    
    // Порядок оксидов в dropdown - это порядок группы в классификации
    // r2o_ro used to be re-sorted here against a hardcoded order. It no longer
    // needs to be: the group is built as r2o followed by ro, so it already
    // arrives in the intended order, and unlike a literal it covers every oxide
    // of the classification (CoO and NiO used to fall into the alphabetical
    // tail of the list because the literal did not mention them).
    const oxide_options = [...oxide_groups[group]];

    const row = document.createElement('tr');
    // Если это R2O оксид и стоит в конце своей группы, добавляем класс для разделителя
    if (is_r2o) {
        row.dataset.isR2o = "true";
    }
    
    // Создаем селект для выбора оксида
    const select_cell = document.createElement('td');
    const select = document.createElement('select');
    select.className = 'oxide-select';
    select.dataset.group = group;
    
    // Добавляем пустой option
    const empty_option = document.createElement('option');
    empty_option.value = '';
    empty_option.textContent = 'Выберите оксид';
    select.appendChild(empty_option);
    
    // Получаем список оксидов, которые еще не выбраны в этой группе
    const used_oxides = get_used_oxides();
    
    // Добавляем оксиды в селект, принадлежащие к нужной группе
    for (const oxide of oxide_options) {
        if (all_oxides[oxide] && (selected_oxide === oxide || !used_oxides.includes(oxide))) {
            const option = document.createElement('option');
            option.value = oxide;
            option.innerHTML = format_oxide_name(oxide);
            option.selected = (oxide === selected_oxide);
            select.appendChild(option);
        }
    }
    
    // Добавляем остальные оксиды из molar_masses.json, которые могут подходить к этой группе
    for (const oxide in all_oxides) {
        if (!oxide_groups.r2o_ro.includes(oxide) && 
            !oxide_groups.r2o3.includes(oxide) && 
            !oxide_groups.ro2.includes(oxide) && 
            (selected_oxide === oxide || !used_oxides.includes(oxide))) {
            // Определяем группу оксида
            const detected_group = get_oxide_group(oxide);
            if (detected_group === group) {
                const option = document.createElement('option');
                option.value = oxide;
                option.innerHTML = format_oxide_name(oxide);
                option.selected = (oxide === selected_oxide);
                select.appendChild(option);
            }
        }
    }
    
    select_cell.appendChild(select);
    
    // Создаем input для значения
    const value_cell = document.createElement('td');
    const input = document.createElement('input');
    input.type = 'number';
    input.className = 'oxide-input';
    input.id = selected_oxide || '';
    input.step = '0.01';
    input.value = value;
    value_cell.appendChild(input);
    
    // Создаем кнопку удаления
    const delete_cell = document.createElement('td');
    const delete_button = document.createElement('button');
    delete_button.type = 'button';
    delete_button.className = 'delete-oxide-btn';
    delete_button.textContent = '✕';
    delete_cell.appendChild(delete_button);
    
    // Собираем row
    row.appendChild(select_cell);
    row.appendChild(value_cell);
    row.appendChild(delete_cell);
    
    // Добавляем в таблицу
    table.appendChild(row);
    
    // Добавляем обработчики событий
    setup_oxide_row_events(select, input, delete_button);
}

// Получаем список уже использованных оксидов
function get_used_oxides() {
    const selects = document.querySelectorAll('.oxide-select');
    const used = [];
    
    selects.forEach(select => {
        if (select.value) {
            used.push(select.value);
        }
    });
    
    return used;
}

// Обновление разделителя между R2O и RO оксидами
function update_r2o_ro_divider() {
    const table = elements.r2o_ro_table;
    if (!table) return;
    
    // Сначала удаляем класс разделителя со всех строк
    const rows = table.querySelectorAll('tr');
    rows.forEach(row => row.classList.remove('r2o-ro-divider'));
    
    // Находим последний R2O оксид
    const r2o_rows = Array.from(rows).filter(row => row.dataset.isR2o === "true");
    const ro_rows = Array.from(rows).filter(row => !row.dataset.isR2o || row.dataset.isR2o !== "true");
    
    // Если есть и R2O и RO оксиды, добавляем разделитель
    if (r2o_rows.length > 0 && ro_rows.length > 0) {
        r2o_rows[r2o_rows.length - 1].classList.add('r2o-ro-divider');
    }
}

// Добавляем обработчики событий для строки оксида
function setup_oxide_row_events(select, input, delete_button) {
    // При изменении выбранного оксида
    select.addEventListener('change', () => {
        // Получаем текущее значение выбранного оксида
        const selected_oxide = select.value;
        
        // Обновляем ID инпута
        input.id = selected_oxide;
        
        // Определяем, является ли это R2O оксидом
        const is_r2o = ['K2O', 'Na2O', 'Li2O'].includes(selected_oxide);
        const row = select.closest('tr');
        
        if (row) {
            if (is_r2o) {
                row.dataset.isR2o = "true";
            } else {
                delete row.dataset.isR2o;
            }
            // Обновляем разделитель
            update_r2o_ro_divider();
        }
        
        // Обновляем current_umf
        current_umf = get_umf_from_inputs();

        // Сохраняем в URL
        save_umf_to_storage(current_umf);

        // The formula is no longer the imported recipe
        mark_glazy_source_stale();

        // Запускаем расчет с debounce
        debounce_solve();
    });

    // При изменении значения
    input.addEventListener('input', () => {
        // Обновляем current_umf
        current_umf = get_umf_from_inputs();

        // Сохраняем в URL
        save_umf_to_storage(current_umf);

        // The formula is no longer the imported recipe
        mark_glazy_source_stale();

        // Запускаем расчет с debounce
        debounce_solve();
    });

    // При нажатии на кнопку удаления
    delete_button.addEventListener('click', () => {
        // Удаляем строку
        const row = delete_button.closest('tr');
        row.remove();

        // Обновляем разделитель
        update_r2o_ro_divider();

        // Обновляем current_umf
        current_umf = get_umf_from_inputs();

        // Сохраняем в URL
        save_umf_to_storage(current_umf);

        // The formula is no longer the imported recipe
        mark_glazy_source_stale();

        // Запускаем расчет с debounce
        debounce_solve();
    });
}

// Get current UMF values from inputs
function get_umf_from_inputs() {
    const oxide_inputs = document.querySelectorAll('.oxide-input');
    const umf = {};
    
    oxide_inputs.forEach(input => {
        if (input.id) {
            const value = parseFloat(input.value);
            if (!isNaN(value) && value > 0) {
                umf[input.id] = value;
            }
        }
    });
    
    return umf;
}

// Показать статус расчета рядом с заголовком
function show_calculation_status(is_loading) {
    if (is_loading) {
        elements.calculation_status.innerHTML = '<span class="loader"></span> Расчет...';
        elements.calculation_status.classList.add('visible');
    } else {
        elements.calculation_status.innerHTML = '';
        elements.calculation_status.classList.remove('visible');
    }
}

// Показать сообщение об ошибке в контейнере решений
function show_error_message(message) {
    elements.solutions_container.innerHTML = '';

    const status_elem = document.createElement('div');
    status_elem.className = 'status-message';
    status_elem.textContent = message;

    elements.solutions_container.appendChild(status_elem);
    render_solve_warnings();
}

// /api/solve answers {solutions, warnings} and used to answer the bare list.
// Both are accepted here, in one place rather than at each call site, so that
// this page works against a server of either vintage.
function unpack_solve_response(data) {
    if (Array.isArray(data)) {
        return { solutions: data, warnings: [] };
    }
    return { solutions: (data && data.solutions) || [], warnings: (data && data.warnings) || [] };
}

// "Мы проигнорировали оксид, который вы просили" is the one thing the user
// cannot read off the answer: the recipe arrives, the target it was fitted to
// simply no longer has the key. It is shown ABOVE the solutions and on the
// "nothing found" path too, because a target of nothing but unknown oxides is
// exactly the request that produces no solutions at all.
function render_solve_warnings() {
    if (!current_solve_warnings || current_solve_warnings.length === 0) {
        return;
    }

    const warnings_elem = document.createElement('div');
    warnings_elem.className = 'status-message';
    warnings_elem.textContent = current_solve_warnings.join('; ');

    elements.solutions_container.appendChild(warnings_elem);
}

// Получить класс цвета в зависимости от погрешности
function get_error_color_class(error_percent) {
    if (error_percent < 1) {
        return 'error-low';
    } else if (error_percent > 10) {
        return 'error-high';
    }
    return 'error-medium';
}

// Debounce функция для предотвращения слишком частых запросов
function debounce_solve() {
    // Если уже запущен таймер, отменяем его
    if (calculate_timer) {
        clearTimeout(calculate_timer);
    }
    
    // Устанавливаем статус расчета только при первом вызове
    if (!is_calculating) {
        show_calculation_status(true);
        is_calculating = true;
    }
    
    // Запускаем таймер на выполнение расчета через 500мс
    calculate_timer = setTimeout(() => {
        solve_recipe();
    }, 500);
}

// Создаем элемент для отображения UMF формулы
function create_umf_element(recipe_umf) {
    const umf_container = document.createElement('div');
    umf_container.className = 'solution-umf';
    
    const umf_title = document.createElement('div');
    umf_title.className = 'solution-umf-title';
    
    // Создаем контейнер для всех групп UMF
    const umf_groups_container = document.createElement('div');
    umf_groups_container.className = 'solution-umf-groups';
    
    // Отбираем только значимые оксиды (> 0.001)
    const filtered_umf = {};
    for (const [oxide, value] of Object.entries(recipe_umf)) {
        if (value > 0.001) {
            filtered_umf[oxide] = value;
        }
    }
    
    // Находим оксиды, которые есть в текущем UMF, но отсутствуют в решении
    const missing_oxides = {};
    for (const [oxide, value] of Object.entries(current_umf)) {
        if (value > 0.001 && !filtered_umf[oxide]) {
            missing_oxides[oxide] = value;
        }
    }
    
    // Создаем группы
    const createGroup = (title, group_id) => {
        const groupContainer = document.createElement('div');
        groupContainer.className = 'solution-umf-group';
        
        const groupTitle = document.createElement('div');
        groupTitle.className = 'solution-umf-group-title';
        groupTitle.innerHTML = title;
        
        const umf_grid = document.createElement('div');
        umf_grid.className = 'solution-umf-grid single-column';
        
        // Фильтруем оксиды в группе для тех, что есть в решении
        const groupOxides = Object.keys(filtered_umf).filter(oxide => {
            const group = get_oxide_group(oxide);
            return group === group_id;
        });
        
        // Фильтруем отсутствующие оксиды в этой группе
        const missingGroupOxides = Object.keys(missing_oxides).filter(oxide => {
            const group = get_oxide_group(oxide);
            return group === group_id;
        });
        
        // Объединяем оба списка
        const allGroupOxides = [...groupOxides, ...missingGroupOxides];
        
        // Если группа пустая, не создаем её
        if (allGroupOxides.length === 0) {
            return null;
        }
        
        // Сортируем оксиды для группы R2O/RO так, чтобы сначала шли R2O, потом RO
        let sortedGroupOxides = [...groupOxides];
        let sortedMissingGroupOxides = [...missingGroupOxides];
        
        if (group_id === 'r2o_ro') {
            // Функция для определения, является ли оксид типом R2O
            const is_r2o = (oxide) => ['Na2O', 'K2O', 'Li2O'].includes(oxide);
            
            // Сортируем существующие оксиды в правильном порядке
            sortedGroupOxides = [
                // Сначала K2O, Na2O, Li2O в этом порядке, если они есть
                ...(sortedGroupOxides.filter(oxide => oxide === 'K2O')),
                ...(sortedGroupOxides.filter(oxide => oxide === 'Na2O')),
                ...(sortedGroupOxides.filter(oxide => oxide === 'Li2O')),
                // Потом остальные R2O, если такие появятся в будущем
                ...(sortedGroupOxides.filter(oxide => is_r2o(oxide) && 
                    !['K2O', 'Na2O', 'Li2O'].includes(oxide))),
                // В конце все оксиды RO
                ...(sortedGroupOxides.filter(oxide => !is_r2o(oxide)))
            ];
            
            // Сортируем отсутствующие оксиды так же
            sortedMissingGroupOxides = [
                ...(sortedMissingGroupOxides.filter(oxide => oxide === 'K2O')),
                ...(sortedMissingGroupOxides.filter(oxide => oxide === 'Na2O')),
                ...(sortedMissingGroupOxides.filter(oxide => oxide === 'Li2O')),
                ...(sortedMissingGroupOxides.filter(oxide => is_r2o(oxide) && 
                    !['K2O', 'Na2O', 'Li2O'].includes(oxide))),
                ...(sortedMissingGroupOxides.filter(oxide => !is_r2o(oxide)))
            ];
        }
        
        // Добавляем оксиды из решения
        sortedGroupOxides.forEach((oxide, index) => {
            const umf_item = document.createElement('div');
            umf_item.className = 'solution-umf-item';
            
            // Добавляем визуальный разделитель между R2O и RO в группе r2o_ro
            if (group_id === 'r2o_ro' &&
                ['Na2O', 'K2O', 'Li2O'].includes(oxide) &&
                index < sortedGroupOxides.length - 1 &&
                !['Na2O', 'K2O', 'Li2O'].includes(sortedGroupOxides[index + 1])) {
                umf_item.classList.add('r2o-ro-divider');
            }
            
            const oxide_name = document.createElement('div');
            oxide_name.className = 'solution-umf-name';
            oxide_name.innerHTML = format_oxide_name(oxide);
            
            const oxide_value = document.createElement('div');
            oxide_value.className = 'solution-umf-value';
            
            // Сравниваем с целевым UMF
            if (current_umf[oxide] !== undefined && current_umf[oxide] > 0) {
                const target_value = current_umf[oxide];
                const solution_value = filtered_umf[oxide];
                const abs_diff = Math.abs(solution_value - target_value);
                
                // Добавляем класс в зависимости от величины абсолютной разницы
                if (abs_diff >= 0 && abs_diff < 0.01) { oxide_value.classList.add('diff-low'); }
                if (abs_diff >= 0.01 && abs_diff < 0.1) { oxide_value.classList.add('diff-medium'); }
                if (abs_diff >= 0.1) { oxide_value.classList.add('diff-high'); }
                
                
                
                // Добавляем тултип с информацией о различии
                oxide_value.title = `Целевое: ${target_value.toFixed(3)}, Разница: ${abs_diff.toFixed(3)}`;
                oxide_value.textContent = filtered_umf[oxide].toFixed(3);
            } else {
                // Если оксида нет в целевом UMF или его значение 0, подсвечиваем красным
                oxide_value.classList.add('diff-high');
                oxide_value.title = 'Этот оксид отсутствует в целевом UMF';
                oxide_value.textContent = '! ' + filtered_umf[oxide].toFixed(3);
            }
            
            umf_item.appendChild(oxide_name);
            umf_item.appendChild(oxide_value);
            umf_grid.appendChild(umf_item);
        });
        
        // Добавляем отсутствующие оксиды
        sortedMissingGroupOxides.forEach((oxide, index) => {
            const umf_item = document.createElement('div');
            umf_item.className = 'solution-umf-item';
            
            // Добавляем визуальный разделитель между R2O и RO в группе r2o_ro для отсутствующих оксидов
            if (group_id === 'r2o_ro' &&
                ['Na2O', 'K2O', 'Li2O'].includes(oxide) &&
                index < sortedMissingGroupOxides.length - 1 &&
                !['Na2O', 'K2O', 'Li2O'].includes(sortedMissingGroupOxides[index + 1])) {
                umf_item.classList.add('r2o-ro-divider');
            }
            
            const oxide_name = document.createElement('div');
            oxide_name.className = 'solution-umf-name';
            oxide_name.innerHTML = format_oxide_name(oxide);
            
            const oxide_value = document.createElement('div');
            oxide_value.className = 'solution-umf-value diff-missing';
            oxide_value.title = `Этот оксид отсутствует в решении. Целевое значение: ${missing_oxides[oxide].toFixed(3)}`;
            oxide_value.textContent = '? ' + missing_oxides[oxide].toFixed(3);
            
            umf_item.appendChild(oxide_name);
            umf_item.appendChild(oxide_value);
            umf_grid.appendChild(umf_item);
        });
        
        groupContainer.appendChild(groupTitle);
        groupContainer.appendChild(umf_grid);
        
        return groupContainer;
    };
    
    // Создаем группы оксидов
    const groupR2O_RO_element = createGroup('R<sub>2</sub>O/RO', 'r2o_ro');
    const groupR2O3_element = createGroup('R<sub>2</sub>O<sub>3</sub>', 'r2o3');
    const groupRO2_element = createGroup('RO<sub>2</sub>', 'ro2');
    
    // Добавляем группы в контейнер
    if (groupR2O_RO_element) umf_groups_container.appendChild(groupR2O_RO_element);
    if (groupR2O3_element) umf_groups_container.appendChild(groupR2O3_element);
    if (groupRO2_element) umf_groups_container.appendChild(groupRO2_element);
    
    // Добавляем информацию о сравнении
    const legend = document.createElement('div');
    legend.className = 'umf-comparison-legend';
    legend.innerHTML = '<div class="legend-title">Разница с целевым UMF:</div>' +
                       '<div class="legend-item"><span class="legend-color diff-high"></span> >0.1</div>' +
                       '<div class="legend-item"><span class="legend-color diff-medium"></span> 0.02-0.1</div>' +
                       '<div class="legend-item"><span class="legend-color diff-low"></span> <0.02</div>'
    
    umf_container.appendChild(umf_title);
    umf_container.appendChild(umf_groups_container);
    umf_container.appendChild(legend);
    
    return umf_container;
}

// Форматирование названий оксидов с подстрочным текстом
function format_oxide_name(oxide) {
    return oxide.replace(/(\d+)/g, '<sub>$1</sub>');
}

// Solve recipe based on current UMF
async function solve_recipe() {
    try {
        // Cleared before anything can display: a banner belongs to the request
        // that produced it, and show_error_message renders it on every path out
        current_solve_warnings = [];

        const umf = get_umf_from_inputs();

        if (Object.keys(umf).length === 0) {
            show_calculation_status(false);
            is_calculating = false;
            show_error_message('Введите значения UMF для решения.');
            return;
        }

        // Call API to solve recipe
        const answer = unpack_solve_response(await api.solve_recipe(umf, {
            max_solutions: 15,  // Запрашиваем больше для полноценной сортировки
            min_materials: use_min_materials, // Используем значение из параметра
            error_tolerance: 0.05
        }));
        const solutions = answer.solutions;
        current_solve_warnings = answer.warnings;

        // Убираем индикатор расчета
        show_calculation_status(false);
        is_calculating = false;
        
        if (solutions && solutions.length > 0) {
            // Новая логика сортировки:
            // 1. Разделяем на группы: ошибка < 1% и остальные
            // 2. В первой группе сортируем по количеству ингредиентов
            // 3. Во второй группе сначала по ошибке, затем по количеству ингредиентов
            
            const low_error_solutions = solutions.filter(s => s.error < 0.01);  // Ошибка < 1%
            const other_solutions = solutions.filter(s => s.error >= 0.01);     // Ошибка >= 1%
            
            // Сортируем группу с низкой ошибкой по количеству ингредиентов
            low_error_solutions.sort((a, b) => a.materials_count - b.materials_count);
            
            // Сортируем остальные сначала по ошибке, затем по количеству ингредиентов
            other_solutions.sort((a, b) => {
                if (Math.abs(a.error - b.error) < 0.001) {
                    return a.materials_count - b.materials_count;
                }
                return a.error - b.error;
            });
            
            // Объединяем результаты: сначала с низкой ошибкой, затем остальные
            current_solutions = [...low_error_solutions, ...other_solutions];
            
            display_solutions();
        } else {
            show_error_message('Не удалось найти подходящие решения.');
        }
    } catch (error) {
        console.error('Error solving recipe:', error);
        show_calculation_status(false);
        is_calculating = false;
        show_error_message(`Ошибка при поиске решения: ${error.message}`);
    }
}

// Функция для отключения материала в списке
function disable_material_in_list(material_name) {
    const material_id = `material_${material_name.replace(/\s+/g, '_')}`;
    const checkbox = document.getElementById(material_id);
    
    if (checkbox) {
        checkbox.checked = false;
        // Вызываем функцию обновления выбранных материалов (определена в index.html)
        if (typeof update_selected_materials === 'function') {
            update_selected_materials();
        }
    }
}

// Display found solutions
function display_solutions() {
    elements.solutions_container.innerHTML = '';

    if (current_solutions.length === 0) {
        show_error_message('Решения не найдены.');
        return;
    }

    render_solve_warnings();

    current_solutions.forEach((solution, index) => {
        const solution_item = document.createElement('div');
        solution_item.className = 'solution-item';
        
        const solution_header = document.createElement('div');
        solution_header.className = 'solution-header';
        
        const solution_title = document.createElement('div');
        solution_title.className = 'solution-title';
        solution_title.textContent = `Решение #${index + 1} (${Object.keys(solution.recipe).length} материалов)`;
        
        // Рассчитываем погрешность в процентах
        const error_percent = solution.error * 100;
        
        const solution_error = document.createElement('div');
        solution_error.className = `solution-error ${get_error_color_class(error_percent)}`;
        solution_error.textContent = `Погрешность: ${error_percent.toFixed(2)}%`;
        
        solution_header.appendChild(solution_title);
        solution_header.appendChild(solution_error);
        
        const solution_recipe = document.createElement('div');
        solution_recipe.className = 'solution-recipe';
        
        // Add each ingredient
        for (const [material, amount] of Object.entries(solution.recipe)) {
            const recipe_item = document.createElement('div');
            recipe_item.className = 'solution-recipe-item';
            
            const material_action = document.createElement('div');
            material_action.className = 'solution-recipe-action';
            
            // Добавляем крестик для отключения материала
            const disable_button = document.createElement('button');
            disable_button.type = 'button';
            disable_button.className = 'disable-material-btn';
            disable_button.innerHTML = '&#10005;'; // Более тонкий символ крестика
            disable_button.title = 'Исключить материал из расчета';
            disable_button.dataset.material = material;
            disable_button.addEventListener('click', function() {
                disable_material_in_list(material);
            });
            
            material_action.appendChild(disable_button);
            
            const material_name = document.createElement('div');
            material_name.className = 'solution-recipe-name';
            material_name.textContent = material;
            
            const material_amount = document.createElement('div');
            material_amount.className = 'solution-recipe-amount';
            material_amount.textContent = `${amount.toFixed(1)}%`;
            
            recipe_item.appendChild(material_action);
            recipe_item.appendChild(material_name);
            recipe_item.appendChild(material_amount);
            solution_recipe.appendChild(recipe_item);
        }
        
        solution_item.appendChild(solution_header);
        solution_item.appendChild(solution_recipe);
        
        // Добавляем UMF формулу, если она есть
        if (solution.recipe_umf) {
            const umf_element = create_umf_element(solution.recipe_umf);
            solution_item.appendChild(umf_element);
        }
        
        elements.solutions_container.appendChild(solution_item);
    });
}

// Show the state of the Glazy import next to the import button
function set_glazy_import_status(message, is_error) {
    const status = elements.glazy_import_status;
    if (!status) return;

    status.textContent = message;
    status.classList.toggle('error', Boolean(is_error));
}

// Import a recipe from glazy.org: load the formula into the target fields,
// show the original recipe and recalculate
async function import_from_glazy() {
    const input = elements.glazy_import_input;
    const button = elements.glazy_import_btn;
    if (!input || !button) return;

    // The disabled button is not enough: Enter in the input reaches this
    // function directly, and a held Enter would fire parallel requests whose
    // first finally block re-enables the button under the others
    if (is_importing_from_glazy) return;

    const recipe_reference = input.value.trim();
    if (!recipe_reference) {
        set_glazy_import_status('Введите ссылку или ID рецепта', true);
        return;
    }

    is_importing_from_glazy = true;
    button.disabled = true;
    set_glazy_import_status('Загрузка…', false);

    try {
        const response = await fetch(`${API_URL}/glazy_import`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ recipe: recipe_reference })
        });

        const data = await response.json().catch(() => null);

        // The current formula is left alone on an error: an unreachable Glazy
        // must not cost the user the recipe already in the fields
        if (!response.ok || !data || data.error) {
            console.error('glazy_import_error:', data || response.status);
            set_glazy_import_status((data && data.message) || 'Не удалось загрузить рецепт с Glazy', true);
            return;
        }

        apply_glazy_import(data);
        set_glazy_import_status('', false);
    } catch (error) {
        console.error('glazy_import_error:', error);
        set_glazy_import_status('Не удалось загрузить рецепт с Glazy', true);
    } finally {
        is_importing_from_glazy = false;
        button.disabled = false;
    }
}

// Apply an imported recipe: refill the oxide tables, show the source card and solve
function apply_glazy_import(data) {
    current_umf = data.umf || {};

    // Same refill as the hashchange handler of index.html: the tables are
    // rebuilt from current_umf, they are not patched in place. Rebuilding fires
    // no input/change events, so it never marks the card stale on its own - and
    // render_glazy_source() below rebuilds the card anyway, clearing any mark
    // left from an earlier hand edit.
    elements.r2o_ro_table.innerHTML = '';
    elements.r2o3_table.innerHTML = '';
    elements.ro2_table.innerHTML = '';
    add_initial_oxides();

    // Keeps the imported formula in the URL, so it survives a reload and can be shared
    save_umf_to_storage(current_umf);

    if (elements.recipe_name) {
        elements.recipe_name.textContent = data.name || `Рецепт #${data.id}`;
    }

    render_glazy_source(data);

    // Looked up on the global object on purpose: index.html replaces
    // window.solve_recipe with a version that also sends the selected materials
    solve_recipe();
}

// Percentage of a component of the original recipe. Additions in Glazy are
// routinely 0.05-0.5%, so one decimal is not enough (0.05 would be shown as
// "+0.1%" and 0.04 as "+0.0%"), while the second decimal is dropped when it is a
// zero so that the base rows keep reading as "40.0%"
function format_glazy_percentage(value) {
    const text = value.toFixed(2);
    return text.endsWith('0') ? text.slice(0, -1) : text;
}

// Build one row of the original recipe table
function create_glazy_component_row(component, is_first_additional) {
    const row = document.createElement('tr');
    if (component.is_additional) {
        row.classList.add('glazy-source-additional');
    }
    if (is_first_additional) {
        row.classList.add('glazy-source-divider');
    }

    const name_cell = document.createElement('td');
    name_cell.className = 'glazy-source-name';
    // Everything coming from Glazy is third-party text and is inserted as text only
    name_cell.textContent = component.name || '';
    if (component.is_additional) {
        name_cell.title = 'Добавка сверх 100% основы';
    }

    const amount_cell = document.createElement('td');
    amount_cell.className = 'glazy-source-amount';
    const percentage = Number(component.percentage) || 0;
    amount_cell.textContent = `${component.is_additional ? '+' : ''}${format_glazy_percentage(percentage)}%`;

    row.appendChild(name_cell);
    row.appendChild(amount_cell);

    return row;
}

// Show the original Glazy recipe as a reference card
function render_glazy_source(data) {
    const container = elements.glazy_source;
    if (!container) return;

    container.innerHTML = '';
    // A fresh import is by definition in sync with the fields again
    container.classList.remove('glazy-source-stale');

    const header = document.createElement('div');
    header.className = 'glazy-source-header';

    const link = document.createElement('a');
    link.href = data.url || `https://glazy.org/recipes/${data.id}`;
    link.target = '_blank';
    link.rel = 'noopener noreferrer';
    link.textContent = data.name || `Рецепт #${data.id}`;
    header.appendChild(link);

    if (data.id !== undefined && data.id !== null) {
        const id_label = document.createElement('span');
        id_label.className = 'glazy-source-id';
        id_label.textContent = `#${data.id}`;
        header.appendChild(id_label);
    }

    container.appendChild(header);

    // Cone names arrive unescaped from the server, so they are plain text here
    const cones = [data.cone_from, data.cone_to].filter(cone => cone);
    if (cones.length > 0) {
        const cones_line = document.createElement('div');
        cones_line.className = 'glazy-source-cones';
        cones_line.textContent = (cones.length === 2 && cones[0] !== cones[1])
            ? `Конус ${cones[0]}–${cones[1]}`
            : `Конус ${cones[0]}`;
        container.appendChild(cones_line);
    }

    const components = Array.isArray(data.components) ? data.components : [];
    if (components.length > 0) {
        const table = document.createElement('table');
        table.className = 'glazy-source-table';

        // Base first, then the additionals: in Glazy an additional sits on top
        // of a 100% base and its percentage is not part of that 100%
        components.filter(component => !component.is_additional)
            .forEach(component => table.appendChild(create_glazy_component_row(component, false)));

        components.filter(component => component.is_additional)
            .forEach((component, index) => table.appendChild(create_glazy_component_row(component, index === 0)));

        container.appendChild(table);
    }

    const umf_glazy = data.umf_glazy || {};
    const glazy_oxides = Object.keys(umf_glazy);
    if (glazy_oxides.length > 0) {
        const umf_block = document.createElement('div');
        umf_block.className = 'glazy-source-umf';

        const umf_title = document.createElement('div');
        umf_title.className = 'glazy-source-subtitle';
        umf_title.textContent = 'UMF по данным Glazy';
        umf_block.appendChild(umf_title);

        const umf_list = document.createElement('div');
        umf_list.className = 'glazy-source-umf-list';

        glazy_oxides.forEach(oxide => {
            const umf_item = document.createElement('span');
            umf_item.className = 'glazy-source-umf-item';

            const oxide_name = document.createElement('span');
            // format_oxide_name writes HTML, but the server filtered these keys
            // against the molar mass table, so they are known oxide names
            oxide_name.innerHTML = format_oxide_name(oxide);

            const oxide_value = document.createElement('span');
            oxide_value.className = 'glazy-source-umf-value';
            oxide_value.textContent = (Number(umf_glazy[oxide]) || 0).toFixed(3);

            umf_item.appendChild(oxide_name);
            umf_item.appendChild(oxide_value);
            umf_list.appendChild(umf_item);
        });

        umf_block.appendChild(umf_list);
        container.appendChild(umf_block);
    }

    // umf_basis_diff is null when Glazy returned no UMF at all: there is then
    // nothing to compare our target against and nothing to warn about
    const basis_diff = (data.umf_basis_diff === null || data.umf_basis_diff === undefined)
        ? null
        : Number(data.umf_basis_diff);

    if (data.umf_source === 'glazy_umf' || (basis_diff !== null && basis_diff > GLAZY_UMF_DIFF_WARNING_THRESHOLD)) {
        const warning = document.createElement('div');
        warning.className = 'glazy-source-warning';
        warning.textContent = 'Внимание: в этом проекте UMF нормируется по другому набору флюсов, чем на glazy.org '
            + '(CuO и другие красители здесь считаются флюсами, а там нет), поэтому загруженные в поля числа — '
            + 'в другом масштабе, чем на странице рецепта в Glazy. Расчёт при этом остаётся внутренне согласованным: '
            + 'решатель считает кандидатов той же нормировкой. Но чтобы повторить такой рецепт, в инвентаре должны '
            + 'быть подходящие материалы — для медной глазури медные, а красители по умолчанию выключены.';
        container.appendChild(warning);
    } else if (basis_diff !== null && basis_diff > GLAZY_UMF_DIFF_NOTE_THRESHOLD) {
        const note = document.createElement('div');
        note.className = 'glazy-source-note';
        note.textContent = 'Glazy нормирует UMF по немного другому набору флюсов, поэтому загруженная в поля целевая формула может отличаться от чисел на glazy.org.';
        container.appendChild(note);
    }

    container.hidden = false;
}

// The formula in the fields no longer matches the imported recipe. The card is
// marked instead of being hidden: the link to the source recipe stays useful.
// Called from every path that changes the formula except the import itself.
function mark_glazy_source_stale() {
    const container = elements.glazy_source;
    if (!container || container.hidden) return;
    if (container.classList.contains('glazy-source-stale')) return;

    container.classList.add('glazy-source-stale');

    const note = document.createElement('div');
    note.className = 'glazy-source-stale-note';
    note.textContent = 'Формула отредактирована вручную и больше не совпадает с этим рецептом.';
    container.appendChild(note);
}

// index.html reaches the mark from its inline hashchange handler
window.mark_glazy_source_stale = mark_glazy_source_stale;

// Setup event listeners
function setup_event_listeners() {
    // Добавляем обработчики для кнопок добавления оксидов
    elements.add_oxide_buttons.forEach(button => {
        button.addEventListener('click', () => {
            const group = button.dataset.group;
            // Если добавляем оксид в группу R2O/RO,
            // приоритезируем оксиды из подгруппы R2O
            if (group === 'r2o_ro') {
                // Получаем список неиспользуемых оксидов
                const used_oxides = get_used_oxides();
                
                // Проверяем, есть ли доступные R2O оксиды
                const r2o_oxides = ['K2O', 'Na2O', 'Li2O'];
                const available_r2o = r2o_oxides.filter(oxide => 
                    !used_oxides.includes(oxide) && all_oxides[oxide]);
                
                // Если есть доступные R2O оксиды, добавляем первый из них
                if (available_r2o.length > 0) {
                    add_oxide_to_table(group, available_r2o[0], 0, true);
                } else {
                    // Иначе добавляем пустую строку
                    add_oxide_to_table(group);
                }
                
                // Обновляем разделитель
                update_r2o_ro_divider();
            } else {
                add_oxide_to_table(group);
            }
        });
    });

    // Glazy import: the button and Enter in the input do the same thing
    const glazy_import_button = elements.glazy_import_btn;
    if (glazy_import_button) {
        glazy_import_button.addEventListener('click', import_from_glazy);
    }

    const glazy_import_input = elements.glazy_import_input;
    if (glazy_import_input) {
        glazy_import_input.addEventListener('keydown', function(event) {
            // A held Enter repeats: import_from_glazy() itself ignores the
            // repeats while a request is in flight
            if (event.key === 'Enter') {
                event.preventDefault();
                import_from_glazy();
            }
        });
    }

    // Добавляем обработчик для переключателя min_materials
    const min_materials_checkbox = document.getElementById('min_materials_toggle');
    if (min_materials_checkbox) {
        min_materials_checkbox.checked = use_min_materials;
        min_materials_checkbox.addEventListener('change', function() {
            use_min_materials = this.checked;
            solve_recipe();
        });
    }

    // Handler of the solver engine switch
    elements.solver_radios.forEach(radio => {
        radio.addEventListener('change', function() {
            if (!this.checked) {
                return;
            }
            current_solver = this.value;
            save_solver_to_storage(current_solver);
            sync_solver_dependent_controls();
            solve_recipe();
        });
    });

    // The switch has to show the stored choice, not only what the markup says
    apply_solver_to_controls();
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', init); 