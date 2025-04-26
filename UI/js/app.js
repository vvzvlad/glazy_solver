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
        const { max_solutions = 5, min_materials = true, error_tolerance = 0.01 } = options;
        
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
    }
};

// App State
let current_umf = {
    'SiO2': 3.144,
    'Al2O3': 0.378,
    'B2O3': 0.265,
    'Na2O': 0.143,
    'K2O': 0.086,
    'MgO': 0.048,
    'CaO': 0.717,
    'SrO': 0.000,
    'Fe2O3': 0.000,
    'TiO2': 0.000
};

let current_solutions = [];
let calculate_timer = null;
let all_oxides = {}; // Будет содержать все доступные оксиды из molar_masses.json
let is_calculating = false;

// Определение групп оксидов
const oxide_groups = {
    'r2o_ro': ['Na2O', 'K2O', 'Li2O', 'MgO', 'CaO', 'SrO', 'BaO', 'ZnO', 'PbO', 'CdO', 'CuO', 'FeO', 'MnO'],
    'r2o3': ['Al2O3', 'B2O3', 'Fe2O3', 'Cr2O3', 'Bi2O3', 'La2O3', 'Y2O3', 'P2O5', 'V2O5'],
    'ro2': ['SiO2', 'TiO2', 'ZrO2', 'SnO2', 'MnO2', 'GeO2']
};

// DOM Elements
const elements = {
    recipe_name: document.getElementById('recipe_name'),
    solutions_container: document.getElementById('solutions_container'),
    r2o_ro_table: document.getElementById('r2o_ro_table'),
    r2o3_table: document.getElementById('r2o3_table'),
    ro2_table: document.getElementById('ro2_table'),
    add_oxide_buttons: document.querySelectorAll('.add-oxide-btn'),
    calculation_status: document.getElementById('calculation_status')
};

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
        // Кодируем JSON в URL-safe формат
        window.location.hash = encodeURIComponent(umf_json);
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
    
    // Check API health
    check_api_health();
    
    // Загружаем формулу из URL
    load_umf_from_storage();
    
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
    // Добавляем стандартные оксиды из current_umf
    for (const [oxide, value] of Object.entries(current_umf)) {
        if (value !== undefined) {
            const group = get_oxide_group(oxide);
            if (group) {
                add_oxide_to_table(group, oxide, value);
            }
        }
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
function add_oxide_to_table(group, selected_oxide = null, value = 0) {
    const table = elements[`${group}_table`];
    if (!table) return;
    
    const row = document.createElement('tr');
    
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
    for (const oxide of oxide_groups[group]) {
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

// Добавляем обработчики событий для строки оксида
function setup_oxide_row_events(select, input, delete_button) {
    // При изменении выбранного оксида
    select.addEventListener('change', () => {
        // Обновляем ID инпута
        input.id = select.value;
        
        // Обновляем current_umf
        current_umf = get_umf_from_inputs();
        
        // Сохраняем в URL
        save_umf_to_storage(current_umf);
        
        // Запускаем расчет
        debounce_solve();
    });
    
    // При изменении значения
    input.addEventListener('input', () => {
        // Обновляем current_umf
        current_umf = get_umf_from_inputs();
        
        // Сохраняем в URL
        save_umf_to_storage(current_umf);
        
        // Запускаем расчет
        debounce_solve();
    });
    
    // При нажатии на кнопку удаления
    delete_button.addEventListener('click', () => {
        // Удаляем строку
        const row = delete_button.closest('tr');
        row.remove();
        
        // Обновляем current_umf
        current_umf = get_umf_from_inputs();
        
        // Сохраняем в URL
        save_umf_to_storage(current_umf);
        
        // Запускаем расчет
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
    
    // Устанавливаем статус расчета
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
        
        // Добавляем оксиды из решения
        groupOxides.forEach(oxide => {
            const umf_item = document.createElement('div');
            umf_item.className = 'solution-umf-item';
            
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
        missingGroupOxides.forEach(oxide => {
            const umf_item = document.createElement('div');
            umf_item.className = 'solution-umf-item';
            
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
        const umf = get_umf_from_inputs();
        
        if (Object.keys(umf).length === 0) {
            show_calculation_status(false);
            is_calculating = false;
            show_error_message('Введите значения UMF для решения.');
            return;
        }
        
        // Call API to solve recipe
        const solutions = await api.solve_recipe(umf, {
            max_solutions: 15,  // Запрашиваем больше для полноценной сортировки
            min_materials: true,
            error_tolerance: 0.05
        });
        
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

// Setup event listeners
function setup_event_listeners() {
    // Добавляем обработчики для кнопок добавления оксидов
    elements.add_oxide_buttons.forEach(button => {
        button.addEventListener('click', () => {
            const group = button.dataset.group;
            add_oxide_to_table(group);
        });
    });
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', init); 