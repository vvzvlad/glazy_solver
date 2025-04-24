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

// DOM Elements
const elements = {
    recipe_name: document.getElementById('recipe_name'),
    solutions_container: document.getElementById('solutions_container'),
    
    // Get all UMF input elements
    SiO2: document.getElementById('SiO2'),
    Al2O3: document.getElementById('Al2O3'),
    B2O3: document.getElementById('B2O3'),
    Na2O: document.getElementById('Na2O'),
    K2O: document.getElementById('K2O'),
    MgO: document.getElementById('MgO'),
    CaO: document.getElementById('CaO'),
    SrO: document.getElementById('SrO'),
    Fe2O3: document.getElementById('Fe2O3'),
    TiO2: document.getElementById('TiO2')
};

// Initialize the app
function init() {
    // Check API health
    check_api_health();
    
    // Set initial UMF values
    display_umf_values();
    
    // Setup event listeners
    setup_event_listeners();
    
    // Автоматически рассчитываем решение при загрузке
    solve_recipe();
}

// Check if API server is running
async function check_api_health() {
    try {
        await api.check_health();
        console.log('API server is running');
    } catch (error) {
        console.error('API server is not available. Some features may not work correctly.');
        show_status_message('API сервер недоступен. Некоторые функции могут не работать корректно.');
    }
}

// Display UMF values in inputs
function display_umf_values() {
    for (const [oxide, value] of Object.entries(current_umf)) {
        const element = elements[oxide];
        if (element) {
            element.value = value.toFixed(3);
        }
    }
}

// Get current UMF values from inputs
function get_umf_from_inputs() {
    const oxide_inputs = document.querySelectorAll('.oxide-input');
    const umf = {};
    
    oxide_inputs.forEach(input => {
        const value = parseFloat(input.value);
        if (!isNaN(value) && value > 0) {
            umf[input.id] = value;
        }
    });
    
    return umf;
}

// Show status message in solutions container
function show_status_message(message, is_loading = false) {
    elements.solutions_container.innerHTML = '';
    
    const status_elem = document.createElement('div');
    status_elem.className = is_loading ? 'loading-status' : 'status-message';
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
    
    // Показываем сообщение о расчете
    show_status_message('Расчет...', true);
    
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
    umf_title.textContent = 'UMF решения:';
    
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
    
    // Разделяем оксиды по группам
    const groupR2O_RO = ['Na2O', 'K2O', 'MgO', 'CaO', 'SrO', 'ZnO', 'BaO', 'Li2O', 'PbO'];
    const groupR2O3 = ['Al2O3', 'B2O3', 'Fe2O3', 'TiO2', 'ZrO2', 'MnO', 'P2O5', 'Cr2O3'];
    const groupRO2 = ['SiO2', 'SnO2'];
    
    // Создаем группы
    const createGroup = (title, oxides) => {
        const groupContainer = document.createElement('div');
        groupContainer.className = 'solution-umf-group';
        
        const groupTitle = document.createElement('div');
        groupTitle.className = 'solution-umf-group-title';
        groupTitle.innerHTML = title;
        
        const umf_grid = document.createElement('div');
        umf_grid.className = 'solution-umf-grid single-column';
        
        // Фильтруем и сортируем оксиды в группе
        const groupOxides = oxides.filter(oxide => filtered_umf[oxide] !== undefined);
        
        // Если группа пустая, не создаем её
        if (groupOxides.length === 0) {
            return null;
        }
        
        groupOxides.forEach(oxide => {
            const umf_item = document.createElement('div');
            umf_item.className = 'solution-umf-item';
            
            const oxide_name = document.createElement('div');
            oxide_name.className = 'solution-umf-name';
            oxide_name.innerHTML = format_oxide_name(oxide);
            
            const oxide_value = document.createElement('div');
            oxide_value.className = 'solution-umf-value';
            oxide_value.textContent = filtered_umf[oxide].toFixed(3);
            
            umf_item.appendChild(oxide_name);
            umf_item.appendChild(oxide_value);
            umf_grid.appendChild(umf_item);
        });
        
        groupContainer.appendChild(groupTitle);
        groupContainer.appendChild(umf_grid);
        
        return groupContainer;
    };
    
    // Создаем группы оксидов
    const groupR2O_RO_element = createGroup('R<sub>2</sub>O/RO', groupR2O_RO);
    const groupR2O3_element = createGroup('R<sub>2</sub>O<sub>3</sub>', groupR2O3);
    const groupRO2_element = createGroup('RO<sub>2</sub>', groupRO2);
    
    // Добавляем группы в контейнер
    if (groupR2O_RO_element) umf_groups_container.appendChild(groupR2O_RO_element);
    if (groupR2O3_element) umf_groups_container.appendChild(groupR2O3_element);
    if (groupRO2_element) umf_groups_container.appendChild(groupRO2_element);
    
    umf_container.appendChild(umf_title);
    umf_container.appendChild(umf_groups_container);
    
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
            show_status_message('Введите значения UMF для решения.');
            return;
        }
        
        // Show loading
        show_status_message('Выполняется решение...', true);
        
        // Call API to solve recipe
        const solutions = await api.solve_recipe(umf, {
            max_solutions: 15,  // Запрашиваем больше для полноценной сортировки
            min_materials: true,
            error_tolerance: 0.05
        });
        
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
            show_status_message('Не удалось найти подходящие решения.');
        }
    } catch (error) {
        console.error('Error solving recipe:', error);
        show_status_message(`Ошибка при поиске решения: ${error.message}`);
    }
}

// Display found solutions
function display_solutions() {
    elements.solutions_container.innerHTML = '';
    
    if (current_solutions.length === 0) {
        show_status_message('Решения не найдены.');
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
            
            const material_name = document.createElement('div');
            material_name.className = 'solution-recipe-name';
            material_name.textContent = material;
            
            const material_amount = document.createElement('div');
            material_amount.className = 'solution-recipe-amount';
            material_amount.textContent = `${amount.toFixed(1)}%`;
            
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
    // Oxide inputs change
    document.querySelectorAll('.oxide-input').forEach(input => {
        input.addEventListener('input', () => {
            current_umf = get_umf_from_inputs();
            debounce_solve();
        });
    });
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', init); 