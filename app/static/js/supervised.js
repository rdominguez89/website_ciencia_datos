// ==============================================
// GLOBAL STATE MANAGEMENT
// ==============================================

let currentDF = null;
let selectedPredictionColumn = null;
let selectedOneHotColumns = new Set();
let selectedStandardScaleColumns = new Set();
let selectedModel = '';
let modelParams = {};
let modelDefaultParams = {};

// Model definitions
const categoricalModels = [
    { name: 'logistic', params: [{ name: 'C', value: '1.0' }, { name: 'penalty', value: 'l2' }, { name: 'solver', value: 'lbfgs' }] },
    { name: 'svc', params: [{ name: 'C', value: '1.0' }, { name: 'kernel', value: 'rbf' }, { name: 'gamma', value: 'scale' }]},
    { name: 'randomforest', params: [{ name: 'n_estimators', value: '100' }, { name: 'max_depth', value: '2' }, { name: 'min_samples_split', value: '2' }]},
    { name: 'gradientboosting', params: [{ name: 'n_estimators', value: '100' }, { name: 'learning_rate', value: '0.1' }, { name: 'max_depth', value: '3' }]},
    { name: 'xgboost', params: [{ name: 'n_estimators', value: '100' }, { name: 'learning_rate', value: '0.1' }, { name: 'max_depth', value: '3' }, { name: 'eval_metric', value: 'logloss' }]},
    { name: 'knn', params: [{ name: 'n_neighbors', value: '5' }]},
    { name: 'decisiontree', params: [{ name: 'max_depth', value: '2' }, { name: 'min_samples_split', value: '2' }]}
];

const numericalModels = [
    { name: 'poly', params: [{ name: 'degree', value: '2' }] },
    { name: 'linear', params: [] },
    { name: 'ridge', params: [{ name: 'alpha', value: '1.0' }] },
    { name: 'lasso', params: [{ name: 'alpha', value: '1.0' }] },
    { name: 'elasticnet', params: [{ name: 'alpha', value: '1.0' }, { name: 'l1_ratio', value: '0.5' }]},
    { name: 'svr', params: [{ name: 'C', value: '1.0' }, { name: 'kernel', value: 'rbf' }, { name: 'gamma', value: 'scale' }]}
];

// ==============================================
// INITIALIZATION AND EVENT LISTENERS
// ==============================================

/**
 * Initialize the application by setting up event listeners
 */
function initializeApp() {
    setupUploadButton();
    initializeModelDefaults();
}

/**
 * Set up the upload button event listener
 */
function setupUploadButton() {
    document.getElementById('uploadBtn').addEventListener('click', handleFileUpload);
}

/**
 * Initialize default model parameters
 */
function initializeModelDefaults() {
    [...categoricalModels, ...numericalModels].forEach(model => {
        modelDefaultParams[model.name] = {};
        model.params.forEach(param => {
            modelDefaultParams[model.name][param.name] = param.value;
        });
    });
}

// ==============================================
// FILE UPLOAD HANDLING
// ==============================================

/**
 * Handle file upload process
 */
async function handleFileUpload() {
    // Reset state before processing new file
    resetState();

    const file = validateFileInput();
    if (!file) return;

    try {
        const uploadData = await uploadFile(file);
        if (uploadData.error) {
            showError(uploadData.error);
        } else {
            await processData(uploadData);
        }
    } catch (error) {
        handleError('Error during file upload:', error);
    }
}

/**
 * Validate the file input and return the file if valid
 */
function validateFileInput() {
    const fileInput = document.getElementById('csvFile');
    const file = fileInput.files[0];
    
    if (!file) {
        alert('Please select a CSV file first');
        return null;
    }
    
    if (file.size > 5 * 1024 * 1024) {
        alert('File size exceeds 5MB limit');
        return null;
    }
    
    return file;
}

/**
 * Upload file to server
 */
async function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData
    });
    return await response.json();
}

// ==============================================
// DATA PROCESSING
// ==============================================

/**
 * Process uploaded data
 */
async function processData(data) {
    currentDF = data.df;

    try {
        const statsData = await getDataStatistics(data);
        renderDataDisplay(statsData);
    } catch (error) {
        handleError('Error processing data:', error);
    }
}

/**
 * Get statistics for the uploaded data
 */
async function getDataStatistics(data) {
    const response = await fetch('/api/statsummary', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ df: data.df })
    });
    const statsData = await response.json();
    
    if (statsData.error) {
        throw new Error(statsData.error);
    }
    
    // Ensure statsData has the expected structure
    return {
        ...statsData,
        stats: {
            rows: statsData.stats?.rows || 0,
            columns: statsData.stats?.columns || 0,
            duplicates: statsData.stats?.duplicates || 0,
            total_null: statsData.stats?.total_null || 0
        }
    };
}

// ==============================================
// DATA RENDERING
// ==============================================

/**
 * Render the main data display
 */
// Update the renderDataDisplay function to call initializeColumnSelections:
function renderDataDisplay(statsData) {
    // Only initialize column selections on first render or if explicitly needed
    if (!document.querySelector('.data-tabs')) {
        initializeColumnSelections();
    }
    
    // Check for null values, with fallback to avoid undefined errors
    const hasNullValues = statsData.stats?.total_null > 0;
    
    // Conditionally include the analysis section
    const analysisSectionHTML = hasNullValues ? '' : renderAnalysisSection(statsData);
    
    // Preserve existing analysis results
    const dataPreview = document.getElementById('dataPreview');
    const existingResults = document.getElementById('analysis-results-area');
    
    const html = `
        ${renderTabNavigation()}
        <div id="tab-content"></div>
        ${renderDatasetOverview(statsData)}
        ${analysisSectionHTML}
    `;
    
    // Update the main content
    dataPreview.innerHTML = html;
    
    // Reattach existing analysis results if they exist
    if (existingResults) {
        dataPreview.appendChild(existingResults);
    }
    
    loadInitialContent(statsData);
    setupTabHandlers(statsData);
    if (!hasNullValues) {
        setupAnalysisControls(statsData);
    }
}

/**
 * Render tab navigation
 */
function renderTabNavigation() {
    return `
        <div class="data-tabs">
            <button class="tab-btn active" data-tab="preview">Preview</button>
            <button class="tab-btn" data-tab="stats">Statistics</button>
            <button class="tab-btn" data-tab="structure">Structure</button>
        </div>
    `;
}

/**
 * Render dataset overview section
 */
function renderDatasetOverview(statsData) {
    const hasNullValues = statsData.stats?.total_null > 0;
    const hasDuplicates = statsData.stats?.duplicates > 0;
    
    let warningMessage = '';
    let showCleaningLink = false;
    
    if (hasNullValues) {
        warningMessage += `
            <div class="warning-message">
                The Analysis cannot continue having null values.
            </div>
        `;
        showCleaningLink = true;
    }
    
    if (hasDuplicates) {
        warningMessage += `
            <div class="warning-message">
                Warning: The dataset contains duplicate rows.
            </div>
        `;
        showCleaningLink = true;
    }
    
    const cleaningLink = showCleaningLink ? `
        <div class="cleaning-link">
            <a href="/cleaning">Go to Data Cleaning</a>
        </div>
    ` : '';
    
    return `
        <div class="data-section">
            <h3>Dataset Overview</h3>
            <div class="stats-grid">
                <div class="stat-card">
                    <span class="stat-value">${statsData.stats?.rows || 0}</span>
                    <span class="stat-label">Rows</span>
                </div>
                <div class="stat-card">
                    <span class="stat-value">${statsData.stats?.columns || 0}</span>
                    <span class="stat-label">Columns</span>
                </div>
                <div class="stat-card">
                    <span class="stat-value">${statsData.stats?.duplicates || 0}</span>
                    <span class="stat-label">Duplicates</span>
                </div>
                <div class="stat-card">
                    <span class="stat-value">${statsData.stats?.total_null || 0}</span>
                    <span class="stat-label">Null Values</span>
                </div>
            </div>
            ${warningMessage}
            ${cleaningLink}
        </div>
    `;
}

/**
 * Render analysis section
 */
function renderAnalysisSection(statsData) {
    // Categorize columns more precisely
    const numericColumns = statsData.dtype_info.filter(col => 
        (col.type === 'float64' || col.type === 'int64') && 
        !(col.unique_values <= 5 && selectedOneHotColumns.has(col.column))
    );
    
    const categoricalColumns = statsData.dtype_info.filter(col => 
        col.type === 'object' || col.type === 'category' || 
        (col.unique_values <= 5 && selectedOneHotColumns.has(col.column))
    );
    
    // Columns that could be treated as categorical (including low-cardinality numeric)
    const potentialCategoricalColumns = statsData.dtype_info.filter(col => 
        col.type === 'object' || col.type === 'category' || col.unique_values <= 5
    );
    
    // Columns that could be treated as numeric (excluding those already selected for one-hot)
    const potentialNumericColumns = statsData.dtype_info.filter(col => 
        (col.type === 'float64' || col.type === 'int64') &&
        !selectedOneHotColumns.has(col.column)
    );
    
    const allcolumns = [...numericColumns, ...categoricalColumns];
    
    const isCategorical = selectedPredictionColumn ? 
        categoricalColumns.some(col => col.column === selectedPredictionColumn) : false;

    // Check if analysis results exist
    const hasResults = !!document.getElementById('analysis-results-area');
    
    // Conditionally render model selection section only if a prediction target is selected
    const modelSelectionHTML = selectedPredictionColumn ? buildModelSelectionSection(isCategorical, statsData) : '';

    // Conditionally render or disable the Clear Results button
    const clearButtonHTML = hasResults ? 
        `<button id="clearResults" class="action-button clear-button">Clear Results</button>` : 
        `<button id="clearResults" class="action-button clear-button" disabled>Clear Results</button>`;

    return `
        <div class="data-section">
            <div class="analysis-section">
                <h3>Data Analysis Setup</h3>
                ${renderPredictionTargetSection(allcolumns, statsData)}
                ${renderOneHotEncodingSection(potentialCategoricalColumns)}
                ${renderStandardScalingSection(potentialNumericColumns)}
                ${renderTrainTestSplitSection()}
                ${modelSelectionHTML}
                <div class="analysis-controls">
                    <button id="runAnalysis" class="action-button">Run Analysis</button>
                    ${clearButtonHTML}
                </div>
            </div>
        </div>
    `;
}

// ==============================================
// ANALYSIS SECTION COMPONENTS
// ==============================================

/**
 * Render prediction target section
 */
function renderPredictionTargetSection(columns, statsData) {
    const selectedColInfo = selectedPredictionColumn ? 
        statsData.dtype_info.find(col => col.column === selectedPredictionColumn) : null;
    const isLowCardNumeric = selectedColInfo && 
                           (selectedColInfo.type === 'float64' || selectedColInfo.type === 'int64') &&
                           selectedColInfo.unique_values <= 5;

    // Determine current treatment
    const currentTreatment = selectedOneHotColumns.has(selectedPredictionColumn) ? 
                           'categorical' : 'numeric';

    const typeToggle = isLowCardNumeric ? `
        <div class="numeric-type-toggle">
            <label>Treat as:</label>
            <label>
                <input type="radio" name="numericAsType" value="numeric" 
                       ${currentTreatment === 'numeric' ? 'checked' : ''}>
                Numeric (Regression)
            </label>
            <label>
                <input type="radio" name="numericAsType" value="categorical"
                       ${currentTreatment === 'categorical' ? 'checked' : ''}>
                Categorical (Classification)
            </label>
        </div>
    ` : '';

    return `
        <div class="analysis-subsection">
            <h4>Select Prediction Target</h4>
            <div class="column-selection">
                ${renderColumnOptions(columns, 'predictionColumn', 'radio')}
            </div>
            ${typeToggle}
        </div>
    `;
}

/**
 * Render one-hot encoding section
 */
function setupStandardScalingControls(statsData) {
    document.querySelectorAll('.standardscale-checkbox').forEach(checkbox => {
        checkbox.addEventListener('change', function() {
            if (this.checked) {
                selectedStandardScaleColumns.add(this.value);
                selectedOneHotColumns.delete(this.value);
                // Re-render to update available options and model types
                renderDataDisplay(statsData);
            } else {
                selectedStandardScaleColumns.delete(this.value);
                renderDataDisplay(statsData);
            }
        });
    });
}

/**
 * Render one-hot encoding section
 */
function renderOneHotEncodingSection(columns) {
    return `
        <div class="analysis-subsection">
            <h4>Select Columns for Label Encoding</h4>
            <div class="column-selection onehot-selection">
                ${renderColumnOptions(columns, 'onehot', 'checkbox')}
            </div>
        </div>
    `;
}

/**
 * Render standard scaling section
 */
function renderStandardScalingSection(columns) {
    const filteredColumns = columns.filter(col => col.column !== selectedPredictionColumn);
    return `
        <div class="analysis-subsection">
            <h4>Select Columns for Standard Scaling</h4>
            <div class="column-selection standardscale-selection">
                ${renderColumnOptions(filteredColumns, 'standardscale', 'checkbox')}
            </div>
        </div>
    `;
}

/**
 * Render train/test split section
 */
function renderTrainTestSplitSection() {
    return `
        <div class="analysis-subsection">
            <h4>Train/Test Split Configuration</h4>
            <div class="split-config">
                <div class="split-control">
                    <label>Training Size (%):</label>
                    <input type="number" id="trainSize" min="1" max="99" value="80" class="split-input">
                    <button class="split-adjust" data-direction="up" data-target="trainSize">↑</button>
                    <button class="split-adjust" data-direction="down" data-target="trainSize">↓</button>
                </div>
                <div class="split-control">
                    <label>Test Size (%):</label>
                    <input type="number" id="testSize" min="1" max="99" value="20" class="split-input" readonly>
                </div>
                <div class="split-control">
                    <label>Random Seed:</label>
                    <input type="number" id="randomSeed" value="42" class="split-input">
                </div>
            </div>
        </div>
    `;
}

/**
 * Render column options (generic for radio or checkbox)
 */
// In renderColumnOptions function, update the input attributes:
function renderColumnOptions(columns, type, inputType) {
    if (columns.length === 0) {
        return `<p>No columns available</p>`;
    }

    let options = '<div class="options-row">';
    columns.forEach((col, index) => {
        const isPredictionColumn = col.column === selectedPredictionColumn;
        const isCategoricalTreatment = selectedOneHotColumns.has(selectedPredictionColumn);
        
        // Determine if the column is checked
        const checked = type === 'predictionColumn' 
            ? isPredictionColumn
            : type === 'onehot' 
                ? selectedOneHotColumns.has(col.column)
                : selectedStandardScaleColumns.has(col.column);
        
        // Determine if the column should be disabled
        const disabled = type === 'onehot'
            ? selectedStandardScaleColumns.has(col.column) || (isPredictionColumn && isCategoricalTreatment) // Disable prediction column if treated as categorical
            : type === 'standardscale'
                ? selectedOneHotColumns.has(col.column) || (isPredictionColumn && isCategoricalTreatment) // Disable in standard scaling if categorical
                : false;

        const isLowCardinalityNumeric = (col.type === 'float64' || col.type === 'int64') && col.unique_values <= 5;
        const typeIndicator = isLowCardinalityNumeric ? ' (numeric, low cardinality)' : 
                            (col.type === 'object' || col.type === 'category') ? ' (categorical)' : ' (numeric)';

        options += `
            <div class="column-option">
                <label>
                    <input type="${inputType}" 
                           name="${type === 'predictionColumn' ? 'predictionColumn' : type + 'Columns'}" 
                           value="${col.column}" 
                           ${checked ? 'checked' : ''} 
                           ${disabled ? 'disabled' : ''}
                           class="${type === 'predictionColumn' ? '' : type + '-checkbox'}">
                    ${col.column}${typeIndicator}
                </label>
            </div>
        `;
        
        if ((index + 1) % 4 === 0 && index + 1 < columns.length) {
            options += '</div><div class="options-row">';
        }
    });
    options += '</div>';
    return options;
}

// ==============================================
// MODEL SELECTION SECTION
// ==============================================

/**
 * Build model selection section
 */
function buildModelSelectionSection(isCategorical, statsData) {
    // Check if prediction target is a low-cardinality numeric column
    const predictionColumnInfo = selectedPredictionColumn ? 
        statsData.dtype_info.find(col => col.column === selectedPredictionColumn) : null;
    
    const isLowCardNumeric = predictionColumnInfo && 
                           (predictionColumnInfo.type === 'float64' || predictionColumnInfo.type === 'int64') &&
                           predictionColumnInfo.unique_values <= 5;

    // Determine final type - either categorical or numeric
    const finalIsCategorical = isCategorical || 
                             (isLowCardNumeric && selectedOneHotColumns.has(selectedPredictionColumn));

    const models = finalIsCategorical ? categoricalModels : numericalModels;
    const modelType = finalIsCategorical ? 'classification' : 'regression';
    
    return `
        <div class="model-selection-section">
            <h4>Select ${modelType} Model</h4>
            <div class="model-type-info">
                ${isLowCardNumeric ? `Treating "${selectedPredictionColumn}" as ${finalIsCategorical ? 'categorical' : 'numeric'}` : ''}
            </div>
            ${renderModelOptions(models, modelType)}
            ${renderModelParameters(models)}
        </div>
    `;
}

/**
 * Render model options
 */
function renderModelOptions(models, modelType) {
    let options = '<div class="model-options-row">';
    
    models.forEach((model, index) => {
        options += `
            <div class="model-option">
                <label>
                    <input type="radio" name="modelType" value="${model.name}" 
                           data-model-type="${modelType}"
                           ${selectedModel === model.name ? 'checked' : ''}>
                    ${model.name}
                </label>
            </div>
        `;
        
        if ((index + 1) % 4 === 0 && index + 1 < models.length) {
            options += '</div><div class="model-options-row">';
        }
    });
    
    options += '</div>';
    return options;
}

/**
 * Render model parameters
 */
function renderModelParameters(models) {
    let paramsHTML = '';
    
    models.forEach(model => {
        if (model.params.length > 0) {
            paramsHTML += `
                <div class="model-params" id="params-${model.name}" 
                     style="display: ${selectedModel === model.name ? 'block' : 'none'}">
                    <h5>${model.name} Parameters</h5>
                    <div class="param-grid">
                        ${renderModelParameterInputs(model)}
                    </div>
                </div>
            `;
        }
    });
    
    return paramsHTML;
}

function renderModelParameterInputs(model) {
    let inputs = '';
    
    model.params.forEach(param => {
        // Use user-defined value if available, otherwise use default
        const paramValue = modelParams[model.name]?.[param.name] !== undefined 
            ? modelParams[model.name][param.name] 
            : param.value;
            
        inputs += `
            <div class="param-control">
                <label>${param.name}:</label>
                <input type="text" class="param-input" 
                       data-model="${model.name}" 
                       data-param="${param.name}" 
                       value="${paramValue}"
                       placeholder="${param.value}">
            </div>
        `;
    });
    
    return inputs;
}

/**
 * Render inputs for model parameters
 */
function renderModelParameterInputs(model) {
    let inputs = '';
    
    model.params.forEach(param => {
        const paramValue = modelParams[model.name]?.[param.name] !== undefined 
            ? modelParams[model.name][param.name] 
            : param.value;
            
        inputs += `
            <div class="param-control">
                <label>${param.name}:</label>
                <input type="text" class="param-input" 
                       data-model="${model.name}" 
                       data-param="${param.name}" 
                       value="${paramValue}"
                       placeholder="${param.value}">
            </div>
        `;
    });
    
    return inputs;
}

// ==============================================
// TAB HANDLING
// ==============================================

/**
 * Set up tab click handlers
 */
function setupTabHandlers(statsData) {
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            loadTabContent(btn.dataset.tab, statsData);
        });
    });
}

/**
 * Load initial tab content
 */
function loadInitialContent(statsData) {
    loadTabContent('preview', statsData);
}

/**
 * Load content for specific tab
 */
function loadTabContent(tabName, statsData) {
    const tabContent = document.getElementById('tab-content');
    
    switch(tabName) {
        case 'preview':
            tabContent.innerHTML = `
                <div id="preview-content">
                    <h3>First 5 Rows</h3>
                    ${statsData.data_head}
                </div>
            `;
            break;
        case 'stats':
            tabContent.innerHTML = `
                <div id="stats-content">
                    ${generateStatsHTML(statsData)}
                </div>
            `;
            break;
        case 'structure':
            tabContent.innerHTML = generateStructureHTML(statsData);
            break;
    }
}

// ==============================================
// STATISTICS AND STRUCTURE DISPLAY
// ==============================================

/**
 * Generate statistics HTML table
 */
function generateStatsHTML(statsData) {
    if (Object.keys(statsData.num_stats).length === 0) {
        return '<p>No statistics available</p>';
    }

    const originalColumns = statsData.column_order;
    const allMetrics = collectAllMetrics(statsData.num_stats);
    const metricsToShow = filterAndSortMetrics(allMetrics);

    let statsHTML = '<h3>Dataframe Statistics</h3><table class="stats-table"><tr><th>Statistic</th>';
    
    originalColumns.forEach(col => {
        statsHTML += `<th>${col}</th>`;
    });
    statsHTML += '</tr>';
    
    metricsToShow.forEach(metric => {
        statsHTML += `<tr><td>${metric}</td>`;
        originalColumns.forEach(col => {
            const value = statsData.num_stats[col][metric];
            statsHTML += `<td>${value !== undefined ? value : '-'}</td>`;
        });
        statsHTML += '</tr>';
    });
    
    statsHTML += '</table>';
    return statsHTML;
}

/**
 * Collect all metrics from numerical stats
 */
function collectAllMetrics(numStats) {
    const allMetrics = new Set();
    Object.values(numStats).forEach(colStats => {
        Object.keys(colStats).forEach(metric => allMetrics.add(metric));
    });
    return allMetrics;
}

/**
 * Filter and sort metrics for display
 */
function filterAndSortMetrics(metrics) {
    const excludedMetrics = ['unique', 'top', 'freq'];
    const order = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'];
    
    return Array.from(metrics)
        .filter(metric => !excludedMetrics.includes(metric))
        .sort((a, b) => order.indexOf(a) - order.indexOf(b));
}

/**
 * Generate structure HTML table
 */
function generateStructureHTML(statsData) {
    let structureHTML = '<h3>Column Information</h3><table class="structure-table"><tr><th>Column</th><th>Type</th><th>Unique Values</th><th>Null Values</th></tr>';
    
    statsData.dtype_info.forEach(col => {
        structureHTML += `
            <tr>
                <td>${col.column}</td>
                <td>${col.type}</td>
                <td>${col.unique_values}</td>
                <td>${col.null_values}</td>
            </tr>
        `;
    });
    
    structureHTML += '</table>';
    return structureHTML;
}

// ==============================================
// ANALYSIS CONTROLS SETUP
// ==============================================

/**
 * Set up all analysis control event listeners
 */
function setupAnalysisControls(statsData) {
    setupPredictionColumnControls(statsData);
    setupOneHotEncodingControls(statsData);
    setupStandardScalingControls(statsData);
    setupTrainTestSplitControls();
    setupModelSelectionControls();
    setupParameterInputControls();
    setupRunAnalysisButton(statsData);
    
    // Set up clear results button only if it exists and is not disabled
    const clearResultsBtn = document.getElementById('clearResults');
    if (clearResultsBtn && !clearResultsBtn.disabled) {
        clearResultsBtn.addEventListener('click', clearAnalysisResults);
    }
}

/**
 * Set up prediction column controls
 */
function setupPredictionColumnControls(statsData) {
    // Handle prediction column selection
    document.querySelectorAll('input[name="predictionColumn"]').forEach(radio => {
        radio.addEventListener('change', function() {
            if (this.checked) {
                selectedPredictionColumn = this.value;
                const colInfo = statsData.dtype_info.find(col => col.column === this.value);
                const isLowCardNumeric = colInfo && 
                                      (colInfo.type === 'float64' || colInfo.type === 'int64') &&
                                      colInfo.unique_values <= 5;
                
                // Set default treatment
                if (isLowCardNumeric) {
                    selectedOneHotColumns.delete(this.value); // Default to numeric
                    selectedStandardScaleColumns.add(this.value);
                } else if (colInfo.type === 'object' || colInfo.type === 'category') {
                    selectedOneHotColumns.add(this.value); // Default to categorical
                    selectedStandardScaleColumns.delete(this.value);
                } else {
                    selectedStandardScaleColumns.add(this.value); // Default to numeric
                    selectedOneHotColumns.delete(this.value);
                }
                
                // Reset model and parameters
                selectedModel = '';
                modelParams = {};
                
                // Initialize default parameters for the first appropriate model
                const isCategorical = selectedOneHotColumns.has(this.value);
                const models = isCategorical ? categoricalModels : numericalModels;
                if (models.length > 0) {
                    selectedModel = models[0].name; // Select the first model
                    modelParams[selectedModel] = { ...modelDefaultParams[selectedModel] }; // Initialize with defaults
                }
                
                // Re-render to update UI, preserving analysis results
                renderDataDisplay(statsData);
            }
        });
    });

    // Handle treatment type change using event delegation
    document.addEventListener('change', function(e) {
        if (e.target && e.target.name === 'numericAsType' && selectedPredictionColumn) {
            if (e.target.value === 'categorical') {
                selectedOneHotColumns.add(selectedPredictionColumn);
                selectedStandardScaleColumns.delete(selectedPredictionColumn);
            } else {
                selectedOneHotColumns.delete(selectedPredictionColumn);
                selectedStandardScaleColumns.add(selectedPredictionColumn);
            }
            
            // Reset model and parameters
            selectedModel = '';
            modelParams = {};
            
            // Initialize default parameters for the first appropriate model
            const isCategorical = e.target.value === 'categorical';
            const models = isCategorical ? categoricalModels : numericalModels;
            if (models.length > 0) {
                selectedModel = models[0].name; // Select the first model
                modelParams[selectedModel] = { ...modelDefaultParams[selectedModel] }; // Initialize with defaults
            }
            
            // Re-render to update UI, preserving analysis results
            renderDataDisplay(statsData);
        }
    });
}

/**
 * Set up one-hot encoding controls
 */
// Update the setupOneHotEncodingControls and setupStandardScalingControls functions:
function setupOneHotEncodingControls(statsData) {
    document.querySelectorAll('.onehot-checkbox').forEach(checkbox => {
        checkbox.addEventListener('change', function() {
            // Prevent changes if this is the prediction column treated as categorical
            if (this.value === selectedPredictionColumn && selectedOneHotColumns.has(selectedPredictionColumn)) {
                this.checked = true; // Force it to stay checked
                return;
            }
            
            if (this.checked) {
                selectedOneHotColumns.add(this.value);
                selectedStandardScaleColumns.delete(this.value);
            } else {
                selectedOneHotColumns.delete(this.value);
            }
            // Re-render to update UI, preserving analysis results
            renderDataDisplay(statsData);
        });
    });
}


/**
 * Set up standard scaling controls
 */
function setupStandardScalingControls(statsData) {
    document.querySelectorAll('.standardscale-checkbox').forEach(checkbox => {
        checkbox.addEventListener('change', function() {
            if (this.checked) {
                selectedStandardScaleColumns.add(this.value);
                selectedOneHotColumns.delete(this.value);
            } else {
                selectedStandardScaleColumns.delete(this.value);
            }
            // If this is the prediction column, update its treatment
            if (this.value === selectedPredictionColumn) {
                if (this.checked) {
                    selectedOneHotColumns.delete(this.value);
                } else {
                    selectedOneHotColumns.add(this.value);
                }
            }
            // Re-render to update UI, preserving analysis results
            renderDataDisplay(statsData);
        });
    });
}

// Add a new function to properly initialize the selections when rendering:
function initializeColumnSelections() {
    // Only initialize if no selections exist to avoid overwriting user choices
    if (selectedOneHotColumns.size === 0 && selectedStandardScaleColumns.size === 0) {
        // Initialize one-hot encoding selections from checkboxes
        selectedOneHotColumns = new Set();
        document.querySelectorAll('.onehot-checkbox:checked').forEach(checkbox => {
            selectedOneHotColumns.add(checkbox.value);
        });

        // Initialize standard scaling selections from checkboxes
        selectedStandardScaleColumns = new Set();
        document.querySelectorAll('.standardscale-checkbox:checked').forEach(checkbox => {
            selectedStandardScaleColumns.add(checkbox.value);
        });
    }
}

/**
 * Set up train/test split controls
 */
function setupTrainTestSplitControls() {
    const trainInput = document.getElementById('trainSize');
    const testInput = document.getElementById('testSize');
    
    trainInput.addEventListener('input', function() {
        const trainValue = parseInt(this.value);
        if (isNaN(trainValue)) {
            this.value = 80;
        } else if (trainValue < 1) {
            this.value = 1;
        } else if (trainValue > 99) {
            this.value = 99;
        }
        testInput.value = 100 - parseInt(this.value);
    });
    
    document.querySelectorAll('.split-adjust').forEach(button => {
        button.addEventListener('click', function() {
            const target = document.getElementById(this.dataset.target);
            let value = parseInt(target.value);
            
            if (this.dataset.direction === 'up') {
                value = Math.min(value + 1, 99);
            } else {
                value = Math.max(value - 1, 1);
            }
            
            target.value = value;
            if (this.dataset.target === 'trainSize') {
                testInput.value = 100 - value;
            }
            target.dispatchEvent(new Event('input'));
        });
    });
}

/**
 * Set up model selection controls
 */
function setupModelSelectionControls() {
    document.querySelectorAll('input[name="modelType"]').forEach(radio => {
        radio.addEventListener('change', function() {
            if (this.checked) {
                selectedModel = this.value;
                
                // Initialize model parameters with defaults if not already set
                if (!modelParams[selectedModel]) {
                    modelParams[selectedModel] = { ...modelDefaultParams[selectedModel] };
                }
                
                // Show parameters for the selected model
                document.querySelectorAll('.model-params').forEach(el => {
                    el.style.display = 'none';
                });
                const paramsSection = document.getElementById(`params-${this.value}`);
                if (paramsSection) paramsSection.style.display = 'block';
            }
        });
    });
}

/**
 * Set up model parameter input controls
 */
function setupParameterInputControls() {
    document.querySelectorAll('.param-input').forEach(input => {
        input.addEventListener('change', function() {
            const model = this.dataset.model;
            const param = this.dataset.param;
            const value = this.value;
            
            if (!modelParams[model]) {
                modelParams[model] = {};
            }
            modelParams[model][param] = value;
        });
    });
}

/**
 * Set up run analysis button
 */
/**
 * Set up run analysis button
 */
function setupRunAnalysisButton(statsData) {
    const runAnalysisBtn = document.getElementById('runAnalysis');
    if (!runAnalysisBtn) return;

    runAnalysisBtn.addEventListener('click', async function() {
        // Disable the button and show loading state
        runAnalysisBtn.disabled = true;
        runAnalysisBtn.textContent = 'Processing...';
        
        try {
            if (!validateAnalysisInputs(statsData)) {
                throw new Error('Invalid analysis inputs');
            }

            const analysisConfig = prepareAnalysisConfig(statsData);
            const analysisResults = await runAnalysis(analysisConfig);
            displayAnalysisResults(analysisResults);
        } catch (error) {
            handleError('Error during analysis:', error);
        } finally {
            // Re-enable the button regardless of success or failure
            runAnalysisBtn.disabled = false;
            runAnalysisBtn.textContent = 'Run Analysis';
        }
    });
}

// ==============================================
// ANALYSIS EXECUTION
// ==============================================

/**
 * Validate analysis inputs before running
 */
function validateAnalysisInputs(statsData) {
    if (!selectedPredictionColumn) {
        alert('Please select a prediction target column');
        return false;
    }
    
    if (!selectedModel) {
        alert('Please select a model');
        return false;
    }

    const trainSize = parseInt(document.getElementById('trainSize').value);
    if (isNaN(trainSize)) {
        alert('Please enter valid numbers for train/test split');
        return false;
    }
    
    return true;
}

/**
 * Prepare analysis configuration object
 */
function prepareAnalysisConfig(statsData) {
    const encoderColumns = statsData.dtype_info.filter(col => col.type === 'object' || col.type === 'category' || col.unique_values <= 5);
    const isCategorical = encoderColumns.some(col => col.column === selectedPredictionColumn) || 
                         selectedOneHotColumns.has(selectedPredictionColumn);
    const models = isCategorical ? categoricalModels : numericalModels;
    const selectedModelConfig = models.find(m => m.name === selectedModel);
    
    // Initialize params with default values if not already set
    let params = modelParams[selectedModel] || {};
    if (selectedModelConfig && Object.keys(params).length === 0) {
        params = { ...modelDefaultParams[selectedModel] };
    }
    
    return {
        df: currentDF,
        prediction_column: selectedPredictionColumn,
        onehot_columns: Array.from(selectedOneHotColumns),
        standardscale_columns: Array.from(selectedStandardScaleColumns),
        train_size: parseInt(document.getElementById('trainSize').value) / 100,
        test_size: parseInt(document.getElementById('testSize').value) / 100,
        random_seed: parseInt(document.getElementById('randomSeed').value),
        model: selectedModel,
        model_params: params
    };
}

/**
 * Run analysis by sending request to server
 */
async function runAnalysis(config) {
    const response = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
    });
    const data = await response.json();
    
    if (data.error) {
        throw new Error(data.error);
    }
    
    return data;
}

// ==============================================
// RESULTS DISPLAY
// ==============================================

/**
 * Display analysis results
 */
async function displayAnalysisResults(data) {
    try {
        // Get or create display area
        let displayArea = document.getElementById('analysis-results-area');
        if (!displayArea) {
            displayArea = document.createElement('div');
            displayArea.id = 'analysis-results-area';
            document.querySelector('.analysis-section').appendChild(displayArea);
        }
        
        // Create results container for this specific result
        const resultsContainer = document.createElement('div');
        resultsContainer.className = 'analysis-results';
        resultsContainer.innerHTML = '<h3>Analysis Results</h3>';
        
        // Add visualization if available
        if (data.image_data) {
            resultsContainer.appendChild(createVisualizationCard(data.image_data));
        }
        
        // Add metrics if available
        if (data.metrics) {
            resultsContainer.appendChild(createMetricsDisplay(data.metrics));
        }
        
        // Add this result to the display area
        if (displayArea.firstChild) {
            displayArea.insertBefore(resultsContainer, displayArea.firstChild);
        } else {
            displayArea.appendChild(resultsContainer);
        }
        
        // Fetch updated stats to ensure valid statsData
        const statsData = await getDataStatistics({ df: currentDF });
        renderDataDisplay(statsData);
    } catch (error) {
        handleError('Error updating UI after displaying results:', error);
    }
}

/**
 * Create visualization card with image
 */
function createVisualizationCard(imageData) {
    const visualizationContainer = document.createElement('div');
    visualizationContainer.className = 'visualization-container';
    
    const imgCard = document.createElement('div');
    imgCard.className = 'visualization-card';
    imgCard.innerHTML = `
        <h4>Analysis Visualization</h4>
        <img src="data:image/png;base64,${imageData}" alt="Analysis Result">
    `;
    
    // Make image clickable for fullscreen
    imgCard.querySelector('img').addEventListener('click', function() {
        createImageOverlay(this.src, this.alt);
    });
    
    visualizationContainer.appendChild(imgCard);
    return visualizationContainer;
}

/**
 * Create metrics display
 */
function createMetricsDisplay(metrics) {
    const metricsDiv = document.createElement('div');
    metricsDiv.className = 'analysis-metrics';
    
    let metricsContent = '<h4>Analysis Metrics</h4><div class="metrics-grid">';
    for (const [key, value] of Object.entries(metrics)) {
        metricsContent += `
            <div class="metric-item">
                <span class="metric-key">${key}:</span>
                <span class="metric-value">${value}</span>
            </div>
        `;
    }
    metricsContent += '</div>';
    
    metricsDiv.innerHTML = metricsContent;
    return metricsDiv;
}

/**
 * Reset all internal state variables when a new CSV file is uploaded
 */
function resetState() {
    selectedPredictionColumn = null;
    selectedOneHotColumns = new Set();
    selectedStandardScaleColumns = new Set();
    selectedModel = '';
    modelParams = {};
    modelDefaultParams = {};
    
    // Reinitialize modelDefaultParams with default values
    [...categoricalModels, ...numericalModels].forEach(model => {
        modelDefaultParams[model.name] = {};
        model.params.forEach(param => {
            modelDefaultParams[model.name][param.name] = param.value;
        });
    });
}

// ==============================================
// IMAGE OVERLAY
// ==============================================

/**
 * Create fullscreen image overlay
 */
function createImageOverlay(src, alt) {
    // Add class to body to hide scrollbar
    document.body.classList.add('overlay-active');

    const overlay = document.createElement('div');
    overlay.className = 'plot-overlay';

    const content = document.createElement('div');
    content.className = 'plot-overlay-content';

    const closeBtn = document.createElement('span');
    closeBtn.className = 'close-overlay';
    closeBtn.innerHTML = '×';

    const img = document.createElement('img');
    img.src = src;
    img.alt = alt;

    content.appendChild(closeBtn);
    content.appendChild(img);
    overlay.appendChild(content);
    document.body.appendChild(overlay);

    function closeOverlay() {
        document.body.removeChild(overlay);
        // Remove class from body to restore scrolling
        document.body.classList.remove('overlay-active');
        document.removeEventListener('keydown', escClose);
    }

    closeBtn.addEventListener('click', closeOverlay);
    overlay.addEventListener('click', function(e) {
        if (e.target === overlay) {
            closeOverlay();
        }
    });

    function escClose(e) {
        if (e.key === 'Escape') {
            closeOverlay();
        }
    }
    document.addEventListener('keydown', escClose);
}

/**
 * Clear analysis results from the display
 */
async function clearAnalysisResults() {
    try {
        const displayArea = document.getElementById('analysis-results-area');
        if (displayArea) {
            displayArea.remove();
        }
        // Fetch updated stats to ensure valid statsData
        const statsData = await getDataStatistics({ df: currentDF });
        renderDataDisplay(statsData);
    } catch (error) {
        handleError('Error updating UI after clearing results:', error);
    }
}

// ==============================================
// ERROR HANDLING
// ==============================================

/**
 * Handle errors consistently
 */
function handleError(message, error) {
    console.error(message, error);
    alert('An error occurred: ' + (error.message || message));
}

/**
 * Show error message to user
 */
function showError(message) {
    alert(message);
}

// ==============================================
// INITIALIZE APPLICATION
// ==============================================

// Start the application when the DOM is fully loaded
document.addEventListener('DOMContentLoaded', initializeApp);