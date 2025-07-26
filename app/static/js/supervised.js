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
let cvState = {
    method: 'none',
    kfoldSplits: 5,
    shuffleSplits: 5,
    shuffleTestSize: 20
};
let selectedColumns = new Set();
let splitSettings = {
    trainSize: 80,
    testSize: 20,
    randomSeed: 42
};

// Model definitions
const categoricalModels = [
    { name: 'logistic', params: [{ name: 'C', value: '1.0' }, { name: 'penalty', value: 'l2' }, { name: 'solver', value: 'lbfgs' }], description: 'Best for simple binary classification with linear data.' },
    { name: 'svc', params: [{ name: 'C', value: '1.0' }, { name: 'kernel', value: 'rbf' }, { name: 'gamma', value: 'scale' }], description: 'Good for small-to-medium datasets with clear margins (linear or kernel-based). Handles nonlinearity well.' },
    { name: 'randomforest', params: [{ name: 'n_estimators', value: '100' }, { name: 'max_depth', value: '2' }, { name: 'min_samples_split', value: '2' }], description: 'Robust general-purpose classifier, handles nonlinearity well. Good for large datasets.'},
    { name: 'gradientboosting', params: [{ name: 'n_estimators', value: '100' }, { name: 'learning_rate', value: '0.1' }, { name: 'max_depth', value: '3' }], description: 'High accuracy for tabular data, great with imbalanced datasets.'},
    { name: 'xgboost', params: [{ name: 'n_estimators', value: '100' }, { name: 'learning_rate', value: '0.1' }, { name: 'max_depth', value: '3' }, { name: 'eval_metric', value: 'logloss' }], description: 'Best for structured data, optimized performance.'},
    { name: 'knn', params: [{ name: 'n_neighbors', value: '5' }], description: 'Works well for small, low-dimensional data with local patterns. Prone to noise and high dimensions.'},
    { name: 'decisiontree', params: [{ name: 'max_depth', value: '2' }, { name: 'min_samples_split', value: '2' }], description: 'Simple and interpretable, but prone to overfitting; good for small datasets.'},
    { name: 'adaboost', params: [{name: 'n_estimators', value: '50'}, {name: 'learning_rate', value: '1.0'}], description: 'Good for boosting weak learners.'},
    { name: 'bagging', params: [{name: 'n_estimators', value: '10'}], description: 'Reduces variance, improves stability.'},
];

const numericalModels = [
    { name: 'poly', params: [{ name: 'degree', value: '2' }], description: 'Accurate for nonlinear relationships when the true trend is polynomial (but prone to overfitting with high degrees).' },
    { name: 'linear', params: [], description: 'Best for simple, linear relationships with low multicollinearity and no overfitting.' },
    { name: 'ridge', params: [{ name: 'alpha', value: '1.0' }], description: 'Better than linear when multicollinearity exists; good for many correlated features.' },
    { name: 'lasso', params: [{ name: 'alpha', value: '1.0' }], description: 'Best when feature selection is needed (sparse data) and some coefficients should be zero.' },
    { name: 'elasticnet', params: [{ name: 'alpha', value: '1.0' }, { name: 'l1_ratio', value: '0.5' }], description: 'Combines Ridge & Lasso benefits; best when both multicollinearity and feature selection are needed.'},
    { name: 'svr', params: [{ name: 'C', value: '1.0' }, { name: 'kernel', value: 'rbf' }, { name: 'gamma', value: 'scale' }], description: 'Effective for complex, nonlinear relationships, especially with kernel tricks for high-dimensional data. Good for small-to-medium datasets.' },
    { name: 'randomforest', params: [{ name: 'n_estimators', value: '100' }, { name: 'max_depth', value: '2' }, { name: 'min_samples_split', value: '2' }], description: 'Robust general-purpose classifier, handles nonlinearity well. Good for large datasets.'},
    { name: 'gradientboosting', params: [{ name: 'n_estimators', value: '100' }, { name: 'learning_rate', value: '0.1' }, { name: 'max_depth', value: '3' }], description: 'High accuracy for tabular data, great with imbalanced datasets.'},
    { name: 'xgboost', params: [{ name: 'n_estimators', value: '100' }, { name: 'learning_rate', value: '0.1' }, { name: 'max_depth', value: '3' }, { name: 'eval_metric', value: 'logloss' }], description: 'Best for structured data, optimized performance.'},
    { name: 'knn', params: [{ name: 'n_neighbors', value: '5' }], description: 'Works well for small, low-dimensional data with local patterns. Prone to noise and high dimensions.'},
    { name: 'decisiontree', params: [{ name: 'max_depth', value: '2' }, { name: 'min_samples_split', value: '2' }], description: 'Simple and interpretable, but prone to overfitting; good for small datasets.'},
    { name: 'adaboost', params: [{name: 'n_estimators', value: '50'}, {name: 'learning_rate', value: '1.0'}], description: 'Good for boosting weak learners.'},
    { name: 'bagging', params: [{name: 'n_estimators', value: '10'}], description: 'Reduces variance, improves stability.'},
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
    // Only reset state for new uploads
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
        
        // Initialize selectedColumns with all columns by default
        statsData.dtype_info.forEach(col => {
            selectedColumns.add(col.column);
        });
        
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
    
    const hasNullValues = statsData.stats?.total_null > 0;
    const analysisSectionHTML = hasNullValues ? '' : renderAnalysisSection(statsData);
    
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
    // Get all columns for the include section
    const allColumns = statsData.dtype_info;
    
    // Filter columns based on selection for other sections
    const availableColumns = allColumns.filter(col => selectedColumns.has(col.column));
    
    // Categorize columns for the other sections
    const numericColumns = availableColumns.filter(col => 
        (col.type === 'float64' || col.type === 'int64') && 
        !(col.unique_values <= 5 && selectedOneHotColumns.has(col.column))
    );
    
    const categoricalColumns = availableColumns.filter(col => 
        col.type === 'object' || col.type === 'category' || 
        (col.unique_values <= 5 && selectedOneHotColumns.has(col.column))
    );
    
    const potentialCategoricalColumns = availableColumns.filter(col => 
        col.type === 'object' || col.type === 'category' || col.unique_values <= 5
    );
    
    const potentialNumericColumns = availableColumns.filter(col => 
        (col.type === 'float64' || col.type === 'int64') &&
        !selectedOneHotColumns.has(col.column)
    );
    
    const isCategorical = selectedPredictionColumn ? 
        categoricalColumns.some(col => col.column === selectedPredictionColumn) : false;

    const hasResults = !!document.getElementById('analysis-results-area');
    const modelSelectionHTML = selectedPredictionColumn ? buildModelSelectionSection(isCategorical, statsData) : '';

    const clearButtonHTML = hasResults ? 
        `<button id="clearResults" class="action-button clear-button">Clear Results</button>` : 
        `<button id="clearResults" class="action-button clear-button" disabled>Clear Results</button>`;

    return `
        <div class="data-section">
            <div class="analysis-section">
                <h3>Data Analysis Setup</h3>
                ${renderColumnSelectionSection(allColumns)}
                ${renderPredictionTargetSection(availableColumns, statsData)}
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

function setupIncludeColumnControls(statsData) {
    document.querySelectorAll('.include-checkbox').forEach(checkbox => {
        checkbox.addEventListener('change', function() {
            if (this.checked) {
                selectedColumns.add(this.value);
            } else {
                selectedColumns.delete(this.value);
                // Also remove from other selections if deselected
                selectedOneHotColumns.delete(this.value);
                selectedStandardScaleColumns.delete(this.value);
                if (selectedPredictionColumn === this.value) {
                    selectedPredictionColumn = null;
                }
            }
            // Re-render to update UI
            renderDataDisplay(statsData);
        });
    });
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
    const isPredictionColumnNumeric = selectedPredictionColumn &&
                                     !selectedOneHotColumns.has(selectedPredictionColumn);

    const balanceMethodOptions = isPredictionColumnNumeric ? `
        <option value="none">None</option>
    ` : `
        <option value="none">None</option>
        <option value="randomoversampler">RandomOverSampler</option>
        <option value="randomundersampler">RandomUnderSampler</option>
        <option value="smoteenn" disabled>SMOTEENN (unavailable)</option>
    `;

    const balanceMethodDisabled = isPredictionColumnNumeric;

    return `
        <div class="analysis-subsection">
            <h4>Train/Test Split Configuration</h4>
            <div class="split-config">
                <div class="split-control">
                    <label>Training Size (%):</label>
                    <input type="number" id="trainSize" min="1" max="99" value="${splitSettings.trainSize}" class="split-input">
                    <button class="split-adjust" data-direction="up" data-target="trainSize">↑</button>
                    <button class="split-adjust" data-direction="down" data-target="trainSize">↓</button>
                </div>
                <div class="split-control">
                    <label>Test Size (%):</label>
                    <input type="number" id="testSize" min="1" max="99" value="${splitSettings.testSize}" class="split-input" readonly>
                </div>
                <div class="split-control">
                    <label>Random Seed:</label>
                    <input type="number" id="randomSeed" value="${splitSettings.randomSeed}" class="split-input">
                </div>
                <div class="split-control">
                    <label>Balance Method:</label>
                    <select id="balanceMethod" class="split-input wider-select" ${balanceMethodDisabled ? 'disabled' : ''}>
                        ${balanceMethodOptions}
                    </select>
                </div>
            </div>
        </div>
    `;
}

/**
 * Render column options (generic for radio or checkbox)
 */
// In renderColumnOptions function, update the input attributes:
function renderColumnOptions(columns, type, inputType, isIncludeSection = false) {
    if (columns.length === 0) {
        return `<p>No columns available</p>`;
    }

    let options = '<div class="options-row">';
    columns.forEach((col, index) => {
        const isPredictionColumn = col.column === selectedPredictionColumn;
        const isCategoricalTreatment = selectedOneHotColumns.has(col.column);
        const isNumericTreatment = selectedStandardScaleColumns.has(col.column);
        
        // Determine actual treatment
        let actualTreatment;
        if (isCategoricalTreatment) {
            actualTreatment = 'categorical';
        } else if (isNumericTreatment) {
            actualTreatment = 'numeric';
        } else {
            // Default treatment based on column type
            actualTreatment = (col.type === 'object' || col.type === 'category') ? 'categorical' : 'numeric';
        }

        // Define checked based on the section
        let checked = false;
        if (isIncludeSection) {
            checked = selectedColumns.has(col.column);
        } else {
            if (type === 'predictionColumn') {
                checked = isPredictionColumn;
            } else if (type === 'onehot') {
                checked = isCategoricalTreatment;
            } else if (type === 'standardscale') {
                checked = isNumericTreatment;
            }
        }
        
        // Determine if the column should be disabled
        let disabled = false;
        if (!isIncludeSection) {
            if (type === 'onehot') {
                disabled = isNumericTreatment || 
                          (isPredictionColumn && actualTreatment === 'categorical');
            } else if (type === 'standardscale') {
                disabled = isCategoricalTreatment || 
                          (isPredictionColumn && actualTreatment === 'numeric');
            }
        }

        const typeIndicator = actualTreatment === 'categorical' ? 
                            (col.unique_values <= 5 ? 'categorical (low card)' : 'categorical') : 
                            'numeric';

        options += `
            <div class="column-option">
                <label>
                    <input type="${inputType}" 
                           name="${isIncludeSection ? 'includeColumn' : type === 'predictionColumn' ? 'predictionColumn' : type + 'Columns'}" 
                           value="${col.column}" 
                           ${checked ? 'checked' : ''} 
                           ${disabled ? 'disabled' : ''}
                           class="${isIncludeSection ? 'include-checkbox' : type === 'predictionColumn' ? '' : type + '-checkbox'}">
                    ${col.column}
                    <span class="type-indicator">${typeIndicator}</span>
                </label>
            </div>
        `;
        
        // Start new row every 3 items
        if ((index + 1) % 3 === 0 && index + 1 < columns.length) {
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
    // Check the actual treatment of the prediction column
    const predictionColumnInfo = selectedPredictionColumn ? 
        statsData.dtype_info.find(col => col.column === selectedPredictionColumn) : null;
    
    const isLowCardNumeric = predictionColumnInfo && 
                           (predictionColumnInfo.type === 'float64' || predictionColumnInfo.type === 'int64') &&
                           predictionColumnInfo.unique_values <= 5;

    // Determine final treatment based on user selection
    const finalIsCategorical = selectedOneHotColumns.has(selectedPredictionColumn) || 
                             (predictionColumnInfo && 
                              (predictionColumnInfo.type === 'object' || predictionColumnInfo.type === 'category'));

    const models = finalIsCategorical ? categoricalModels : numericalModels;
    const modelType = finalIsCategorical ? 'classification' : 'regression';
    
    return `
        <div class="model-selection-section">
            ${renderModelOptions(models, modelType)}
            ${renderCrossValidationOptions()}
            ${renderModelParameters(models)}
        </div>
    `;
}

/**
 * Render model options
 */
function renderModelOptions(models, modelType) {
    let options = `
        <div class="model-selection-container">
            <h4>Select ${modelType} Model</h4>
            <div class="model-options-grid">
    `;
    
    models.forEach(model => {
        const isDisabled = model.name.toLowerCase() === 'xgboost'; // Add other disabled models here
        const disabledClass = isDisabled ? 'disabled-model-option' : '';
        const disabledAttr = isDisabled ? 'disabled' : '';
        
        options += `
            <div class="model-option-box ${disabledClass}">
                <label>
                    <input type="radio" name="modelType" value="${model.name}" 
                           data-model-type="${modelType}"
                           ${selectedModel === model.name ? 'checked' : ''}
                           ${disabledAttr}>
                    <span class="model-name">${model.name}</span>
                    ${isDisabled ? '<span class="model-unavailable">(Temporarily unavailable)</span>' : ''}
                    <span class="model-description">${model.description || ''}</span>
                </label>
            </div>
        `;
    });
    
    options += `
            </div>
        </div>
    `;
    return options;
}

/**
 * Render model parameters
 */
function renderModelParameters(models) {
    let paramsHTML = '<div class="model-params-section">';
    
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
    
    paramsHTML += '</div>';
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
        // Get the current value from modelParams or fall back to default
        const currentValue = modelParams[model.name]?.[param.name] || param.value;
        
        inputs += `
            <div class="param-control">
                <label>${param.name}:</label>
                <input type="text" class="param-input" 
                       data-model="${model.name}" 
                       data-param="${param.name}" 
                       value="${currentValue}"
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
                    <div class="table-responsive-desktop">
                        ${statsData.data_head}
                    </div>
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
        // ... other cases as needed
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
    setupIncludeColumnControls(statsData);
    setupPredictionColumnControls(statsData);
    setupOneHotEncodingControls(statsData);
    setupStandardScalingControls(statsData);
    setupTrainTestSplitControls();
    setupModelSelectionControls();
    setupParameterInputControls();
    setupRunAnalysisButton(statsData);
    setupCrossValidationControls();
    
    const clearResultsBtn = document.getElementById('clearResults');
    if (clearResultsBtn && !clearResultsBtn.disabled) {
        clearResultsBtn.addEventListener('click', clearAnalysisResults);
    }
}

/**
 * Set up prediction column controls
 */
function setupPredictionColumnControls(statsData) {
    document.querySelectorAll('input[name="predictionColumn"]').forEach(radio => {
        radio.addEventListener('change', function() {
            if (this.checked) {
                selectedPredictionColumn = this.value;
                const colInfo = statsData.dtype_info.find(col => col.column === this.value);
                // Determine if this is a low-cardinality numeric column
                const isLowCardNumeric = colInfo && 
                                      (colInfo.type === 'float64' || colInfo.type === 'int64') &&
                                      colInfo.unique_values <= 5;

                // Set default treatment based on column type and cardinality
                if (isLowCardNumeric) {
                    // For low-cardinality numeric, default to numeric treatment (regression)
                    // But allow user to override via the radio buttons
                    selectedOneHotColumns.delete(this.value);
                    selectedStandardScaleColumns.add(this.value);
                } else if (colInfo.type === 'object' || colInfo.type === 'category') {
                    // For true categorical columns, default to categorical treatment
                    selectedOneHotColumns.add(this.value);
                    selectedStandardScaleColumns.delete(this.value);
                } else {
                    // For regular numeric columns
                    selectedStandardScaleColumns.add(this.value);
                    selectedOneHotColumns.delete(this.value);
                }

                // Reset model and parameters
                selectedModel = '';
                modelParams = {};
                
                // Initialize default parameters for the first appropriate model
                const finalIsCategorical = selectedOneHotColumns.has(this.value);
                const models = finalIsCategorical ? categoricalModels : numericalModels;
                if (models.length > 0) {
                    selectedModel = models[0].name;
                    modelParams[selectedModel] = { ...modelDefaultParams[selectedModel] };
                }
                
                // Re-render to update UI
                renderDataDisplay(statsData);
            }
        });
    });

    // Handle treatment type change for low-cardinality numeric columns
    document.addEventListener('change', function(e) {
        if (e.target && e.target.name === 'numericAsType' && selectedPredictionColumn) {
            const colInfo = statsData.dtype_info.find(col => col.column === selectedPredictionColumn);
            const isLowCardNumeric = colInfo && 
                                   (colInfo.type === 'float64' || colInfo.type === 'int64') &&
                                   colInfo.unique_values <= 5;

            if (isLowCardNumeric) {
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
                
                // Initialize default parameters for the new model type
                const finalIsCategorical = e.target.value === 'categorical';
                const models = finalIsCategorical ? categoricalModels : numericalModels;
                if (models.length > 0) {
                    selectedModel = models[0].name;
                    modelParams[selectedModel] = { ...modelDefaultParams[selectedModel] };
                }
                
                // Re-render to update UI
                renderDataDisplay(statsData);
            }
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
    const randomSeedInput = document.getElementById('randomSeed');
    const balanceMethodSelect = document.getElementById('balanceMethod'); // Get the balance method select element

    // Initialize from state
    if (trainInput) trainInput.value = splitSettings.trainSize;
    if (testInput) testInput.value = splitSettings.testSize;
    if (randomSeedInput) randomSeedInput.value = splitSettings.randomSeed;
    if (balanceMethodSelect) balanceMethodSelect.value = splitSettings.balanceMethod || 'none'; // Initialize balance method

    trainInput?.addEventListener('input', function() {
        let trainValue = parseInt(this.value);
        if (isNaN(trainValue)) {
            trainValue = 80;
        }
        trainValue = Math.max(1, Math.min(99, trainValue));
        this.value = trainValue;
        splitSettings.trainSize = trainValue;
        splitSettings.testSize = 100 - trainValue;
        testInput.value = splitSettings.testSize;
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
                splitSettings.trainSize = value;
                splitSettings.testSize = 100 - value;
                testInput.value = splitSettings.testSize;
            } else if (this.dataset.target === 'randomSeed') {
                splitSettings.randomSeed = value;
            }
            target.dispatchEvent(new Event('input'));
        });
    });

    randomSeedInput?.addEventListener('input', function() {
        splitSettings.randomSeed = parseInt(this.value) || 42;
    });

    // Add event listener for balance method selection
    balanceMethodSelect?.addEventListener('change', function() {
        splitSettings.balanceMethod = this.value;
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
                    modelParams[selectedModel] = {};
                    const models = [...categoricalModels, ...numericalModels];
                    const modelConfig = models.find(m => m.name === selectedModel);
                    if (modelConfig) {
                        modelConfig.params.forEach(param => {
                            modelParams[selectedModel][param.name] = param.value;
                        });
                    }
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
    // Use event delegation for dynamically created inputs
    document.addEventListener('input', function(e) {  // Changed from 'change' to 'input'
        if (e.target && e.target.classList.contains('param-input')) {
            const model = e.target.dataset.model;
            const param = e.target.dataset.param;
            const value = e.target.value;
            
            if (!modelParams[model]) {
                modelParams[model] = {};
            }
            
            // Store the parameter value
            modelParams[model][param] = value;
        }
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

/**
 * Render column selection section
 */
function renderColumnSelectionSection(columns) {
    return `
        <div class="analysis-subsection">
            <h4>Select Columns to Include in Analysis</h4>
            <div class="column-selection">
                ${renderColumnOptions(columns, 'includeColumn', 'checkbox', true)}
            </div>
        </div>
    `;
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
 * coment
 */

function prepareAnalysisConfig(statsData) {
    // Determine if this is a classification or regression problem
    const isCategorical = selectedOneHotColumns.has(selectedPredictionColumn) || 
                         statsData.dtype_info.some(col => 
                             col.column === selectedPredictionColumn && 
                             (col.type === 'object' || col.type === 'category')
                         );

    const models = isCategorical ? categoricalModels : numericalModels;
    const selectedModelConfig = models.find(m => m.name === selectedModel);
    
    // Get parameters - use user-modified values if available, otherwise use defaults
    let params = {};
    if (selectedModelConfig) {
        selectedModelConfig.params.forEach(param => {
            // Check if we have a user-modified value in modelParams
            if (modelParams[selectedModel] && modelParams[selectedModel][param.name] !== undefined) {
                params[param.name] = modelParams[selectedModel][param.name];
            } else {
                // Fall back to default value
                params[param.name] = param.value;
            }
        });
    }
    
    // Get CV parameters
    const cvMethod = document.querySelector('input[name="cvMethod"]:checked').value;
    let cvParams = {};
    
    if (cvState.method === 'kfold') {
        cvParams = {
            method: 'kfold',
            n_splits: cvState.kfoldSplits
        };
    } else if (cvState.method === 'shufflesplit') {
        cvParams = {
            method: 'shufflesplit',
            n_splits: cvState.shuffleSplits,
            test_size: cvState.shuffleTestSize / 100
        };
    } else if (cvMethod === 'loo') {
        cvParams = {
            method: 'loo'
        };
    }

    return {
        df: currentDF,
        columns_to_use: Array.from(selectedColumns),
        prediction_column: selectedPredictionColumn,
        onehot_columns: Array.from(selectedOneHotColumns),
        standardscale_columns: Array.from(selectedStandardScaleColumns),
        train_size: splitSettings.trainSize / 100,
        test_size: splitSettings.testSize / 100,
        random_seed: splitSettings.randomSeed,
        model: selectedModel,
        model_params: params,
        cross_validation: cvParams,
        balance_method: splitSettings.balanceMethod || 'none',
        // Add explicit type information
        column_types: statsData.dtype_info.map(col => ({
            name: col.column,
            dtype: col.type,
            unique_values: col.unique_values,
            // Add how the column should be treated based on user selections
            treatment: selectedOneHotColumns.has(col.column) ? 'categorical' : 
                     selectedStandardScaleColumns.has(col.column) ? 'numeric' :
                     col.type === 'object' || col.type === 'category' ? 'categorical' : 'numeric'
        }))
    };
}

/**
 * Run analysis by sending request to server
 */
async function runAnalysis(config) {
    const runAnalysisBtn = document.getElementById('runAnalysis');
    try {
        // Store current selections before running analysis
        const currentState = {
            selectedColumns: new Set(selectedColumns),
            selectedPredictionColumn,
            selectedOneHotColumns: new Set(selectedOneHotColumns),
            selectedStandardScaleColumns: new Set(selectedStandardScaleColumns),
            selectedModel,
            modelParams: JSON.parse(JSON.stringify(modelParams)),
            cvState: {...cvState}
        };

        runAnalysisBtn.disabled = true;
        runAnalysisBtn.textContent = 'Processing...';
        
        const response = await fetch('/api/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });
        const data = await response.json();
        if (data.error) {
            throw new Error(data.error);
        }

        // Restore selections after analysis completes
        selectedColumns = new Set(currentState.selectedColumns);
        selectedPredictionColumn = currentState.selectedPredictionColumn;
        selectedOneHotColumns = new Set(currentState.selectedOneHotColumns);
        selectedStandardScaleColumns = new Set(currentState.selectedStandardScaleColumns);
        selectedModel = currentState.selectedModel;
        modelParams = JSON.parse(JSON.stringify(currentState.modelParams));
        cvState = {...currentState.cvState};

        return data;
    } catch (error) {
        throw error;
    } finally {
        if (runAnalysisBtn) {
            runAnalysisBtn.disabled = false;
            runAnalysisBtn.textContent = 'Run Analysis';
        }
    }
}

// ==============================================
// RESULTS DISPLAY
// ==============================================

/**
 * Display analysis results
 */
async function displayAnalysisResults(data) {
    try {
        // Store current statsData reference
        const currentStatsData = await getDataStatistics({ df: currentDF });
        
        // Get or create display area
        let displayArea = document.getElementById('analysis-results-area');
        if (!displayArea) {
            displayArea = document.createElement('div');
            displayArea.id = 'analysis-results-area';
            document.querySelector('.analysis-section').appendChild(displayArea);
        }
        
        // Create results container
        const resultsContainer = document.createElement('div');
        resultsContainer.className = 'analysis-results';
        
        // Add visualization if available
        if (data.image_data) {
            resultsContainer.appendChild(createVisualizationCard(data.image_data));
        }
        
        // Add metrics if available
        if (data.metrics) {
            resultsContainer.appendChild(createMetricsDisplay(data.metrics));
        }
        
        // Add results to display area
        if (displayArea.firstChild) {
            displayArea.insertBefore(resultsContainer, displayArea.firstChild);
        } else {
            displayArea.appendChild(resultsContainer);
        }
        
        // Re-render while preserving state
        renderDataDisplay(currentStatsData);
        
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
        <div class="thumbnail-container">
            <img src="data:image/png;base64,${imageData}" alt="Analysis Result" class="thumbnail-image">
        </div>
    `;
    
    // Make image clickable for fullscreen
    const thumbnail = imgCard.querySelector('img');
    thumbnail.addEventListener('click', function() {
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

function renderCrossValidationOptions() {
    return `
        <div class="cv-options">
            <h4>Cross Validation</h4>
            <div class="cv-option">
                <label>
                    <input type="radio" name="cvMethod" value="none" 
                           ${cvState.method === 'none' ? 'checked' : ''}>
                    None (simple train/test split)
                </label>
            </div>
            <div class="cv-option">
                <label>
                    <input type="radio" name="cvMethod" value="kfold"
                           ${cvState.method === 'kfold' ? 'checked' : ''}>
                    K-Fold
                    <input type="number" id="kfoldSplits" class="cv-param" 
                           min="2" max="20" value="${cvState.kfoldSplits}"
                           ${cvState.method !== 'kfold' ? 'disabled' : ''}>
                    splits
                </label>
            </div>
            <div class="cv-option">
                <label>
                    <input type="radio" name="cvMethod" value="shufflesplit"
                           ${cvState.method === 'shufflesplit' ? 'checked' : ''}>
                    Shuffle Split
                    <input type="number" id="shuffleSplits" class="cv-param" 
                           min="1" max="20" value="${cvState.shuffleSplits}"
                           ${cvState.method !== 'shufflesplit' ? 'disabled' : ''}>
                    splits,
                    <input type="number" id="shuffleTestSize" class="cv-param" 
                           min="1" max="99" value="${cvState.shuffleTestSize}"
                           ${cvState.method !== 'shufflesplit' ? 'disabled' : ''}>
                    % test size
                </label>
            </div>
            <div class="cv-option">
                <label>
                    <input type="radio" name="cvMethod" value="loo" disabled>
                    Leave-One-Out (LOO) (CPU expensive, disabled)
                </label>
            </div>
        </div>
    `;
}

function setupCrossValidationControls() {
    // Enable/disable parameter inputs based on CV selection
    document.querySelectorAll('input[name="cvMethod"]').forEach(radio => {
        radio.addEventListener('change', function() {
            // Update state
            cvState.method = this.value;
            
            // Enable/disable parameter inputs
            document.getElementById('kfoldSplits').disabled = this.value !== 'kfold';
            document.getElementById('shuffleSplits').disabled = this.value !== 'shufflesplit';
            document.getElementById('shuffleTestSize').disabled = this.value !== 'shufflesplit';
        });
    });

    // Update state when parameter values change
    document.getElementById('kfoldSplits')?.addEventListener('change', function() {
        cvState.kfoldSplits = parseInt(this.value);
    });
    
    document.getElementById('shuffleSplits')?.addEventListener('change', function() {
        cvState.shuffleSplits = parseInt(this.value);
    });
    
    document.getElementById('shuffleTestSize')?.addEventListener('change', function() {
        cvState.shuffleTestSize = parseInt(this.value);
    });
}

/**
 * Reset all internal state variables when a new CSV file is uploaded
 */
function resetState() {
    // Only reset if we don't have any existing selections
    // if (selectedColumns.size === 0) {
    //     // Initialize with all columns if we have statsData
    //     if (currentDF && currentDF.columns) {
    //         currentDF.columns.forEach(col => {
    //             selectedColumns.add(col);
    //         });
    //     }
    // }
    
    // Don't reset these unless we're doing a fresh upload

    selectedPredictionColumn = null;
    selectedOneHotColumns = new Set();
    selectedStandardScaleColumns = new Set();
    selectedModel = '';
    modelParams = {};
    columnsToInclude = new Set();
    selectedColumns = new Set();
    splitSettings = {
        trainSize: 80,
        testSize: 20,
        randomSeed: 42
    };
}

// ==============================================
// IMAGE OVERLAY
// ==============================================

/**
 * Create fullscreen image overlay
 */
function createImageOverlay(src, alt) {
    // Create image first and wait for it to load
    const img = new Image();
    img.src = src;
    img.alt = alt;
    
    // Create overlay structure but keep it hidden
    const overlay = document.createElement('div');
    overlay.className = 'plot-overlay';
    overlay.style.opacity = '0'; // Start transparent
    overlay.style.transition = 'opacity 0.3s ease'; // Smooth fade-in

    const content = document.createElement('div');
    content.className = 'plot-overlay-content';
    content.style.visibility = 'hidden'; // Hide content initially

    const closeBtn = document.createElement('span');
    closeBtn.className = 'close-overlay';
    closeBtn.innerHTML = '×';

    // Apply image styles
    img.style.maxWidth = '100%';
    img.style.maxHeight = '100%';
    img.style.objectFit = 'contain';
    img.style.display = 'block';

    content.appendChild(closeBtn);
    content.appendChild(img);
    overlay.appendChild(content);

    img.onload = function() {
        // Adjust for portrait images
        if (img.naturalHeight > img.naturalWidth) {
            content.style.maxWidth = `${(img.naturalWidth / img.naturalHeight) * 90}vh`;
        }
        
        // Add to DOM and make visible
        document.body.classList.add('overlay-active');
        document.body.appendChild(overlay);
        
        // Trigger fade-in after a brief timeout to ensure layout is ready
        setTimeout(() => {
            overlay.style.opacity = '1';
            content.style.visibility = 'visible';
        }, 10);
    };

    // Error handling in case image fails to load
    img.onerror = function() {
        console.error('Failed to load overlay image');
        content.innerHTML = '<div class="image-error">Failed to load image</div>';
        document.body.appendChild(overlay);
        overlay.style.opacity = '1';
    };

    function closeOverlay() {
        overlay.style.opacity = '0';
        setTimeout(() => {
            if (overlay.parentNode) {
                document.body.removeChild(overlay);
            }
            document.body.classList.remove('overlay-active');
            document.removeEventListener('keydown', escClose);
        }, 300); // Match transition duration
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