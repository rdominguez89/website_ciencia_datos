// Global variables to maintain state
let currentDF = null;
let selectedPredictionColumn = null;
let selectedOneHotColumns = new Set();
let selectedStandardScaleColumns = new Set();
let selectedModel = '';
let modelParams = {};
let modelDefaultParams = {};  // Add this line

// Main upload button handler
document.getElementById('uploadBtn').addEventListener('click', async function() {
    const fileInput = document.getElementById('csvFile');
    const file = fileInput.files[0];
    
    if (!file) {
        alert('Please select a CSV file first');
        return;
    }
    
    if (file.size > 5 * 1024 * 1024) {
        alert('File size exceeds 5MB limit');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        const uploadData = await response.json();
        
        if (uploadData.error) {
            alert(uploadData.error);
        } else {
            await processData(uploadData);
        }
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred while uploading the file');
    }
});

// Central data processing function
async function processData(data) {
    currentDF = data.df;
    
    try {
        const statsResponse = await fetch('/api/statsummary', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                df: currentDF
            })
        });
        const statsData = await statsResponse.json();
        
        if (statsData.error) {
            alert(statsData.error);
            return;
        }

        // Render the data visualization
        renderDataDisplay(statsData);
        
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred while processing the data');
    }
}

// Main rendering function
function renderDataDisplay(statsData) {           
    // Generate column checkboxes for prediction
    const numericColumns = statsData.dtype_info.filter(col => col.type === 'float64' || col.type === 'int64');
    const categoricalColumns = statsData.dtype_info.filter(col => col.type === 'object' || col.type === 'category');
    const encoderColumns = statsData.dtype_info.filter(col => col.type === 'object' || col.type === 'category' || col.unique_values <= 5);
    const allcolumns = [...numericColumns, ...categoricalColumns];

    let predictionOptions = '<div class="options-row">';
    allcolumns.forEach((col, index) => {
        predictionOptions += `
            <div class="column-option">
                <label>
                    <input type="radio" name="predictionColumn" value="${col.column}" 
                           ${selectedPredictionColumn === col.column ? 'checked' : ''}>
                    ${col.column}
                </label>
            </div>
        `;
        // Add closing and opening div for new row after every 4 items (adjustable)
        if ((index + 1) % 4 === 0 && index + 1 < allcolumns.length) {
            predictionOptions += '</div><div class="options-row">';
        }
    });
    predictionOptions += '</div>';
    
    // Generate checkboxes for OneHotEncoder
    let oneHotOptions = '<div class="options-row">';
    encoderColumns.forEach((col, index) => {
        oneHotOptions += `
            <div class="column-option">
                <label>
                    <input type="checkbox" class="onehot-checkbox" value="${col.column}" 
                           ${selectedOneHotColumns.has(col.column) ? 'checked' : ''}
                           ${selectedStandardScaleColumns.has(col.column) ? 'disabled' : ''}>
                    ${col.column}
                </label>
            </div>
        `;
        if ((index + 1) % 5 === 0 && index + 1 < encoderColumns.length) {
            oneHotOptions += '</div><div class="options-row">';
        }
    });
    oneHotOptions += '</div>';
    
    // Generate checkboxes for StandardScaling - only show numeric columns (excluding prediction column)
    let standardScaleOptions = '<div class="options-row">';
    numericColumns.forEach((col, index) => {
        // Only show if it's not the prediction column and it's a numeric type
        if (col.column !== selectedPredictionColumn) {
            standardScaleOptions += `
                <div class="column-option">
                    <label>
                        <input type="checkbox" class="standardscale-checkbox" value="${col.column}" 
                               ${selectedStandardScaleColumns.has(col.column) ? 'checked' : ''}
                               ${selectedOneHotColumns.has(col.column) ? 'disabled' : ''}>
                        ${col.column}
                    </label>
                </div>
            `;
            if ((index + 1) % 4 === 0 && index + 1 < numericColumns.length) {
                standardScaleOptions += '</div><div class="options-row">';
            }
        }
    });
    standardScaleOptions += '</div>';
    
    // Determine if prediction column is categorical or numerical
    const isCategorical = selectedPredictionColumn ? 
    categoricalColumns.some(col => col.column === selectedPredictionColumn) : false;
    
    // Model selection and parameters
    const modelSelectionSection = buildModelSelectionSection(isCategorical);
    
    const analysisSection = `
        <div class="analysis-section">
            <h3>Data Analysis Setup</h3>
            
            <div class="analysis-subsection">
                <h4>Select Prediction Target</h4>
                <div class="column-selection">
                    ${predictionOptions || '<p>No columns available for prediction</p>'}
                </div>
            </div>
            
            <div class="analysis-subsection">
                <h4>Select Columns for OneHot Encoding</h4>
                <div class="column-selection onehot-selection">
                    ${oneHotOptions || '<p>No categorical columns available for one-hot encoding</p>'}
                </div>
            </div>
            
            <div class="analysis-subsection">
                <h4>Select Columns for Standard Scaling</h4>
                <div class="column-selection standardscale-selection">
                    ${standardScaleOptions || '<p>No numeric columns available for standard scaling</p>'}
                </div>
            </div>
            
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
            
            ${modelSelectionSection}
            
            <button id="runAnalysis" class="action-button">Run Analysis</button>
        </div>
    `;
    
    const html = `
        <div class="data-tabs">
            <button class="tab-btn active" data-tab="preview">Preview</button>
            <button class="tab-btn" data-tab="stats">Statistics</button>
            <button class="tab-btn" data-tab="structure">Structure</button>
        </div>
        
        <div id="tab-content">
            <!-- Initial content will be loaded here -->
        </div>
        
        <div class="data-section">
            <h3>Dataset Overview</h3>
            <div class="stats-grid">
                <div class="stat-card">
                    <span class="stat-value">${statsData.stats.rows}</span>
                    <span class="stat-label">Rows</span>
                </div>
                <div class="stat-card">
                    <span class="stat-value">${statsData.stats.columns}</span>
                    <span class="stat-label">Columns</span>
                </div>
                <div class="stat-card">
                    <span class="stat-value">${statsData.stats.duplicates}</span>
                    <span class="stat-label">Duplicates</span>
                </div>
                <div class="stat-card">
                    <span class="stat-value">${statsData.stats.total_null}</span>
                    <span class="stat-label">Null Values</span>
                </div>
            </div>
        </div>
        <div class="data-section">
            ${analysisSection}
        </div>
    `;
        
    document.getElementById('dataPreview').innerHTML = html;
    
    // Load initial tab content
    loadInitialContent(statsData);
    
    // Add tab click handlers
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            loadTabContent(btn.dataset.tab, statsData);
        });
    });
    
    // Add event listeners for the new analysis controls
    setupAnalysisControls(statsData);
}

// Global model definitions
const categoricalModels = [
    { name: 'logistic', params: [{ name: 'C', value: '1.0' },
        { name: 'penalty', value: 'l2' },
        { name: 'solver', value: 'lbfgs' },
    ] },
    { name: 'svc', params: [
        { name: 'C', value: '1.0' },
        { name: 'kernel', value: 'rbf' },
        { name: 'gamma', value: 'scale' }
    ]},
    { name: 'randomforest', params: [
        { name: 'n_estimators', value: '100' },
        { name: 'max_depth', value: '2' },
        { name: 'min_samples_split', value: '2' }
    ]},
    { name: 'gradientboosting', params: [
        { name: 'n_estimators', value: '100' },
        { name: 'learning_rate', value: '0.1' },
        { name: 'max_depth', value: '3' }
    ]},
    { name: 'xgboost', params: [
        { name: 'n_estimators', value: '100' },
        { name: 'learning_rate', value: '0.1' },
        { name: 'max_depth', value: '3' },
        { name: 'eval_metric', value: 'logloss' }
    ]},
    { name: 'knn', params: [
        { name: 'n_neighbors', value: '5' }
    ]},
    { name: 'decisiontree', params: [
        { name: 'max_depth', value: '2' },
        { name: 'min_samples_split', value: '2' }
    ]}
];

const numericalModels = [
    { name: 'poly', params: [{ name: 'degree', value: '2' }] },
    { name: 'linear', params: [] },
    { name: 'ridge', params: [{ name: 'alpha', value: '1.0' }] },
    { name: 'lasso', params: [{ name: 'alpha', value: '1.0' }] },
    { name: 'elasticnet', params: [
        { name: 'alpha', value: '1.0' },
        { name: 'l1_ratio', value: '0.5' }
    ]},
    { name: 'svr', params: [
        { name: 'C', value: '1.0' },
        { name: 'kernel', value: 'rbf' },
        { name: 'gamma', value: 'scale' }
    ]}
];

// Global variables
let params = {};


function buildModelSelectionSection(isCategorical) {
    const models = isCategorical ? categoricalModels : numericalModels;
    const modelType = isCategorical ? 'categorical' : 'numerical';
    
    // Initialize modelDefaultParams if it doesn't exist
    if (Object.keys(modelDefaultParams).length === 0) {
        models.forEach(model => {
            modelDefaultParams[model.name] = {};
            model.params.forEach(param => {
                modelDefaultParams[model.name][param.name] = param.value;
            });
        });
    }

    let modelOptions = `<div class="model-selection-section">
        <h4>Select ${isCategorical ? 'Classification' : 'Regression'} Model</h4>
        <div class="model-options-row">`;
    
    // Model radio buttons
    models.forEach((model, index) => {
        modelOptions += `
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
            modelOptions += '</div><div class="model-options-row">';
        }
    });
    
    modelOptions += `</div>`;
    
    // Model parameters
    models.forEach(model => {
        if (model.params.length > 0) {
            modelOptions += `
                <div class="model-params" id="params-${model.name}" 
                     style="display: ${selectedModel === model.name ? 'block' : 'none'}">
                    <h5>${model.name} Parameters</h5>
                    <div class="param-grid">
            `;
            
            model.params.forEach(param => {
                const paramValue = modelParams[model.name]?.[param.name] !== undefined 
                    ? modelParams[model.name][param.name] 
                    : param.value;
                modelOptions += `
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
            
            modelOptions += `</div></div>`;
        }
    });

    return modelOptions;
}

function setupAnalysisControls(statsData) {
    // Store the current dataframe
    currentDF = statsData.df;
    
    // Prediction column selection
    document.querySelectorAll('input[name="predictionColumn"]').forEach(radio => {
        radio.addEventListener('change', function() {
            if (this.checked) {
                // Store current checkbox states before re-rendering
                const currentOneHot = new Set(selectedOneHotColumns);
                const currentStandardScale = new Set(selectedStandardScaleColumns);
                
                selectedPredictionColumn = this.value;
                // Reset model selection when target changes
                selectedModel = '';
                modelParams = {};
                
                // Restore checkbox states after re-render
                setTimeout(() => {
                    selectedOneHotColumns = currentOneHot;
                    selectedStandardScaleColumns = currentStandardScale;
                    renderDataDisplay(statsData);
                }, 0);
            }
        });
    });
    
    // OneHotEncoder checkbox handling
    document.querySelectorAll('.onehot-checkbox').forEach(checkbox => {
        checkbox.addEventListener('change', function() {
            if (this.checked) {
                selectedOneHotColumns.add(this.value);
                selectedStandardScaleColumns.delete(this.value);
            } else {
                selectedOneHotColumns.delete(this.value);
            }
            renderDataDisplay(statsData);
        });
    });
    
    // StandardScaling checkbox handling
    document.querySelectorAll('.standardscale-checkbox').forEach(checkbox => {
        checkbox.addEventListener('change', function() {
            if (this.checked) {
                selectedStandardScaleColumns.add(this.value);
                selectedOneHotColumns.delete(this.value);
            } else {
                selectedStandardScaleColumns.delete(this.value);
            }
            renderDataDisplay(statsData);
        });
    });
    
    // Train/test split controls
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
    
    // Model selection handling
    document.querySelectorAll('input[name="modelType"]').forEach(radio => {
        radio.addEventListener('change', function() {
            if (this.checked) {
                selectedModel = this.value;
                // Hide all parameter sections
                document.querySelectorAll('.model-params').forEach(el => {
                    el.style.display = 'none';
                });
                // Show selected model's parameters
                const paramsSection = document.getElementById(`params-${this.value}`);
                if (paramsSection) paramsSection.style.display = 'block';
            }
        });
    });
    
    // Model parameter input handling
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
    
    // Run analysis button
    document.getElementById('runAnalysis')?.addEventListener('click', async function() {
        if (!selectedPredictionColumn) {
            alert('Please select a prediction target column');
            return;
        }
        
        if (!selectedModel) {
            alert('Please select a model');
            return;
        }

        const trainSize = parseInt(document.getElementById('trainSize').value) / 100;
        const testSize = parseInt(document.getElementById('testSize').value) / 100;
        const randomSeed = parseInt(document.getElementById('randomSeed').value);
        
        if (isNaN(trainSize)) {
            alert('Please enter valid numbers for train/test split');
            return;
        }

        // Determine if this is a categorical problem
        const encoderColumns = statsData.dtype_info.filter(col => col.type === 'object' || col.type === 'category' || col.unique_values <= 5);
        const isCategorical = encoderColumns.some(col => col.column === selectedPredictionColumn);
        // Get the appropriate model list
        const models = isCategorical ? categoricalModels : numericalModels;
        const selectedModelConfig = models.find(m => m.name === selectedModel);

        // Initialize params with default values
        params = {};
        if (selectedModelConfig) {
            selectedModelConfig.params.forEach(param => {
                params[param.name] = modelParams[selectedModel]?.[param.name] ?? param.value;
            });
        }

        try {
            const response = await fetch('/api/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    df: currentDF,
                    prediction_column: selectedPredictionColumn,
                    onehot_columns: Array.from(selectedOneHotColumns),
                    standardscale_columns: Array.from(selectedStandardScaleColumns),
                    train_size: trainSize,
                    test_size: testSize,
                    random_seed: randomSeed,
                    model: selectedModel,
                    model_params: params
                })
            });
            
        const data = await response.json();

        if (data.error) {
            alert(data.error);
        } else {            
            // Get or create display area (DON'T clear previous results)
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
            
            // Create visualization container
            const visualizationContainer = document.createElement('div');
            visualizationContainer.className = 'visualization-container';
            
            // Display the image if it exists
            if (data.image_data) {
                const imgCard = document.createElement('div');
                imgCard.className = 'visualization-card';
                imgCard.innerHTML = `
                    <h4>Analysis Visualization</h4>
                    <img src="data:image/png;base64,${data.image_data}" 
                        alt="Analysis Result">
                `;
                visualizationContainer.appendChild(imgCard);
                
                // Make image clickable for fullscreen
                imgCard.querySelector('img').addEventListener('click', function() {
                    // Create overlay div
                    const overlay = document.createElement('div');
                    overlay.className = 'image-overlay';
                    overlay.style.position = 'fixed';
                    overlay.style.top = '0';
                    overlay.style.left = '0';
                    overlay.style.width = '100%';
                    overlay.style.height = '100%';
                    overlay.style.backgroundColor = 'rgba(0,0,0,0.9)';
                    overlay.style.zIndex = '1000';
                    overlay.style.display = 'flex';
                    overlay.style.justifyContent = 'center';
                    overlay.style.alignItems = 'center';
                    
                    // Create close button (X)
                    const closeBtn = document.createElement('span');
                    closeBtn.innerHTML = '&times;';
                    closeBtn.className = 'close-overlay';
                    closeBtn.style.position = 'absolute';
                    closeBtn.style.top = '20px';
                    closeBtn.style.right = '40px';
                    closeBtn.style.color = 'white';
                    closeBtn.style.fontSize = '50px';
                    closeBtn.style.fontWeight = 'bold';
                    closeBtn.style.cursor = 'pointer';
                    closeBtn.addEventListener('click', function() {
                        document.body.removeChild(overlay);
                    });
                    
                    // Create fullscreen image
                    const fullscreenImg = document.createElement('img');
                    fullscreenImg.src = this.src;
                    fullscreenImg.alt = this.alt;
                    fullscreenImg.style.maxHeight = '90vh';
                    fullscreenImg.style.maxWidth = '90vw';
                    
                    // Add click to close functionality (click outside image)
                    overlay.addEventListener('click', function(e) {
                        if (e.target === overlay) {
                            document.body.removeChild(overlay);
                        }
                    });
                    
                    overlay.appendChild(closeBtn);
                    overlay.appendChild(fullscreenImg);
                    document.body.appendChild(overlay);
                });
            }
            
            resultsContainer.appendChild(visualizationContainer);
            
            // Display metrics if they exist (without <pre> tag)
            if (data.metrics) {
                const metricsDiv = document.createElement('div');
                metricsDiv.className = 'analysis-metrics';
                
                let metricsContent = '<h4>Analysis Metrics</h4><div class="metrics-grid">';
                for (const [key, value] of Object.entries(data.metrics)) {
                    metricsContent += `
                        <div class="metric-item">
                            <span class="metric-key">${key}:</span>
                            <span class="metric-value">${value}</span>
                        </div>
                    `;
                }
                metricsContent += '</div>';
                
                metricsDiv.innerHTML = metricsContent;
                resultsContainer.appendChild(metricsDiv);
            }
            
            // // Add download button for the image
            // if (data.image_data) {
            //     const downloadBtn = document.createElement('button');
            //     downloadBtn.className = 'action-button';
            //     downloadBtn.textContent = 'Download Image';
            //     downloadBtn.style.margin = '10px 0';
            //     downloadBtn.addEventListener('click', function() {
            //         const link = document.createElement('a');
            //         link.href = `data:image/png;base64,${data.image_data}`;
            //         link.download = 'analysis_result.png';
            //         link.click();
            //     });
            //     resultsContainer.appendChild(downloadBtn);
            // }
            
            // Add this result to the display area (appends to existing content)
            // Insert new result at the top of the display area
            if (displayArea.firstChild) {
                displayArea.insertBefore(resultsContainer, displayArea.firstChild);
            } else {
                displayArea.appendChild(resultsContainer);
            }
        }
  
        } catch (error) {
            console.error('Error during analysis:', error);
            alert('An error occurred during analysis');
        }
    });
}

// Helper functions
function loadInitialContent(statsData) {
    const tabContent = document.getElementById('tab-content');
    tabContent.innerHTML = `
        <div id="preview-content">
            <h3>First 5 Rows</h3>
            ${statsData.data_head}
        </div>
    `;
}

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

function generateStatsHTML(statsData) {
    let statsHTML = '<h3>Dataframe Statistics</h3>';
    
    if (Object.keys(statsData.num_stats).length > 0) {
        const originalColumns = statsData.column_order;
        const allMetrics = new Set();
        Object.values(statsData.num_stats).forEach(colStats => {
            Object.keys(colStats).forEach(metric => allMetrics.add(metric));
        });
        
        const metricsToShow = Array.from(allMetrics)
            .filter(metric => !['unique', 'top', 'freq'].includes(metric))
            .sort((a, b) => {
                const order = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'];
                return order.indexOf(a) - order.indexOf(b);
            });
        
        statsHTML += '<table class="stats-table"><tr><th>Statistic</th>';
        
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
    } else {
        statsHTML += '<p>No statistics available</p>';
    }
    
    return statsHTML;
}

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

// Function to create the overlay
function createImageOverlay(src, alt) {
    // Create overlay elements
    const overlay = document.createElement('div');
    overlay.className = 'plot-overlay';
    
    const content = document.createElement('div');
    content.className = 'plot-overlay-content';
    
    const closeBtn = document.createElement('span');
    closeBtn.className = 'close-overlay';
    closeBtn.innerHTML = '&times;';
    
    const img = document.createElement('img');
    img.src = src;
    img.alt = alt;
    
    // Build the overlay
    content.appendChild(closeBtn);
    content.appendChild(img);
    overlay.appendChild(content);
    document.body.appendChild(overlay);
    
    // Close functionality
    function closeOverlay() {
        document.body.removeChild(overlay);
    }
    
    closeBtn.addEventListener('click', closeOverlay);
    overlay.addEventListener('click', function(e) {
        if (e.target === overlay) {
            closeOverlay();
        }
    });
    
    // Close with ESC key
    document.addEventListener('keydown', function escClose(e) {
        if (e.key === 'Escape') {
            closeOverlay();
            document.removeEventListener('keydown', escClose);
        }
    });
}