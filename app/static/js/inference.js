// filepath: /media/raul3/M2SDD/website_ciencia_datos_dev/app/static/js/inference.js

// Main Initialization
document.addEventListener('DOMContentLoaded', initializeInference);

function initializeInference() {
    const dataPreview = document.getElementById('dataPreview');
    renderHypothesisOptions(dataPreview);
}

// Hypothesis Section
function renderHypothesisOptions(dataPreview) {
    const hypothesisOptionsHTML = createHypothesisOptionsHTML();
    dataPreview.innerHTML = hypothesisOptionsHTML;
    setupHypothesisSelectListener(dataPreview);
}

function createHypothesisOptionsHTML() {
    return `
        <div id="hypothesisSection">
            <h3>Select Hypothesis Test:</h3>
            <select id="hypothesisSelect">
                <option value="">Select from list</option>
                <option value="h0" disabled>One-Sample t-test (H0) (developing)</option>
                <option value="correlation">Correlations</option>
                <option value="probability">Probability Space</option>
            </select>
        </div>
    `;
}

function setupHypothesisSelectListener(dataPreview) {
    const hypothesisSelect = document.getElementById('hypothesisSelect');
    hypothesisSelect.addEventListener('change', () => handleHypothesisChange(dataPreview));
}

function handleHypothesisChange(dataPreview) {
    const hypothesis = document.getElementById('hypothesisSelect').value;
    clearBelowHypothesis(dataPreview);

    if (hypothesis === 'h0' || hypothesis === 'correlation') {
        renderDataInputOptions(dataPreview, hypothesis);
    } else if (hypothesis === 'probability') {
        renderDistributionOptions(dataPreview);
    }
}

function clearBelowHypothesis(dataPreview) {
    const elementsToClear = Array.from(dataPreview.children).slice(1);
    elementsToClear.forEach(element => element.remove());
}

// Data Input Section
function renderDataInputOptions(dataPreview, hypothesis) {
    renderObjectInput(dataPreview, hypothesis);
}

function renderObjectInput(dataPreview, hypothesis) {
    const objectInputHTML = createObjectInputHTML();
    dataPreview.insertAdjacentHTML('beforeend', objectInputHTML);
    setupCSVUploadListeners(dataPreview, hypothesis);
    setupObjectAnalysisListener(dataPreview, hypothesis);
}

function createObjectInputHTML() {
    return `
        <h3>Enter Data as a dictionary (Option 1) or upload a CSV (Option 2):</h3>
        <p>Example: <code>{"col_1": [1, 2, 3], "col_2": [4, 5, 6]}</code></p>
        <div style="display: flex;">
            <textarea id="objectDataInput" rows="6" cols="25" style="margin-right: 10px;"></textarea>
            <div class="upload-section" style="text-align: right;">
                <label for="csvFile" class="custom-file-upload">
                    <span id="fileLabel">Choose CSV File</span>
                    <input type="file" id="csvFile" accept=".csv" style="display: none;">
                </label>
                <button id="uploadBtn">Upload Chosen CSV (Option 2)</button>
            </div>
        </div>
        <div class="object-input-actions" style="display: flex; justify-content: space-between; align-items: center;">
            <button id="analyzeObjectBtn">Upload Dictionary (Option 1)</button>
        </div>
    `;
}

function setupCSVUploadListeners(dataPreview, hypothesis) {
    const uploadBtn = document.querySelector('.upload-section #uploadBtn');
    const csvFile = document.querySelector('.upload-section #csvFile');
    
    // Initially disable the upload button
    uploadBtn.disabled = true;

    // Listen for changes in the file input
    csvFile.addEventListener('change', function() {
        // Enable the upload button if a file has been selected
        uploadBtn.disabled = !csvFile.files.length;
    });
    
    uploadBtn.addEventListener('click', function() {
        handleCSVUpload(dataPreview, hypothesis, csvFile);
    });
}

function handleCSVUpload(dataPreview, hypothesis, csvFile) {
    resetAnalysis(dataPreview);
    const file = csvFile.files[0];
    if (file.size > MAX_FILE_SIZE) {
        alert("File size exceeds 5MB limit.");
        return;
    }
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            const csvData = e.target.result;
            const parsedData = parseCSVData(csvData);
            if (Object.keys(parsedData).length === 0) {
                // If parsing returns an empty object because of limitations, do not proceed.
                return;
            }
            if (hypothesis === 'h0') {
                renderH0Inputs(dataPreview, parsedData);
            } else if (hypothesis === 'correlation') {
                renderCorrelationInputs(dataPreview, parsedData);
            }
        }
        reader.readAsText(file);
    }
}

function setupObjectAnalysisListener(dataPreview, hypothesis) {
    const analyzeObjectBtn = document.getElementById('analyzeObjectBtn');
    analyzeObjectBtn.addEventListener('click', function() {
        handleObjectAnalysis(dataPreview, hypothesis);
    });
}

function handleObjectAnalysis(dataPreview, hypothesis) {
    resetAnalysis(dataPreview);
    try {
        const objectDataInput = document.getElementById('objectDataInput').value;
        const objectData = JSON.parse(objectDataInput);
        if (hypothesis === 'h0') {
            renderH0Inputs(dataPreview, objectData);
        } else if (hypothesis === 'correlation') {
            const existingCorrelation = document.getElementById('correlationInputs');
            if (existingCorrelation) existingCorrelation.remove();
            renderCorrelationInputs(dataPreview, objectData);
        }
    } catch (e) {
        alert('Invalid JSON format');
    }
}

// Correlation Section
function renderCorrelationInputs(dataPreview, data) {
    removeExistingElement('correlationInputs');
    const correlationInputsHTML = createCorrelationInputsHTML(data);
    insertAboveResults(dataPreview, correlationInputsHTML);
    setupCorrelationTestListener(data);
}

function createCorrelationInputsHTML(data) {
    const numericColumns = Object.keys(data).filter(col => data[col].every(item => typeof item === 'number'));
    
    return `
        <div id="correlationInputs">
            <label>Column 1:</label>
            <select id="column1Select">
                ${numericColumns.map(col => `<option value="${col}">${col}</option>`).join('')}
            </select>
            <label>Column 2:</label>
            <select id="column2Select">
                ${numericColumns.map((col, index) => `<option value="${col}" ${index === 1 ? 'selected' : ''}>${col}</option>`).join('')}
            </select>
            <label>Correlation Method:</label>
            <select id="correlationMethodSelect">
                <option value="pearson">Pearson</option>
                <option value="spearman">Spearman's</option>
                <option value="kendall">Kendall's</option>
                <option value="pointbiserial">Point-Biserial</option>
                <option value="cramer">Cramér's</option>
            </select>
            <label>
                <input type="checkbox" id="producePlotCheckbox"> Produce Plot
            </label>
            <button id="runCorrelationTestBtn">Run Test</button>
        </div>
    `;
}

function setupCorrelationTestListener(data) {
    const runCorrelationTestBtn = document.getElementById('runCorrelationTestBtn');
    runCorrelationTestBtn.addEventListener('click', function() {
        const column1 = document.getElementById('column1Select').value;
        const column2 = document.getElementById('column2Select').value;
        const method = document.getElementById('correlationMethodSelect').value;
        const producePlot = document.getElementById('producePlotCheckbox').checked; // Default is unchecked
        sendDataForAnalysis(data, 'correlation', column1, column2, method, producePlot);
    });
    addClearResultsButton('correlationInputs');
}

// H0 Test Section
function renderH0Inputs(dataPreview, data) {
    const h0InputsHTML = createH0InputsHTML();
    dataPreview.insertAdjacentHTML('beforeend', h0InputsHTML);
    setupH0TestListener(data);
}

function createH0InputsHTML() {
    return `
        <div id="h0Inputs">
            <label>Population Mean (μ):</label>
            <input type="number" id="populationMean">
            <label>Significance Level (α):</label>
            <input type="number" id="significanceLevel" value="0.05">
            <button id="runH0TestBtn">Run Test</button>
        </div>
    `;
}

function setupH0TestListener(data) {
    const runH0TestBtn = document.getElementById('runH0TestBtn');
    runH0TestBtn.addEventListener('click', function() {
        const populationMean = document.getElementById('populationMean').value;
        const significanceLevel = document.getElementById('significanceLevel').value;
        sendDataForAnalysis(data, 'h0', populationMean, significanceLevel);
    });
}

// Probability Distribution Section
function renderDistributionOptions(dataPreview) {
    clearDistributionElements();
    const distributionOptionsHTML = createDistributionOptionsHTML();
    dataPreview.insertAdjacentHTML('beforeend', distributionOptionsHTML);
    setupProbabilitySpaceListener(dataPreview);
}

function clearDistributionElements() {
    const idsToClear = ['h0Inputs', 'pearsonInputs', 'distributionInputs', 'probabilitySpaceDiv', 'correlationInputs'];
    idsToClear.forEach(id => removeExistingElement(id));
}

function createDistributionOptionsHTML() {
    return `
        <h3>Select Distribution:</h3>
        <select id="distributionSelect">
            <option value="">Select from list</option>
            <option value="binomial">Binomial (n: number of trials; k: number of successes)</option>
            <option value="multinomial">Multinomial (k: number of successes)</option>
            <option value="poisson">Poisson (λ: average events; n: seek frequency)</option>
            <option value="uniform">Uniform (a: minimum; b: maximum)</option>
            <option value="normal">Normal (μ: mean; σ: standard deviation; condition; seek value)</option>
        </select>
    `;
}

function setupProbabilitySpaceListener(dataPreview) {
    const distributionSelect = document.getElementById('distributionSelect');
    distributionSelect.addEventListener('change', function() {
        handleDistributionChange(dataPreview, this.value);
    });
}

function handleDistributionChange(dataPreview, distribution) {
    removeExistingElement('distributionInputs');
    removeExistingElement('probabilitySpaceDiv');
    
    if (distribution) {
        renderDistributionInputs(dataPreview, distribution);
        setupProbabilityInput(dataPreview, distribution);
    }
}

function renderDistributionInputs(dataPreview, distribution) {
    const distributionInputsHTML = createDistributionInputsHTML(distribution);
    insertAboveResults(dataPreview, distributionInputsHTML);
    
    if (distribution === 'normal') {
        setupNormalDistributionListeners();
    }
}

function createDistributionInputsHTML(distribution) {
    let inputs = '';
    
    if (distribution === 'binomial') {
        inputs = `
            <label>Number of trials (n):</label>
            <input type="number" id="binomialTrials" value="10" min="1">
            <label>Number of successes (k):</label>
            <input type="number" id="binomialSuccesses" value="1" min="0">
        `;
    } else if (distribution === 'poisson') {
        inputs = `
            <label>Lambda (λ) average in time:</label>
            <input type="number" id="poissonLambda" value="3" min="0" step="1">
            <label>Time frequency (n):</label>
            <input type="number" id="poissonTimeFrequency" value="1" min="0" step="1">
        `;
    } else if (distribution === 'uniform') {
        inputs = `
            <label>Total Space Interval:</label>
            <input type="number" id="uniformTotal" value="1">
            <label>Favorable Space Interval:</label>
            <input type="number" id="uniformFavorable" value="1">
        `;
    } else if (distribution === 'normal') {
        inputs = `
            <label>Mean (μ):</label>
            <input type="number" id="normalMean" value="2">
            <label>Standard deviation (σ):</label>
            <input type="number" id="normalStdDev" value="1" min="0">
            <label>Condition:</label>
            <select id="normalCondition">
                <option value=">=">>=</option>
                <option value="<="><=</option>
                <option value="range">Range (a ≤ x ≤ b)</option>
                <option value="out_of_range">Out of Range (x ≤ a and x ≥ b)</option>
            </select>
            <div id="rangeInputs" style="display:none;">
                <label>Lower Bound (a):</label>
                <input type="number" id="normalLowerBound" value="1">
                <label>Upper Bound (b):</label>
                <input type="number" id="normalUpperBound" value="2">
            </div>
            <div id="seekOptionsContainer">
                <div id="seekValueContainer">
                    <label>
                        <input type="checkbox" id="normalSeekValueCheckbox" checked>
                        Seek Value:
                    </label>
                    <input type="number" id="normalSeekValue" value="1">
                </div>
                <div id="seekProbabilityContainer">
                    <label>
                        <input type="checkbox" id="normalSeekProbabilityCheckbox">
                        Seek Probability:
                    </label>
                    <input type="number" id="normalSeekProbability" value="50" min="0" max="100" step="1" disabled>
                </div>
            </div>
            <label>
                <input type="checkbox" id="normalProducePlotCheckbox">
                Produce Plot
            </label>
        `;
    }
    
    return `<div id="distributionInputs">${inputs}</div>`;
}

function setupNormalDistributionListeners() {
    const normalCondition = document.getElementById('normalCondition');
    const rangeInputs = document.getElementById('rangeInputs');
    const normalLowerBound = document.getElementById('normalLowerBound');
    const normalUpperBound = document.getElementById('normalUpperBound');
    const seekValueCheckbox = document.getElementById('normalSeekValueCheckbox');
    const seekProbabilityCheckbox = document.getElementById('normalSeekProbabilityCheckbox');
    const seekValueInput = document.getElementById('normalSeekValue');
    const seekProbabilityInput = document.getElementById('normalSeekProbability');
    
    // Set initial state for non-range conditions.
    normalCondition.value = ">=";
    rangeInputs.style.display = 'none';
    seekValueInput.style.display = 'inline-block';
    seekProbabilityInput.style.display = 'inline-block';

    // Helper to update the default range values based on the selected seek option.
    function updateRangeDefaults() {
        if (normalCondition.value === 'range' || normalCondition.value === 'out_of_range') {
            if (seekValueCheckbox.checked) {
                normalLowerBound.value = 1;
                normalUpperBound.value = 2;
            } else if (seekProbabilityCheckbox.checked) {
                normalLowerBound.value = 5;
                normalUpperBound.value = 95;
            }
        }
    }

    normalCondition.addEventListener('change', function() {
        if (this.value === 'range' || this.value === 'out_of_range') {
            rangeInputs.style.display = 'block';
            // Hide the seek input fields while keeping checkboxes visible
            seekValueInput.style.display = 'none';
            seekProbabilityInput.style.display = 'none';
            // Do not update defaults here in order to preserve user-entered values.
            // updateRangeDefaults();  <-- Removed this call.
        } else {
            rangeInputs.style.display = 'none';
            // Show the seek input fields based on current checkbox states
            seekValueInput.style.display = 'inline-block';
            seekProbabilityInput.style.display = 'inline-block';
        }
    });
    
    // Setup mutual exclusion for Seek Value and Seek Probability checkboxes,
    // and manage enabling/disabling of their corresponding input fields.
    seekValueCheckbox.addEventListener('change', function() {
        if (this.checked) {
            seekValueInput.disabled = false;
            seekProbabilityCheckbox.checked = false;
            seekProbabilityInput.disabled = true;
            updateRangeDefaults(); // update defaults when seek option changes
        }
    });
    
    seekProbabilityCheckbox.addEventListener('change', function() {
        if (this.checked) {
            seekProbabilityInput.disabled = false;
            seekValueCheckbox.checked = false;
            seekValueInput.disabled = true;
            updateRangeDefaults(); // update defaults when seek option changes
        } else {
            seekValueCheckbox.checked = true;
            seekValueInput.disabled = false;
        }
    });
}

function setupProbabilityInput(dataPreview, distribution) {
    if (distribution === 'poisson') {
        renderProbabilityInput(dataPreview, false, distribution);
    } else if (['binomial', 'multinomial', 'uniform', 'normal'].includes(distribution)) {
        renderProbabilityInput(dataPreview, true, distribution);
    }
}

function renderProbabilityInput(dataPreview, showProbability, distType) {
    removeExistingElement('probabilitySpaceDiv');
    
    const probabilitySpaceHTML = createProbabilitySpaceHTML(showProbability, distType);
    insertAboveResults(dataPreview, probabilitySpaceHTML);
    
    const probabilitySpaceDiv = document.getElementById('probabilitySpaceDiv');
    
    if (showProbability && (distType === 'multinomial' || 
        (distType !== 'binomial' && distType !== 'uniform' && distType !== 'normal'))) {
        setupOutcomeControls(distType);
    }
    
    setupProbabilityAnalysisButton(distType);
    addClearResultsButton('probabilitySpaceDiv');
}

function createProbabilitySpaceHTML(showProbability, distType) {
    let html = '';
    
    if (showProbability) {
        if (distType === 'multinomial') {
            html = createMultinomialHTML();
        } else if (distType === 'binomial') {
            html = createBinomialHTML();
        } else if (distType === 'uniform') {
            html = createUniformHTML();
        } else if (distType === 'normal') {
            html = createNormalHTML();
        } else {
            html = createDefaultProbabilityHTML();
        }
    } else {
        html = createPoissonHTML();
    }
    
    return html;
}

function createMultinomialHTML() {
    return `
        <div id="probabilitySpaceDiv">
            <h3>Define outcomes and their probabilities</h3>
            <div id="probabilitySpace">
                <div class="outcome">
                    <label>Outcome 1:</label>
                    <input type="text" class="outcomeName" value="Outcome 1">
                    <label>Event Probability:</label>
                    <input type="number" class="outcomeProbability" value="0.5" min="0" max="1" step="0.01">
                    <label>Number of Successes:</label>
                    <input type="number" class="outcomeSuccesses" value="1" min="1" step="1">
                </div>
            </div>
            <button id="addOutcomeBtn">Add Outcome</button>
        </div>
    `;
}

function createBinomialHTML() {
    return `
        <div id="probabilitySpaceDiv">
            <h3>Define outcomes and their probabilities</h3>
            <div id="probabilitySpace">
                <div class="outcome">
                    <label>Outcome 1:</label>
                    <input type="text" class="outcomeName" value="Outcome 1">
                    <label>Event Probability:</label>
                    <input type="number" class="outcomeProbability" value="0.5" min="0" max="1" step="0.01">
                </div>
            </div>
            <button id="addOutcomeBtn" hidden>Add Outcome</button>
        </div>
    `;
}

function createUniformHTML() {
    return `
        <div id="probabilitySpaceDiv">
            <h3>Define outcomes</h3>
            <div id="probabilitySpace">
                <div class="outcome">
                    <label>Outcome 1:</label>
                    <input type="text" class="outcomeName" value="Outcome 1">
                </div>
            </div>
            <button id="addOutcomeBtn" hidden>Add Outcome</button>
        </div>
    `;
}

function createNormalHTML() {
    return `
        <div id="probabilitySpaceDiv">
            <h3>Define outcome</h3>
            <div id="probabilitySpace">
                <div class="outcome">
                    <label>Outcome 1:</label>
                    <input type="text" class="outcomeName" value="Outcome 1">
                </div>
            </div>
            <button id="addOutcomeBtn" hidden>Add Outcome</button>
        </div>
    `;
}

function createDefaultProbabilityHTML() {
    return `
        <div id="probabilitySpaceDiv">
            <h3>Define outcomes and their probabilities</h3>
            <div id="probabilitySpace">
                <div class="outcome">
                    <label>Outcome 1:</label>
                    <input type="text" class="outcomeName" value="Outcome 1">
                    <label>Event Probability:</label>
                    <input type="number" class="outcomeProbability" value="0.5" min="0" max="1" step="0.01">
                </div>
            </div>
            <button id="addOutcomeBtn">Add Outcome</button>
        </div>
    `;
}

function createPoissonHTML() {
    return `
        <div id="probabilitySpaceDiv">
            <h3>Define outcomes</h3>
            <div id="probabilitySpace">
                <div class="outcome">
                    <label>Outcome 1:</label>
                    <input type="text" class="outcomeName" value="Outcome 1">
                </div>
            </div>
            <button id="addOutcomeBtn" hidden>Add Outcome</button>
        </div>
    `;
}

function setupOutcomeControls(distType) {
    const addOutcomeBtn = document.getElementById('addOutcomeBtn');
    addOutcomeBtn.addEventListener('click', function() {
        addNewOutcome(distType);
    });

    if (distType === 'multinomial' || (distType !== 'binomial' && distType !== 'uniform' && distType !== 'normal')) {
        const removeOutcomeBtn = createRemoveOutcomeButton();
        document.getElementById('probabilitySpaceDiv').appendChild(removeOutcomeBtn);
    }
}

function addNewOutcome(distType) {
    const outcomeCount = document.querySelectorAll('.outcome').length + 1;
    let previousProbability = 0.5;
    const previousOutcomes = document.querySelectorAll('.outcome');
    
    if (previousOutcomes.length > 0) {
        const lastOutcome = previousOutcomes[previousOutcomes.length - 1];
        previousProbability = parseFloat(lastOutcome.querySelector('.outcomeProbability')?.value || "0.5");
    }

    let newOutcomeHTML = createNewOutcomeHTML(distType, outcomeCount, previousProbability);
    document.getElementById('probabilitySpace').insertAdjacentHTML('beforeend', newOutcomeHTML);
    
    const removeOutcomeBtn = document.getElementById('removeOutcomeBtn');
    if (removeOutcomeBtn) {
        removeOutcomeBtn.disabled = document.querySelectorAll('.outcome').length <= 1;
    }
}

function createNewOutcomeHTML(distType, outcomeCount, previousProbability) {
    if (distType === 'multinomial') {
        return `
            <div class="outcome">
                <label>Outcome ${outcomeCount}:</label>
                <input type="text" class="outcomeName" value="Outcome ${outcomeCount}">
                <label>Event Probability:</label>
                <input type="number" class="outcomeProbability" value="${previousProbability}" min="0" max="1" step="0.01">
                <label>Number of Successes:</label>
                <input type="number" class="outcomeSuccesses" value="1" min="1" step="1">
            </div>
        `;
    } else {
        return `
            <div class="outcome">
                <label>Outcome ${outcomeCount}:</label>
                <input type="text" class="outcomeName" value="Outcome ${outcomeCount}">
                <label>Event Probability:</label>
                <input type="number" class="outcomeProbability" value="${previousProbability}" min="0" max="1" step="0.01">
            </div>
        `;
    }
}

function createRemoveOutcomeButton() {
    const removeOutcomeBtn = document.createElement('button');
    removeOutcomeBtn.id = 'removeOutcomeBtn';
    removeOutcomeBtn.textContent = 'Remove Outcome';
    removeOutcomeBtn.disabled = true;
    
    removeOutcomeBtn.addEventListener('click', function() {
        const outcomeElements = document.querySelectorAll('.outcome');
        if (outcomeElements.length > 1) {
            outcomeElements[outcomeElements.length - 1].remove();
        }
        this.disabled = document.querySelectorAll('.outcome').length <= 1;
    });
    
    return removeOutcomeBtn;
}

function setupProbabilityAnalysisButton(distType) {
    const analyzeProbabilityBtn = document.createElement('button');
    analyzeProbabilityBtn.id = 'analyzeProbabilityBtn';
    analyzeProbabilityBtn.textContent = 'Analyze';
    document.getElementById('probabilitySpaceDiv').appendChild(analyzeProbabilityBtn);
    
    analyzeProbabilityBtn.addEventListener('click', async function() {
        analyzeProbabilityBtn.disabled = true;
        try {
            await handleProbabilityAnalysis(distType);
        } finally {
            analyzeProbabilityBtn.disabled = false;
        }
    });
}

async function handleProbabilityAnalysis(distType) {
    const outcomes = collectOutcomes(distType);
    
    if (distType !== 'uniform' && distType !== 'normal') {
        const totalProbability = outcomes.reduce((sum, outcome) => sum + (outcome.probability || 0), 0);
        if (totalProbability > 1) {
            alert("The total event probability exceeds 1. Please adjust your probabilities.");
            return;
        }
    }
    
    const params = collectDistributionParams(distType);
    if (params === null) return; // Validation failed
    await sendDataForAnalysis(outcomes, distType, params);
}

function collectOutcomes(distType) {
    const outcomes = [];
    
    if (distType !== 'uniform' && distType !== 'normal') {
        document.querySelectorAll('.outcome').forEach(outcome => {
            const name = outcome.querySelector('.outcomeName').value;
            let probability = 1;
            const probElem = outcome.querySelector('.outcomeProbability');
            if (probElem) probability = parseFloat(probElem.value);
            
            if (distType === 'multinomial') {
                const successesElem = outcome.querySelector('.outcomeSuccesses');
                let successes = 0;
                if (successesElem) successes = parseInt(successesElem.value, 10);
                outcomes.push({name, probability, successes});
            } else {
                outcomes.push({name, probability});
            }
        });
    } else {
        document.querySelectorAll('.outcome').forEach(outcome => {
            const name = outcome.querySelector('.outcomeName').value;
            outcomes.push({name});
        });
    }
    
    return outcomes;
}

function collectDistributionParams(distType) {
    const params = {};
    
    if (distType === 'binomial') {
        params.trials = document.getElementById('binomialTrials').value;
        params.successes = document.getElementById('binomialSuccesses').value;
    } else if (distType === 'poisson') {
        params.lambda = document.getElementById('poissonLambda').value;
        params.timeFrequency = document.getElementById('poissonTimeFrequency').value;
    } else if (distType === 'uniform') {
        params.total = document.getElementById('uniformTotal').value;
        params.favorable = document.getElementById('uniformFavorable').value;
        if (parseFloat(params.total) <= parseFloat(params.favorable)) {
            alert("Total Space Interval must be greater than Favorable Space Interval for Uniform distribution.");
            return null;
        }
    } else if (distType === 'normal') {
        params.mean = document.getElementById('normalMean').value;
        params.stdDev = document.getElementById('normalStdDev').value;
        params.condition = document.getElementById('normalCondition').value;
        if (params.condition === 'range' || params.condition === 'out_of_range') {
            params.lowerBound = document.getElementById('normalLowerBound').value;
            params.upperBound = document.getElementById('normalUpperBound').value;
        }
        const seekValueCheckbox = document.getElementById('normalSeekValueCheckbox');
        const seekProbabilityCheckbox = document.getElementById('normalSeekProbabilityCheckbox');
        if (seekValueCheckbox.checked) {
            params.seekOption = "value";
            params.seekValue = document.getElementById('normalSeekValue').value;
        } else if (seekProbabilityCheckbox.checked) {
            params.seekOption = "probability";
            params.seekProbability = document.getElementById('normalSeekProbability').value;
        }
        // New: pass the produce plot flag
        params.producePlot = document.getElementById('normalProducePlotCheckbox').checked;
    }
    
    return params;
}

// Data Processing
function parseCSVData(csvData) {
    const lines = csvData.split('\n').filter(line => line.trim() !== "");
    
    // Check row limit (excluding header)
    if (lines.length - 1 > MAX_ROWS) {
        alert("CSV file exceeds maximum allowed rows (" + MAX_ROWS + ").");
        return {};
    }
    const headers = lines[0].split(',').map(header => header.trim());
    // Check column limit
    if (headers.length > MAX_COLUMNS) {
        alert("CSV file exceeds maximum allowed columns (" + MAX_COLUMNS + ").");
        return {};
    }

    const data = {};
    headers.forEach(header => {
        data[header] = [];
    });

    for (let i = 1; i < lines.length; i++) {
        const values = lines[i].split(',');
        if (values.every(value => value === "")) continue;
        for (let j = 0; j < headers.length; j++) {
            let value = values[j] ? values[j].trim() : "";
            const num = Number(value);
            if (!isNaN(num) && value !== "") {
                data[headers[j]].push(num);
            } else {
                data[headers[j]].push(value);
            }
        }
    }
    return data;
}

// API Communication
async function sendDataForAnalysis(data, testType, ...args) {
    let requestBody = {
        data: data,
        testType: testType
    };

    if (testType === 'h0') {
        requestBody.populationMean = args[0];
        requestBody.significanceLevel = args[1];
    } else if (testType === 'correlation') {
        requestBody.column1 = args[0];
        requestBody.column2 = args[1];
        requestBody.correlationMethod = args[2];
        requestBody.producePlot = args[3] || false; // Default to false if not provided
    } else {
        requestBody.params = args[0];
        requestBody.distribution = testType;
        requestBody.testType = 'test_distribution';
    }
    
    try {
        const response = await fetch('/api/analyze_inference', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestBody)
        });
        
        const result = await response.json();
        if (!result.success) {
            alert("Error: " + result.message);
        } else {
            displayResults(result);
        }
    } catch (error) {
        alert("⚠️ Analysis failed!\n\n" + error.message);
    }
}

// Results Display
function displayResults(result) {
    const dataPreview = document.getElementById('dataPreview');
    let resultsSection = document.getElementById('analysis-results-area');
    
    // If results section doesn't exist, create it and add a "Results" title
    if (!resultsSection) {
        resultsSection = document.createElement('div');
        resultsSection.id = 'analysis-results-area';
        
        const title = document.createElement('h3');
        title.className = 'results-title';
        title.textContent = 'Results';
        resultsSection.appendChild(title);
        
        dataPreview.insertAdjacentElement('beforeend', resultsSection);
    }
    
    // Create new result entry
    const resultContainer = document.createElement('div');
    resultContainer.className = 'result-entry';
    
    const messageElement = document.createElement('p');
    messageElement.textContent = result.message;
    resultContainer.appendChild(messageElement);
    
    if (result.plot && result.plot.trim() !== "") {
        const plotImg = document.createElement('img');
        plotImg.src = `data:image/png;base64,${result.plot}`;
        plotImg.alt = 'Inference Plot';
        plotImg.className = 'thumbnail-image';
        plotImg.addEventListener('click', function() {
            createImageOverlay(this.src, this.alt);
        });
        resultContainer.appendChild(plotImg);
    }
    
    // Append the new result entry below the title
    // (i.e. after the first child, which is the title)
    resultsSection.insertBefore(resultContainer, resultsSection.childNodes[1] || null);
}

// Utility Functions
function resetAnalysis(dataPreview) {
    const elementsToRemove = ['h0Inputs', 'distributionInputs', 'probabilitySpaceDiv', 'pearsonInputs'];
    elementsToRemove.forEach(id => removeExistingElement(id));
}

function removeExistingElement(id) {
    const element = document.getElementById(id);
    if (element) element.remove();
}

function insertAboveResults(dataPreview, html) {
    const resultsSection = document.getElementById('analysis-results-area');
    if (resultsSection) {
        resultsSection.insertAdjacentHTML('beforebegin', html);
    } else {
        dataPreview.insertAdjacentHTML('beforeend', html);
    }
}

function addClearResultsButton(parentId) {
    const parent = document.getElementById(parentId);
    if (parent && !document.getElementById('clearResultsBtn')) {
        const clearResultsBtn = document.createElement('button');
        clearResultsBtn.id = 'clearResultsBtn';
        clearResultsBtn.textContent = 'Clear Results';
        parent.appendChild(clearResultsBtn);
        clearResultsBtn.addEventListener('click', function() {
            removeExistingElement('analysis-results-area');
        });
    }
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

function showErrorToast(message) {
    const toast = document.createElement('div');
    toast.className = 'error-toast';
    toast.textContent = message;
    toast.style.position = 'fixed';
    toast.style.bottom = '20px';
    toast.style.right = '20px';
    toast.style.backgroundColor = '#f56565';
    toast.style.color = 'white';
    toast.style.padding = '15px';
    toast.style.borderRadius = '4px';
    toast.style.zIndex = '1000';
    toast.style.boxShadow = '0 2px 10px rgba(0,0,0,0.2)';
    toast.style.animation = 'fadeIn 0.3s';
    
    document.body.appendChild(toast);
    
    setTimeout(() => {
        toast.style.animation = 'fadeOut 0.3s';
        setTimeout(() => {
            document.body.removeChild(toast);
        }, 300);
    }, 3000);
}

// Add this to your CSS:
/*
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes fadeOut {
    from { opacity: 1; transform: translateY(0); }
    to { opacity: 0; transform: translateY(20px); }
}
*/

// ==============================================
// INITIALIZE APPLICATION
// ==============================================

// Start the application when the DOM is fully loaded
document.addEventListener('DOMContentLoaded', initializeInference);

// Add these constants near the top (after other global code)
const MAX_FILE_SIZE = 5 * 1024 * 1024; // 5MB limit
const MAX_ROWS = 1000;
const MAX_COLUMNS = 13;