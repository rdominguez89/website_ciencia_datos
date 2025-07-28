// Global State Management
let cleanedData = null;
let currentDF = null;

// Initialization Functions
function initializeEventListeners() {
    // Main upload button handler
    const uploadBtn = document.getElementById('uploadBtn');
    const fileInput = document.getElementById('csvFile');

    // Initially disable the upload button
    uploadBtn.disabled = true;

    // Listen for changes in the file input
    fileInput.addEventListener('change', function() {
        // Enable the upload button if a file has been selected
        uploadBtn.disabled = !fileInput.files.length;
    });

    uploadBtn.addEventListener('click', handleFileUpload);
}

// File Upload and Processing
async function handleFileUpload() {
    if (cleanedData) {
        await processData(cleanedData);
        cleanedData = null;
        return;
    }

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
}

// Data Processing
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

        renderDataDisplay(statsData);
        setupEventListeners(statsData);
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred while processing the data');
    }
}

// UI Rendering
function renderDataDisplay(statsData) {
    const nullsDisplay = statsData.stats.columns_with_null_count > 0 ?
        `<div class="null-columns">
            <strong>Columns with nulls:</strong>
            ${Object.entries(statsData.stats.columns_with_nulls).map(([col, count]) =>
                `<span class="null-pill">${col}: ${count}</span>`
            ).join('')}
        </div>` :
        '<div class="no-nulls">No columns with null values</div>';

    const duplicatesDisplay = statsData.stats.duplicates > 0 ?
        `<div class="null-columns">
            <strong>Duplicated rows:</strong>
            <span class="null-pill">${statsData.stats.duplicates} duplicates found</span>
        </div>` :
        '<div class="no-nulls">No duplicated rows found</div>';

    const actionsDisplay = (statsData.stats.total_null > 0 || statsData.stats.duplicates > 0) ?
        `<div class="actions-section">
            <h3>Data Cleaning Actions</h3>
            ${statsData.stats.total_null > 0 ?
                `<div class="action-item">
                    <label>
                        <input type="checkbox" id="removeNulls" checked>
                        Remove rows with null values (${statsData.stats.total_null} nulls found)
                    </label>
                </div>` : ''
            }
            ${statsData.stats.duplicates > 0 ?
                `<div class="action-item">
                    <label>
                        <input type="checkbox" id="removeDuplicates" checked>
                        Remove duplicate rows (${statsData.stats.duplicates} duplicates found)
                    </label>
                </div>` : ''
            }
            <button id="runActions" class="action-button">Clean Data</button>
        </div>` :
        `<div class="actions-section">
            <h3>Data Analysis</h3>
            <button id="createVisualizations" class="action-button">Create Preview Exploration Plots</button>
        </div>`;

    const html = `
        <div class="data-tabs">
            <button class="tab-btn active" data-tab="preview">Preview</button>
            <button class="tab-btn" data-tab="stats">Statistics</button>
            <button class="tab-btn" data-tab="structure">Structure</button>
            <button class="tab-btn" data-tab="nulls">Null Values</button>
            <button class="tab-btn" data-tab="duplicates">Duplicates</button>
        </div>
        <div id="tab-content"></div>
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
            ${nullsDisplay}
            ${duplicatesDisplay}
            ${actionsDisplay}
        </div>
    `;

    document.getElementById('dataPreview').innerHTML = html;
    loadInitialContent(statsData);
    setupTabEventListeners(statsData);
}

// Event Listener Setup
function setupEventListeners(statsData) {
    setupCleaningActions(statsData);
    setupVisualizationActions(statsData);
}

function setupTabEventListeners(statsData) {
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            loadTabContent(btn.dataset.tab, statsData);
        });
    });
}

function setupCleaningActions(statsData) {
    document.getElementById('runActions')?.addEventListener('click', async function() {
        const removeNulls = document.getElementById('removeNulls')?.checked || false;
        const removeDuplicates = document.getElementById('removeDuplicates')?.checked || false;

        if (!removeNulls && !removeDuplicates) {
            alert('Please select at least one action');
            return;
        }

        try {
            const response = await fetch('/api/clean_data', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    remove_nulls: removeNulls,
                    remove_duplicates: removeDuplicates,
                    df: currentDF
                })
            });
            const data = await response.json();
            cleanedData = data;
            document.getElementById('uploadBtn').click();
        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred while cleaning the data');
        }
    });
}

function setupVisualizationActions(statsData) {
    document.getElementById('createVisualizations')?.addEventListener('click', async function() {
        const createBtn = this;
        createBtn.disabled = true;
        createBtn.textContent = 'Creating Preview Exploration Plots...';

        try {
            const response = await fetch('/api/create_visualizations', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    df: currentDF
                })
            });
            const data = await response.json();

            renderVisualizations(data, statsData);
            createBtn.disabled = false;
            createBtn.textContent = 'Create Preview Exploration Plots';
        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred while creating visualizations');
            createBtn.disabled = false;
            createBtn.textContent = 'Create Preview Exploration Plots';
        }
    });
}

// Visualization Rendering
function renderVisualizations(data, statsData) {
    const visualizationsContainer = document.createElement('div');
    visualizationsContainer.id = 'visualizations-container';
    visualizationsContainer.className = 'visualizations';

    visualizationsContainer.innerHTML = `
        <h3>Preview Exploration Plots</h3>
        ${data.boxplots.length > 0 ? `
            <div class="plot-section">
                <h4>Boxplots</h4>
                <div class="visualization-grid">
                    ${data.boxplots.map(vis => `
                        <div class="visualization-card">
                            <h5>${vis.title}</h5>
                            <img src="data:image/png;base64,${vis.image_data}" alt="${vis.title}">
                        </div>
                    `).join('')}
                </div>
            </div>
        ` : ''}
        ${data.distributions.length > 0 ? `
            <div class="plot-section">
                <h4>Histogram/counts plots</h4>
                <div class="visualization-grid">
                    ${data.distributions.map(vis => `
                        <div class="visualization-card">
                            <h5>${vis.title}</h5>
                            <img src="data:image/png;base64,${vis.image_data}" alt="${vis.title}">
                        </div>
                    `).join('')}
                </div>
            </div>
        ` : ''}
    `;

    const dataPreview = document.getElementById('dataPreview');
    const existingVisualizations = document.getElementById('visualizations-container');
    if (existingVisualizations) {
        dataPreview.removeChild(existingVisualizations);
    }
    dataPreview.appendChild(visualizationsContainer);

    setupOutlierManagement(data, visualizationsContainer);
    setupCorrelationPlots(visualizationsContainer);
    setupImageClickHandlers();
    document.querySelector('.actions-section')?.remove();
}

// Outlier Management
function setupOutlierManagement(data, visualizationsContainer) {
    if (data.outliers_data && Object.keys(data.outliers_data).length > 0) {
        const outlierColumns = Object.keys(data.outliers_data).filter(col =>
            data.outliers_data[col].total_outliers > 0
        );

        if (outlierColumns.length > 0) {
            const outlierSection = document.createElement('div');
            outlierSection.className = 'outliers-section';
            outlierSection.innerHTML = `
                <h3>Outlier Management</h3>
                ${outlierColumns.map(col => {
                    const outliers = data.outliers_data[col];
                    return `
                        <div class="outlier-column">
                            <div class="outlier-controls">
                                <button class="action-button" data-column="${col}">Select All</button>
                                <button class="action-button" data-column="${col}">Deselect All</button>
                            </div>
                            <div class="outlier-table-container">
                                <table class="outlier-table">
                                    <thead>
                                        <tr>
                                            <th>Select</th>
                                            <th>Row Index</th>
                                            <th>${col}</th>
                                            <th>Other Columns</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        ${outliers.outlier_rows.map((row, i) => `
                                            <tr>
                                                <td><input type="checkbox" class="outlier-checkbox" data-column="${col}" data-index="${row.index || row._index}" checked></td>
                                                <td>${row.index || row._index}</td>
                                                <td>${row[col]}</td>
                                                <td>${Object.entries(row)
                                                    .filter(([k, v]) => k !== col && k !== 'index' && k !== '_index')
                                                    .map(([k, v]) => `${k}: ${v}`)
                                                    .join(', ')}</td>
                                            </tr>
                                        `).join('')}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    `;
                }).join('')}
                <button id="removeOutliersBtn" class="action-button" disabled>Remove Selected Outliers</button>
            `;

            visualizationsContainer.insertBefore(outlierSection, document.getElementById('correlationPlotBtn'));
            setupOutlierEventListeners();
        }
    } else {
        const noOutliersMsg = document.createElement('div');
        noOutliersMsg.className = 'no-outliers';
        noOutliersMsg.innerHTML = '<p><strong>No outliers detected in any columns</strong></p>';
        visualizationsContainer.insertBefore(noOutliersMsg, document.getElementById('correlationPlotBtn'));
    }
}

function setupOutlierEventListeners() {
    const updateButtonStates = () => {
        const removeBtn = document.getElementById('removeOutliersBtn');
        const anyChecked = document.querySelectorAll('.outlier-checkbox:checked').length > 0;

        if (removeBtn) {
            removeBtn.disabled = !anyChecked;
        }
    };

    // Select All buttons
    document.querySelectorAll('.outlier-controls button:nth-child(1)').forEach(btn => {
        btn.addEventListener('click', function() {
            const column = this.dataset.column;
            document.querySelectorAll(`.outlier-checkbox[data-column="${column}"]`).forEach(cb => {
                cb.checked = true;
            });
            updateButtonStates();
        });
    });

    // Deselect All buttons
    document.querySelectorAll('.outlier-controls button:nth-child(2)').forEach(btn => {
        btn.addEventListener('click', function() {
            const column = this.dataset.column;
            document.querySelectorAll(`.outlier-checkbox[data-column="${column}"]`).forEach(cb => {
                cb.checked = false;
            });
            updateButtonStates();
        });
    });

    // Checkbox change events
    document.querySelectorAll('.outlier-checkbox').forEach(checkbox => {
        checkbox.addEventListener('change', updateButtonStates);
    });

    // Remove Outliers button
    document.getElementById('removeOutliersBtn')?.addEventListener('click', async function() {
        if (this.disabled) return;

        const selectedOutliers = [];
        document.querySelectorAll('.outlier-checkbox:checked').forEach(checkbox => {
            selectedOutliers.push({
                column: checkbox.dataset.column,
                index: parseInt(checkbox.dataset.index)
            });
        });

        try {
            const response = await fetch('/api/remove_outliers', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    df: currentDF,
                    outliers: selectedOutliers
                })
            });

            const data = await response.json();
            if (data.error) {
                alert(data.error);
            } else {
                cleanedData = data;
                document.getElementById('uploadBtn').click();
            }
        } catch (error) {
            console.error('Error removing outliers:', error);
            alert('An error occurred while removing outliers');
        }
    });

    updateButtonStates();
}

// Correlation Plots
function setupCorrelationPlots(visualizationsContainer) {
    const correlationBtn = document.createElement('button');
    correlationBtn.id = 'correlationPlotBtn';
    correlationBtn.className = 'action-button';
    correlationBtn.textContent = 'Show Correlation Plots';
    visualizationsContainer.appendChild(correlationBtn);

    correlationBtn.addEventListener('click', async function() {
        if (this.disabled) return;
        this.parentNode.removeChild(this);

        try {
            const response = await fetch('/api/create_correlation_plots', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    df: currentDF
                })
            });

            const data = await response.json();
            if (data.error) {
                alert(data.error);
            } else {
                const correlationContainer = document.createElement('div');
                correlationContainer.id = 'correlation-container';
                correlationContainer.innerHTML = `
                    <h3>Correlation Plots</h3>
                    <div class="plot-section">
                        <div class="visualization-grid">
                            ${data.plots.map(plot => `
                                <div class="visualization-card">
                                    <h5>${plot.title}</h5>
                                    <img src="data:image/png;base64,${plot.image_data}" alt="${plot.title}">
                                </div>
                            `).join('')}
                        </div>
                    </div>
                `;

                const existingCorrelation = document.getElementById('correlation-container');
                if (existingCorrelation) {
                    existingCorrelation.remove();
                }

                visualizationsContainer.appendChild(correlationContainer);
                setupCorrelationImageHandlers();
                setupSaveSection(visualizationsContainer);
            }
        } catch (error) {
            console.error('Error creating correlation plots:', error);
            alert('An error occurred while creating correlation plots');
        }
    });
}

// Save Functionality
function setupSaveSection(visualizationsContainer) {
    const saveSection = document.createElement('div');
    saveSection.className = 'save-section';
    saveSection.innerHTML = `
        <h3>Save Processed Data</h3>
        <div class="column-selection">
            <h4>Select Columns to Save (All Selected by Default)</h4>
            <div class="column-checkboxes">
                ${Object.keys(currentDF[0]).map(col => `
                    <label>
                        <input type="checkbox" name="saveColumns" value="${col}" checked>
                        ${col}
                    </label>
                `).join('')}
            </div>
        </div>
        <div class="save-controls">
            <button id="saveDataBtn" class="action-button">Save as CSV</button>
        </div>
    `;

    visualizationsContainer.appendChild(saveSection);

    document.getElementById('saveDataBtn').addEventListener('click', function() {
        const selectedColumns = Array.from(
            document.querySelectorAll('input[name="saveColumns"]:checked')
        ).map(checkbox => checkbox.value);

        const dataToSave = currentDF.map(row => {
            const filteredRow = {};
            selectedColumns.forEach(col => {
                filteredRow[col] = row[col];
            });
            return filteredRow;
        });

        const headers = selectedColumns.join(',');
        const csvRows = dataToSave.map(row =>
            selectedColumns.map(col =>
                typeof row[col] === 'string' ? `"${row[col].replace(/"/g, '""')}"` : row[col]
            ).join(',')
        );
        const csvContent = [headers, ...csvRows].join('\n');

        const fileName = 'cleaned_dataframe';
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = fileName.endsWith('.csv') ? fileName : `${fileName}.csv`;

        const userConfirmed = confirm('Do you want to download the CSV file?');
        if (userConfirmed) {
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
        }
        URL.revokeObjectURL(url);
    });
}

// Image Handling
function setupCorrelationImageHandlers() {
    document.querySelectorAll('#correlation-container img').forEach(img => {
        img.addEventListener('click', function() {
            createImageOverlay(this.src, this.alt);
        });
    });
}

function setupImageClickHandlers() {
    document.querySelectorAll('.visualization-card img').forEach(img => {
        img.addEventListener('click', function() {
            createImageOverlay(this.src, this.alt);
        });
    });
}

/**
 * Create fullscreen image overlay with responsive sizing
 */
function createImageOverlay(src, alt) {
    // Add overflow hidden to body to prevent scrolling
    document.body.classList.add('overlay-active');
    
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
    
    // Apply responsive image sizing
    img.style.maxWidth = '100%';
    img.style.maxHeight = 'calc(100vh - 60px)'; // Account for padding and close button
    img.style.objectFit = 'contain';
    img.style.display = 'block';
    
    // Build overlay
    content.appendChild(closeBtn);
    content.appendChild(img);
    overlay.appendChild(content);
    document.body.appendChild(overlay);
    
    // Close functionality
    const closeOverlay = () => {
        document.body.removeChild(overlay);
        document.body.classList.remove('overlay-active');
        window.removeEventListener('resize', handleResize);
    };
    
    closeBtn.addEventListener('click', closeOverlay);
    overlay.addEventListener('click', (e) => {
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
    
    // Handle window resize
    const handleResize = () => {
        // Adjust image size based on current viewport
        const maxHeight = window.innerHeight - 60; // Account for padding
        img.style.maxHeight = `${maxHeight}px`;
    };
    
    window.addEventListener('resize', handleResize);
    
    // Initial size calculation
    handleResize();
}

// Tab Content Rendering
function loadInitialContent(statsData) {
    const tabContent = document.getElementById('tab-content');
    tabContent.innerHTML = `
        <div id="preview-content">
            <h3>First 5 Rows</h3>
            <div class="table-responsive-desktop">
                ${statsData.data_head}
            </div>
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
        case 'nulls':
            tabContent.innerHTML = generateNullsHTML(statsData);
            break;
        case 'duplicates':
            tabContent.innerHTML = generateDuplicatesHTML(statsData);
            break;
    }
}

// Helper Functions for Tab Content
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

function generateNullsHTML(statsData) {
    let nullsHTML = `
        <h3>Null Value Analysis</h3>
        <p>Total null values: ${statsData.stats.total_null}</p>
        <p>Columns with null values: ${statsData.stats.columns_with_null_count}/${statsData.stats.columns}</p>
    `;

    if (statsData.stats.columns_with_null_count > 0) {
        nullsHTML += '<h4>Columns with null values:</h4><ul class="null-list">';
        for (const [col, count] of Object.entries(statsData.stats.columns_with_nulls)) {
            nullsHTML += `<li><strong>${col}</strong>: ${count} null values</li>`;
        }
        nullsHTML += '</ul>';
    } else {
        nullsHTML = '<p><strong>No columns contain null values<strong></p>';
    }

    return nullsHTML;
}

function generateDuplicatesHTML(statsData) {
    let duplicatesHTML = `
        <h3>Duplicated Rows Analysis</h3>
        <p>Total duplicated rows: ${statsData.stats.duplicates}</p>
    `;

    if (statsData.stats.duplicates > 0) {
        duplicatesHTML += '<h4>Duplicated rows:</h4>';
        duplicatesHTML += `${statsData.duplicated_rows}`;
    } else {
        duplicatesHTML = '<p><strong>No duplicated rows found<strong></p>';
    }

    return duplicatesHTML;
}

// Initialize the application
initializeEventListeners();