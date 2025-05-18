document.addEventListener('DOMContentLoaded', () => {
    const API_BASE_URL = 'http://localhost:8001/api'; // Should match main app's API
    const ITEMS_PER_PAGE = 10;

    let allProjectData = [];
    let currentCheckerProject = null;
    let currentPage = 1;

    // DOM Elements
    const projectDropdown = document.getElementById('project-dropdown-checker');
    const loadProjectDataBtn = document.getElementById('load-project-data-btn');
    const checkerDashboard = document.getElementById('checker-dashboard');
    const currentCheckerProjectTitle = document.getElementById('current-checker-project-title');
    const checkerItemsContainer = document.getElementById('checker-items-container');
    const checkerLogsDiv = document.getElementById('checker-logs');

    // Pagination Elements
    const prevPageBtn = document.getElementById('prev-page-btn');
    const nextPageBtn = document.getElementById('next-page-btn');
    const pageInfoSpan = document.getElementById('page-info');
    const prevPageBtnBottom = document.getElementById('prev-page-btn-bottom');
    const nextPageBtnBottom = document.getElementById('next-page-btn-bottom');
    const pageInfoSpanBottom = document.getElementById('page-info-bottom');
    const paginationControlsBottom = document.getElementById('pagination-controls-bottom');


    // --- Logging ---
    function logCheckerMessage(message, type = 'info') {
        const logEntry = document.createElement('div');
        logEntry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
        logEntry.classList.add(type === 'error' ? 'log-error' : (type === 'success' ? 'log-success' : 'log-info'));
        // Prepend to see latest logs first, or append
        checkerLogsDiv.prepend(logEntry); 
    }

    // --- Project Selection ---
    async function fetchCheckerProjects() {
        try {
            const response = await fetch(`${API_BASE_URL}/projects`); // Uses main project list endpoint
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            const data = await response.json();
            populateProjectDropdown(data.projects);
        } catch (error) {
            logCheckerMessage(`Error fetching projects: ${error.message}`, 'error');
        }
    }

    function populateProjectDropdown(projects) {
        projectDropdown.innerHTML = '<option value="">Válassz projektet...</option>';
        projects.forEach(projectName => {
            const option = document.createElement('option');
            option.value = projectName;
            option.textContent = projectName;
            projectDropdown.appendChild(option);
        });
    }

    loadProjectDataBtn.addEventListener('click', () => {
        currentCheckerProject = projectDropdown.value;
        if (!currentCheckerProject) {
            alert('Kérlek válassz egy projektet!');
            return;
        }
        currentCheckerProjectTitle.textContent = `Projekt: ${currentCheckerProject}`;
        checkerDashboard.classList.remove('hidden');
        paginationControlsBottom.classList.remove('hidden');
        logCheckerMessage(`Loading data for project: ${currentCheckerProject}`);
        fetchAndRenderCheckerData();
    });

    // --- Data Fetching and Rendering ---
    async function fetchAndRenderCheckerData() {
        if (!currentCheckerProject) return;
        checkerItemsContainer.innerHTML = '<p class="loading-message">Projekt adatok betöltése...</p>';
        try {
            const response = await fetch(`${API_BASE_URL}/check/${currentCheckerProject}/data`);
            if (!response.ok) {
                 const errData = await response.json();
                 throw new Error(errData.detail || `HTTP error! status: ${response.status}`);
            }
            allProjectData = await response.json();
            logCheckerMessage(`Successfully loaded ${allProjectData.length} items for ${currentCheckerProject}.`, 'success');
            currentPage = 1;
            renderCurrentPage();
        } catch (error) {
            logCheckerMessage(`Error loading checker data: ${error.message}`, 'error');
            checkerItemsContainer.innerHTML = `<p class="loading-message error">Hiba az adatok betöltése közben: ${error.message}</p>`;
        }
    }

    function renderCurrentPage() {
        checkerItemsContainer.innerHTML = '';
        if (allProjectData.length === 0) {
            checkerItemsContainer.innerHTML = '<p class="loading-message">Nincsenek megjeleníthető adatok.</p>';
            updatePaginationControls();
            return;
        }

        const startIndex = (currentPage - 1) * ITEMS_PER_PAGE;
        const endIndex = startIndex + ITEMS_PER_PAGE;
        const pageData = allProjectData.slice(startIndex, endIndex);

        pageData.forEach(item => {
            const itemDiv = document.createElement('div');
            itemDiv.classList.add('checker-item');
            if (item.overlong) {
                itemDiv.classList.add('overlong');
            }

            let overlongInfo = 'N/A';
            if (item.trans_duration > 0) {
                overlongInfo = `Duration: ${item.trans_duration.toFixed(2)}s. `;
                if (item.allowed_interval !== null) { // Check for null instead of Infinity
                     overlongInfo += `Allowed: ${item.allowed_interval.toFixed(2)}s. `;
                } else {
                    overlongInfo += `Allowed: N/A (last item). `;
                }
                if (item.overlong) {
                    overlongInfo += `Overlong by: ${item.diff_seconds.toFixed(2)}s.`;
                } else {
                    overlongInfo += `Within limit.`;
                }
            }


            itemDiv.innerHTML = `
                <h4>${item.basename}</h4>
                <div class="columns">
                    <div class="column">
                        <h5>Eredeti (Splits)</h5>
                        <p class="info-text">Szöveg:</p>
                        <textarea readonly>${item.splits_txt || ''}</textarea>
                        ${item.splits_wav_url ? `<audio class="audio-player" controls src="${item.splits_wav_url}"></audio>` : '<p class="info-text">Eredeti hang nem található.</p>'}
                    </div>
                    <div class="column">
                        <h5>Fordított (Translated Splits)</h5>
                        <p class="info-text">Szöveg (szerkeszthető):</p>
                        <textarea id="text-${item.basename}">${item.translated_txt || ''}</textarea> 
                        ${item.trans_wav_url ? `<audio class="audio-player" controls src="${item.trans_wav_url}"></audio>` : '<p class="info-text">Fordított hang nem található.</p>'}
                        <p class="info-text">${overlongInfo}</p>
                        <div class="actions">
                            <button class="save-text-btn" data-basename="${item.basename}">Szöveg Mentése (és hang törlése)</button>
                            <button class="delete-audio-btn" data-basename="${item.basename}">Fordított Hang Törlése</button>
                        </div>
                    </div>
                </div>
            `;
            checkerItemsContainer.appendChild(itemDiv);
        });

        // Add event listeners for new buttons
        document.querySelectorAll('.save-text-btn').forEach(button => {
            button.addEventListener('click', handleSaveText);
        });
        document.querySelectorAll('.delete-audio-btn').forEach(button => {
            button.addEventListener('click', handleDeleteAudio);
        });
        updatePaginationControls();
    }
    
    function updatePaginationControls() {
        const totalPages = Math.ceil(allProjectData.length / ITEMS_PER_PAGE) || 1;
        pageInfoSpan.textContent = `Oldal: ${currentPage} / ${totalPages}`;
        pageInfoSpanBottom.textContent = `Oldal: ${currentPage} / ${totalPages}`;

        prevPageBtn.disabled = currentPage === 1;
        nextPageBtn.disabled = currentPage === totalPages;
        prevPageBtnBottom.disabled = currentPage === 1;
        nextPageBtnBottom.disabled = currentPage === totalPages;
    }

    // --- Event Handlers for Actions ---
    async function handleSaveText(event) {
        const basename = event.target.dataset.basename;
        const newText = document.getElementById(`text-${basename}`).value;
        logCheckerMessage(`Saving text for ${basename}...`);
        try {
            const response = await fetch(`${API_BASE_URL}/check/${currentCheckerProject}/item/${basename}/save_text`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: newText })
            });
            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || `HTTP error! status: ${response.status}`);
            }
            const result = await response.json();
            logCheckerMessage(result.message, 'success');
            fetchAndRenderCheckerData(); // Refresh data
        } catch (error) {
            logCheckerMessage(`Error saving text for ${basename}: ${error.message}`, 'error');
        }
    }

    async function handleDeleteAudio(event) {
        const basename = event.target.dataset.basename;
        if (!confirm(`Biztosan törlöd a fordított hangot és JSON-t a "${basename}" elemhez?`)) {
            return;
        }
        logCheckerMessage(`Deleting translated audio for ${basename}...`);
        try {
            const response = await fetch(`${API_BASE_URL}/check/${currentCheckerProject}/item/${basename}/delete_audio`, {
                method: 'DELETE'
            });
            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || `HTTP error! status: ${response.status}`);
            }
            const result = await response.json();
            logCheckerMessage(result.message, 'success');
            fetchAndRenderCheckerData(); // Refresh data
        } catch (error) {
            logCheckerMessage(`Error deleting audio for ${basename}: ${error.message}`, 'error');
        }
    }

    // --- Pagination Logic ---
    function changePage(delta) {
        const totalPages = Math.ceil(allProjectData.length / ITEMS_PER_PAGE);
        const newPage = currentPage + delta;
        if (newPage >= 1 && newPage <= totalPages) {
            currentPage = newPage;
            renderCurrentPage();
        }
    }

    prevPageBtn.addEventListener('click', () => changePage(-1));
    nextPageBtn.addEventListener('click', () => changePage(1));
    prevPageBtnBottom.addEventListener('click', () => changePage(-1));
    nextPageBtnBottom.addEventListener('click', () => changePage(1));

    // Initial Load
    fetchCheckerProjects();
});
