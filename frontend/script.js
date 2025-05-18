document.addEventListener('DOMContentLoaded', () => {
    const API_BASE_URL = 'http://localhost:8001/api'; // Backend API URL
    let currentProject = null;
    let currentScript = null;
    let socket = null;
    let scriptRunning = false; // Flag to track script execution

    // DOM Elements
    const newProjectNameInput = document.getElementById('new-project-name');
    const mkvFileUploadInput = document.getElementById('mkv-file-upload');
    const createProjectBtn = document.getElementById('create-project-btn');
    const projectSelectDropdown = document.getElementById('project-select-dropdown'); // Updated ID

    const hfTokenInput = document.getElementById('hf-token-input');
    const deeplApiKeyInput = document.getElementById('deepl-api-key-input');
    const saveApiKeysBtn = document.getElementById('save-api-keys-btn');

    const scriptRunnerSection = document.getElementById('script-runner');
    const currentProjectTitle = document.getElementById('current-project-title');
    const scriptButtonsDiv = document.getElementById('script-buttons');
    const scriptParamsDiv = document.getElementById('script-params');
    const runScriptBtn = document.getElementById('run-script-btn');

    const downloadSection = document.getElementById('download-section');
    const downloadableFilesListDiv = document.getElementById('downloadable-files-list');
    const refreshDownloadsBtn = document.getElementById('refresh-downloads-btn');

    const logsDiv = document.getElementById('logs');

    // --- API Key Management ---
    async function fetchApiKeys() {
        try {
            const response = await fetch(`${API_BASE_URL}/keys`);
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            const keys = await response.json();
            hfTokenInput.value = keys.hf_token || '';
            deeplApiKeyInput.value = keys.deepL_api_key || '';
        } catch (error) {
            logMessage(`Error fetching API keys: ${error.message}`, 'error');
        }
    }

    saveApiKeysBtn.addEventListener('click', async () => {
        const hfToken = hfTokenInput.value.trim();
        const deeplApiKey = deeplApiKeyInput.value.trim();
        try {
            const response = await fetch(`${API_BASE_URL}/keys`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ hf_token: hfToken, deepL_api_key: deeplApiKey })
            });
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
            }
            const data = await response.json();
            logMessage(data.message);
        } catch (error) {
            logMessage(`Error saving API keys: ${error.message}`, 'error');
        }
    });


    // --- Project Management ---
    async function fetchProjects() {
            try {
                const response = await fetch(`${API_BASE_URL}/projects`);
                if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
                const data = await response.json();
                renderProjects(data.projects);
            } catch (error) {
                logMessage(`Error fetching projects: ${error.message}`, 'error');
            }
        }

        function renderProjects(projects) {
            // Clear previous options but keep the placeholder
            projectSelectDropdown.innerHTML = '<option value="">Válassz egy projektet...</option>';
            projects.forEach(projectName => {
                const option = document.createElement('option');
                option.value = projectName;
                option.textContent = projectName;
                projectSelectDropdown.appendChild(option);
            });
        }

        projectSelectDropdown.addEventListener('change', () => {
            const selectedProjectName = projectSelectDropdown.value;
            if (selectedProjectName) {
                selectProject(selectedProjectName);
            } else {
                // Optionally hide script runner if no project is selected
                scriptRunnerSection.classList.add('hidden');
                downloadSection.classList.add('hidden');
                currentProjectTitle.textContent = '';
                currentProject = null;
                if (socket) {
                    socket.close();
                }
            }
        });

        createProjectBtn.addEventListener('click', async () => {
            const projectName = newProjectNameInput.value.trim();
            if (!projectName) {
                alert('Projekt név megadása kötelező!');
                return;
            }

            const formData = new FormData();
            formData.append('project_name', projectName);
            const mkvFile = mkvFileUploadInput.files[0];
            if (mkvFile) {
                formData.append('mkv_file', mkvFile);
            }

            try {
                const response = await fetch(`${API_BASE_URL}/projects`, {
                    method: 'POST',
                    body: formData // FormData sets Content-Type automatically
                });
                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
                }
                const data = await response.json();
                logMessage(data.message + (data.mkv_uploaded ? ` Fájl feltöltve: ${data.mkv_uploaded}` : ''));
                newProjectNameInput.value = '';
                mkvFileUploadInput.value = ''; // Clear file input
                fetchProjects();
            } catch (error) {
                logMessage(`Error creating project: ${error.message}`, 'error');
            }
        });

        function selectProject(projectName) {
            currentProject = projectName;
            currentProjectTitle.textContent = `Aktuális Projekt: ${projectName}`;
            scriptRunnerSection.classList.remove('hidden');
            downloadSection.classList.remove('hidden');
            scriptParamsDiv.classList.add('hidden');
            runScriptBtn.classList.add('hidden');
            currentScript = null;
            renderScriptButtons();
            fetchDownloadableFiles(projectName);
            connectWebSocket(projectName);
            // Set dropdown value to the selected project
            projectSelectDropdown.value = projectName; 
        }

        // --- Download Management ---
        refreshDownloadsBtn.addEventListener('click', () => {
            if (currentProject) {
                fetchDownloadableFiles(currentProject);
            }
        });

        async function fetchDownloadableFiles(projectName) {
            downloadableFilesListDiv.innerHTML = '<p>Fájlok betöltése...</p>';
            try {
                const response = await fetch(`${API_BASE_URL}/projects/${projectName}/files/downloadable`);
                if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
                const data = await response.json();
                renderDownloadableFiles(data.files, projectName);
            } catch (error) {
                logMessage(`Error fetching downloadable files: ${error.message}`, 'error');
                downloadableFilesListDiv.innerHTML = '<p>Hiba a fájlok lekérése közben.</p>';
            }
        }

        function renderDownloadableFiles(files, projectName) {
            downloadableFilesListDiv.innerHTML = '';
            if (files.length === 0) {
                downloadableFilesListDiv.innerHTML = '<p>Nincsenek letölthető fájlok.</p>';
                return;
            }
            files.forEach(fileName => {
                const fileItem = document.createElement('div');
                fileItem.classList.add('download-item');
                
                const fileNameSpan = document.createElement('span');
                fileNameSpan.textContent = fileName;
                fileItem.appendChild(fileNameSpan);

                const downloadBtn = document.createElement('button');
                downloadBtn.textContent = 'Letöltés';
                downloadBtn.classList.add('download-btn');
                downloadBtn.addEventListener('click', () => {
                    window.location.href = `${API_BASE_URL}/projects/${projectName}/files/download/${fileName}`;
                });
                fileItem.appendChild(downloadBtn);
                downloadableFilesListDiv.appendChild(fileItem);
            });
        }


        // --- WebSocket ---
        function connectWebSocket(projectName) {
            if (socket) {
                socket.close();
            }
            socket = new WebSocket(`ws://localhost:8001/ws/projects/${projectName}/status`);
            socket.onopen = () => {
                logMessage(`WebSocket connected for project: ${projectName}`);
                // If a script was running and we reconnected, ensure UI is in correct state
                if (scriptRunning) {
                    runScriptBtn.disabled = true;
                    runScriptBtn.textContent = 'Script Folyamatban...';
                }
            }
            socket.onmessage = (event) => {
                const message = event.data;
                logMessage(message);

                if (message.includes("[INFO] Starting:")) {
                    scriptRunning = true;
                    runScriptBtn.disabled = true;
                    runScriptBtn.textContent = 'Script Folyamatban...';
                    // Optionally, disable other script buttons too
                    disableAllScriptButtons(true);
                } else if (message.includes("[INFO] Finished:")) {
                    scriptRunning = false;
                    runScriptBtn.disabled = false;
                    runScriptBtn.textContent = 'Script Futtatása';
                     // Re-enable script buttons
                    disableAllScriptButtons(false);
                    if (message.includes("completed successfully")) {
                        logMessage(`Script ${currentScript} sikeresen lefutott.`, 'success');
                        // You could add logic here to auto-select the next script or highlight it
                    } else if (message.includes("failed")) {
                        logMessage(`Script ${currentScript} hibával fejeződött be.`, 'error');
                    }
                    // Refresh downloadable files list after certain scripts
                    if (currentScript === 'merge_video' || currentScript === 'run_normalise_and_cut') { // Add other relevant scripts
                        fetchDownloadableFiles(currentProject);
                    }
                }
            };
            socket.onerror = (error) => {
                logMessage(`WebSocket error: ${JSON.stringify(error)}`, 'error');
                // Reset UI if connection fails during script run
                if(scriptRunning) {
                    scriptRunning = false;
                    runScriptBtn.disabled = false;
                    runScriptBtn.textContent = 'Script Futtatása';
                    disableAllScriptButtons(false);
                }
            };
            socket.onclose = () => {
                logMessage(`WebSocket disconnected for project: ${projectName}`);
                 // If script was running and socket closes, assume interruption
                if(scriptRunning) {
                    logMessage('WebSocket lezárult script futás közben. Ellenőrizd a logokat a szerveren.', 'error');
                    scriptRunning = false;
                    runScriptBtn.disabled = false;
                    runScriptBtn.textContent = 'Script Futtatása';
                    disableAllScriptButtons(false);
                }
            };
        }

        function disableAllScriptButtons(disabled) {
            Array.from(scriptButtonsDiv.children).forEach(btn => {
                btn.disabled = disabled;
            });
        }

        // --- Script Running ---
        const SCRIPTS = {
            separate_audio: { name: "Hang Leválasztása", params: [
                { id: "device", label: "Eszköz (cuda/cpu)", type: "select", options: ["cuda", "cpu"], default: "cuda"},
                { id: "keep_full_audio", label: "Teljes hangsáv megtartása", type: "checkbox", default: false },
                { id: "non_speech_silence", label: "Nem beszéd részek némítása", type: "checkbox", default: false },
                { id: "chunk_size", label: "Chunk méret (perc)", type: "number", default: 5 }, // Corrected unit
                { id: "model", label: "Modell", type: "select", dynamicOptions: "demucs_models", default: "htdemucs" }
            ]},
            transcribe_align: { name: "Átírás és Igazítás (WhisperX)", params: [
                { id: "hf_token", label: "HuggingFace Token (mentett kulcsot használ, ha üres)", type: "text" },
                { id: "language", label: "Nyelv (pl. en, hu)", type: "text", default: "en" }
            ]},
            audio_split: { name: "Hangsáv Felosztása (Splitter)", params: [] },
            translate: { name: "Fordítás (DeepL)", params: [
                { id: "auth_key", label: "DeepL API Kulcs (mentett kulcsot használ, ha üres)", type: "text" },
                { id: "input_language", label: "Forrásnyelv", type: "text", default: "EN" },
                { id: "output_language", label: "Célnyelv", type: "text", default: "HU" }
            ]},
            generate_tts: { name: "TTS Generálás (F5-TTS)", params: [
                { id: "tts_subdir", label: "TTS Modell", type: "select", dynamicOptions: "tts_models", required: true },
                { id: "speed", label: "Sebesség", type: "number", step: 0.1, default: 1.0 },
                { id: "nfe_step", label: "NFE Lépés", type: "number", default: 32 },
                { id: "norm_selection", label: "Normalizáló", type: "select", dynamicOptions: "normalizers", default: "hun" },
                { id: "seed", label: "Seed (-1 random)", type: "number", default: -1 },
                { id: "remove_silence", label: "Csend eltávolítása", type: "checkbox", default: false }
            ]},
            transcribe_align_chunks: { name: "Chunkok Átírása és Igazítása (VAD)", params: [
                { id: "splits_lang", label: "Splits Nyelv (pl. en, hu)", type: "text", required: true },
                { id: "translated_splits_lang", label: "Fordított Splits Nyelv (pl. en, hu)", type: "text", required: true }
            ]},
            normalize_cut: { name: "Normalizálás és Vágás", params: [
                { id: "min_db", label: "Min DB", type: "number", step: 0.1, default: -40.0 },
                { id: "delete_empty", label: "Üres fájlok törlése", type: "checkbox", default: false }
            ]},
            inspect_repair: { name: "Ellenőrzés és Javítás (check_app)", params: [] },
            merge_chunks_bg: { name: "Chunkok Egyesítése Háttérzajjal", params: [] },
            merge_video: { name: "Videó és Hang Egyesítése", params: [
                { id: "language", label: "Nyelv (pl. HUN)", type: "text", default: "HUN" }
            ]}
        };

        function renderScriptButtons() {
            scriptButtonsDiv.innerHTML = '';
            Object.keys(SCRIPTS).forEach(scriptId => {
                const button = document.createElement('button');
                button.textContent = SCRIPTS[scriptId].name;
                button.dataset.scriptId = scriptId;
                button.addEventListener('click', () => selectScript(scriptId));
                scriptButtonsDiv.appendChild(button);
            });
        }

        function selectScript(scriptId) {
            currentScript = scriptId;
            renderScriptParams(scriptId);
            scriptParamsDiv.classList.remove('hidden');
            runScriptBtn.classList.remove('hidden');
            // Highlight selected script button
            Array.from(scriptButtonsDiv.children).forEach(btn => {
                btn.classList.toggle('active', btn.dataset.scriptId === scriptId);
            });
        }

        async function renderScriptParams(scriptId) {
            scriptParamsDiv.innerHTML = '';
            const scriptConfig = SCRIPTS[scriptId];
            if (!scriptConfig || scriptConfig.params.length === 0) {
                scriptParamsDiv.innerHTML = '<p>Nincsenek specifikus paraméterek ehhez a scripthez.</p>';
                return;
            }

            for (const param of scriptConfig.params) {
                const paramGroup = document.createElement('div');
                paramGroup.classList.add('param-group');
                const label = document.createElement('label');
                label.setAttribute('for', param.id);
                label.textContent = `${param.label}${param.required ? '*' : ''}: `;
                paramGroup.appendChild(label);

                let input;
                if (param.type === 'select') {
                    input = document.createElement('select');
                    if (param.dynamicOptions) {
                        try {
                            const response = await fetch(`${API_BASE_URL}/${param.dynamicOptions.replace('_', '-')}`); // e.g. /api/tts-models
                            if (!response.ok) throw new Error(`Failed to load ${param.dynamicOptions}`);
                            const data = await response.json();
                            const optionsArray = data[param.dynamicOptions] || [];
                            optionsArray.forEach(opt => {
                                const option = document.createElement('option');
                                option.value = opt;
                                option.textContent = opt;
                                input.appendChild(option);
                            });
                        } catch (error) {
                            logMessage(`Error loading options for ${param.label}: ${error.message}`, 'error');
                            const option = document.createElement('option');
                            option.value = "";
                            option.textContent = "Hiba a betöltéskor";
                            input.appendChild(option);
                        }
                    } else if (param.options) {
                         param.options.forEach(opt => {
                            const option = document.createElement('option');
                            option.value = opt;
                            option.textContent = opt;
                            input.appendChild(option);
                        });
                    }
                    input.value = param.default || (input.options.length > 0 ? input.options[0].value : "");
                } else if (param.type === 'checkbox') {
                    input = document.createElement('input');
                    input.type = 'checkbox';
                    input.checked = param.default || false;
                } else {
                    input = document.createElement('input');
                    input.type = param.type || 'text';
                    if (param.default !== undefined) input.value = param.default;
                    if (param.step) input.step = param.step;
                }
                input.id = param.id;
                input.name = param.id;
                if (param.required) input.required = true;
                paramGroup.appendChild(input);
                scriptParamsDiv.appendChild(paramGroup);
            }
        }


        runScriptBtn.addEventListener('click', async () => {
            if (!currentProject || !currentScript) {
                alert('Projekt és script kiválasztása kötelező!');
                return;
            }
            if (scriptRunning) {
                alert('Egy script már fut. Kérlek várj amíg befejeződik.');
                return;
            }

            const params = {};
            let validParams = true;
            SCRIPTS[currentScript].params.forEach(param => {
                if (!validParams) return;
                const inputElement = document.getElementById(param.id);
                if (inputElement) {
                    if (param.type === 'checkbox') {
                        params[param.id] = inputElement.checked;
                    } else if (param.type === 'number') {
                        const val = parseFloat(inputElement.value);
                        if (isNaN(val) && param.required) {
                            alert(`A(z) '${param.label}' mező érvényes számot kell tartalmazzon!`);
                            validParams = false;
                            return;
                        }
                        params[param.id] = isNaN(val) ? (param.default !== undefined ? param.default : null) : val;

                    } else {
                        params[param.id] = inputElement.value;
                    }

                    if (param.required && (params[param.id] === null || params[param.id] === '') && param.type !== 'checkbox') {
                         alert(`A(z) '${param.label}' mező kitöltése kötelező!`);
                         validParams = false;
                         return;
                    }
                }
            });

            if (!validParams) {
                return; // Stop if validation failed
            }
            
            // Clear previous script-specific error/success messages from log
            clearScriptStatusMessages();

            scriptRunning = true; // Set flag before API call
            runScriptBtn.disabled = true;
            runScriptBtn.textContent = 'Script Folyamatban...';
            disableAllScriptButtons(true);

            try {
                // No need to log "Running script..." here, backend will send "[INFO] Starting..."
                const response = await fetch(`${API_BASE_URL}/projects/${currentProject}/run/${currentScript}`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(params)
                });
                // The actual success/failure is now handled by WebSocket messages "[INFO] Finished..."
                // However, we still need to check for HTTP errors from the API call itself.
                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
                }
                // const data = await response.json(); // This response is minimal, like {"status": "completed/failed"}
                // logMessage(`API call for ${currentScript} responded: ${data.status}`); // Less important now
            } catch (error) {
               logMessage(`Error initiating script ${currentScript}: ${error.message}`, 'error');
               scriptRunning = false; // Reset flag on API call error
               runScriptBtn.disabled = false;
               runScriptBtn.textContent = 'Script Futtatása';
               disableAllScriptButtons(false);
            }
        });

        function clearScriptStatusMessages() {
            const messagesToRemove = logsDiv.querySelectorAll('.script-status-message');
            messagesToRemove.forEach(msg => msg.remove());
        }

        // --- Logging ---
        function logMessage(message, type = 'info') {
            const logEntry = document.createElement('div');
            logEntry.textContent = message;
            
            if (type === 'success') {
                logEntry.classList.add('log-success', 'script-status-message');
            } else if (type === 'error') {
                 logEntry.classList.add('log-error');
                 if (message.startsWith('Script') && (message.includes('sikeresen lefutott') || message.includes('hibával fejeződött be'))) {
                    logEntry.classList.add('script-status-message');
                 }
            } else {
                logEntry.classList.add('log-info');
            }
            
            logsDiv.appendChild(logEntry);
            logsDiv.scrollTop = logsDiv.scrollHeight; // Auto-scroll
        }

        // Initial Load
        fetchApiKeys();
        fetchProjects();
    });
