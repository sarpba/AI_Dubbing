        const PROJECT_CONFIG = window.__PROJECT_CONFIG__ || {};
        const PROJECT_NAME = PROJECT_CONFIG.projectName || '';
        const DEFAULT_CHUNK_SIZE_BYTES = 99 * 1024 * 1024;
        let workflowPollInterval = null;
        let workflowLogInterval = null;
        let currentWorkflowJobId = null;
        let currentLogJobId = null;
        let workflowInfoDismissed = false;
        let workflowKeysModal = null;
        let workflowStepModal = null;
        let workflowParamsModal = null;
        let workflowSaveModal = null;
        let pendingWorkflowPayload = null;
        let pendingWorkflowContext = null;
        let availableScripts = [];
        let defaultWorkflow = [];
        let workflowSteps = [];
        let workflowAutoSaveEnabled = false;
        let workflowAutoSaveTimer = null;
        let workflowAutoSavePending = false;
        let workflowSaving = false;
        let workflowSuppressAutoSave = false;
        let workflowLastSavedSnapshot = null;
        let selectedStepIndex = null;
        let selectedStepType = null;
        let selectedWidgetConfig = null;
        let currentRunContext = null;
        let cycleState = null;
        let workflowTemplates = [];
        let currentTemplateId = null;
        let workflowInitialized = false;
        let workflowContextMenu = null;
        let workflowContextState = {
            targetIndex: null,
            targetRow: null
        };

        const workflowKeyFieldMap = {
            chatgpt: { wrapperId: 'workflowKeyFieldChatgpt', inputId: 'workflowKeyChatgpt', payloadKey: 'chatgpt_api_key' },
            deepl: { wrapperId: 'workflowKeyFieldDeepl', inputId: 'workflowKeyDeepl', payloadKey: 'deepl_api_key' },
            huggingface: { wrapperId: 'workflowKeyFieldHuggingface', inputId: 'workflowKeyHuggingface', payloadKey: 'huggingface_token' }
        };
        const SECRET_PARAM_NAMES = PROJECT_CONFIG.secretParamNames || [];
        const SECRET_VALUE_PLACEHOLDER = '***';
        const AUDIO_EXTENSIONS = new Set(PROJECT_CONFIG.audioExtensions || []);
        const VIDEO_EXTENSIONS = new Set(PROJECT_CONFIG.videoExtensions || []);
        const TEXT_PREVIEW_EXTENSIONS = new Set(['.json', '.txt', '.bak', '.log', '.srt']);
        const translateInline = (key, replacements = {}, fallback) => {
            if (window.translateText && typeof window.translateText === 'function') {
                return window.translateText(key, replacements, fallback);
            }
            if (typeof fallback === 'string') {
                return fallback;
            }
            return key;
        };
        const t = (key, replacements = {}, fallback) => translateInline(key, replacements, fallback);
        let videoPreviewModal = null;
        let jsonPreviewModal = null;

        const workflowWidgetApi = window.ProjectWorkflowWidgets || {};
        const workflowStateApi = window.ProjectWorkflowState || {};
        const workflowWidgets = workflowWidgetApi.createWorkflowWidgets ? workflowWidgetApi.createWorkflowWidgets(t) : [];

        function findWidgetById(widgetId) {
            return workflowWidgetApi.findWidgetById(workflowWidgets, widgetId);
        }

        function normalizeWorkflowStep(step) {
            return workflowWidgetApi.normalizeWorkflowStep(step, workflowWidgets);
        }

        function normalizeWorkflowStepList(steps) {
            return workflowWidgetApi.normalizeWorkflowStepList(steps, workflowWidgets);
        }

        function buildRunStepFromWorkflowStep(step) {
            return workflowWidgetApi.buildRunStepFromWorkflowStep(step, cloneObject);
        }

        function collectWorkflowSegment(startIndex = 0) {
            return workflowWidgetApi.collectWorkflowSegment(workflowSteps, cloneObject, startIndex);
        }

        function collectScriptStepsForRun(startIndex = 0) {
            return workflowWidgetApi.collectScriptStepsForRun(workflowSteps, cloneObject, startIndex);
        }

        function collectWorkflowState() {
            return workflowWidgetApi.collectWorkflowState(workflowSteps, cloneSteps);
        }

        function buildWorkflowSnapshot() {
            return workflowWidgetApi.buildWorkflowSnapshot(workflowSteps, currentTemplateId, cloneSteps);
        }

        function hasEnabledScriptStep(steps) {
            return workflowWidgetApi.hasEnabledScriptStep(steps);
        }

        function resetWorkflowAutoSave() {
            if (workflowAutoSaveTimer) {
                clearTimeout(workflowAutoSaveTimer);
                workflowAutoSaveTimer = null;
            }
            workflowAutoSavePending = false;
        }

        function markWorkflowDirty() {
            if (!workflowAutoSaveEnabled || workflowSuppressAutoSave) {
                return;
            }
            workflowAutoSavePending = true;
            scheduleWorkflowAutoSave();
        }

        function scheduleWorkflowAutoSave(delay = 500) {
            if (workflowAutoSaveTimer) {
                clearTimeout(workflowAutoSaveTimer);
            }
            workflowAutoSaveTimer = setTimeout(runWorkflowAutoSave, delay);
        }

        async function runWorkflowAutoSave() {
            return workflowStateApi.runWorkflowAutoSave({
                projectName: PROJECT_NAME,
                t,
                buildWorkflowSnapshot,
                hasEnabledScriptStep,
                scheduleWorkflowAutoSave,
                getWorkflowAutoSaveEnabled: () => workflowAutoSaveEnabled,
                getWorkflowSuppressAutoSave: () => workflowSuppressAutoSave,
                getWorkflowAutoSavePending: () => workflowAutoSavePending,
                getWorkflowSaving: () => workflowSaving,
                getWorkflowLastSavedSnapshot: () => workflowLastSavedSnapshot,
                setWorkflowAutoSaveTimer: value => { workflowAutoSaveTimer = value; },
                setWorkflowAutoSavePending: value => { workflowAutoSavePending = value; },
                setWorkflowSaving: value => { workflowSaving = value; },
                setWorkflowLastSavedSnapshot: value => { workflowLastSavedSnapshot = value; }
            });
        }

        async function persistProjectWorkflowState(payload) {
            return workflowStateApi.persistProjectWorkflowState(PROJECT_NAME, payload);
        }

        async function loadProjectWorkflowState() {
            return workflowStateApi.loadProjectWorkflowState(PROJECT_NAME, t);
        }

        async function loadDefaultWorkflowTemplate(templates) {
            return workflowStateApi.loadDefaultWorkflowTemplate(templates, t);
        }

        function collectScriptStepsBefore(index, requiredCount) {
            const collected = [];
            let count = 0;
            for (let i = index - 1; i >= 0 && count < requiredCount; i--) {
                const step = workflowSteps[i];
                if (!step || step.type === 'widget') {
                    continue;
                }
                if (step.enabled === false) {
                    continue;
                }
                collected.push(step);
                count++;
            }
            collected.reverse();
            return {
                steps: collected,
                collectedCount: count,
                requiredCount
            };
        }

        function validateStepCollection(stepCollection) {
            const errors = [];
            const missingSummary = [];
            let hasRunnableStep = false;
            (stepCollection || []).forEach(step => {
                if (!step || step.type === 'widget') {
                    return;
                }
                if (step.enabled === false) {
                    return;
                }
                hasRunnableStep = true;
                const script = findScriptById(step.script);
                if (!script) {
                    errors.push(t('workflow.errors.unknown_script', { name: step.script }));
                    return;
                }
                const missing = getMissingParams(step, script);
                if (missing.length) {
                    missingSummary.push({ script, missing });
                }
            });
            return {
                valid: hasRunnableStep && errors.length === 0 && missingSummary.length === 0,
                hasRunnableStep,
                errors,
                missingSummary
            };
        }

        function buildRunPayload(startIndex = 0, segmentInfo = null) {
            const segment = segmentInfo || collectWorkflowSegment(startIndex);
            return {
                steps: segment.steps.map(step => ({
                    script: step.script,
                    enabled: true,
                    halt_on_fail: step.halt_on_fail !== false,
                    params: cloneObject(step.params)
                })),
                template_id: currentTemplateId,
                workflow_state: collectWorkflowState()
            };
        }

        function findPreviousEnabledWidgetIndex(startIndex, widgetId) {
            for (let i = startIndex - 1; i >= 0; i--) {
                const step = workflowSteps[i];
                if (!step || step.type !== 'widget' || step.enabled === false) {
                    continue;
                }
                if (step.widget === widgetId) {
                    return i;
                }
            }
            return null;
        }

        function resolveTranslatedSplitLoopRestart(widgetIndex) {
            const reviewIndex = findPreviousEnabledWidgetIndex(widgetIndex, 'reviewContinue');
            if (reviewIndex === null) {
                return {
                    error: t('workflow.messages.segment_loop_review_missing')
                };
            }
            let restartIndex = null;
            for (let i = reviewIndex + 1; i < widgetIndex; i++) {
                const step = workflowSteps[i];
                if (!step || step.enabled === false) {
                    continue;
                }
                if (step.type === 'widget') {
                    return {
                        error: t('workflow.messages.segment_loop_order_invalid')
                    };
                }
                if (restartIndex === null) {
                    restartIndex = i;
                }
            }
            if (restartIndex === null) {
                return {
                    error: t('workflow.messages.segment_loop_restart_missing')
                };
            }
            return {
                reviewIndex,
                restartIndex
            };
        }

        async function fetchTranslatedSplitProgress() {
            const response = await fetch(`/api/translated-split-progress/${encodeURIComponent(PROJECT_NAME)}`, {
                cache: 'no-store'
            });
            const result = await response.json();
            if (!response.ok || !result.success || !result.progress) {
                throw new Error(result && result.error ? result.error : t('workflow.errors.segment_loop_status_failed'));
            }
            return result.progress;
        }

        function cloneSteps(steps) {
            const cloned = JSON.parse(JSON.stringify(steps || []));
            return normalizeWorkflowStepList(cloned);
        }

        function cloneObject(obj) {
            return JSON.parse(JSON.stringify(obj || {}));
        }

        function getScriptParamDefaults(script) {
            const defaults = new Map();
            if (!script) {
                return defaults;
            }
            (script.parameters || []).forEach(param => {
                if (!param || !param.name || defaults.has(param.name)) {
                    return;
                }
                if (Object.prototype.hasOwnProperty.call(param, 'default')) {
                    const value = param.default;
                    if (value !== undefined && value !== null) {
                        defaults.set(param.name, value);
                    }
                }
            });
            const raw = script.raw || {};
            ['required', 'optional'].forEach(key => {
                const list = raw[key];
                (list || []).forEach(param => {
                    if (!param || !param.name || defaults.has(param.name)) {
                        return;
                    }
                    const value = param.default;
                    if (value !== undefined && value !== null) {
                        defaults.set(param.name, value);
                    }
                });
            });
            return defaults;
        }

        function normalizeFlagValue(value) {
            if (value === undefined || value === null) {
                return undefined;
            }
            if (typeof value === 'string') {
                const trimmed = value.trim().toLowerCase();
                if (trimmed === 'true' || trimmed === '1') {
                    return true;
                }
                if (trimmed === 'false' || trimmed === '0') {
                    return false;
                }
                return trimmed.length > 0;
            }
            return Boolean(value);
        }

        function getFlagMode(param) {
            return (param && param.flag_mode) || 'standard';
        }

        function getParamLabel(param) {
            return (param && (param.ui_name || param.name)) || '';
        }

        function getDisplayFlagValue(param, rawValue, defaultValue) {
            const normalizedRaw = normalizeFlagValue(rawValue);
            const normalizedDefault = normalizeFlagValue(defaultValue);
            const effectiveRaw = normalizedRaw !== undefined ? normalizedRaw : normalizedDefault;
            if (effectiveRaw === undefined) {
                return false;
            }
            if (getFlagMode(param) === 'negative_only_negative') {
                return !effectiveRaw;
            }
            return effectiveRaw;
        }

        function getStoredFlagValue(param, displayValue) {
            if (getFlagMode(param) === 'negative_only_negative') {
                return !displayValue;
            }
            return displayValue;
        }

        function serializeDatasetBool(value) {
            if (value === undefined) {
                return 'none';
            }
            return value ? 'true' : 'false';
        }

        function parseDatasetBool(value) {
            if (value === 'true') {
                return true;
            }
            if (value === 'false') {
                return false;
            }
            return undefined;
        }

        function buildWorkdirUrl(relativePath) {
            const segments = [PROJECT_NAME];
            if (relativePath) {
                relativePath.split('/').forEach(segment => {
                    if (segment) {
                        segments.push(segment);
                    }
                });
            }
            const encoded = segments.map(part => encodeURIComponent(part));
            return `/workdir/${encoded.join('/')}`;
        }

        function getFileExtension(name) {
            if (!name || typeof name !== 'string') {
                return '';
            }
            const dotIndex = name.lastIndexOf('.');
            if (dotIndex === -1) {
                return '';
            }
            return name.substring(dotIndex).toLowerCase();
        }

        function createMetadataElement(entry) {
            const value = entry && entry.duration_display;
            if (!value) {
                return null;
            }
            const span = document.createElement('span');
            span.className = 'file-browser-meta small text-muted';
            span.textContent = value;
            return span;
        }

        function createFailedOriginalTextElement(entry) {
            const value = entry && entry.failed_original_text_display;
            if (!value) {
                return null;
            }
            const span = document.createElement('span');
            span.className = 'file-browser-note small text-muted';
            span.textContent = value;
            return span;
        }

        function createFileTreeList(entries) {
            const list = document.createElement('ul');
            list.className = 'file-browser-tree list-unstyled mb-0';
            (entries || []).forEach(entry => {
                const item = document.createElement('li');
                if (entry.type === 'directory') {
                    const details = document.createElement('details');
                    details.className = 'file-browser-directory';
                    details.dataset.path = entry.path || '';
                    if (entry.highlight_class) {
                        entry.highlight_class.split(/\s+/).forEach(cls => {
                            if (cls) {
                                details.classList.add(cls);
                            }
                        });
                        details.dataset.highlight = entry.highlight_class;
                    }

                    const summary = document.createElement('summary');
                    summary.className = 'file-browser-summary';
                    const label = document.createElement('span');
                    label.className = 'file-browser-label';
                    label.textContent = entry.name || t('files.unnamed_folder', {}, '(unnamed folder)');
                    const actions = document.createElement('div');
                    actions.className = 'file-browser-actions';
                    const uploadBtn = document.createElement('button');
                    uploadBtn.type = 'button';
                    uploadBtn.className = 'btn btn-sm btn-outline-secondary file-upload-trigger';
                    uploadBtn.dataset.path = entry.path || '';
                    uploadBtn.title = t('file_browser.upload_title');
                    uploadBtn.textContent = t('file_browser.upload_button');
                    actions.append(uploadBtn);

                    const clearBtn = document.createElement('button');
                    clearBtn.type = 'button';
                    clearBtn.className = 'btn btn-sm btn-outline-danger file-directory-clear-btn';
                    clearBtn.dataset.path = entry.path || '';
                    clearBtn.title = t('file_browser.clear_title');
                    clearBtn.textContent = t('file_browser.clear_button');
                    actions.append(clearBtn);

                    summary.append(label);
                    summary.append(actions);
                    details.append(summary);

                    const children = document.createElement('div');
                    children.className = 'file-browser-children';
                    if (entry.children && entry.children.length) {
                        children.appendChild(createFileTreeList(entry.children));
                    }
                    details.append(children);

                    item.appendChild(details);
                } else {
                    const fileRow = document.createElement('div');
                    fileRow.className = 'file-browser-item';
                    fileRow.dataset.filePath = entry.path || '';
                    fileRow.dataset.fileName = entry.name || '';
                    const extension = getFileExtension(entry.name || '');
                    fileRow.dataset.fileExtension = extension;
                    if (entry.enable_failed_move) {
                        fileRow.dataset.enableFailedMove = 'true';
                    }

                    const link = document.createElement('a');
                    link.className = 'file-browser-link';
                    link.href = buildWorkdirUrl(entry.path || '');
                    link.target = '_blank';
                    link.rel = 'noopener';
                    link.textContent = entry.name || t('files.unnamed_file', {}, '(unnamed file)');

                    fileRow.appendChild(link);

                    const actions = document.createElement('div');
                    actions.className = 'file-browser-actions';

                    const deleteBtn = document.createElement('button');
                    deleteBtn.type = 'button';
                    deleteBtn.className = 'btn btn-sm btn-outline-danger file-delete-btn';
                    deleteBtn.title = t('file_browser.delete_title');
                    deleteBtn.textContent = t('file_browser.delete_button');
                    actions.append(deleteBtn);

                    if (entry.enable_failed_move && extension === '.wav') {
                        const moveBtn = document.createElement('button');
                        moveBtn.type = 'button';
                        moveBtn.className = 'btn btn-sm btn-outline-primary file-move-to-translated-btn';
                        moveBtn.title = t('file_browser.move_title');
                        moveBtn.textContent = t('file_browser.move_button');
                        actions.append(moveBtn);
                    }

                    fileRow.appendChild(actions);
                    const metadataElement = createMetadataElement(entry);
                    if (metadataElement) {
                        fileRow.insertBefore(metadataElement, actions);
                    }
                    const noteElement = createFailedOriginalTextElement(entry);
                    if (noteElement) {
                        fileRow.insertBefore(noteElement, actions);
                    }
                    if (entry.duration_from_name !== undefined && entry.duration_from_name !== null) {
                        fileRow.dataset.durationFromName = String(entry.duration_from_name);
                    }
                    if (entry.duration_actual !== undefined && entry.duration_actual !== null) {
                        fileRow.dataset.durationActual = String(entry.duration_actual);
                    }
                    if (entry.duration_display) {
                        fileRow.dataset.durationDisplay = entry.duration_display;
                    }
                    if (entry.failed_original_text !== undefined && entry.failed_original_text !== null) {
                        fileRow.dataset.failedOriginalText = entry.failed_original_text;
                    }
                    if (entry.failed_original_text_display) {
                        fileRow.dataset.failedOriginalTextDisplay = entry.failed_original_text_display;
                    }
                    item.appendChild(fileRow);
                }
                list.appendChild(item);
            });
            return list;
        }

        function updateDirectoryHighlight(detailsElement, highlightClass) {
            if (!detailsElement) {
                return;
            }
            const previous = detailsElement.dataset.highlight || '';
            if (previous) {
                previous.split(/\s+/).forEach(cls => {
                    if (cls) {
                        detailsElement.classList.remove(cls);
                    }
                });
            }
            if (highlightClass && highlightClass.trim()) {
                highlightClass.split(/\s+/).forEach(cls => {
                    if (cls) {
                        detailsElement.classList.add(cls);
                    }
                });
                detailsElement.dataset.highlight = highlightClass;
            } else {
                delete detailsElement.dataset.highlight;
            }
        }

        function updateFailedLegend(visible) {
            const legend = document.getElementById('failedGenerationLegend');
            if (!legend) {
                return;
            }
            if (visible) {
                legend.classList.remove('d-none');
            } else {
                legend.classList.add('d-none');
            }
        }

        async function refreshDirectory(detailsElement) {
            const childrenContainer = detailsElement.querySelector('.file-browser-children');
            if (!childrenContainer) {
                return;
            }
            const relativePath = detailsElement.dataset.path || '';
            childrenContainer.innerHTML = `<div class="small text-muted">${t('files.loading', {}, 'Loading...')}</div>`;
            try {
                const url = `/api/project-tree/${encodeURIComponent(PROJECT_NAME)}?path=${encodeURIComponent(relativePath)}`;
                const response = await fetch(url);
                const result = await response.json();
                if (!response.ok || !result.success) {
                    throw new Error(result.error || t('files.load_failed'));
                }
                const entries = result.entries || [];
                updateDirectoryHighlight(detailsElement, result.current_highlight || '');
                if (Object.prototype.hasOwnProperty.call(result, 'has_highlights')) {
                    updateFailedLegend(Boolean(result.has_highlights));
                }
                childrenContainer.innerHTML = '';
                if (!entries.length) {
                    const empty = document.createElement('div');
                    empty.className = 'small text-muted';
                    empty.textContent = t('files.empty_folder', {}, 'Folder is empty.');
                    childrenContainer.appendChild(empty);
                } else {
                    childrenContainer.appendChild(createFileTreeList(entries));
                }
            } catch (error) {
                console.error('Directory refresh failed:', error);
                childrenContainer.innerHTML = `<div class="text-danger small">${t('files.load_failed', {}, 'Unable to load folder contents.')}</div>`;
            }
        }

        function cssEscape(value) {
            if (window.CSS && typeof window.CSS.escape === 'function') {
                return window.CSS.escape(value);
            }
            return (value || '').replace(/[^a-zA-Z0-9_\-]/g, (char) => `\\${char}`);
        }

        async function reloadFileBrowser(targetPath = '') {
            const browserRoot = document.getElementById('projectFileBrowser');
            if (!browserRoot) {
                return;
            }
            const normalized = (targetPath || '').replace(/\\/g, '/').replace(/^\/+|\/+$/g, '');

            if (!normalized) {
                browserRoot.innerHTML = `<div class="small text-muted">${t('files.refreshing', {}, 'Refreshing...')}</div>`;
                try {
                    const url = `/api/project-tree/${encodeURIComponent(PROJECT_NAME)}?path=`;
                    const response = await fetch(url);
                    const result = await response.json();
                    if (!response.ok || !result.success) {
                        throw new Error(result.error || t('files.load_failed'));
                    }
                    const entries = result.entries || [];
                    browserRoot.innerHTML = '';
                    if (!entries.length) {
                        const empty = document.createElement('div');
                        empty.className = 'small text-muted';
                        empty.textContent = t('files.empty_folder', {}, 'Folder is empty.');
                        browserRoot.appendChild(empty);
                    } else {
                        browserRoot.appendChild(createFileTreeList(entries));
                    }
                    if (Object.prototype.hasOwnProperty.call(result, 'has_highlights')) {
                        updateFailedLegend(Boolean(result.has_highlights));
                    }
                } catch (error) {
                    console.error('Root refresh failed:', error);
                    browserRoot.innerHTML = `<div class="text-danger small">${t('files.refresh_failed', {}, 'Unable to refresh file list.')}</div>`;
                }
                return;
            }

            const selector = `details.file-browser-directory[data-path="${cssEscape(normalized)}"]`;
            const details = browserRoot.querySelector(selector);
            if (details) {
                await refreshDirectory(details);
                details.setAttribute('open', '');
            } else {
                await reloadFileBrowser('');
            }
        }

        function openUploadDialog(targetPath = '') {
            const uploadInput = document.getElementById('projectFileUploadInput');
            if (!uploadInput) {
                return;
            }
            uploadInput.dataset.targetPath = targetPath || '';
            uploadInput.value = '';
            uploadInput.click();
        }

        async function uploadFileToPath(targetPath, file) {
            const formData = new FormData();
            formData.append('projectName', PROJECT_NAME);
            formData.append('targetPath', targetPath || '');
            formData.append('file', file);

            const response = await fetch('/api/project-file/upload', {
                method: 'POST',
                body: formData
            });

            let result = {};
            try {
                result = await response.json();
            } catch (error) {
                // ignore JSON parse error, handled below
            }

            if (!response.ok || !result.success) {
                throw new Error((result && result.error) || t('js.errors.upload_failed', {}, 'Failed to upload file.'));
            }
            return result;
        }

        async function deleteProjectFile(filePath, { ignoreMissing = false } = {}) {
            const response = await fetch(`/api/project-file/${encodeURIComponent(PROJECT_NAME)}`, {
                method: 'DELETE',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ path: filePath })
            });

            if (response.status === 404 && ignoreMissing) {
                return { success: true, skipped: true };
            }

            let result = {};
            try {
                result = await response.json();
            } catch (error) {
                // ignore
            }

            if (!response.ok || !result.success) {
                throw new Error((result && result.error) || t('js.errors.delete_failed', {}, 'Failed to delete file.'));
            }
            return result;
        }

        async function clearProjectDirectory(directoryPath) {
            const response = await fetch(`/api/project-directory/${encodeURIComponent(PROJECT_NAME)}`, {
                method: 'DELETE',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ path: directoryPath })
            });

            let result = {};
            try {
                result = await response.json();
            } catch (error) {
                // ignore
            }

            if (!response.ok || !result.success) {
                throw new Error((result && result.error) || t('js.errors.clear_directory_failed', {}, 'Failed to clear directory contents.'));
            }
            return result;
        }

        async function moveFailedGenerationFileRequest(filePath) {
            const response = await fetch('/api/project-file/move-failed', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    projectName: PROJECT_NAME,
                    sourcePath: filePath
                })
            });

            let result = {};
            try {
                result = await response.json();
            } catch (error) {
                // ignore
            }

            if (!response.ok || !result.success) {
                throw new Error((result && result.error) || t('js.errors.move_failed', {}, 'Failed to move file.'));
            }
            return result;
        }

        async function uploadFilesToTtsRequest(targetPath, files) {
            const orderedKeys = ['model', 'vocab', 'config'];
            let lastResponse = null;
            let processedFiles = 0;
            const totalFiles = orderedKeys.filter(key => files[key]).length || 1;
            for (const key of orderedKeys) {
                const fileObj = files[key];
                if (!fileObj) {
                    continue;
                }
                lastResponse = await uploadSingleTtsFileChunk(targetPath, key, fileObj, processedFiles, totalFiles);
                processedFiles += 1;
            }
            return lastResponse || { success: true };
        }

        function createTtsChunkUploadId(prefix = 'tts') {
            if (window.crypto && window.crypto.randomUUID) {
                return `${prefix}-${crypto.randomUUID()}`;
            }
            return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
        }

        function getTtsChunkSizeBytes(fileSize) {
            if (!fileSize) {
                return DEFAULT_CHUNK_SIZE_BYTES;
            }
            return Math.min(DEFAULT_CHUNK_SIZE_BYTES, fileSize);
        }

        function formatBytes(bytes) {
            if (!bytes) {
                return '0 B';
            }
            const units = ['B', 'KB', 'MB', 'GB', 'TB'];
            const unitIndex = Math.floor(Math.log(bytes) / Math.log(1024));
            const value = bytes / Math.pow(1024, unitIndex);
            return `${value.toFixed(2)} ${units[unitIndex]}`;
        }

        async function uploadSingleTtsFileChunk(targetPath, fileKey, fileObj, filesDone, totalFiles) {
            const chunkSizeBytes = getTtsChunkSizeBytes(fileObj.size);
            const totalChunks = Math.max(1, Math.ceil(fileObj.size / chunkSizeBytes));
            const uploadId = createTtsChunkUploadId(fileKey);
            let uploadedBytes = 0;
            if (window.__updateTtsProgress) {
                window.__updateTtsProgress(0, t('tts.upload_prepare', { fileKey }, `Preparing upload (${fileKey})...`));
            }

            for (let index = 0; index < totalChunks; index++) {
                const start = index * chunkSizeBytes;
                const end = Math.min(fileObj.size, start + chunkSizeBytes);
                const chunk = fileObj.slice(start, end);
                const formData = new FormData();
                if (targetPath) {
                    formData.append('targetPath', targetPath);
                }
                formData.append('chunkUploadId', uploadId);
                formData.append('chunkFileKey', fileKey);
                formData.append('chunkIndex', index);
                formData.append('totalChunks', totalChunks);
                formData.append('chunkSize', chunkSizeBytes);
                formData.append('fileName', fileObj.name);
                formData.append('file', chunk, fileObj.name);

                const response = await fetch('/api/tts-upload', {
                    method: 'POST',
                    body: formData
                });
                let result = {};
                try {
                    result = await response.json();
                } catch (error) {
                    // ignore
                }
                if (!response.ok || !result.success) {
                    throw new Error((result && result.error) || t('js.errors.tts_upload_failed', {}, 'Failed to upload file to the TTS directory.'));
                }

                uploadedBytes = end;
                const chunkPercent = Math.round((uploadedBytes / fileObj.size) * 100);
                const overallPercent = Math.round(((filesDone + (uploadedBytes / fileObj.size)) / totalFiles) * 100);
                if (window.__updateTtsProgress) {
                    window.__updateTtsProgress(
                        overallPercent,
                        t('tts.chunk_status', {
                            fileKey,
                            current: index + 1,
                            total: totalChunks,
                            chunkPercent,
                            uploaded: formatBytes(uploadedBytes),
                            totalBytes: formatBytes(fileObj.size)
                        }, `File (${fileKey}) chunk ${index + 1}/${totalChunks} – ${chunkPercent}% (${formatBytes(uploadedBytes)} / ${formatBytes(fileObj.size)})`)
                    );
                }

                if (result.completed) {
                    if (window.__updateTtsProgress) {
                        window.__updateTtsProgress(
                            Math.round(((filesDone + 1) / totalFiles) * 100),
                            t('tts.upload_complete', { fileKey }, `File (${fileKey}) upload complete.`),
                            false,
                            filesDone + 1 === totalFiles
                        );
                    }
                    return result;
                }
            }

            if (window.__updateTtsProgress) {
                window.__updateTtsProgress(
                    Math.round(((filesDone + 1) / totalFiles) * 100),
                    t('tts.upload_complete', { fileKey }, `File (${fileKey}) upload complete.`),
                    false,
                    filesDone + 1 === totalFiles
                );
            }
            return { success: true };
        }

        function extractDownloadFilename(dispositionHeader) {
            if (!dispositionHeader || typeof dispositionHeader !== 'string') {
                return null;
            }
            const utf8Match = dispositionHeader.match(/filename\*=(?:UTF-8'')?([^;]+)/i);
            if (utf8Match && utf8Match[1]) {
                const rawValue = utf8Match[1].trim().replace(/^\"|\"$/g, '');
                try {
                    return decodeURIComponent(rawValue.replace(/\+/g, '%20'));
                } catch (error) {
                    return rawValue;
                }
            }
            const asciiMatch = dispositionHeader.match(/filename=\"?([^\";]+)\"?/i);
            if (asciiMatch && asciiMatch[1]) {
                return asciiMatch[1].trim();
            }
            return null;
        }

        async function requestProjectBackup() {
            if (!PROJECT_NAME) {
                alert(t('backup.project_missing'));
                return;
            }
            const backupBtn = document.getElementById('projectBackupBtn');
            if (backupBtn) {
                backupBtn.dataset.originalLabel = backupBtn.dataset.originalLabel || backupBtn.textContent;
                backupBtn.disabled = true;
                backupBtn.textContent = t('backup.working');
            }
            try {
                const response = await fetch('/api/project-backup', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        projectName: PROJECT_NAME
                    })
                });
                const contentType = response.headers.get('content-type') || '';
                if (!response.ok) {
                    let errorMessage = t('backup.create_failed');
                    if (contentType.includes('application/json')) {
                        try {
                            const result = await response.json();
                            if (result && result.error) {
                                errorMessage = result.error;
                            }
                        } catch (error) {
                            // ignore parse error
                        }
                    }
                    throw new Error(errorMessage);
                }

                const blob = await response.blob();
                const disposition = response.headers.get('content-disposition') || '';
                const fallbackName = `${PROJECT_NAME || 'project'}_backup.tar.gz`;
                const filename = extractDownloadFilename(disposition) || fallbackName;
                const urlCreator = window.URL || window.webkitURL;
                const downloadUrl = urlCreator.createObjectURL(blob);
                const tempLink = document.createElement('a');
                tempLink.href = downloadUrl;
                tempLink.download = filename;
                document.body.appendChild(tempLink);
                tempLink.click();
                document.body.removeChild(tempLink);
                urlCreator.revokeObjectURL(downloadUrl);
            } catch (error) {
                console.error('Projekt backup hiba:', error);
                alert(error.message || t('backup.create_failed'));
            } finally {
                if (backupBtn) {
                    backupBtn.disabled = false;
                    const original = backupBtn.dataset.originalLabel || t('files.download_backup');
                    backupBtn.textContent = original;
                }
            }
        }

        function showVideoPreview(fileName, fileUrl) {
            const modalElement = document.getElementById('videoPreviewModal');
            if (!modalElement) {
                window.open(fileUrl, '_blank');
                return;
            }
            if (!videoPreviewModal) {
                videoPreviewModal = new bootstrap.Modal(modalElement);
            }

            const titleElement = document.getElementById('videoPreviewTitle');
            if (titleElement) {
                titleElement.textContent = fileName || t('video_modal.default_title', {}, 'Play video');
            }

            const videoElement = document.getElementById('videoPreviewPlayer');
            if (!videoElement) {
                window.open(fileUrl, '_blank');
                return;
            }

            videoElement.pause();
            videoElement.src = fileUrl;
            videoElement.load();

            const handleShown = () => {
                videoElement.play().catch(() => {
                    // Autoplay might be blocked; user can start manually.
                });
            };

            if (modalElement.classList.contains('show')) {
                handleShown();
            } else {
                modalElement.addEventListener('shown.bs.modal', handleShown, { once: true });
            }

            videoPreviewModal.show();
        }

        async function showTextPreview(fileName, fileUrl, extension) {
            const modalElement = document.getElementById('jsonPreviewModal');
            if (!modalElement) {
                window.open(fileUrl, '_blank');
                return;
            }
            if (!jsonPreviewModal) {
                jsonPreviewModal = new bootstrap.Modal(modalElement);
            }
            const titleElement = document.getElementById('jsonPreviewTitle');
            if (titleElement) {
                titleElement.textContent = fileName || t('json_modal.default_title', {}, 'File preview');
            }
            const contentElement = document.getElementById('jsonPreviewContent');
            if (contentElement) {
                contentElement.textContent = t('json_modal.loading', {}, 'Loading...');
            }
            jsonPreviewModal.show();

            try {
                const response = await fetch(fileUrl, {
                    headers: {
                        'Accept': 'application/json, text/plain;q=0.9, */*;q=0.8'
                    }
                });
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}`);
                }
                const text = await response.text();
                let formatted = text;
                if (extension === '.json') {
                    try {
                        const parsed = JSON.parse(text);
                        formatted = JSON.stringify(parsed, null, 2);
                    } catch (error) {
                        // fallback to raw text
                    }
                }
                if (contentElement) {
                    contentElement.textContent = formatted || t('json_modal.empty', {}, 'Empty file.');
                }
            } catch (error) {
                console.error('Text preview failed:', error);
                if (contentElement) {
                    contentElement.textContent = t('json_modal.load_error', { error: error.message || error }, `Failed to load file.\n${error.message || error}`);
                }
            }
        }

        function initPreviewModals() {
            const audioModalElement = document.getElementById('audioPreviewModal');
            if (audioModalElement && (!window.AudioTrimmer || typeof AudioTrimmer.init !== 'function')) {
                audioModalElement.addEventListener('hidden.bs.modal', () => {
                    const titleElement = document.getElementById('audioPreviewTitle');
                    if (titleElement) {
                        titleElement.textContent = '';
                    }
                });
            }

            const videoModalElement = document.getElementById('videoPreviewModal');
            if (videoModalElement) {
                videoPreviewModal = new bootstrap.Modal(videoModalElement);
                videoModalElement.addEventListener('hidden.bs.modal', () => {
                    const videoElement = document.getElementById('videoPreviewPlayer');
                    if (videoElement) {
                        videoElement.pause();
                        videoElement.removeAttribute('src');
                        videoElement.load();
                    }
                    const titleElement = document.getElementById('videoPreviewTitle');
                    if (titleElement) {
                        titleElement.textContent = '';
                    }
                });
            }

            const jsonModalElement = document.getElementById('jsonPreviewModal');
            if (jsonModalElement) {
                jsonPreviewModal = new bootstrap.Modal(jsonModalElement);
                jsonModalElement.addEventListener('hidden.bs.modal', () => {
                    const contentElement = document.getElementById('jsonPreviewContent');
                    if (contentElement) {
                        contentElement.textContent = '';
                    }
                    const titleElement = document.getElementById('jsonPreviewTitle');
                    if (titleElement) {
                        titleElement.textContent = '';
                    }
                });
            }
        }

        function initProjectFileActions() {
            const uploadInput = document.getElementById('projectFileUploadInput');
            if (uploadInput) {
                uploadInput.addEventListener('change', async (event) => {
                    const files = Array.from(event.target.files || []);
                    if (!files.length) {
                        return;
                    }
                    const targetPath = uploadInput.dataset.targetPath || '';
                    try {
                        await uploadFileToPath(targetPath, files[0]);
                        alert(t('notifications.file_upload_success'));
                        await reloadFileBrowser(targetPath);
                    } catch (error) {
                        alert(error.message || t('notifications.file_upload_failed'));
                    } finally {
                        uploadInput.value = '';
                        delete uploadInput.dataset.targetPath;
                    }
                });
            }

            const rootUploadBtn = document.getElementById('rootUploadBtn');
            if (rootUploadBtn) {
                rootUploadBtn.addEventListener('click', () => openUploadDialog(''));
            }

            const refreshBtn = document.getElementById('refreshFileBrowserBtn');
            if (refreshBtn) {
                refreshBtn.addEventListener('click', () => {
                    reloadFileBrowser('');
                });
            }

            const backupBtn = document.getElementById('projectBackupBtn');
            if (backupBtn) {
                backupBtn.addEventListener('click', () => {
                    requestProjectBackup();
                });
            }
        }

        function initTtsControls() {
            const uploadTargetInput = document.getElementById('ttsUploadTargetInput');
            const modelInput = document.getElementById('ttsUploadModelInput');
            const vocabInput = document.getElementById('ttsUploadVocabInput');
            const configInput = document.getElementById('ttsUploadConfigInput');
            const uploadBtn = document.getElementById('ttsUploadBtn');
            const progressBar = document.getElementById('ttsUploadProgressBar');
            const progressText = document.getElementById('ttsUploadProgressText');

            if (uploadBtn && modelInput && vocabInput && configInput) {
                uploadBtn.addEventListener('click', async () => {
                    const modelFile = modelInput.files && modelInput.files[0];
                    const vocabFile = vocabInput.files && vocabInput.files[0];
                    const configFile = configInput.files && configInput.files[0];
                    if (!modelFile || !vocabFile || !configFile) {
                        alert(t('tts.validation.all_files'));
                        return;
                    }
                    if (!vocabFile.name.toLowerCase().endsWith('.txt')) {
                        alert(t('tts.validation.vocab_extension'));
                        return;
                    }
                    if (!configFile.name.toLowerCase().endsWith('.json')) {
                        alert(t('tts.validation.config_extension'));
                        return;
                    }
                    const targetPath = uploadTargetInput ? uploadTargetInput.value.trim() : '';
                    const targetError = document.getElementById('ttsUploadTargetError');
                    if (!targetPath) {
                        if (targetError) {
                            targetError.classList.remove('d-none');
                        }
                        alert(t('tts.validation.target_required'));
                        return;
                    }
                    if (targetError) {
                        targetError.classList.add('d-none');
                    }
                    const originalLabel = uploadBtn.dataset.originalLabel || uploadBtn.textContent;
                    uploadBtn.dataset.originalLabel = originalLabel;
                    uploadBtn.disabled = true;
                    modelInput.disabled = true;
                    vocabInput.disabled = true;
                    configInput.disabled = true;
                    uploadBtn.textContent = t('tts.button_uploading', {}, 'Uploading...');
                    resetTtsProgress();
                    try {
                        const result = await uploadFilesToTtsRequest(targetPath, {
                            model: modelFile,
                            vocab: vocabFile,
                            config: configFile
                        });
                        alert(result.message || t('notifications.files_upload_success'));
                        modelInput.value = '';
                        vocabInput.value = '';
                        configInput.value = '';
                        setTimeout(() => location.reload(), 600);
                    } catch (error) {
                        console.error('TTS feltöltési hiba:', error);
                        alert(error.message || t('notifications.files_upload_failed'));
                    } finally {
                        resetTtsProgress();
                        uploadBtn.disabled = false;
                        modelInput.disabled = false;
                        vocabInput.disabled = false;
                        configInput.disabled = false;
                        uploadBtn.textContent = originalLabel;
                    }
                });
            }

            function resetTtsProgress() {
                if (progressBar) {
                    progressBar.style.width = '0%';
                    progressBar.classList.remove('bg-success', 'bg-danger');
                    progressBar.textContent = '0%';
                }
                if (progressText) {
                    progressText.textContent = '';
                }
            }

            function updateTtsProgress(percent, text, isError = false, isComplete = false) {
                if (progressBar) {
                    progressBar.style.width = `${percent}%`;
                    progressBar.textContent = `${percent}%`;
                    progressBar.classList.toggle('bg-success', isComplete && !isError);
                    progressBar.classList.toggle('bg-danger', isError);
                }
                if (progressText) {
                    progressText.textContent = text || '';
                }
            }

            window.__updateTtsProgress = updateTtsProgress;
        }

        function initFileBrowser() {
            const browserRoot = document.getElementById('projectFileBrowser');
            if (!browserRoot) {
                return;
            }

            browserRoot.addEventListener('click', async (event) => {
                const uploadButton = event.target.closest('.file-upload-trigger');
                if (uploadButton && browserRoot.contains(uploadButton)) {
                    event.preventDefault();
                    event.stopPropagation();
                    openUploadDialog(uploadButton.dataset.path || '');
                    return;
                }

                const clearDirectoryButton = event.target.closest('.file-directory-clear-btn');
                if (clearDirectoryButton && browserRoot.contains(clearDirectoryButton)) {
                    event.preventDefault();
                    event.stopPropagation();
                    const detailsElement = clearDirectoryButton.closest('details.file-browser-directory');
                    if (!detailsElement) {
                        return;
                    }
                    const directoryPath = clearDirectoryButton.dataset.path || detailsElement.dataset.path || '';
                    if (!directoryPath) {
                        alert(t('directories.path_missing'));
                        return;
                    }
                    const labelElement = detailsElement.querySelector('.file-browser-label');
                    const directoryName = (labelElement && labelElement.textContent) ? labelElement.textContent.trim() : directoryPath;
                    const confirmed = confirm(t('directories.clear_confirm', { name: directoryName }));
                    if (!confirmed) {
                        return;
                    }
                    clearDirectoryButton.disabled = true;
                    try {
                        await clearProjectDirectory(directoryPath);
                        alert(t('directories.clear_success'));
                        await refreshDirectory(detailsElement);
                        detailsElement.setAttribute('open', '');
                    } catch (error) {
                        alert(error.message || t('directories.clear_error'));
                    } finally {
                        clearDirectoryButton.disabled = false;
                    }
                    return;
                }

                const deleteButton = event.target.closest('.file-delete-btn');
                if (deleteButton && browserRoot.contains(deleteButton)) {
                    event.preventDefault();
                    event.stopPropagation();
                    const fileRow = deleteButton.closest('.file-browser-item');
                    if (!fileRow) {
                        return;
                    }
                    const filePath = fileRow.dataset.filePath || '';
                    if (!filePath) {
                        return;
                    }
                    const fileName = fileRow.dataset.fileName || filePath.split('/').pop();
                    const confirmed = confirm(t('directories.delete_confirm', { name: fileName }));
                    if (!confirmed) {
                        return;
                    }
                    try {
                        await deleteProjectFile(filePath);
                        const parentPath = filePath.includes('/') ? filePath.substring(0, filePath.lastIndexOf('/')) : '';
                        alert(t('directories.delete_success'));
                        await reloadFileBrowser(parentPath);
                    } catch (error) {
                        alert(error.message || t('directories.delete_error'));
                    }
                    return;
                }

                const moveButton = event.target.closest('.file-move-to-translated-btn');
                if (moveButton && browserRoot.contains(moveButton)) {
                    event.preventDefault();
                    event.stopPropagation();
                    const fileRow = moveButton.closest('.file-browser-item');
                    if (!fileRow) {
                        return;
                    }
                    const filePath = fileRow.dataset.filePath || '';
                    if (!filePath) {
                        return;
                    }
                    const fileName = fileRow.dataset.fileName || filePath.split('/').pop();
                    const confirmed = confirm(t('directories.move_confirm', { name: fileName }));
                    if (!confirmed) {
                        return;
                    }
                    moveButton.disabled = true;
                    try {
                        await moveFailedGenerationFileRequest(filePath);
                        const parentPath = filePath.includes('/') ? filePath.substring(0, filePath.lastIndexOf('/')) : '';
                        alert(t('directories.move_success'));
                        await reloadFileBrowser(parentPath);
                    } catch (error) {
                        alert(error.message || t('directories.move_error'));
                    } finally {
                        moveButton.disabled = false;
                    }
                    return;
                }

                const fileLink = event.target.closest('.file-browser-link');
                if (fileLink && browserRoot.contains(fileLink)) {
                    if (event.metaKey || event.ctrlKey || event.shiftKey || event.button !== 0) {
                        return;
                    }
                    const fileRow = fileLink.closest('.file-browser-item');
                    if (!fileRow) {
                        return;
                    }
                    const fileName = fileRow.dataset.fileName || '';
                    const filePath = fileRow.dataset.filePath || '';
                    const extension = (fileRow.dataset.fileExtension || '').toLowerCase();

                    if (AUDIO_EXTENSIONS.has(extension)) {
                        event.preventDefault();
                        event.stopPropagation();
                        if (window.AudioTrimmer && typeof AudioTrimmer.showPreview === 'function') {
                            AudioTrimmer.showPreview(fileName, buildWorkdirUrl(filePath), filePath);
                        } else {
                            window.open(buildWorkdirUrl(filePath), '_blank');
                        }
                        return;
                    }

                    if (VIDEO_EXTENSIONS.has(extension)) {
                        event.preventDefault();
                        event.stopPropagation();
                        showVideoPreview(fileName, buildWorkdirUrl(filePath));
                        return;
                    }

                    if (TEXT_PREVIEW_EXTENSIONS.has(extension)) {
                        event.preventDefault();
                        event.stopPropagation();
                        await showTextPreview(fileName, buildWorkdirUrl(filePath), extension);
                        return;
                    }

                    return;
                }

                const summary = event.target.closest('summary.file-browser-summary');
                if (!summary || !browserRoot.contains(summary)) {
                    return;
                }

                const details = summary.closest('details.file-browser-directory');
                if (!details) {
                    return;
                }

                if (event.target.closest('.file-browser-actions')) {
                    return;
                }

                event.preventDefault();

                if (details.dataset.loading === 'true') {
                    return;
                }

                const isOpen = details.hasAttribute('open');
                if (isOpen) {
                    details.removeAttribute('open');
                    return;
                }

                details.dataset.loading = 'true';
                try {
                    await refreshDirectory(details);
                    details.setAttribute('open', '');
                } finally {
                    delete details.dataset.loading;
                }
            });
        }

        function findScriptById(scriptId) {
            if (!scriptId) return null;
            return availableScripts.find(entry => entry.id === scriptId) || null;
        }

        function getMissingParams(step, script) {
            if (!script || !script.parameters) {
                return [];
            }
            const defaults = getScriptParamDefaults(script);
            return script.parameters
                .filter(param => param.required && param.type !== 'flag' && !param.autofill)
                .filter(param => {
                    const value = step.params ? step.params[param.name] : undefined;
                    let hasValue = value !== undefined && value !== null;
                    if (hasValue && typeof value === 'string' && value.trim() === '') {
                        hasValue = false;
                    }
                    if (hasValue) {
                        return false;
                    }
                    if (defaults.has(param.name)) {
                        const defaultValue = defaults.get(param.name);
                        if (defaultValue !== undefined && defaultValue !== null) {
                            if (!(typeof defaultValue === 'string' && defaultValue.trim() === '')) {
                                return false;
                            }
                        }
                    }
                    return true;
                })
                .map(param => param.name);
        }

        function describeStepParameters(step, script, missing) {
            if (missing && missing.length) {
                return t('workflow.messages.missing_required', { missing: missing.join(', ') }, `Missing required parameters: ${missing.join(', ')}`);
            }
            const summary = [];
            const defaults = getScriptParamDefaults(script);
            if (script && script.parameters) {
                script.parameters.forEach(param => {
                    const value = step.params ? step.params[param.name] : undefined;
                    const defaultValue = defaults.has(param.name) ? defaults.get(param.name) : undefined;
                    const hasValue = value !== undefined && value !== null && !(typeof value === 'string' && value.trim() === '');
                    if (param.type === 'flag') {
                        const resolvedValue = hasValue ? value : defaultValue;
                        if (resolvedValue !== undefined) {
                            const displayValue = getDisplayFlagValue(param, resolvedValue, defaultValue);
                            summary.push(`${getParamLabel(param)}=${displayValue ? 'on' : 'off'}`);
                        }
                    } else if (param.secret) {
                        if (hasValue) {
                            summary.push(`${getParamLabel(param)}=${SECRET_VALUE_PLACEHOLDER}`);
                        }
                    } else if (hasValue) {
                        summary.push(`${getParamLabel(param)}=${value}`);
                    }
                });
            }
            Object.entries(step.params || {}).forEach(([key, value]) => {
                if (!script || !(script.parameters || []).some(param => param.name === key)) {
                    if (SECRET_PARAM_NAMES.includes(key)) {
                        summary.push(`${key}=${SECRET_VALUE_PLACEHOLDER}`);
                    } else if (value !== undefined && value !== null && !(typeof value === 'string' && value.trim() === '')) {
                        summary.push(`${key}=${value}`);
                    }
                }
            });
            return summary.length ? summary.join(', ') : t('workflow.messages.no_parameters');
        }

        function describeWidgetParameters(widget, step) {
            if (!widget || !Array.isArray(widget.parameters) || !widget.parameters.length) {
                return '';
            }
            const params = step && step.params ? step.params : {};
            const summary = [];
            widget.parameters.forEach(param => {
                if (!param || !param.name) {
                    return;
                }
                const label = param.label || param.name;
                const hasOwnValue = Object.prototype.hasOwnProperty.call(params, param.name);
                let value = hasOwnValue ? params[param.name] : param.default;
                if (value === undefined || value === null || (typeof value === 'string' && value.trim() === '')) {
                    return;
                }
                summary.push(`${label}: ${value}`);
            });
            return summary.join(' • ');
        }

        function formatScriptDirectoryName(script, step) {
            const rawPath = (script && script.script) || (step && step.script) || '';
            if (typeof rawPath === 'string' && rawPath) {
                const normalized = rawPath.replace(/\\/g, '/');
                const segments = normalized.split('/').filter(Boolean);
                if (segments.length >= 2) {
                    const parentDir = segments[segments.length - 2];
                    if (parentDir) {
                        return parentDir.replace(/_/g, ' ');
                    }
                } else if (segments.length === 1) {
                    return segments[0].replace(/_/g, ' ');
                }
            }
            if (script && script.display_name) {
                return script.display_name;
            }
            if (step && typeof step.script === 'string') {
                return step.script.replace(/_/g, ' ');
            }
            return t('workflow.errors.unknown_script_short', {}, 'Unknown script');
        }

        function updateMissingBadge() {
            const badge = document.getElementById('workflowMissingBadge');
            if (!badge) return;
            const hasMissing = workflowSteps.some(step =>
                step.type !== 'widget' &&
                step.enabled !== false &&
                getMissingParams(step, findScriptById(step.script)).length
            );
            badge.classList.toggle('d-none', !hasMissing);
        }

        function findTemplateById(templateId) {
            if (!templateId) {
                return null;
            }
            return workflowTemplates.find(item => item.id === templateId) || null;
        }

        function populateWorkflowTemplateSelect(selectedId) {
            const select = document.getElementById('workflowTemplateSelect');
            if (!select) {
                return;
            }
            const previousValue = selectedId !== undefined ? selectedId : select.value;
            select.innerHTML = '';

            const placeholder = document.createElement('option');
            placeholder.value = '';
            placeholder.textContent = t('workflow.templates.placeholder', {}, 'Select a template...');
            select.append(placeholder);

            workflowTemplates.forEach(template => {
                const option = document.createElement('option');
                option.value = template.id;
                option.textContent = template.name || template.id;
                select.append(option);
            });

            const targetValue = selectedId !== undefined ? selectedId : previousValue;
            if (targetValue && Array.from(select.options).some(option => option.value === targetValue)) {
                select.value = targetValue;
                currentTemplateId = targetValue;
            } else {
                select.value = '';
                if (selectedId !== undefined) {
                    currentTemplateId = null;
                }
            }
        }

        async function refreshWorkflowTemplates(selectedId = currentTemplateId) {
            try {
                const response = await fetch('/api/workflow-templates');
                const result = await response.json();
                if (!response.ok || !result.success) {
                    throw new Error(result.error || t('workflow.templates.list_failed'));
                }
                workflowTemplates = result.templates || [];
                populateWorkflowTemplateSelect(selectedId);
            } catch (error) {
                console.error('Workflow sablon lista frissítési hiba:', error);
            }
        }

        async function loadWorkflowTemplateById(templateId, showMessage = true) {
            if (!templateId) {
                alert(t('workflow.templates.select_prompt'));
                return;
            }
            try {
                const response = await fetch(`/api/workflow-template/${encodeURIComponent(templateId)}`);
                const result = await response.json();
                if (!response.ok || !result.success) {
                    throw new Error(result.error || t('workflow.templates.load_failed'));
                }
                const template = result.template || {};
                currentTemplateId = template.id || templateId;
                workflowSteps = cloneSteps(template.steps || []);
                defaultWorkflow = cloneSteps(template.steps || []);
                markWorkflowDirty();
                populateWorkflowTemplateSelect(currentTemplateId);
                renderWorkflowSteps();
                if (showMessage) {
                    updateInfoBox('info', t('workflow.templates.loaded', { name: template.name || currentTemplateId }));
                }
            } catch (error) {
                console.error('Workflow sablon betöltési hiba:', error);
                alert(t('workflow.templates.load_error', { error: error.message }));
            }
        }

        function showSaveWorkflowModal() {
            const modalElement = document.getElementById('workflowSaveModal');
            if (!modalElement) {
                return;
            }
            const nameInput = document.getElementById('workflowSaveName');
            const descriptionInput = document.getElementById('workflowSaveDescription');
            const overwriteCheckbox = document.getElementById('workflowSaveOverwrite');
            const overwriteInfo = document.getElementById('workflowSaveOverwriteInfo');
            const errorBox = document.getElementById('workflowSaveError');

            const currentTemplate = findTemplateById(currentTemplateId);
            if (nameInput) {
                nameInput.value = currentTemplate ? (currentTemplate.name || '') : '';
            }
            if (descriptionInput) {
                descriptionInput.value = currentTemplate ? (currentTemplate.description || '') : '';
            }
            if (overwriteCheckbox) {
                overwriteCheckbox.checked = false;
                overwriteCheckbox.disabled = !currentTemplateId;
            }
            if (overwriteInfo) {
                if (currentTemplate) {
                    overwriteInfo.textContent = t('workflow.templates.overwrite_current', {
                        name: currentTemplate.name || '',
                        id: currentTemplate.id || ''
                    });
                    overwriteInfo.classList.remove('d-none');
                } else {
                    overwriteInfo.classList.add('d-none');
                }
            }
            if (errorBox) {
                errorBox.classList.add('d-none');
                errorBox.textContent = '';
            }
            if (!workflowSaveModal) {
                workflowSaveModal = new bootstrap.Modal(modalElement);
            }
            workflowSaveModal.show();
        }

        async function saveWorkflowTemplateFromModal() {
            const nameInput = document.getElementById('workflowSaveName');
            const descriptionInput = document.getElementById('workflowSaveDescription');
            const overwriteCheckbox = document.getElementById('workflowSaveOverwrite');
            const errorBox = document.getElementById('workflowSaveError');

            const name = nameInput ? nameInput.value.trim() : '';
            const description = descriptionInput ? descriptionInput.value.trim() : '';
            const overwrite = overwriteCheckbox ? overwriteCheckbox.checked : false;

            if (errorBox) {
                errorBox.classList.add('d-none');
                errorBox.textContent = '';
            }

            if (overwrite && !currentTemplateId) {
                if (errorBox) {
                    errorBox.textContent = t('workflow.templates.none_selected_overwrite');
                    errorBox.classList.remove('d-none');
                }
                return;
            }

            if (!name && !overwrite) {
                if (errorBox) {
                    errorBox.textContent = t('workflow.templates.name_required');
                    errorBox.classList.remove('d-none');
                }
                return;
            }

            const workflowState = collectWorkflowState();
            const body = {
                name,
                description,
                steps: workflowState,
                overwrite
            };
            if (overwrite && currentTemplateId) {
                body.template_id = currentTemplateId;
            }

            try {
                const response = await fetch('/api/save-workflow-template', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(body)
                });
                const result = await response.json();
                if (!response.ok || !result.success) {
                throw new Error(result.error || t('workflow.templates.save_failed'));
                }
                if (workflowSaveModal) {
                    workflowSaveModal.hide();
                }
                const savedTemplate = result.template || {};
                currentTemplateId = savedTemplate.id || currentTemplateId;
                defaultWorkflow = cloneSteps(workflowState);
                await refreshWorkflowTemplates(currentTemplateId);
                populateWorkflowTemplateSelect(currentTemplateId);
                updateInfoBox('success', t('workflow.templates.saved', { name: savedTemplate.name || currentTemplateId }));
            } catch (error) {
                console.error('Workflow sablon mentési hiba:', error);
                if (errorBox) {
                    errorBox.textContent = error.message;
                    errorBox.classList.remove('d-none');
                }
            }
        }

        function initWorkflowButtons() {
            const templateSelect = document.getElementById('workflowTemplateSelect');
            if (templateSelect) {
                templateSelect.addEventListener('change', event => {
                    const value = event.target.value || '';
                    if (!value) {
                        currentTemplateId = null;
                        return;
                    }
                    if (value === currentTemplateId) {
                        return;
                    }
                    currentTemplateId = value;
                    loadWorkflowTemplateById(currentTemplateId);
                });
            }

            const saveTemplateBtn = document.getElementById('saveWorkflowBtn');
            if (saveTemplateBtn) {
                saveTemplateBtn.addEventListener('click', showSaveWorkflowModal);
            }

            const resetBtn = document.getElementById('resetWorkflowBtn');
            if (resetBtn) {
                resetBtn.addEventListener('click', () => {
                    if (!defaultWorkflow.length) {
                        alert(t('workflow.templates.default_missing'));
                        return;
                    }
                    if (!confirm(t('workflow.templates.reset_confirm'))) {
                        return;
                    }
                    workflowSteps = cloneSteps(defaultWorkflow);
                    markWorkflowDirty();
                    renderWorkflowSteps();
                    updateInfoBox('info', t('workflow.templates.default_loaded'));
                });
            }

            const startBtn = document.getElementById('startWorkflowBtn');
            if (startBtn) {
                startBtn.addEventListener('click', handleStartWorkflow);
            }

            const stopBtn = document.getElementById('stopWorkflowBtn');
            if (stopBtn) {
                stopBtn.addEventListener('click', stopWorkflow);
            }

            const searchInput = document.getElementById('workflowStepSearch');
            if (searchInput) {
                searchInput.addEventListener('input', event => populateWorkflowStepList(event.target.value || ''));
            }
        }

        function initWorkflowContextMenu() {
            if (workflowContextMenu) {
                return;
            }
            workflowContextMenu = document.createElement('div');
            workflowContextMenu.id = 'workflowContextMenu';
            workflowContextMenu.className = 'workflow-context-menu';
            workflowContextMenu.addEventListener('contextmenu', event => event.preventDefault());
            document.body.append(workflowContextMenu);

            document.addEventListener('click', event => {
                if (!workflowContextMenu || !workflowContextMenu.classList.contains('show')) {
                    return;
                }
                if (workflowContextMenu.contains(event.target)) {
                    return;
                }
                hideWorkflowContextMenu();
            });

            document.addEventListener('keydown', event => {
                if (event.key === 'Escape') {
                    hideWorkflowContextMenu();
                }
            });

            window.addEventListener('resize', hideWorkflowContextMenu);
            document.addEventListener('scroll', hideWorkflowContextMenu, true);

            const stepsTable = document.getElementById('workflowStepsTable');
            if (stepsTable) {
                stepsTable.addEventListener('contextmenu', event => {
                    const row = event.target.closest('tr[data-step-index]');
                    const rawIndex = row ? parseInt(row.dataset.stepIndex, 10) : NaN;
                    const targetIndex = Number.isNaN(rawIndex) ? null : rawIndex;
                    showWorkflowContextMenu(event, row || null, targetIndex);
                });
            }
        }

        function hideWorkflowContextMenu() {
            if (!workflowContextMenu) {
                return;
            }
            workflowContextMenu.classList.remove('show');
            workflowContextMenu.innerHTML = '';
            if (workflowContextState.targetRow) {
                workflowContextState.targetRow.classList.remove('workflow-context-target');
            }
            workflowContextState.targetIndex = null;
            workflowContextState.targetRow = null;
        }

        function showWorkflowContextMenu(event, rowElement, targetIndex) {
            event.preventDefault();
            if (!workflowContextMenu) {
                return;
            }

            hideWorkflowContextMenu();

            workflowContextState.targetIndex = typeof targetIndex === 'number' ? targetIndex : null;
            workflowContextState.targetRow = rowElement || null;
            if (workflowContextState.targetRow) {
                workflowContextState.targetRow.classList.add('workflow-context-target');
            }

            const positions = [
                { key: 'above', label: t('workflow.context_menu.insert_above') },
                { key: 'below', label: t('workflow.context_menu.insert_below') }
            ];

            positions.forEach(position => {
                const item = document.createElement('div');
                item.className = 'workflow-context-item has-submenu';
                item.textContent = position.label;
                const submenu = buildWorkflowInsertSubmenu(position.key);
                item.append(submenu);
                workflowContextMenu.append(item);
            });

            const pageX = event.pageX;
            const pageY = event.pageY;
            workflowContextMenu.style.left = `${pageX}px`;
            workflowContextMenu.style.top = `${pageY}px`;
            workflowContextMenu.classList.add('show');

            requestAnimationFrame(() => {
                const rect = workflowContextMenu.getBoundingClientRect();
                let adjustedLeft = pageX;
                let adjustedTop = pageY;
                const viewportRight = window.scrollX + window.innerWidth;
                const viewportBottom = window.scrollY + window.innerHeight;
                if (rect.right > viewportRight) {
                    adjustedLeft = viewportRight - rect.width - 8;
                }
                if (rect.bottom > viewportBottom) {
                    adjustedTop = viewportBottom - rect.height - 8;
                }
                workflowContextMenu.style.left = `${Math.max(window.scrollX + 8, adjustedLeft)}px`;
                workflowContextMenu.style.top = `${Math.max(window.scrollY + 8, adjustedTop)}px`;
            });
        }

        function buildWorkflowInsertSubmenu(position) {
            const submenu = document.createElement('div');
            submenu.className = 'workflow-context-submenu';
            const scriptTree = buildScriptDirectoryTree();
            const hasDirectoryContent = scriptTree.children.size > 0;
            const hasRootScripts = scriptTree.scripts.length > 0;
            const hasAnyScripts = hasDirectoryContent || hasRootScripts;

            if (workflowWidgets.length) {
                const widgetItem = document.createElement('div');
                widgetItem.className = 'workflow-context-item has-submenu';
                widgetItem.textContent = 'Widgetek';
                const widgetSubmenu = document.createElement('div');
                widgetSubmenu.className = 'workflow-context-submenu';

                workflowWidgets.forEach(widget => {
                    const widgetOption = document.createElement('div');
                    widgetOption.className = 'workflow-context-item';
                    widgetOption.textContent = widget.name;
                    widgetOption.addEventListener('click', event => {
                        event.preventDefault();
                        event.stopPropagation();
                        insertWorkflowWidget(widget.id, position);
                    });
                    widgetSubmenu.append(widgetOption);
                });

                widgetItem.append(widgetSubmenu);
                submenu.append(widgetItem);
            }

            if (workflowWidgets.length && hasAnyScripts) {
                const divider = document.createElement('div');
                divider.className = 'workflow-context-divider';
                submenu.append(divider);
            }

            if (hasRootScripts) {
                const rootItem = document.createElement('div');
                rootItem.className = 'workflow-context-item has-submenu';
                rootItem.textContent = t('workflow.file_tree.root_label');
                const rootSubmenu = document.createElement('div');
                rootSubmenu.className = 'workflow-context-submenu';
                appendScriptsToMenu(rootSubmenu, scriptTree.scripts, position);
                rootItem.append(rootSubmenu);
                submenu.append(rootItem);
            }

            appendDirectoryTreeToMenu(submenu, scriptTree.children, position);

            if (!workflowWidgets.length && !hasAnyScripts) {
                const empty = document.createElement('div');
                empty.className = 'workflow-context-item disabled';
                empty.textContent = t('workflow.file_tree.empty');
                submenu.append(empty);
            }

            return submenu;
        }

        function buildScriptDirectoryTree() {
            const createNode = name => ({
                name,
                children: new Map(),
                scripts: []
            });

            const root = createNode('');

            availableScripts.forEach(script => {
                const normalizedPath = String(script.script || '').replace(/\\/g, '/');
                let relativePath = normalizedPath;
                if (relativePath.startsWith('scripts/')) {
                    relativePath = relativePath.slice('scripts/'.length);
                }

                const pathParts = relativePath.split('/').filter(Boolean);
                const directories = pathParts.slice(0, -1);
                const fileName = pathParts[pathParts.length - 1] || (script.display_name || script.script || script.id);

                let current = root;
                directories.forEach(segment => {
                    if (!current.children.has(segment)) {
                        current.children.set(segment, createNode(segment));
                    }
                    current = current.children.get(segment);
                });

                current.scripts.push({
                    data: script,
                    fileName
                });
            });

            return root;
        }

        function appendDirectoryTreeToMenu(parentMenu, childrenMap, position) {
            const sortedChildren = Array
                .from(childrenMap.values())
                .sort((childA, childB) => childA.name.localeCompare(childB.name));

            sortedChildren.forEach(child => {
                const hasNestedDirectories = child.children.size > 0;
                const scriptCount = child.scripts.length;

                if (!hasNestedDirectories && scriptCount === 1) {
                    const singleItem = document.createElement('div');
                    singleItem.className = 'workflow-context-item';
                    singleItem.textContent = child.name;
                    singleItem.addEventListener('click', event => {
                        event.preventDefault();
                        event.stopPropagation();
                        insertWorkflowScript(child.scripts[0].data.id, position);
                    });
                    parentMenu.append(singleItem);
                    return;
                }

                const dirItem = document.createElement('div');
                dirItem.className = 'workflow-context-item has-submenu';
                dirItem.textContent = child.name;
                const childSubmenu = document.createElement('div');
                childSubmenu.className = 'workflow-context-submenu';

                appendDirectoryTreeToMenu(childSubmenu, child.children, position);
                appendScriptsToMenu(childSubmenu, child.scripts, position);

                dirItem.append(childSubmenu);
                parentMenu.append(dirItem);
            });
        }

        function appendScriptsToMenu(menuElement, scripts, position) {
            const sortedScripts = scripts
                .slice()
                .sort((entryA, entryB) => {
                    const labelA = getScriptMenuLabel(entryA).toLowerCase();
                    const labelB = getScriptMenuLabel(entryB).toLowerCase();
                    return labelA.localeCompare(labelB);
                });

            sortedScripts.forEach(entry => {
                const scriptOption = document.createElement('div');
                scriptOption.className = 'workflow-context-item';
                scriptOption.textContent = getScriptMenuLabel(entry);
                scriptOption.addEventListener('click', event => {
                    event.preventDefault();
                    event.stopPropagation();
                    insertWorkflowScript(entry.data.id, position);
                });
                menuElement.append(scriptOption);
            });
        }

        function getScriptMenuLabel(entry) {
            if (!entry || !entry.data) {
                return '';
            }
            return entry.data.display_name || entry.fileName || entry.data.script || entry.data.id || '';
        }

        function calculateInsertIndex(position) {
            if (typeof workflowContextState.targetIndex !== 'number' || Number.isNaN(workflowContextState.targetIndex)) {
                return workflowSteps.length;
            }
            if (position === 'above') {
                return Math.max(0, Math.min(workflowContextState.targetIndex, workflowSteps.length));
            }
            return Math.max(0, Math.min(workflowContextState.targetIndex + 1, workflowSteps.length));
        }

        function insertWorkflowScript(scriptId, position) {
            const insertIndex = calculateInsertIndex(position);
            hideWorkflowContextMenu();
            addWorkflowStep(scriptId, insertIndex);
        }

        function insertWorkflowWidget(widgetId, position) {
            const insertIndex = calculateInsertIndex(position);
            hideWorkflowContextMenu();
            addWorkflowWidget(widgetId, insertIndex);
        }

        function initWorkflowModals() {
            const stepModalElement = document.getElementById('workflowStepModal');
            if (stepModalElement) {
                workflowStepModal = new bootstrap.Modal(stepModalElement);
                stepModalElement.addEventListener('shown.bs.modal', () => {
                    const searchInput = document.getElementById('workflowStepSearch');
                    if (searchInput) {
                        searchInput.value = '';
                        searchInput.focus();
                        populateWorkflowStepList('');
                    }
                });
            }

            const paramsModalElement = document.getElementById('workflowParamsModal');
            if (paramsModalElement) {
                workflowParamsModal = new bootstrap.Modal(paramsModalElement);
                paramsModalElement.addEventListener('hidden.bs.modal', () => {
                    selectedStepIndex = null;
                    selectedStepType = null;
                    selectedWidgetConfig = null;
                    const alertBox = document.getElementById('workflowParamsAlert');
                    if (alertBox) {
                        alertBox.classList.add('d-none');
                        alertBox.textContent = '';
                    }
                });
                const saveBtn = document.getElementById('workflowParamsSaveBtn');
                if (saveBtn) {
                    saveBtn.addEventListener('click', saveWorkflowParams);
                }
            }

            const saveModalElement = document.getElementById('workflowSaveModal');
            if (saveModalElement) {
                workflowSaveModal = new bootstrap.Modal(saveModalElement);
                const saveConfirmBtn = document.getElementById('workflowSaveConfirmBtn');
                if (saveConfirmBtn) {
                    saveConfirmBtn.addEventListener('click', saveWorkflowTemplateFromModal);
                }
            }
        }

        function initWorkflowKeyModal() {
            const modalElement = document.getElementById('workflowKeysModal');
            if (modalElement) {
                workflowKeysModal = new bootstrap.Modal(modalElement);
                modalElement.addEventListener('hidden.bs.modal', () => {
                    resetWorkflowKeyModal();
                    pendingWorkflowPayload = null;
                });
                const submitBtn = document.getElementById('workflowKeysSubmitBtn');
                if (submitBtn) {
                    submitBtn.addEventListener('click', saveWorkflowKeys);
                }
            }
        }

        async function loadWorkflowOptions(projectName) {
            const statusText = document.getElementById('workflowStatusText');
            const startButton = document.getElementById('startWorkflowBtn');
            resetWorkflowAutoSave();
            workflowAutoSaveEnabled = false;
            workflowSuppressAutoSave = true;
            try {
                const response = await fetch(`/api/workflow-options/${encodeURIComponent(projectName)}`);
                const result = await response.json();
                if (!response.ok || !result.success) {
                    throw new Error(result.error || t('workflow.errors.unknown_error'));
                }

                availableScripts = result.scripts || [];
                workflowTemplates = result.templates || [];
                const defaults = result.defaults || {};

                const [savedState, defaultTemplateSteps] = await Promise.all([
                    loadProjectWorkflowState(),
                    loadDefaultWorkflowTemplate(workflowTemplates)
                ]);

                if (Array.isArray(defaultTemplateSteps)) {
                    defaultWorkflow = cloneSteps(defaultTemplateSteps);
                } else {
                    defaultWorkflow = cloneSteps(defaults.workflow || []);
                }

                let initialTemplateId = defaults.selected_template || currentTemplateId || (workflowTemplates.length ? workflowTemplates[0].id : null);
                let initialSteps;
                if (savedState && Array.isArray(savedState.steps)) {
                    initialSteps = cloneSteps(savedState.steps);
                    if (Object.prototype.hasOwnProperty.call(savedState, 'template_id')) {
                        initialTemplateId = savedState.template_id;
                    }
                } else {
                    initialSteps = cloneSteps(defaultWorkflow);
                }

                workflowSteps = initialSteps;
                workflowInitialized = true;
                currentTemplateId = initialTemplateId;
                workflowSuppressAutoSave = false;

                populateWorkflowTemplateSelect(currentTemplateId);

                renderWorkflowSteps();
                renderWorkflowStatus(result.latest_job);

                if (result.latest_job && ['queued', 'running', 'cancelling'].includes(result.latest_job.status)) {
                    currentWorkflowJobId = result.latest_job.job_id;
                    startWorkflowPolling(result.latest_job.job_id, true);
                }

                const hasScriptSteps = workflowSteps.some(step => step.type !== 'widget');
                const hasEnabledScripts = workflowSteps.some(step => step.type !== 'widget' && step.enabled !== false);
                const runnableSegment = collectScriptStepsForRun(0);
                if (statusText) {
                    if (!hasScriptSteps) {
                        statusText.textContent = t('workflow.status_texts.add_steps');
                    } else if (!hasEnabledScripts) {
                        statusText.textContent = t('workflow.status_texts.enable_steps');
                    } else if (!runnableSegment.length) {
                        statusText.textContent = t('workflow.status_texts.no_steps_before_widget');
                    } else {
                        statusText.textContent = t('workflow.status_texts.ready');
                    }
                }

                if (startButton) {
                    startButton.disabled = !runnableSegment.length;
                }

                workflowLastSavedSnapshot = JSON.stringify(buildWorkflowSnapshot());
                workflowAutoSaveEnabled = true;
            } catch (error) {
                console.error('Hiba a workflow opciók betöltésekor:', error);
                if (statusText) {
                    statusText.textContent = t('workflow.status_texts.error', { error: error.message }, `Hiba: ${error.message}`);
                }
                if (startButton) {
                    startButton.disabled = true;
                }
                updateInfoBox('danger', t('workflow.messages.options_load_failed', { error: error.message }));
            } finally {
                workflowSuppressAutoSave = false;
            }
        }

        function renderWorkflowSteps() {
            const tbody = document.getElementById('workflowStepsBody');
            if (!tbody) {
                return;
            }
            tbody.innerHTML = '';
            if (!workflowSteps.length) {
                const row = document.createElement('tr');
                row.className = 'text-muted';
                const cell = document.createElement('td');
                cell.colSpan = 5;
                cell.textContent = t('workflow.table_empty');
                row.append(cell);
                tbody.append(row);
                updateMissingBadge();
                const startButtonEmpty = document.getElementById('startWorkflowBtn');
                if (startButtonEmpty) {
                    startButtonEmpty.disabled = true;
                }
                return;
            }

            workflowSteps.forEach((step, index) => {
                normalizeWorkflowStep(step);
                const isWidget = step.type === 'widget';
                const script = !isWidget ? findScriptById(step.script) : null;
                const widget = isWidget ? findWidgetById(step.widget) : null;
                const row = document.createElement('tr');
                row.classList.add('workflow-row');
                row.dataset.stepIndex = String(index);
                if (isWidget) {
                    row.classList.add('workflow-row-widget');
                }
                if (step.enabled === false) {
                    row.classList.add('workflow-row-disabled', 'text-muted');
                }
                const missing = (!isWidget && step.enabled !== false) ? getMissingParams(step, script) : [];
                if (missing.length) {
                    row.classList.add('workflow-row-missing');
                }

                const enabledCell = document.createElement('td');
                const enabledWrapper = document.createElement('div');
                enabledWrapper.className = 'form-check form-switch';
                const enabledCheckbox = document.createElement('input');
                enabledCheckbox.type = 'checkbox';
                enabledCheckbox.className = 'form-check-input';
                enabledCheckbox.checked = step.enabled !== false;
                enabledCheckbox.addEventListener('change', () => {
                    step.enabled = enabledCheckbox.checked;
                    markWorkflowDirty();
                    renderWorkflowSteps();
                });
                enabledWrapper.append(enabledCheckbox);
                enabledCell.append(enabledWrapper);
                row.append(enabledCell);

                const infoCell = document.createElement('td');
                if (isWidget) {
                    const title = document.createElement('div');
                    title.className = 'fw-semibold';
                    title.textContent = widget ? widget.name : t('workflow.labels_extra.special_widget');
                    infoCell.append(title);

                    if (widget && widget.description) {
                        const desc = document.createElement('div');
                        desc.className = 'small text-muted';
                        desc.textContent = widget.description;
                        infoCell.append(desc);
                    }

                    const buttonRow = document.createElement('div');
                    buttonRow.className = 'd-flex flex-wrap gap-2 mt-2';

                    const reviewBtn = document.createElement('button');
                    reviewBtn.type = 'button';
                    reviewBtn.className = 'btn btn-sm btn-outline-info';
                    reviewBtn.textContent = widget && widget.reviewLabel ? widget.reviewLabel : 'Review';
                    reviewBtn.disabled = step.enabled === false;
                    reviewBtn.addEventListener('click', () => {
                        const url = `/review/${encodeURIComponent(PROJECT_NAME)}`;
                        window.open(url, '_blank', 'noopener');
                    });

                    const continueBtn = document.createElement('button');
                    continueBtn.type = 'button';
                    continueBtn.className = 'btn btn-sm btn-success';
                    continueBtn.textContent = widget && widget.continueLabel ? widget.continueLabel : 'Continue';
                    const hasRemainingScripts = collectScriptStepsForRun(index + 1).length > 0;
                    continueBtn.disabled = step.enabled === false || !hasRemainingScripts;
                    if (!hasRemainingScripts) {
                        continueBtn.title = t('workflow.messages.no_more_steps');
                    }
                    continueBtn.addEventListener('click', () => handleWidgetContinue(index));

                    buttonRow.append(reviewBtn, continueBtn);
                    infoCell.append(buttonRow);

                    if (widget && Array.isArray(widget.parameters) && widget.parameters.length) {
                        const summary = describeWidgetParameters(widget, step);
                        if (summary) {
                            const summaryDiv = document.createElement('div');
                            summaryDiv.className = 'small text-muted mt-2';
                            summaryDiv.textContent = summary;
                            infoCell.append(summaryDiv);
                        }
                    }
                } else {
                    const titleWrap = document.createElement('div');
                    const titleRow = document.createElement('div');
                    titleRow.className = 'd-flex align-items-center gap-2 flex-wrap';
                    const title = document.createElement('div');
                    title.className = 'fw-semibold mb-0';
                    title.textContent = formatScriptDirectoryName(script, step);
                    titleRow.append(title);
                    if (script && script.api) {
                        const apiBadge = document.createElement('span');
                        apiBadge.className = 'badge bg-warning text-dark script-api-badge';
                        apiBadge.textContent = `${String(script.api).toUpperCase()} API`;
                        titleRow.append(apiBadge);
                    }
                    titleWrap.append(titleRow);

                    if (script && script.description) {
                        const desc = document.createElement('div');
                        desc.className = 'small text-muted';
                        desc.textContent = script.description;
                        titleWrap.append(desc);
                    }

                    infoCell.append(titleWrap);

                    const summary = document.createElement('div');
                    summary.className = 'small mt-1';
                    summary.textContent = describeStepParameters(step, script, missing);
                    infoCell.append(summary);

                    if (script && script.api) {
                        const apiInfoLine = document.createElement('div');
                        apiInfoLine.className = 'small mt-1 script-api-highlight';
                        const apiUsageLabel = t('workflow.labels_extra.api_usage');
                        apiInfoLine.textContent = `${apiUsageLabel}: ${String(script.api).toUpperCase()}`;
                        infoCell.append(apiInfoLine);
                    }

                    if (script && script.required_keys && script.required_keys.length) {
                        const keyLine = document.createElement('div');
                        keyLine.className = 'small mt-1';
                        script.required_keys.forEach(key => {
                            const badge = document.createElement('span');
                            badge.className = 'badge bg-secondary me-1';
                            badge.textContent = key;
                            keyLine.append(badge);
                        });
                        infoCell.append(keyLine);
                    }
                }
                row.append(infoCell);

                const envCell = document.createElement('td');
                if (isWidget) {
                    const widgetBadge = document.createElement('span');
                    widgetBadge.className = 'badge workflow-env-badge';
                    widgetBadge.textContent = t('workflow.labels_extra.widget_badge', {}, 'Widget');
                    envCell.append(widgetBadge);
                } else if (script && script.environment) {
                    const envBadge = document.createElement('span');
                    envBadge.className = 'badge workflow-env-badge';
                    envBadge.textContent = script.environment;
                    envCell.append(envBadge);
                } else {
                    envCell.innerHTML = '&mdash;';
                }
                row.append(envCell);

                const haltCell = document.createElement('td');
                if (isWidget) {
                    haltCell.innerHTML = '&mdash;';
                } else {
                    const haltWrapper = document.createElement('div');
                    haltWrapper.className = 'form-check form-switch';
                    const haltCheckbox = document.createElement('input');
                    haltCheckbox.type = 'checkbox';
                    haltCheckbox.className = 'form-check-input';
                    haltCheckbox.checked = step.halt_on_fail !== false;
                    haltCheckbox.addEventListener('change', () => {
                        step.halt_on_fail = haltCheckbox.checked;
                        markWorkflowDirty();
                        renderWorkflowSteps();
                    });
                    const haltLabel = document.createElement('label');
                    haltLabel.className = 'form-check-label small';
                    haltLabel.textContent = haltCheckbox.checked
                        ? t('workflow.labels_extra.halt_on_fail')
                        : t('workflow.labels_extra.continue_on_fail');
                    haltWrapper.append(haltCheckbox, haltLabel);
                    haltCell.append(haltWrapper);
                }
                row.append(haltCell);

                const actionsCell = document.createElement('td');
                actionsCell.className = 'text-end';
                if (!isWidget) {
                    const editBtn = document.createElement('button');
                    editBtn.type = 'button';
                    editBtn.className = 'btn btn-sm btn-outline-primary me-2';
                    editBtn.textContent = t('workflow.buttons_extra.parameters');
                    editBtn.addEventListener('click', () => openWorkflowParams(index));
                    actionsCell.append(editBtn);
                } else if (widget && Array.isArray(widget.parameters) && widget.parameters.length) {
                    const editBtn = document.createElement('button');
                    editBtn.type = 'button';
                    editBtn.className = 'btn btn-sm btn-outline-primary me-2';
                    editBtn.textContent = t('workflow.buttons_extra.parameters');
                    editBtn.addEventListener('click', () => openWorkflowWidgetParams(index));
                    actionsCell.append(editBtn);
                }

                const upBtn = document.createElement('button');
                upBtn.type = 'button';
                upBtn.className = 'btn btn-sm btn-outline-secondary me-1';
                upBtn.innerHTML = '&uarr;';
                upBtn.disabled = index === 0;
                upBtn.addEventListener('click', () => moveWorkflowStep(index, -1));

                const downBtn = document.createElement('button');
                downBtn.type = 'button';
                downBtn.className = 'btn btn-sm btn-outline-secondary me-1';
                downBtn.innerHTML = '&darr;';
                downBtn.disabled = index === workflowSteps.length - 1;
                downBtn.addEventListener('click', () => moveWorkflowStep(index, 1));

                const removeBtn = document.createElement('button');
                removeBtn.type = 'button';
                removeBtn.className = 'btn btn-sm btn-outline-danger';
                removeBtn.innerHTML = '&times;';
                removeBtn.addEventListener('click', () => removeWorkflowStep(index));

                actionsCell.append(upBtn, downBtn, removeBtn);
                row.append(actionsCell);

                tbody.append(row);
            });

            if (cycleState) {
                const widgetIndex = cycleState.widgetIndex;
                const currentIteration = cycleState.activeIteration !== null
                    ? cycleState.activeIteration
                    : Math.max(0, cycleState.nextIteration - 1);
                updateCycleDisplay(widgetIndex, currentIteration, cycleState.totalIterations);
            }

            updateMissingBadge();

            const startButton = document.getElementById('startWorkflowBtn');
            if (startButton) {
                const runnableSegment = collectScriptStepsForRun(0);
                startButton.disabled = runnableSegment.length === 0;
            }

            const statusText = document.getElementById('workflowStatusText');
            if (statusText && !currentWorkflowJobId) {
                const hasAnyScript = workflowSteps.some(step => step.type !== 'widget');
                const hasEnabledScripts = workflowSteps.some(step => step.type !== 'widget' && step.enabled !== false);
                const runnableSegment = collectScriptStepsForRun(0);
                if (!hasAnyScript) {
                    statusText.textContent = t('workflow.status_texts.add_steps');
                } else if (!hasEnabledScripts) {
                    statusText.textContent = t('workflow.status_texts.enable_steps');
                } else if (!runnableSegment.length) {
                    statusText.textContent = t('workflow.status_texts.no_steps_before_widget');
                } else {
                    statusText.textContent = t('workflow.status_texts.ready');
                }
            }
        }

        function openWorkflowStepPicker() {
            populateWorkflowStepList('');
            if (workflowStepModal) {
                workflowStepModal.show();
            }
        }

        function populateWorkflowStepList(filterText) {
            const list = document.getElementById('workflowStepList');
            if (!list) {
                return;
            }
            list.innerHTML = '';
            const term = (filterText || '').toLowerCase();
            const scripts = availableScripts
                .slice()
                .sort((a, b) => (a.display_name || a.script).localeCompare(b.display_name || b.script));
            const filtered = term
                ? scripts.filter(script =>
                    (script.display_name && script.display_name.toLowerCase().includes(term)) ||
                    (script.script && script.script.toLowerCase().includes(term)))
                : scripts;

            if (!filtered.length) {
                const empty = document.createElement('div');
                empty.className = 'list-group-item text-muted';
                empty.textContent = t('workflow.search.no_results');
                list.append(empty);
                return;
            }

            const normalizeScriptPath = (path) => (path || '').replace(/\\/g, '/');

            const createScriptListItem = (script) => {
                const item = document.createElement('div');
                item.className = 'list-group-item ps-4';
                const header = document.createElement('div');
                header.className = 'd-flex justify-content-between align-items-start gap-3';

                const titleWrap = document.createElement('div');
                const titleRow = document.createElement('div');
                titleRow.className = 'd-flex align-items-center gap-2 flex-wrap';
                const title = document.createElement('div');
                title.className = 'fw-semibold';
                title.textContent = script.display_name || script.script;
                titleRow.append(title);
                if (script.api) {
                    const apiBadge = document.createElement('span');
                    apiBadge.className = 'badge bg-warning text-dark script-api-badge';
                    apiBadge.textContent = `${String(script.api).toUpperCase()} API`;
                    titleRow.append(apiBadge);
                }
                titleWrap.append(titleRow);
                const subtitle = document.createElement('div');
                subtitle.className = 'small text-muted';
                subtitle.textContent = script.description || script.script;
                titleWrap.append(subtitle);

                if (script.description && script.script && script.description !== script.script) {
                    const pathInfo = document.createElement('div');
                    pathInfo.className = 'small text-muted';
                    pathInfo.textContent = script.script;
                    titleWrap.append(pathInfo);
                }

                const addBtn = document.createElement('button');
                addBtn.type = 'button';
                addBtn.className = 'btn btn-sm btn-primary';
                addBtn.textContent = t('workflow.buttons.add_step');
                addBtn.addEventListener('click', () => {
                    addWorkflowStep(script.id);
                    if (workflowStepModal) {
                        workflowStepModal.hide();
                    }
                });

                header.append(titleWrap, addBtn);
                item.append(header);

                if (script.environment) {
                    const envInfo = document.createElement('div');
                    envInfo.className = 'small mt-1';
                    envInfo.innerHTML = `<strong>${t('workflow.details.environment_label')}</strong> ${script.environment}`;
                    item.append(envInfo);
                }

                if (script.api) {
                    const apiInfo = document.createElement('div');
                    apiInfo.className = 'small mt-1 script-api-highlight';
                    apiInfo.innerHTML = `<strong>${t('workflow.details.api_label', {}, 'API:')}</strong> ${String(script.api).toUpperCase()}`;
                    item.append(apiInfo);
                }

                const requiredParams = (script.parameters || []).filter(param => param.required && !param.autofill);
                if (requiredParams.length) {
                    const reqInfo = document.createElement('div');
                    reqInfo.className = 'small mt-1';
                    reqInfo.innerHTML = `<strong>${t('workflow.details.required_params_label')}</strong> ${requiredParams.map(param => param.name).join(', ')}`;
                    item.append(reqInfo);
                }

                if (script.required_keys && script.required_keys.length) {
                    const keyInfo = document.createElement('div');
                    keyInfo.className = 'small mt-1';
                    keyInfo.innerHTML = `<strong>${t('workflow.details.required_keys_label')}</strong> ${script.required_keys.join(', ')}`;
                    item.append(keyInfo);
                }

                if (script.notes) {
                    const notes = document.createElement('div');
                    notes.className = 'small mt-2 text-muted';
                    notes.textContent = script.notes;
                    item.append(notes);
                }

                return item;
            };

            const groupMap = new Map();
            filtered.forEach(script => {
                const normalizedPath = normalizeScriptPath(script.script);
                const lastSlashIndex = normalizedPath.lastIndexOf('/');
                const groupKey = lastSlashIndex > -1 ? normalizedPath.slice(0, lastSlashIndex) : '';
                if (!groupMap.has(groupKey)) {
                    groupMap.set(groupKey, []);
                }
                groupMap.get(groupKey).push(script);
            });

            const compareGroupNames = (a, b) => {
                if (a === b) {
                    return 0;
                }
                if (!a) {
                    return -1;
                }
                if (!b) {
                    return 1;
                }
                return a.localeCompare(b);
            };

            const groupedEntries = Array.from(groupMap.entries()).sort((entryA, entryB) => compareGroupNames(entryA[0], entryB[0]));

            groupedEntries.forEach(([groupName, scriptsInGroup]) => {
                const header = document.createElement('div');
                header.className = 'list-group-item list-group-item-secondary fw-semibold';
                header.textContent = groupName ? groupName : t('workflow.groups.root_label');
                list.append(header);

                scriptsInGroup.forEach(script => {
                    list.append(createScriptListItem(script));
                });
            });
        }

        function addWorkflowStep(scriptId, insertIndex = workflowSteps.length) {
            const script = findScriptById(scriptId);
            if (!script) {
                alert(t('workflow.errors.unknown_script', { name: scriptId }));
                return;
            }
            const currentMatch = workflowSteps.find(step => step && step.type !== 'widget' && step.script === script.id);
            const template = currentMatch || (defaultWorkflow || []).find(step => step.script === script.id);
            const newStep = {
                script: script.id,
                enabled: true,
                halt_on_fail: template ? (template.halt_on_fail !== undefined ? template.halt_on_fail : true) : true,
                params: template ? cloneObject(template.params) : {}
            };
            let targetIndex = parseInt(insertIndex, 10);
            if (Number.isNaN(targetIndex)) {
                targetIndex = workflowSteps.length;
            }
            targetIndex = Math.max(0, Math.min(targetIndex, workflowSteps.length));
            workflowSteps.splice(targetIndex, 0, newStep);
            markWorkflowDirty();
            renderWorkflowSteps();
            updateInfoBox('secondary', t('workflow.messages.script_added', { name: script.display_name || script.script }));
        }

        function addWorkflowWidget(widgetId, insertIndex = workflowSteps.length) {
            const widget = findWidgetById(widgetId);
            if (!widget) {
                alert(t('workflow.errors.unknown_widget', { name: widgetId }));
                return;
            }
            const defaultParams = {};
            if (Array.isArray(widget.parameters)) {
                widget.parameters.forEach(param => {
                    if (param && param.name && param.default !== undefined) {
                        defaultParams[param.name] = param.default;
                    }
                });
            }
            const newStep = {
                type: 'widget',
                widget: widget.id,
                enabled: true,
                params: defaultParams
            };
            let targetIndex = parseInt(insertIndex, 10);
            if (Number.isNaN(targetIndex)) {
                targetIndex = workflowSteps.length;
            }
            targetIndex = Math.max(0, Math.min(targetIndex, workflowSteps.length));
            workflowSteps.splice(targetIndex, 0, newStep);
            markWorkflowDirty();
            renderWorkflowSteps();
            updateInfoBox('secondary', t('workflow.messages.widget_added', { name: widget.name }));
        }

        async function continueWorkflowAfterWidget(widgetIndex) {
            const validation = validateWorkflowSegment(widgetIndex + 1);
            if (!validation.hasRunnableStep) {
                updateInfoBox('info', t('workflow.messages.no_more_steps'));
                return;
            }
            if (!validation.valid) {
                const messages = [...validation.errors];
                validation.missingSummary.forEach(item => {
                    messages.push(t('workflow.messages.missing_params_list', {
                        name: item.script.display_name || item.script.script,
                        missing: item.missing.join(', ')
                    }));
                });
                updateInfoBox('warning', messages.join(' | '));
                updateMissingBadge();
                return;
            }

            const segment = collectWorkflowSegment(widgetIndex + 1);
            if (!segment.steps.length) {
                updateInfoBox('info', t('workflow.messages.no_more_steps'));
                return;
            }
            const payload = buildRunPayload(widgetIndex + 1, segment);
            const context = {
                type: 'segment',
                startIndex: widgetIndex + 1,
                widgetIndex: segment.widgetIndex
            };

            try {
                const ready = await ensureWorkflowKeys(payload, context);
                if (!ready) {
                    return;
                }
                await executeWorkflow(payload, context);
            } catch (error) {
                console.error('Widget folytatási hiba:', error);
                alert(t('workflow.errors.continue_failed', { error: error.message }));
            }
        }

        async function handleTranslatedSplitLoopWidget(widgetIndex, { auto = false } = {}) {
            const widgetStep = workflowSteps[widgetIndex];
            if (!widgetStep || widgetStep.type !== 'widget') {
                return;
            }
            const allowedMissingRaw = widgetStep.params ? widgetStep.params.allowed_missing_segments : 0;
            const allowedMissing = Number.parseInt(String(allowedMissingRaw ?? '0'), 10);
            if (!Number.isInteger(allowedMissing) || allowedMissing < 0) {
                updateInfoBox('warning', t('workflow.messages.segment_loop_allowed_invalid'));
                return;
            }

            let progress = null;
            try {
                progress = await fetchTranslatedSplitProgress();
            } catch (error) {
                console.error('Translated split állapot lekérdezési hiba:', error);
                updateInfoBox('danger', t('workflow.messages.segment_loop_check_failed', { error: error.message }));
                return;
            }

            const expectedSegments = Number.parseInt(String(progress.expected_segments ?? 0), 10) || 0;
            const completedSegments = Number.parseInt(String(progress.completed_segments ?? 0), 10) || 0;
            const missingSegments = Number.parseInt(
                String(progress.missing_segments ?? Math.max(expectedSegments - completedSegments, 0)),
                10
            ) || 0;

            if (missingSegments > allowedMissing) {
                const restartInfo = resolveTranslatedSplitLoopRestart(widgetIndex);
                if (restartInfo.error) {
                    updateInfoBox('warning', restartInfo.error);
                    return;
                }
                updateInfoBox('secondary', t('workflow.messages.segment_loop_retry', {
                    completed: completedSegments,
                    total: expectedSegments,
                    missing: missingSegments,
                    allowed: allowedMissing,
                    json: progress.json_file_name || ''
                }));
                await startSegmentRun(restartInfo.restartIndex);
                return;
            }

            updateInfoBox('success', t('workflow.messages.segment_loop_continue', {
                completed: completedSegments,
                total: expectedSegments,
                missing: missingSegments,
                allowed: allowedMissing,
                json: progress.json_file_name || ''
            }));
            await continueWorkflowAfterWidget(widgetIndex);
        }

        async function handleWidgetContinue(widgetIndex) {
            if (currentWorkflowJobId) {
                updateInfoBox('warning', t('workflow.messages.already_running'));
                return;
            }
            const widgetStep = workflowSteps[widgetIndex];
            if (!widgetStep || widgetStep.type !== 'widget') {
                return;
            }
            if (widgetStep.enabled === false) {
                updateInfoBox('info', t('workflow.messages.widget_disabled'));
                return;
            }
            const widget = findWidgetById(widgetStep.widget);
            if (widget && widget.id === 'cycleWidget') {
                await startCycleSequence(widgetIndex, { auto: false });
                return;
            }
            if (widget && widget.id === 'translatedSplitLoopWidget') {
                await handleTranslatedSplitLoopWidget(widgetIndex, { auto: false });
                return;
            }
            await continueWorkflowAfterWidget(widgetIndex);
        }

        function moveWorkflowStep(index, delta) {
            const newIndex = index + delta;
            if (newIndex < 0 || newIndex >= workflowSteps.length) {
                return;
            }
            const [step] = workflowSteps.splice(index, 1);
            workflowSteps.splice(newIndex, 0, step);
            markWorkflowDirty();
            renderWorkflowSteps();
        }

        function removeWorkflowStep(index) {
            const [removed] = workflowSteps.splice(index, 1);
            markWorkflowDirty();
            renderWorkflowSteps();
            if (removed) {
                let label = removed.script;
                if (removed.type === 'widget') {
                    const widget = findWidgetById(removed.widget);
                    label = widget ? widget.name : t('workflow.labels_extra.widget_badge', {}, 'Widget');
                }
                updateInfoBox('secondary', t('workflow.messages.step_removed', { name: label }));
            }
        }

        function openWorkflowParams(index) {
            const step = workflowSteps[index];
            if (!step) {
                return;
            }
            if (step.type === 'widget') {
                return;
            }
            const script = findScriptById(step.script);
            if (!script) {
                alert(t('workflow.errors.unknown_script', { name: step.script }));
                return;
            }
            selectedStepIndex = index;
            selectedStepType = 'script';
            selectedWidgetConfig = null;
            const titleEl = document.getElementById('workflowParamsTitle');
            if (titleEl) {
                titleEl.textContent = t('workflow.modals.params.script_title', {
                    name: script.display_name || script.script
                }, `${script.display_name || script.script} – parameters`);
            }
            renderWorkflowParamsForm(script, step);
            if (workflowParamsModal) {
                workflowParamsModal.show();
            }
        }

        function openWorkflowWidgetParams(index) {
            const step = workflowSteps[index];
            if (!step || step.type !== 'widget') {
                return;
            }
            const widget = findWidgetById(step.widget);
            if (!widget) {
                alert(t('workflow.errors.unknown_widget', { name: step.widget }));
                return;
            }
            selectedStepIndex = index;
            selectedStepType = 'widget';
            selectedWidgetConfig = widget;
            const titleEl = document.getElementById('workflowParamsTitle');
            if (titleEl) {
                titleEl.textContent = t('workflow.modals.params.widget_title', {
                    name: widget.name
                }, `${widget.name} – settings`);
            }
            renderWidgetParamsForm(widget, step);
            if (workflowParamsModal) {
                workflowParamsModal.show();
            }
        }

        function updateWorkflowParamsHelp(source) {
            const helpBox = document.getElementById('workflowParamsHelp');
            if (!helpBox) {
                return;
            }
            let helpText = '';
            let isMarkdown = false;
            if (source && typeof source.help_markdown === 'string') {
                helpText = source.help_markdown.trim();
                isMarkdown = true;
            } else if (source && typeof source.help === 'string') {
                helpText = source.help.trim();
            }
            if (!helpText) {
                helpBox.classList.add('d-none');
                helpBox.innerHTML = '';
                return;
            }
            helpBox.classList.remove('d-none');
            helpBox.innerHTML = '';
            const title = document.createElement('h6');
            title.className = 'mb-2';
            title.textContent = t('workflow.modals.params.help_title', {}, 'Help');
            const block = document.createElement('div');
            block.className = 'small bg-body-tertiary p-3 border rounded';
            if (isMarkdown) {
                let renderedHtml = '';
                const markedRef = window.marked;
                if (markedRef) {
                    if (typeof markedRef === 'function') {
                        try {
                            renderedHtml = markedRef(helpText);
                        } catch (err) {
                            console.warn('Markdown render hiba (function):', err);
                        }
                    } else if (typeof markedRef.parse === 'function') {
                        try {
                            renderedHtml = markedRef.parse(helpText);
                        } catch (err) {
                            console.warn('Markdown render hiba (parse):', err);
                        }
                    }
                }
                if (renderedHtml) {
                    block.classList.add('markdown-content');
                    block.innerHTML = renderedHtml;
                } else {
                    block.style.whiteSpace = 'pre-wrap';
                    block.style.fontFamily = 'var(--bs-body-font-family, inherit)';
                    block.textContent = helpText;
                }
            } else {
                block.style.whiteSpace = 'pre-wrap';
                block.style.fontFamily = 'var(--bs-body-font-family, inherit)';
                block.textContent = helpText;
            }
            helpBox.append(title, block);
        }

        function renderWorkflowParamsForm(script, step) {
            const container = document.getElementById('workflowParamsContainer');
            const alertBox = document.getElementById('workflowParamsAlert');
            if (alertBox) {
                alertBox.classList.add('d-none');
                alertBox.textContent = '';
            }
            if (!container) {
                updateWorkflowParamsHelp(script);
                return;
            }
            container.innerHTML = '';
            const params = script.parameters || [];
            const scriptDefaults = getScriptParamDefaults(script);
            if (!params.length) {
                const placeholder = document.createElement('div');
                placeholder.className = 'text-muted';
                placeholder.textContent = t('workflow.modals.params.no_settings');
                container.append(placeholder);
                updateWorkflowParamsHelp(script);
                return;
            }
            params.forEach(param => {
                const group = document.createElement('div');
                const rawValue = step.params ? step.params[param.name] : undefined;
                const hasExplicitDefault = param && Object.prototype.hasOwnProperty.call(param, 'default');
                let defaultValue = hasExplicitDefault ? param.default : undefined;
                if ((defaultValue === undefined || defaultValue === null) && scriptDefaults.has(param.name)) {
                    defaultValue = scriptDefaults.get(param.name);
                }
                const hasDefault = defaultValue !== undefined && defaultValue !== null && !(typeof defaultValue === 'string' && defaultValue.trim() === '');
                const defaultString = hasDefault ? String(defaultValue) : '';
                if (param.autofill) {
                    group.className = 'border rounded p-2 bg-body-tertiary';
                    const label = document.createElement('div');
                    label.className = 'fw-semibold';
                    label.textContent = getParamLabel(param);
                    const help = document.createElement('div');
                    help.className = 'small text-muted';
                    help.textContent = param.autofill === 'project_name'
                        ? t('workflow.labels_extra.auto_value_project')
                        : t('workflow.labels_extra.auto_value_path');
                    group.append(label, help);
                    container.append(group);
                    return;
                }

                if (param.type === 'flag') {
                    group.className = 'form-check form-switch';
                    const input = document.createElement('input');
                    input.type = 'checkbox';
                    input.className = 'form-check-input';
                    input.id = `workflow-param-${param.name}`;
                    input.dataset.param = param.name;
                    input.dataset.type = param.type;
                    input.dataset.flagMode = getFlagMode(param);
                    const defaultFlag = hasDefault ? normalizeFlagValue(defaultValue) : undefined;
                    input.checked = getDisplayFlagValue(param, rawValue, defaultValue);
                    input.dataset.defaultRawValue = serializeDatasetBool(defaultFlag);
                    const label = document.createElement('label');
                    label.className = 'form-check-label';
                    label.setAttribute('for', input.id);
                    label.textContent = getParamLabel(param);
                    group.append(input, label);
                    if (param.description) {
                        const descriptionInfo = document.createElement('div');
                        descriptionInfo.className = 'form-text';
                        descriptionInfo.textContent = param.description;
                        group.append(descriptionInfo);
                    }
                    if (defaultFlag !== undefined) {
                        const defaultInfo = document.createElement('div');
                        defaultInfo.className = 'form-text';
                        defaultInfo.textContent = getDisplayFlagValue(param, defaultFlag, defaultFlag)
                            ? t('workflow.labels_extra.default_flag_on')
                            : t('workflow.labels_extra.default_flag_off');
                        group.append(defaultInfo);
                    }
                } else {
                    group.className = 'form-group';
                    const label = document.createElement('label');
                    label.className = 'form-label fw-semibold';
                    label.setAttribute('for', `workflow-param-${param.name}`);
                    label.textContent = getParamLabel(param) + (param.required ? ' *' : '');
                    group.append(label);
                    const input = document.createElement('input');
                    input.className = 'form-control';
                    input.id = `workflow-param-${param.name}`;
                    input.dataset.param = param.name;
                    input.dataset.type = param.type;
                    let helper = null;
                    let clearBtn = null;
                    if (param.secret) {
                        const storedValue = rawValue !== undefined && rawValue !== null ? String(rawValue) : '';
                        const hasExisting = storedValue.trim() !== '';
                        input.type = 'password';
                        input.autocomplete = 'off';
                        input.dataset.secret = 'true';
                        input.dataset.cleared = 'false';
                        input.dataset.hasExisting = hasExisting ? 'true' : 'false';
                        if (hasExisting) {
                            input.placeholder = '••••••';
                            input.dataset.original = storedValue;
                        } else {
                            input.placeholder = t('workflow.labels_extra.key_placeholder', {}, 'API kulcs');
                        }
                        input.value = '';
                        helper = document.createElement('div');
                        helper.className = 'form-text';
                        helper.textContent = hasExisting
                            ? t('workflow.labels_extra.key_saved_hint')
                            : t('workflow.labels_extra.key_enter_value');
                        if (hasExisting) {
                            clearBtn = document.createElement('button');
                            clearBtn.type = 'button';
                            clearBtn.className = 'btn btn-sm btn-outline-danger mt-2';
                            clearBtn.textContent = t('workflow.labels_extra.key_clear_button');
                            clearBtn.addEventListener('click', () => {
                                input.value = '';
                                input.placeholder = t('workflow.labels_extra.key_clear_placeholder');
                                input.dataset.cleared = 'true';
                                input.dataset.original = '';
                                input.dataset.hasExisting = 'false';
                                helper.textContent = t('workflow.labels_extra.key_clear_hint');
                            });
                        }
                        input.addEventListener('input', () => {
                            const trimmed = input.value.trim();
                            if (trimmed.length === 0) {
                                if (input.dataset.cleared === 'true') {
                                    helper.textContent = t('workflow.labels_extra.key_clear_hint');
                                } else if (input.dataset.hasExisting === 'true') {
                                    helper.textContent = t('workflow.labels_extra.key_saved_hint');
                                } else {
                                    helper.textContent = t('workflow.labels_extra.key_enter_value');
                                }
                            } else {
                                input.dataset.cleared = 'false';
                                helper.textContent = t('workflow.labels_extra.key_new_hint');
                            }
                        });
                    } else {
                        input.type = 'text';
                        if (rawValue !== undefined && rawValue !== null) {
                            input.value = String(rawValue);
                        } else {
                            input.value = '';
                            if (hasDefault) {
                                input.placeholder = defaultString;
                            }
                        }
                    }
                    group.append(input);
                    if (param.description) {
                        const descriptionInfo = document.createElement('div');
                        descriptionInfo.className = 'form-text';
                        descriptionInfo.textContent = param.description;
                        group.append(descriptionInfo);
                    }
                    if (helper) {
                        group.append(helper);
                    }
                    if (clearBtn) {
                        group.append(clearBtn);
                    }
                }

                if (param.flags && param.flags.length) {
                    const flagInfo = document.createElement('div');
                    flagInfo.className = 'form-text';
                    flagInfo.textContent = `${t('workflow.labels_extra.flags_prefix')}: ${param.flags.join(', ')}`;
                    group.append(flagInfo);
                }

                container.append(group);
            });
            updateWorkflowParamsHelp(script);
        }

        function renderWidgetParamsForm(widget, step) {
            const container = document.getElementById('workflowParamsContainer');
            const alertBox = document.getElementById('workflowParamsAlert');
            if (alertBox) {
                alertBox.classList.add('d-none');
                alertBox.textContent = '';
            }
            if (!container) {
                updateWorkflowParamsHelp(widget);
                return;
            }
            container.innerHTML = '';
            const parameters = Array.isArray(widget && widget.parameters) ? widget.parameters : [];
            if (!parameters.length) {
                const placeholder = document.createElement('div');
                placeholder.className = 'text-muted';
                placeholder.textContent = t('workflow.modals.params.no_widget_settings');
                container.append(placeholder);
                updateWorkflowParamsHelp(widget);
                return;
            }
            parameters.forEach(param => {
                if (!param || !param.name) {
                    return;
                }
                const group = document.createElement('div');
                group.className = 'mb-3';
                const label = document.createElement('label');
                label.className = 'form-label';
                label.setAttribute('for', `widgetParam_${param.name}`);
                label.textContent = param.label || param.name;
                group.append(label);

                const input = document.createElement('input');
                input.id = `widgetParam_${param.name}`;
                input.className = 'form-control';
                input.dataset.param = param.name;
                input.dataset.type = param.type || 'text';
                input.dataset.widget = 'true';
                if (param.type === 'number') {
                    input.type = 'number';
                    if (param.min !== undefined) {
                        input.min = String(param.min);
                    }
                    if (param.max !== undefined) {
                        input.max = String(param.max);
                    }
                    if (param.step !== undefined) {
                        input.step = String(param.step);
                    } else {
                        input.step = '1';
                    }
                } else {
                    input.type = 'text';
                }
                if (param.required) {
                    input.required = true;
                }
                const params = step && step.params ? step.params : {};
                const hasOwnValue = Object.prototype.hasOwnProperty.call(params, param.name);
                const value = hasOwnValue ? params[param.name] : param.default;
                if (value !== undefined && value !== null) {
                    input.value = String(value);
                } else {
                    input.value = '';
                }
                group.append(input);
                if (param.helper) {
                    const helper = document.createElement('div');
                    helper.className = 'form-text';
                    helper.textContent = param.helper;
                    group.append(helper);
                }
                container.append(group);
            });
            updateWorkflowParamsHelp(widget);
        }

        function saveWorkflowParams() {
            if (selectedStepIndex === null) {
                return;
            }
            const step = workflowSteps[selectedStepIndex];
            if (!step) {
                return;
            }
            if (selectedStepType === 'widget') {
                saveWidgetParams(step, selectedWidgetConfig || findWidgetById(step.widget));
                return;
            }
            if (step.type === 'widget') {
                return;
            }
            const script = findScriptById(step.script);
            if (!script) {
                return;
            }
            const container = document.getElementById('workflowParamsContainer');
            const alertBox = document.getElementById('workflowParamsAlert');
            if (!container) {
                return;
            }
            const inputs = container.querySelectorAll('[data-param]');
            const newParams = cloneObject(step.params);
            inputs.forEach(input => {
                const name = input.dataset.param;
                const type = input.dataset.type;
                if (type === 'flag') {
                    const flagMode = input.dataset.flagMode || 'standard';
                    const defaultValue = parseDatasetBool(input.dataset.defaultRawValue || 'none');
                    const storedValue = flagMode === 'negative_only_negative'
                        ? !input.checked
                        : input.checked;
                    if (defaultValue !== undefined && storedValue === defaultValue) {
                        delete newParams[name];
                    } else {
                        newParams[name] = storedValue;
                    }
                } else {
                    const isSecret = input.dataset.secret === 'true';
                    const rawValue = input.value;
                    const trimmedValue = rawValue.trim();
                    if (isSecret) {
                        const cleared = input.dataset.cleared === 'true';
                        const hasExisting = input.dataset.hasExisting === 'true';
                        const originalValue = input.dataset.original || '';
                        if (cleared) {
                            delete newParams[name];
                        } else if (trimmedValue === '') {
                            if (hasExisting && originalValue) {
                                newParams[name] = originalValue;
                            } else {
                                delete newParams[name];
                            }
                        } else {
                            newParams[name] = trimmedValue;
                        }
                    } else {
                        if (trimmedValue === '') {
                            delete newParams[name];
                        } else {
                            newParams[name] = trimmedValue;
                        }
                    }
                }
            });

            const missing = getMissingParams({ ...step, params: newParams }, script);
            if (missing.length && alertBox) {
                alertBox.textContent = t('workflow.messages.missing_required', { missing: missing.join(', ') });
                alertBox.classList.remove('d-none');
                return;
            }

            step.params = newParams;
            markWorkflowDirty();
            if (workflowParamsModal) {
                workflowParamsModal.hide();
            }
            renderWorkflowSteps();
        }

        function saveWidgetParams(step, widget) {
            const container = document.getElementById('workflowParamsContainer');
            const alertBox = document.getElementById('workflowParamsAlert');
            if (!container) {
                return;
            }
            const parameters = Array.isArray(widget && widget.parameters) ? widget.parameters : [];
            if (!parameters.length) {
                step.params = {};
                markWorkflowDirty();
                if (workflowParamsModal) {
                    workflowParamsModal.hide();
                }
                renderWorkflowSteps();
                return;
            }
            const newParams = {};
            const errors = [];
            parameters.forEach(param => {
                if (!param || !param.name) {
                    return;
                }
                const input = container.querySelector(`[data-param="${param.name}"]`);
                if (!input) {
                    return;
                }
                const rawValue = input.value ?? '';
                const trimmed = rawValue.trim();
                const fieldLabel = param.label || param.name;
                if (!trimmed) {
                    if (param.required) {
                        errors.push(fieldLabel);
                        return;
                    }
                    if (param.default !== undefined) {
                        newParams[param.name] = param.default;
                    }
                    return;
                }
                if ((param.type || 'text') === 'number') {
                    if (!/^[-+]?\d+$/.test(trimmed)) {
                        errors.push(t('workflow.validation.number_invalid', { field: fieldLabel }));
                        return;
                    }
                    const numericValue = Number.parseInt(trimmed, 10);
                    if (param.min !== undefined && numericValue < param.min) {
                        errors.push(t('workflow.validation.min_value', { field: fieldLabel, min: param.min }));
                        return;
                    }
                    if (param.max !== undefined && numericValue > param.max) {
                        errors.push(t('workflow.validation.max_value', { field: fieldLabel, max: param.max }));
                        return;
                    }
                    newParams[param.name] = numericValue;
                } else {
                    newParams[param.name] = trimmed;
                }
            });
            if (errors.length) {
                if (alertBox) {
                    alertBox.textContent = t('workflow.messages.check_fields', { fields: errors.join(', ') });
                    alertBox.classList.remove('d-none');
                }
                return;
            }
            step.params = newParams;
            markWorkflowDirty();
            if (workflowParamsModal) {
                workflowParamsModal.hide();
            }
            renderWorkflowSteps();
        }

        function validateWorkflowSegment(startIndex = 0) {
            const stepsToValidate = [];
            for (let i = startIndex; i < workflowSteps.length; i++) {
                const step = workflowSteps[i];
                if (step.type === 'widget') {
                    if (step.enabled !== false) {
                        break;
                    }
                    continue;
                }
                if (step.enabled === false) {
                    continue;
                }
                stepsToValidate.push(step);
            }
            return validateStepCollection(stepsToValidate);
        }

        async function startSegmentRun(startIndex = 0) {
            const segment = collectWorkflowSegment(startIndex);
            if (!segment.steps.length) {
                updateInfoBox('info', startIndex === 0
                    ? t('workflow.status_texts.no_steps_before_widget')
                    : t('workflow.messages.no_more_steps'));
                return false;
            }
            const payload = buildRunPayload(startIndex, segment);
            const context = {
                type: 'segment',
                startIndex,
                widgetIndex: segment.widgetIndex
            };
            try {
                const ready = await ensureWorkflowKeys(payload, context);
                if (!ready) {
                    return false;
                }
                await executeWorkflow(payload, context);
                return true;
            } catch (error) {
                console.error('Workflow indítási hiba:', error);
                alert(t('workflow.errors.start_failed', { error: error.message }));
                return false;
            }
        }

        function updateCycleDisplay(widgetIndex, iteration, total) {
            const row = document.querySelector(`tr[data-step-index="${widgetIndex}"]`);
            if (!row) {
                return;
            }
            const infoCell = row.querySelector('td:nth-child(2)');
            if (!infoCell) {
                return;
            }
            let display = infoCell.querySelector('.cycle-counter-display');
            if (!display) {
                display = document.createElement('div');
                display.className = 'cycle-counter-display';
                infoCell.append(display);
            }
            const safeIteration = Math.max(0, iteration);
            display.textContent = safeIteration === 0
                ? t('workflow.cycle.preparing')
                : t('workflow.cycle.progress', { current: safeIteration, total });
        }

        function clearCycleDisplay(widgetIndex) {
            const row = document.querySelector(`tr[data-step-index="${widgetIndex}"]`);
            if (!row) {
                return;
            }
            const infoCell = row.querySelector('td:nth-child(2)');
            if (!infoCell) {
                return;
            }
            const display = infoCell.querySelector('.cycle-counter-display');
            if (display) {
                display.remove();
            }
        }

        function prepareCycleState(widgetIndex) {
            const widgetStep = workflowSteps[widgetIndex];
            if (!widgetStep || widgetStep.type !== 'widget') {
                return null;
            }
            const widget = findWidgetById(widgetStep.widget);
            if (!widget || widget.id !== 'cycleWidget') {
                return null;
            }
            const params = widgetStep.params || {};
            const repeatRaw = params.repeat_count;
            const stepBackRaw = params.step_back;
            const repeatCount = Number.parseInt(String(repeatRaw ?? ''), 10);
            const stepBackCount = Number.parseInt(String(stepBackRaw ?? ''), 10);
            if (!Number.isInteger(repeatCount) || repeatCount < 1) {
                updateInfoBox('warning', t('workflow.messages.cycle_repeat_invalid'));
                return null;
            }
            if (!Number.isInteger(stepBackCount) || stepBackCount < 1) {
                updateInfoBox('warning', t('workflow.messages.cycle_step_back_invalid'));
                return null;
            }
            const {
                steps: cycleSteps,
                collectedCount
            } = collectScriptStepsBefore(widgetIndex, stepBackCount);
            if (!cycleSteps.length) {
                updateInfoBox('info', t('workflow.messages.no_steps_before_cycle'));
                return null;
            }
            if (collectedCount < stepBackCount) {
                updateInfoBox('warning', t('workflow.messages.cycle_not_enough_steps'));
                return null;
            }
            const validation = validateStepCollection(cycleSteps);
            if (!validation.hasRunnableStep) {
                updateInfoBox('info', t('workflow.messages.no_steps_before_cycle'));
                return null;
            }
            if (!validation.valid) {
                const messages = [...validation.errors];
                validation.missingSummary.forEach(item => {
                    messages.push(t('workflow.messages.missing_params_list', {
                        name: item.script.display_name || item.script.script,
                        missing: item.missing.join(', ')
                    }));
                });
                updateInfoBox('warning', messages.join(' | '));
                updateMissingBadge();
                return null;
            }
            const runSteps = cycleSteps.map(step => buildRunStepFromWorkflowStep(step));
            return {
                widgetStep,
                widget,
                repeatCount,
                stepBackCount,
                runSteps
            };
        }

        async function startCycleSequence(widgetIndex, { auto = false } = {}) {
            if (cycleState) {
                return;
            }
            const state = prepareCycleState(widgetIndex);
            if (!state) {
                return;
            }
            cycleState = {
                widgetIndex,
                totalIterations: state.repeatCount,
                nextIteration: 1,
                runSteps: state.runSteps,
                stepBackCount: state.stepBackCount,
                autoTriggered: auto,
                active: false,
                activeIteration: null
            };
            updateCycleDisplay(widgetIndex, 0, state.repeatCount);
            updateInfoBox('secondary', t('workflow.messages.cycle_start', { count: state.repeatCount }));
            try {
                await launchNextCycleIteration();
            } catch (error) {
                console.error('Ciklus indítási hiba:', error);
                updateInfoBox('danger', t('workflow.messages.cycle_start_failed', { error: error.message }));
                clearCycleDisplay(widgetIndex);
                cycleState = null;
            }
        }

        async function startCycleIterationExecution(payload, context) {
            if (!cycleState || cycleState.widgetIndex !== context.widgetIndex) {
                await executeWorkflow(payload, context);
                return;
            }
            cycleState.active = true;
            cycleState.activeIteration = context.iteration;
            updateCycleDisplay(context.widgetIndex, context.iteration, context.totalIterations);
            try {
                await executeWorkflow(payload, context);
                cycleState.nextIteration = Math.max(cycleState.nextIteration, context.iteration + 1);
            } catch (error) {
                cycleState.active = false;
                cycleState.activeIteration = null;
                throw error;
            }
        }

        async function launchNextCycleIteration() {
            if (!cycleState) {
                return;
            }
            if (cycleState.active) {
                return;
            }
            if (cycleState.nextIteration > cycleState.totalIterations) {
                finalizeCycleSequence(true);
                return;
            }
            const iteration = cycleState.nextIteration;
            const payload = {
                steps: cycleState.runSteps.map(step => ({
                    script: step.script,
                    enabled: true,
                    halt_on_fail: step.halt_on_fail !== false,
                    params: cloneObject(step.params)
                })),
                template_id: currentTemplateId,
                workflow_state: collectWorkflowState()
            };
            const context = {
                type: 'cycle',
                widgetIndex: cycleState.widgetIndex,
                iteration,
                totalIterations: cycleState.totalIterations
            };
            const ready = await ensureWorkflowKeys(payload, context);
            if (!ready) {
                return;
            }
            await startCycleIterationExecution(payload, context);
        }

        function finalizeCycleSequence(success) {
            if (!cycleState) {
                return;
            }
            const widgetIndex = cycleState.widgetIndex;
            clearCycleDisplay(widgetIndex);
            cycleState = null;
            if (success) {
                updateInfoBox('success', t('workflow.messages.cycle_complete'));
                setTimeout(() => {
                    startSegmentRun(widgetIndex + 1);
                }, 0);
            } else {
                updateInfoBox('danger', t('workflow.messages.cycle_cancelled'));
            }
        }

        function autoTriggerWidget(widgetIndex) {
            const widgetStep = workflowSteps[widgetIndex];
            if (!widgetStep || widgetStep.type !== 'widget') {
                return;
            }
            const widget = findWidgetById(widgetStep.widget);
            if (!widget || widgetStep.enabled === false) {
                return;
            }
            if (widget.id === 'cycleWidget') {
                startCycleSequence(widgetIndex, { auto: true }).catch(error => {
                    console.error('Automatikus ciklus hiba:', error);
                });
            } else if (widget.id === 'translatedSplitLoopWidget') {
                handleTranslatedSplitLoopWidget(widgetIndex, { auto: true }).catch(error => {
                    console.error('Automatikus translated split loop hiba:', error);
                });
            } else {
                updateInfoBox('info', t('workflow.messages.widget_continue_hint', { name: widget.name }));
            }
        }

        function handleRunContextJobUpdate(job) {
            if (!job || !currentRunContext || currentRunContext.jobId !== job.job_id) {
                return;
            }
            if (currentRunContext.completionHandled) {
                return;
            }
            if (job.status === 'completed') {
                currentRunContext.completionHandled = true;
                if (currentRunContext.type === 'segment') {
                    const context = currentRunContext;
                    currentRunContext = null;
                    if (context.widgetIndex !== null && context.widgetIndex !== undefined) {
                        autoTriggerWidget(context.widgetIndex);
                    } else {
                        updateInfoBox('success', t('workflow.messages.segment_success'));
                    }
                } else if (currentRunContext.type === 'cycle') {
                    currentRunContext.completionHandled = true;
                    currentRunContext = null;
                    if (cycleState) {
                        cycleState.active = false;
                        cycleState.activeIteration = null;
                        setTimeout(() => {
                            launchNextCycleIteration().catch(error => {
                                console.error('Ciklus folytatás hiba:', error);
                                updateInfoBox('danger', t('workflow.messages.cycle_next_failed'));
                                finalizeCycleSequence(false);
                            });
                        }, 0);
                    }
                } else {
                    currentRunContext = null;
                }
            } else if (job.status === 'failed' || job.status === 'cancelled') {
                currentRunContext.completionHandled = true;
                if (cycleState) {
                    cycleState.active = false;
                    cycleState.activeIteration = null;
                    finalizeCycleSequence(false);
                }
                currentRunContext = null;
            }
        }

        async function handleStartWorkflow() {
            const validation = validateWorkflowSegment(0);
            if (!validation.hasRunnableStep) {
                updateInfoBox('info', t('workflow.status_texts.no_steps_before_widget'));
                return;
            }
            if (!validation.valid) {
                const messages = [...validation.errors];
                if (validation.missingSummary.length) {
                    validation.missingSummary.forEach(item => {
                        const script = item.script;
                        messages.push(t('workflow.messages.missing_params_list', {
                            name: script.display_name || script.script,
                            missing: item.missing.join(', ')
                        }));
                    });
                }
                updateInfoBox('warning', messages.join(' | '));
                updateMissingBadge();
                return;
            }

            const segment = collectWorkflowSegment(0);
            if (!segment.steps.length) {
                updateInfoBox('info', t('workflow.status_texts.no_steps_before_widget'));
                return;
            }
            const payload = buildRunPayload(0, segment);
            const context = {
                type: 'segment',
                startIndex: 0,
                widgetIndex: segment.widgetIndex
            };
            try {
                const ready = await ensureWorkflowKeys(payload, context);
                if (!ready) {
                    return;
                }
                await executeWorkflow(payload, context);
            } catch (error) {
                console.error('Workflow indítási hiba:', error);
                alert(t('workflow.errors.start_failed', { error: error.message }));
            }
        }

        async function ensureWorkflowKeys(payload, context = null) {
            const response = await fetch(`/api/workflow-key-status/${encodeURIComponent(PROJECT_NAME)}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ steps: payload.steps })
            });
            const result = await response.json();
            if (!response.ok || !result.success) {
                throw new Error(result.error || t('workflow.errors.key_status_failed'));
            }
            const missing = Object.entries(result.keys || {}).filter(([, info]) => info.required && !info.present);
            if (!missing.length) {
                pendingWorkflowPayload = null;
                pendingWorkflowContext = null;
                return true;
            }
            const missingList = missing.map(([key, info]) => ({ key, label: info.label || key }));
            prepareWorkflowKeyModal(missingList);
            pendingWorkflowPayload = payload;
            pendingWorkflowContext = context || null;
            if (workflowKeysModal) {
                workflowKeysModal.show();
            } else {
                alert(t('workflow.keys.missing_alert', {
                    keys: missingList.map(item => item.label).join(', ')
                }));
            }
            return false;
        }

        function prepareWorkflowKeyModal(missingList) {
            Object.entries(workflowKeyFieldMap).forEach(([key, cfg]) => {
                const wrapper = document.getElementById(cfg.wrapperId);
                const input = document.getElementById(cfg.inputId);
                if (wrapper) {
                    const isRequired = missingList.some(item => item.key === key);
                    wrapper.classList.toggle('d-none', !isRequired);
                }
                if (input) {
                    input.value = '';
                }
            });
            const info = document.getElementById('workflowKeysInfo');
            if (info) {
                const labels = missingList.map(item => item.label).join(', ');
                info.textContent = t('workflow.keys.list_instructions', { keys: labels });
            }
        }

        function resetWorkflowKeyModal() {
            Object.values(workflowKeyFieldMap).forEach(cfg => {
                const wrapper = document.getElementById(cfg.wrapperId);
                if (wrapper) {
                    wrapper.classList.add('d-none');
                }
                const input = document.getElementById(cfg.inputId);
                if (input) {
                    input.value = '';
                }
            });
            const info = document.getElementById('workflowKeysInfo');
            if (info) {
                info.textContent = t('workflow.modals.api_keys.instructions');
            }
        }

        async function saveWorkflowKeys() {
            if (!pendingWorkflowPayload) {
                if (workflowKeysModal) {
                    workflowKeysModal.hide();
                }
                return;
            }
            const submitBtn = document.getElementById('workflowKeysSubmitBtn');
            const spinner = document.getElementById('workflowKeysSpinner');
            const payload = {};
            let firstEmpty = null;

            Object.values(workflowKeyFieldMap).forEach(cfg => {
                const wrapper = document.getElementById(cfg.wrapperId);
                if (wrapper && !wrapper.classList.contains('d-none')) {
                    const input = document.getElementById(cfg.inputId);
                    const value = (input ? input.value : '').trim();
                    if (!value) {
                        if (!firstEmpty && input) {
                            firstEmpty = input;
                        }
                    } else {
                        payload[cfg.payloadKey] = value;
                    }
                }
            });

            if (firstEmpty) {
                firstEmpty.focus();
                alert(t('workflow.keys.fill_all'));
                return;
            }
            if (!Object.keys(payload).length) {
                if (workflowKeysModal) {
                    workflowKeysModal.hide();
                }
                return;
            }

            if (submitBtn) {
                submitBtn.disabled = true;
            }
            if (spinner) {
                spinner.classList.remove('d-none');
            }

            try {
                const response = await fetch('/save-workflow-keys', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const result = await response.json();
                if (!response.ok || !result.success) {
                    throw new Error(result.error || t('workflow.errors.keys_save_failed_generic'));
                }
                const payloadToRun = pendingWorkflowPayload;
                const contextToRun = pendingWorkflowContext;
                pendingWorkflowPayload = null;
                pendingWorkflowContext = null;
                if (workflowKeysModal) {
                    workflowKeysModal.hide();
                }
                if (!payloadToRun) {
                    return;
                }
                if (contextToRun && contextToRun.type === 'cycle') {
                    await startCycleIterationExecution(payloadToRun, contextToRun);
                } else {
                    await executeWorkflow(payloadToRun, contextToRun || null);
                }
            } catch (error) {
                console.error('API kulcs mentése sikertelen:', error);
                alert(t('workflow.errors.keys_save_failed', { error: error.message }));
            } finally {
                if (submitBtn) {
                    submitBtn.disabled = false;
                }
                if (spinner) {
                    spinner.classList.add('d-none');
                }
            }
        }

        async function executeWorkflow(payload, context = null) {
            const statusText = document.getElementById('workflowStatusText');
            const startButton = document.getElementById('startWorkflowBtn');
            const spinner = document.getElementById('workflowSpinner');
            const stopButton = document.getElementById('stopWorkflowBtn');
            const stopSpinner = document.getElementById('stopWorkflowSpinner');
            const stopLabel = document.getElementById('stopWorkflowLabel');
            const logContent = document.getElementById('workflowLogContent');
            const logLink = document.getElementById('workflowLogLink');

            if (context) {
                currentRunContext = {
                    ...context,
                    jobId: null,
                    completionHandled: false
                };
            } else {
                currentRunContext = null;
            }

            currentWorkflowJobId = null;
            stopLogPolling(true);

            if (startButton) {
                startButton.disabled = true;
                startButton.classList.add('btn-processing');
            }
            if (spinner) {
                spinner.classList.remove('d-none');
            }
            if (statusText) {
                statusText.textContent = t('workflow.run.starting');
            }
            updateInfoBox('secondary', t('workflow.run.starting_info'));

            if (stopButton) {
                stopButton.classList.remove('d-none');
                stopButton.disabled = true;
            }
            if (stopSpinner) {
                stopSpinner.classList.add('d-none');
            }
            if (stopLabel) {
                stopLabel.textContent = t('workflow.buttons_extra.stop');
            }
            if (logLink) {
                logLink.innerHTML = '';
            }
            if (logContent) {
                logContent.textContent = t('workflow.log_messages.log_initializing');
            }
            updateLogStatus(t('workflow.run.waiting_log'), 'muted');

            try {
                const response = await fetch(`/api/run-workflow/${encodeURIComponent(PROJECT_NAME)}`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                const result = await response.json();
                if (!response.ok || !result.success) {
                    throw new Error(result.error || t('workflow.errors.start_unknown'));
                }
                if (context) {
                    currentRunContext = {
                        ...context,
                        jobId: result.job_id,
                        completionHandled: false
                    };
                } else {
                    currentRunContext = null;
                }
                if (statusText) {
                    statusText.textContent = t('workflow.run.started');
                }
                startWorkflowPolling(result.job_id);
            } catch (error) {
                console.error('Workflow indítási hiba:', error);
                if (statusText) {
                    statusText.textContent = t('workflow.status_texts.error', { error: error.message }, `Hiba: ${error.message}`);
                }
                if (startButton) {
                    startButton.disabled = false;
                    startButton.classList.remove('btn-processing');
                }
                if (spinner) {
                    spinner.classList.add('d-none');
                }
                if (stopButton) {
                    stopButton.classList.add('d-none');
                    stopButton.disabled = false;
                }
                updateInfoBox('danger', t('workflow.errors.start_failed', { error: error.message }));
                updateLogStatus(t('workflow.log_status'), 'muted');
                currentRunContext = null;
            }
        }

        function startWorkflowPolling(jobId, triggerImmediate = true) {
            stopWorkflowPolling();
            currentWorkflowJobId = jobId;
            if (triggerImmediate) {
                pollWorkflowStatus();
            }
            workflowPollInterval = setInterval(pollWorkflowStatus, 5000);
        }

        function stopWorkflowPolling() {
            if (workflowPollInterval) {
                clearInterval(workflowPollInterval);
                workflowPollInterval = null;
            }
        }

        async function pollWorkflowStatus() {
            if (!currentWorkflowJobId) {
                return;
            }
            try {
                const response = await fetch(`/api/workflow-status/${currentWorkflowJobId}`);
                const result = await response.json();
                if (!response.ok || !result.success) {
                    throw new Error(result.error || t('workflow.errors.status_check_failed'));
                }
                const job = result.job;
                renderWorkflowStatus(job);
                if (['completed', 'failed', 'cancelled'].includes(job.status)) {
                    stopWorkflowPolling();
                    stopLogPolling();
                    currentWorkflowJobId = null;
                }
            } catch (error) {
                console.error('Állapot lekérdezési hiba:', error);
                stopWorkflowPolling();
                stopLogPolling();
                currentWorkflowJobId = null;
            }
        }

        function renderWorkflowStatus(job) {
            const badge = document.getElementById('workflowStatusBadge');
            const statusText = document.getElementById('workflowStatusText');
            const logLink = document.getElementById('workflowLogLink');
            const startButton = document.getElementById('startWorkflowBtn');
            const startSpinner = document.getElementById('workflowSpinner');
            const stopButton = document.getElementById('stopWorkflowBtn');
            const stopSpinner = document.getElementById('stopWorkflowSpinner');
            const stopLabel = document.getElementById('stopWorkflowLabel');
            const hint = document.getElementById('workflowStatusHint');

            const statusLabels = {
                queued: t('workflow.status_labels.queued'),
                running: t('workflow.status_labels.running'),
                completed: t('workflow.status_labels.completed'),
                failed: t('workflow.status_labels.failed'),
                cancelling: t('workflow.status_labels.cancelling'),
                cancelled: t('workflow.status_labels.cancelled')
            };

            if (!job) {
                if (badge) {
                    badge.textContent = t('workflow.status_none');
                    setWorkflowBadge(null);
                }
                if (statusText) {
                    statusText.textContent = collectScriptStepsForRun(0).length
                        ? t('workflow.status_texts.never_run')
                        : t('workflow.status_texts.add_steps');
                }
                if (logLink) {
                    logLink.innerHTML = '';
                }
                if (startButton) {
                    startButton.disabled = collectScriptStepsForRun(0).length === 0;
                    startButton.classList.remove('btn-processing');
                }
                if (startSpinner) {
                    startSpinner.classList.add('d-none');
                }
                if (stopButton) {
                    stopButton.classList.add('d-none');
                    stopButton.disabled = false;
                }
                if (hint) {
                    hint.textContent = t('workflow.hint');
                }
                updateLogStatus(t('workflow.log_status'));
                return;
            }

            if (badge) {
                badge.textContent = statusLabels[job.status] || job.status || t('workflow.status_labels.unknown');
                setWorkflowBadge(job.status);
            }

            const activeStatuses = ['queued', 'running', 'cancelling'];
            const isActive = activeStatuses.includes(job.status);

            if (isActive) {
                if (startButton) {
                    startButton.disabled = true;
                    startButton.classList.add('btn-processing');
                }
                if (startSpinner) {
                    startSpinner.classList.remove('d-none');
                }
                if (stopButton) {
                    stopButton.classList.remove('d-none');
                    stopButton.disabled = Boolean(job.cancel_requested);
                }
                if (stopSpinner) {
                    stopSpinner.classList.toggle('d-none', Boolean(job.cancel_requested));
                }
                if (stopLabel) {
                    stopLabel.textContent = job.cancel_requested
                        ? t('workflow.stop.requesting_label')
                        : t('workflow.buttons_extra.stop');
                }
                const activeJobId = currentWorkflowJobId || job.job_id;
                if (activeJobId) {
                    startLogPolling(activeJobId);
                }
            } else {
                if (startButton) {
                    startButton.disabled = collectScriptStepsForRun(0).length === 0;
                    startButton.classList.remove('btn-processing');
                }
                if (startSpinner) {
                    startSpinner.classList.add('d-none');
                }
                if (stopButton) {
                    stopButton.classList.add('d-none');
                    stopButton.disabled = false;
                }
                if (stopSpinner) {
                    stopSpinner.classList.add('d-none');
                }
                if (stopLabel) {
                    stopLabel.textContent = t('workflow.buttons_extra.stop');
                }
                stopLogPolling();
            }

            if (statusText) {
                const stepInfo = job.current_step && job.current_step.display_name
                    ? `${job.current_step.index}/${job.current_step.total} – ${job.current_step.display_name}`
                    : '';
                statusText.textContent = [job.message || '', stepInfo].filter(Boolean).join(' ');
            }

            if (hint) {
                if (job.status === 'queued') {
                    hint.textContent = t('workflow.log_messages.queued');
                } else if (job.status === 'running') {
                    hint.textContent = t('workflow.log_messages.running');
                } else if (job.status === 'cancelling') {
                    hint.textContent = t('workflow.log_messages.cancelling');
                } else if (job.status === 'failed') {
                    hint.textContent = t('workflow.log_messages.failed');
                } else if (job.status === 'cancelled') {
                    hint.textContent = t('workflow.log_messages.cancelled');
                } else if (job.status === 'completed') {
                    hint.textContent = t('workflow.log_messages.completed');
                }
            }

            if (logLink) {
                if (job.log && job.log.url) {
                    logLink.innerHTML = `<a href="${job.log.url}" target="_blank" rel="noopener">${t('workflow.log_messages.open_log')}</a>`;
                } else {
                    logLink.innerHTML = '';
                }
            }

            if (job.status === 'failed') {
                updateLogStatus(t('workflow.log_messages.failed_short'), 'danger');
            } else if (job.status === 'cancelled') {
                updateLogStatus(t('workflow.log_messages.cancelled_short'), 'danger');
            } else if (job.status === 'completed') {
                updateLogStatus(t('workflow.log_messages.completed_short'), 'success');
            } else if (isActive) {
                updateLogStatus(t('workflow.log_messages.running_short'), 'info');
            }

            handleRunContextJobUpdate(job);
        }

        function setWorkflowBadge(status) {
            const badge = document.getElementById('workflowStatusBadge');
            if (!badge) return;
            badge.classList.remove('bg-secondary', 'bg-primary', 'bg-success', 'bg-danger', 'bg-warning', 'bg-info');
            switch (status) {
                case 'queued':
                    badge.classList.add('bg-info');
                    break;
                case 'running':
                    badge.classList.add('bg-primary');
                    break;
                case 'completed':
                    badge.classList.add('bg-success');
                    break;
                case 'failed':
                    badge.classList.add('bg-danger');
                    break;
                case 'cancelling':
                case 'cancelled':
                    badge.classList.add('bg-warning');
                    break;
                default:
                    badge.classList.add('bg-secondary');
                    break;
            }
        }

        function updateInfoBox(level, message) {
            const infoBox = document.getElementById('workflowInfo');
            if (!infoBox) {
                return;
            }
            infoBox.classList.remove('alert-info', 'alert-secondary', 'alert-warning', 'alert-danger', 'alert-success');
            const classMap = {
                info: 'alert-info',
                success: 'alert-success',
                warning: 'alert-warning',
                danger: 'alert-danger',
                secondary: 'alert-secondary'
            };
            infoBox.classList.add(classMap[level] || 'alert-secondary');
            if (message !== undefined) {
                infoBox.textContent = message;
            }
            infoBox.classList.remove('d-none');
        }

        function updateLogStatus(text, tone = 'muted') {
            const statusEl = document.getElementById('workflowLogStatus');
            if (!statusEl) return;
            statusEl.textContent = text;
            statusEl.classList.remove('text-muted', 'text-danger', 'text-success', 'text-info');
            const toneClass = {
                muted: 'text-muted',
                danger: 'text-danger',
                success: 'text-success',
                info: 'text-info'
            }[tone] || 'text-muted';
            statusEl.classList.add(toneClass);
        }

        function startLogPolling(jobId, immediate = false) {
            stopLogPolling();
            currentLogJobId = jobId;
            if (immediate) {
                fetchWorkflowLog();
            }
            workflowLogInterval = setInterval(fetchWorkflowLog, 4000);
        }

        function stopLogPolling(silent = false) {
            if (workflowLogInterval) {
                clearInterval(workflowLogInterval);
                workflowLogInterval = null;
            }
            if (!silent) {
                currentLogJobId = null;
            }
        }

        async function fetchWorkflowLog() {
            if (!currentWorkflowJobId && !currentLogJobId) {
                return;
            }
            const jobId = currentWorkflowJobId || currentLogJobId;
            try {
                const response = await fetch(`/api/workflow-log/${jobId}`);
                const result = await response.json();
                if (!response.ok || !result.success) {
                    throw new Error(result.error || t('workflow.errors.log_fetch_failed'));
                }
                const logContent = document.getElementById('workflowLogContent');
                if (logContent) {
                    logContent.textContent = result.log || '—';
                }
                updateLogStatus(
                    result.completed ? t('workflow.log_messages.completed_short') : t('workflow.log_messages.running_short'),
                    result.completed ? 'success' : 'info'
                );
            } catch (error) {
                console.error('Log olvasási hiba:', error);
                updateLogStatus(t('workflow.log_messages.log_refresh_failed'), 'danger');
            }
        }

        async function stopWorkflow() {
            if (!currentWorkflowJobId) {
                return;
            }
            const stopButton = document.getElementById('stopWorkflowBtn');
            const stopSpinner = document.getElementById('stopWorkflowSpinner');
            const stopLabel = document.getElementById('stopWorkflowLabel');
            const statusText = document.getElementById('workflowStatusText');

            if (stopButton) {
                stopButton.disabled = true;
            }
            if (stopSpinner) {
                stopSpinner.classList.remove('d-none');
            }
            if (stopLabel) {
                stopLabel.textContent = t('workflow.stop.requesting');
            }

            try {
                const response = await fetch(`/api/stop-workflow/${currentWorkflowJobId}`, {
                    method: 'POST'
                });
                const result = await response.json();
                if (!response.ok || !result.success) {
                    throw new Error(result.error || t('workflow.stop.unknown_error'));
                }
                if (statusText) {
                    statusText.textContent = result.message || t('workflow.log_messages.stop_pending');
                }
                updateLogStatus(t('workflow.log_messages.stop_pending'), 'info');
            } catch (error) {
                console.error('Workflow megszakítási hiba:', error);
                alert(t('workflow.errors.stop_failed', { error: error.message }));
                if (stopButton) {
                    stopButton.disabled = false;
                }
                if (stopSpinner) {
                    stopSpinner.classList.add('d-none');
                }
                if (stopLabel) {
                    stopLabel.textContent = t('workflow.buttons_extra.stop');
                }
            }
        }


        document.addEventListener('DOMContentLoaded', () => {
            const languageSelect = document.getElementById('languageSelect');
            const t = typeof window.translateText === 'function'
                ? window.translateText
                : (key, replacements = {}, fallback) => (fallback || key);

            if (languageSelect) {
                languageSelect.addEventListener('change', event => {
                    const selected = event.target.value;
                    const targetUrl = new URL(window.location.href);
                    if (selected) {
                        targetUrl.searchParams.set('lang', selected);
                    } else {
                        targetUrl.searchParams.delete('lang');
                    }
                    window.location.assign(targetUrl.toString());
                });
            }

            initPreviewModals();
            if (window.AudioTrimmer && typeof AudioTrimmer.init === 'function') {
                AudioTrimmer.init({
                    projectName: PROJECT_NAME,
                    buildWorkdirUrl: buildWorkdirUrl,
                    refreshDirectory: refreshDirectory,
                    reloadFileBrowser: reloadFileBrowser,
                    cssEscape: cssEscape
                });
            }
            initProjectFileActions();
            initTtsControls();
            initFileBrowser();
            initWorkflowModals();
            initWorkflowButtons();
            initWorkflowContextMenu();
            initWorkflowKeyModal();
            const refreshLogBtn = document.getElementById('refreshWorkflowLogBtn');
            if (refreshLogBtn) {
                refreshLogBtn.addEventListener('click', () => fetchWorkflowLog());
            }
            loadWorkflowOptions(PROJECT_NAME);
        });
