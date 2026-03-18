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

        const fileBrowserApi = (window.ProjectFileBrowser && window.ProjectFileBrowser.createProjectFileBrowser)
            ? window.ProjectFileBrowser.createProjectFileBrowser({
                projectName: PROJECT_NAME,
                defaultChunkSizeBytes: DEFAULT_CHUNK_SIZE_BYTES,
                audioExtensions: AUDIO_EXTENSIONS,
                videoExtensions: VIDEO_EXTENSIONS,
                textPreviewExtensions: TEXT_PREVIEW_EXTENSIONS,
                t,
                getFileExtension,
                buildWorkdirUrl,
                createMetadataElement,
                createFailedOriginalTextElement
            })
            : null;

        function createFileTreeList(entries) {
            return fileBrowserApi.createFileTreeList
                ? fileBrowserApi.createFileTreeList(entries)
                : document.createElement('ul');
        }

        async function refreshDirectory(detailsElement) {
            if (!fileBrowserApi || !fileBrowserApi.refreshDirectory) {
                return;
            }
            return fileBrowserApi.refreshDirectory(detailsElement);
        }

        function cssEscape(value) {
            if (!fileBrowserApi || !fileBrowserApi.cssEscape) {
                if (window.CSS && typeof window.CSS.escape === 'function') {
                    return window.CSS.escape(value);
                }
                return (value || '').replace(/[^a-zA-Z0-9_\-]/g, (char) => `\\${char}`);
            }
            return fileBrowserApi.cssEscape(value);
        }

        async function reloadFileBrowser(targetPath = '') {
            if (!fileBrowserApi) {
                return;
            }
            return fileBrowserApi.reloadFileBrowser(targetPath);
        }

        function openUploadDialog(targetPath = '') {
            if (!fileBrowserApi) {
                return;
            }
            fileBrowserApi.openUploadDialog(targetPath);
        }

        async function uploadFilesToTtsRequest(targetPath, files) {
            if (!fileBrowserApi) {
                return { success: false };
            }
            return fileBrowserApi.uploadFilesToTtsRequest(targetPath, files);
        }

        function initPreviewModals() {
            if (fileBrowserApi) {
                fileBrowserApi.initPreviewModals();
            }
        }

        function initProjectFileActions() {
            if (fileBrowserApi) {
                fileBrowserApi.initProjectFileActions();
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
            if (fileBrowserApi) {
                fileBrowserApi.initFileBrowser();
            }
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

        const workflowEditorApi = (window.ProjectWorkflowEditor && window.ProjectWorkflowEditor.createProjectWorkflowEditor)
            ? window.ProjectWorkflowEditor.createProjectWorkflowEditor({
                projectName: PROJECT_NAME,
                t,
                cloneSteps,
                markWorkflowDirty,
                updateInfoBox,
                handleStartWorkflow,
                stopWorkflow,
                collectWorkflowState,
                collectScriptStepsForRun,
                normalizeWorkflowStep,
                findWidgetById,
                findScriptById,
                getMissingParams,
                describeStepParameters,
                describeWidgetParameters,
                formatScriptDirectoryName,
                updateMissingBadge,
                handleWidgetContinue,
                openWorkflowParams,
                openWorkflowWidgetParams,
                moveWorkflowStep,
                removeWorkflowStep,
                updateCycleDisplay,
                addWorkflowStep,
                getWorkflowTemplates: () => workflowTemplates,
                setWorkflowTemplates: value => { workflowTemplates = value; },
                getCurrentTemplateId: () => currentTemplateId,
                setCurrentTemplateId: value => { currentTemplateId = value; },
                getDefaultWorkflow: () => defaultWorkflow,
                setDefaultWorkflow: value => { defaultWorkflow = value; },
                getWorkflowSteps: () => workflowSteps,
                setWorkflowSteps: value => { workflowSteps = value; },
                getWorkflowSaveModal: () => workflowSaveModal,
                setWorkflowSaveModal: value => { workflowSaveModal = value; },
                getWorkflowStepModal: () => workflowStepModal,
                getCycleState: () => cycleState,
                getCurrentWorkflowJobId: () => currentWorkflowJobId,
                getAvailableScripts: () => availableScripts
            })
            : null;

        function findTemplateById(templateId) {
            return workflowEditorApi ? workflowEditorApi.findTemplateById(templateId) : null;
        }

        function populateWorkflowTemplateSelect(selectedId) {
            if (workflowEditorApi) {
                workflowEditorApi.populateWorkflowTemplateSelect(selectedId);
            }
        }

        async function refreshWorkflowTemplates(selectedId = currentTemplateId) {
            if (!workflowEditorApi) {
                return;
            }
            return workflowEditorApi.refreshWorkflowTemplates(selectedId);
        }

        async function loadWorkflowTemplateById(templateId, showMessage = true) {
            if (!workflowEditorApi) {
                return;
            }
            return workflowEditorApi.loadWorkflowTemplateById(templateId, showMessage);
        }

        function showSaveWorkflowModal() {
            if (workflowEditorApi) {
                workflowEditorApi.showSaveWorkflowModal();
            }
        }

        async function saveWorkflowTemplateFromModal() {
            if (!workflowEditorApi) {
                return;
            }
            return workflowEditorApi.saveWorkflowTemplateFromModal();
        }

        function initWorkflowButtons() {
            if (workflowEditorApi) {
                workflowEditorApi.initWorkflowButtons();
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
            if (workflowEditorApi) {
                workflowEditorApi.renderWorkflowSteps();
            }
        }

        function openWorkflowStepPicker() {
            if (workflowEditorApi) {
                workflowEditorApi.openWorkflowStepPicker();
            }
        }

        function populateWorkflowStepList(filterText) {
            if (workflowEditorApi) {
                workflowEditorApi.populateWorkflowStepList(filterText);
            }
        }

        /* removed from workspace:
           template CRUD, workflow button wiring, workflow table render, step picker list
           now handled by project-workflow-editor.js
        */

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
