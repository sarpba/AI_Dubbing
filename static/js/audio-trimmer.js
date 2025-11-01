(function (global) {
    'use strict';

    const MIN_TRIM_DURATION_SECONDS = 0.05;
    const TRIM_SLIDER_STEP = 0.01;

    const state = {
        initialized: false,
        projectName: '',
        buildWorkdirUrl: null,
        refreshDirectory: null,
        reloadFileBrowser: null,
        cssEscape: null,
        options: {},
        modalElement: null,
        modalInstance: null,
        waveSurfer: null,
        waveSurferRegions: null,
        waveSurferRegion: null,
        audioTrimInProgress: false,
        currentAudioFilePath: '',
        currentAudioFileName: '',
        elements: {
            waveform: null,
            waveformLoading: null,
            startDisplay: null,
            endDisplay: null,
            durationDisplay: null,
            startSlider: null,
            endSlider: null,
            resetButton: null,
            statusText: null,
            errorAlert: null,
            saveSpinner: null,
            saveButton: null,
            title: null
        }
    };

    function applyOptions(options) {
        state.options = Object.assign({
            modalId: 'audioPreviewModal',
            waveformElementId: 'audioWaveform',
            waveformLoadingId: 'audioWaveformLoading',
            startDisplayId: 'trimStartDisplay',
            endDisplayId: 'trimEndDisplay',
            durationDisplayId: 'trimDurationDisplay',
            startSliderId: 'trimStartRange',
            endSliderId: 'trimEndRange',
            resetButtonId: 'trimResetSelectionBtn',
            statusTextId: 'trimStatusText',
            errorAlertId: 'trimErrorAlert',
            saveSpinnerId: 'trimSaveSpinner',
            saveButtonId: 'trimSaveBtn',
            titleId: 'audioPreviewTitle',
            fileBrowserRootId: 'projectFileBrowser',
            minTrimDurationSeconds: MIN_TRIM_DURATION_SECONDS,
            trimSliderStep: TRIM_SLIDER_STEP
        }, options || {});
    }

    function cacheElements() {
        const ids = state.options;
        state.elements.waveform = document.getElementById(ids.waveformElementId);
        state.elements.waveformLoading = document.getElementById(ids.waveformLoadingId);
        state.elements.startDisplay = document.getElementById(ids.startDisplayId);
        state.elements.endDisplay = document.getElementById(ids.endDisplayId);
        state.elements.durationDisplay = document.getElementById(ids.durationDisplayId);
        state.elements.startSlider = document.getElementById(ids.startSliderId);
        state.elements.endSlider = document.getElementById(ids.endSliderId);
        state.elements.resetButton = document.getElementById(ids.resetButtonId);
        state.elements.statusText = document.getElementById(ids.statusTextId);
        state.elements.errorAlert = document.getElementById(ids.errorAlertId);
        state.elements.saveSpinner = document.getElementById(ids.saveSpinnerId);
        state.elements.saveButton = document.getElementById(ids.saveButtonId);
        state.elements.title = document.getElementById(ids.titleId);
        state.modalElement = document.getElementById(ids.modalId);
    }

    function setupControlListeners() {
        const startSlider = state.elements.startSlider;
        const endSlider = state.elements.endSlider;
        const resetButton = state.elements.resetButton;
        const saveButton = state.elements.saveButton;

        if (startSlider) {
            startSlider.disabled = true;
            startSlider.addEventListener('input', handleTrimStartSliderInput);
        }
        if (endSlider) {
            endSlider.disabled = true;
            endSlider.addEventListener('input', handleTrimEndSliderInput);
        }
        if (resetButton) {
            resetButton.disabled = true;
            resetButton.addEventListener('click', () => {
                clearTrimSelection();
            });
        }
        if (saveButton) {
            saveButton.disabled = true;
            saveButton.addEventListener('click', () => {
                saveTrimSelection();
            });
        }
        if (state.modalElement) {
            state.modalElement.addEventListener('hidden.bs.modal', handleModalHidden);
        }
    }

    function init(options) {
        if (state.initialized) {
            // Allow runtime updates of project-dependent values.
            state.projectName = options && options.projectName ? options.projectName : state.projectName;
            state.buildWorkdirUrl = options && options.buildWorkdirUrl ? options.buildWorkdirUrl : state.buildWorkdirUrl;
            state.refreshDirectory = options && options.refreshDirectory ? options.refreshDirectory : state.refreshDirectory;
            state.reloadFileBrowser = options && options.reloadFileBrowser ? options.reloadFileBrowser : state.reloadFileBrowser;
            state.cssEscape = options && options.cssEscape ? options.cssEscape : state.cssEscape;
            return;
        }

        applyOptions(options);
        cacheElements();

        state.projectName = options && options.projectName ? options.projectName : '';
        state.buildWorkdirUrl = options && options.buildWorkdirUrl ? options.buildWorkdirUrl : null;
        state.refreshDirectory = options && options.refreshDirectory ? options.refreshDirectory : null;
        state.reloadFileBrowser = options && options.reloadFileBrowser ? options.reloadFileBrowser : null;
        state.cssEscape = options && options.cssEscape ? options.cssEscape : null;

        setupControlListeners();
        state.initialized = true;
    }

    function ensureInitialized() {
        if (!state.initialized) {
            throw new Error('AudioTrimmer has not been initialized.');
        }
    }

    function toggleWaveformLoading(isLoading, message) {
        const overlay = state.elements.waveformLoading;
        if (!overlay) {
            return;
        }
        overlay.textContent = message || 'Betöltés...';
        if (isLoading) {
            overlay.classList.remove('d-none');
        } else {
            overlay.classList.add('d-none');
        }
    }

    function setTrimStatus(message) {
        const statusElement = state.elements.statusText;
        if (!statusElement) {
            return;
        }
        statusElement.textContent = message || '';
    }

    function setTrimError(message) {
        const errorElement = state.elements.errorAlert;
        if (!errorElement) {
            return;
        }
        if (message) {
            errorElement.textContent = message;
            errorElement.classList.remove('d-none');
        } else {
            errorElement.textContent = '';
            errorElement.classList.add('d-none');
        }
    }

    function toggleTrimSavingSpinner(visible) {
        const spinner = state.elements.saveSpinner;
        if (!spinner) {
            return;
        }
        if (visible) {
            spinner.classList.remove('d-none');
        } else {
            spinner.classList.add('d-none');
        }
    }

    function setTrimSaveEnabled(shouldEnable) {
        const saveButton = state.elements.saveButton;
        if (!saveButton) {
            return;
        }
        const enabled = shouldEnable && !state.audioTrimInProgress;
        saveButton.disabled = !enabled;
    }

    function updateTrimSelectionDisplay() {
        const region = state.waveSurferRegion;
        const duration = state.waveSurfer ? state.waveSurfer.getDuration() : 0;
        const startElement = state.elements.startDisplay;
        const endElement = state.elements.endDisplay;
        const durationElement = state.elements.durationDisplay;
        const resetButton = state.elements.resetButton;
        const startSlider = state.elements.startSlider;
        const endSlider = state.elements.endSlider;

        if (region && duration > 0) {
            const start = Math.max(0, region.start || 0);
            const end = Math.max(0, region.end || 0);
            const trimmedDuration = Math.max(0, end - start);
            if (startElement) {
                startElement.textContent = `${start.toFixed(2)} s`;
            }
            if (endElement) {
                endElement.textContent = `${end.toFixed(2)} s`;
            }
            if (durationElement) {
                durationElement.textContent = `${trimmedDuration.toFixed(2)} s`;
            }
            if (resetButton) {
                resetButton.disabled = false;
            }
            if (startSlider) {
                startSlider.disabled = false;
                startSlider.min = '0';
                startSlider.max = duration.toFixed(2);
                startSlider.step = state.options.trimSliderStep.toString();
                startSlider.value = start.toFixed(2);
            }
            if (endSlider) {
                endSlider.disabled = false;
                endSlider.min = '0';
                endSlider.max = duration.toFixed(2);
                endSlider.step = state.options.trimSliderStep.toString();
                endSlider.value = end.toFixed(2);
            }
            setTrimSaveEnabled(true);
        } else {
            if (startElement) {
                startElement.textContent = '—';
            }
            if (endElement) {
                endElement.textContent = '—';
            }
            if (durationElement) {
                durationElement.textContent = '—';
            }
            if (resetButton) {
                resetButton.disabled = true;
            }
            if (startSlider) {
                startSlider.disabled = true;
                startSlider.value = '0';
            }
            if (endSlider) {
                endSlider.disabled = true;
                endSlider.value = '0';
            }
            setTrimSaveEnabled(false);
        }
    }

    function clearTrimSelection(skipUpdate) {
        const waveSurfer = state.waveSurfer;
        const duration = waveSurfer ? waveSurfer.getDuration() : 0;
        if (duration > 0) {
            if (state.waveSurferRegion) {
                try {
                    state.waveSurferRegion.setOptions({ start: 0, end: duration });
                } catch (error) {
                    console.debug('A kijelölés visszaállítása sikertelen:', error);
                }
            } else if (state.waveSurferRegions && typeof state.waveSurferRegions.addRegion === 'function') {
                try {
                    state.waveSurferRegion = state.waveSurferRegions.addRegion({
                        start: 0,
                        end: duration,
                        color: 'rgba(13, 110, 253, 0.2)'
                    });
                } catch (error) {
                    console.debug('Nem sikerült új kijelölést létrehozni:', error);
                }
            }
        }

        if (!skipUpdate) {
            updateTrimSelectionDisplay();
        }
    }

    function handleTrimStartSliderInput() {
        const region = state.waveSurferRegion;
        const waveSurfer = state.waveSurfer;
        const startSlider = state.elements.startSlider;
        const endSlider = state.elements.endSlider;

        if (!region || !waveSurfer || !startSlider) {
            return;
        }
        const duration = waveSurfer.getDuration() || 0;
        if (duration <= 0) {
            return;
        }
        let start = parseFloat(startSlider.value);
        if (!Number.isFinite(start)) {
            return;
        }
        start = Math.min(Math.max(0, start), duration);
        let end = Math.max(0, region.end || duration);
        if (end - start < state.options.minTrimDurationSeconds) {
            end = Math.min(duration, start + state.options.minTrimDurationSeconds);
            if (endSlider) {
                endSlider.value = end.toFixed(2);
            }
        }
        try {
            region.setOptions({ start, end });
        } catch (error) {
            console.debug('Nem sikerült frissíteni a kijelölés kezdetét:', error);
        }
        updateTrimSelectionDisplay();
    }

    function handleTrimEndSliderInput() {
        const region = state.waveSurferRegion;
        const waveSurfer = state.waveSurfer;
        const startSlider = state.elements.startSlider;
        const endSlider = state.elements.endSlider;

        if (!region || !waveSurfer || !endSlider) {
            return;
        }
        const duration = waveSurfer.getDuration() || 0;
        if (duration <= 0) {
            return;
        }
        let end = parseFloat(endSlider.value);
        if (!Number.isFinite(end)) {
            return;
        }
        end = Math.min(Math.max(0, end), duration);
        let start = Math.max(0, region.start || 0);
        if (end - start < state.options.minTrimDurationSeconds) {
            start = Math.max(0, end - state.options.minTrimDurationSeconds);
            if (startSlider) {
                startSlider.value = start.toFixed(2);
            }
        }
        try {
            region.setOptions({ start, end });
        } catch (error) {
            console.debug('Nem sikerült frissíteni a kijelölés végét:', error);
        }
        updateTrimSelectionDisplay();
    }

    function resetTrimState() {
        state.currentAudioFilePath = '';
        state.currentAudioFileName = '';
        state.audioTrimInProgress = false;
        setTrimStatus('');
        setTrimError('');
        toggleTrimSavingSpinner(false);
        toggleWaveformLoading(false);
        if (state.elements.startSlider) {
            state.elements.startSlider.disabled = true;
            state.elements.startSlider.value = '0';
            state.elements.startSlider.min = '0';
            state.elements.startSlider.max = '0';
        }
        if (state.elements.endSlider) {
            state.elements.endSlider.disabled = true;
            state.elements.endSlider.value = '0';
            state.elements.endSlider.min = '0';
            state.elements.endSlider.max = '0';
        }
        state.waveSurferRegion = null;
        updateTrimSelectionDisplay();
    }

    function refreshFileBrowserAfterTrim(relativePath) {
        if (!relativePath) {
            return;
        }
        const browserRoot = document.getElementById(state.options.fileBrowserRootId);
        if (!browserRoot) {
            return;
        }
        const cssEscape = state.cssEscape || ((value) => (value || '').replace(/[^a-zA-Z0-9_\-]/g, (char) => `\\${char}`));

        const fileSelector = `.file-browser-item[data-file-path="${cssEscape(relativePath)}"]`;
        let fileRow = null;
        try {
            fileRow = browserRoot.querySelector(fileSelector);
        } catch (error) {
            console.debug('Nem sikerült elérni a fájlt a listában:', error);
        }

        if (fileRow && state.refreshDirectory) {
            const parentDirectory = fileRow.closest('details.file-browser-directory');
            if (parentDirectory) {
                state.refreshDirectory(parentDirectory);
            }
            return;
        }

        const parentPath = relativePath.split('/').slice(0, -1).join('/');
        if (parentPath && state.refreshDirectory) {
            const parentSelector = `details.file-browser-directory[data-path="${cssEscape(parentPath)}"]`;
            let parentDetails = null;
            try {
                parentDetails = browserRoot.querySelector(parentSelector);
            } catch (error) {
                console.debug('Nem sikerült elérni a szülő könyvtárat:', error);
            }
            if (parentDetails) {
                state.refreshDirectory(parentDetails);
                return;
            }
        }

        if (state.reloadFileBrowser) {
            state.reloadFileBrowser('');
        }
    }

    function initWaveSurfer() {
        if (!global.WaveSurfer || typeof global.WaveSurfer.create !== 'function') {
            console.warn('WaveSurfer könyvtár nem érhető el.');
            return null;
        }
        if (state.waveSurfer) {
            return state.waveSurfer;
        }

        if (!state.elements.waveform) {
            console.warn('Hiányzik a hullámforma konténer.');
            return null;
        }

        const computedStyles = getComputedStyle(document.documentElement);
        const waveColor = computedStyles.getPropertyValue('--bs-border-color') || '#adb5bd';
        const progressColor = computedStyles.getPropertyValue('--bs-primary-color') || computedStyles.getPropertyValue('--bs-primary') || '#0d6efd';
        const cursorColor = computedStyles.getPropertyValue('--bs-primary') || '#0d6efd';

        state.waveSurfer = global.WaveSurfer.create({
            container: state.elements.waveform,
            waveColor: waveColor.trim() || '#adb5bd',
            progressColor: progressColor.trim() || '#0d6efd',
            cursorColor: cursorColor.trim() || '#0d6efd',
            height: 128,
            responsive: true,
            normalize: true
        });

        const regionsFactory = (global.WaveSurfer && global.WaveSurfer.Regions && typeof global.WaveSurfer.Regions.create === 'function')
            ? global.WaveSurfer.Regions
            : (global.RegionsPlugin && typeof global.RegionsPlugin.create === 'function' ? global.RegionsPlugin : null);

        if (!regionsFactory) {
            console.warn('WaveSurfer Regions plugin nem érhető el, a trimmelés nem használható.');
            setTrimError('A hullámforma kijelöléshez szükséges bővítmény nem érhető el.');
        } else {
            state.waveSurferRegions = state.waveSurfer.registerPlugin(regionsFactory.create({
                dragSelection: {
                    slop: 5
                }
            }));

            if (state.waveSurferRegions && typeof state.waveSurferRegions.on === 'function') {
                state.waveSurferRegions.on('region-created', function (region) {
                    if (state.waveSurferRegion && state.waveSurferRegion.id !== region.id) {
                        try {
                            state.waveSurferRegion.remove();
                        } catch (error) {
                            console.debug('Korábbi kijelölés eltávolítása sikertelen:', error);
                        }
                    }
                    state.waveSurferRegion = region;
                    if (region && typeof region.setOptions === 'function') {
                        region.setOptions({ color: 'rgba(13, 110, 253, 0.2)' });
                    }
                    if (region && typeof region.on === 'function') {
                        region.on('update-end', function () {
                            updateTrimSelectionDisplay();
                        });
                        region.on('remove', function () {
                            if (state.waveSurferRegion && state.waveSurferRegion.id === region.id) {
                                state.waveSurferRegion = null;
                                updateTrimSelectionDisplay();
                            }
                        });
                    }
                    updateTrimSelectionDisplay();
                });

                state.waveSurferRegions.on('region-updated', function (region) {
                    if (state.waveSurferRegion && region && region.id === state.waveSurferRegion.id) {
                        updateTrimSelectionDisplay();
                    }
                });
            }
        }

        state.waveSurfer.on('ready', function () {
            const duration = state.waveSurfer ? state.waveSurfer.getDuration() || 0 : 0;
            setTrimError('');
            if (state.waveSurferRegions && typeof state.waveSurferRegions.clearRegions === 'function') {
                state.waveSurferRegions.clearRegions();
            }
            if (duration > 0 && state.waveSurferRegions && typeof state.waveSurferRegions.addRegion === 'function') {
                try {
                    state.waveSurferRegion = state.waveSurferRegions.addRegion({
                        start: 0,
                        end: duration,
                        color: 'rgba(13, 110, 253, 0.2)',
                        resize: true,
                        drag: true
                    });
                } catch (error) {
                    console.debug('Nem sikerült alapértelmezett kijelölést létrehozni:', error);
                }
            } else if (duration <= 0) {
                setTrimError('A hangfájl hossza nem állapítható meg.');
            }
            toggleWaveformLoading(false);
            if (duration > 0) {
                setTrimStatus('A teljes fájl kijelölve. A hullámformán vagy a csúszkákkal módosíthatod a határokat.');
            } else {
                setTrimStatus('');
            }
            updateTrimSelectionDisplay();
            setTrimSaveEnabled(duration > 0);
            try {
                state.waveSurfer.stop();
            } catch (error) {
                // ignore
            }
            const playPromise = state.waveSurfer.play(0);
            if (playPromise && typeof playPromise.catch === 'function') {
                playPromise.catch((error) => {
                    console.debug('Automatikus lejátszás blokkolva lehet:', error);
                });
            }
        });

        state.waveSurfer.on('loading', function (progress) {
            if (typeof progress === 'number') {
                const message = `Betöltés: ${Math.round(progress)}%`;
                toggleWaveformLoading(true, message);
                setTrimStatus(message);
            } else {
                toggleWaveformLoading(true, 'Betöltés...');
                setTrimStatus('Audió betöltése...');
            }
        });

        state.waveSurfer.on('error', function (error) {
            console.error('WaveSurfer hiba:', error);
            toggleWaveformLoading(true, 'Nem sikerült betölteni az audiót.');
            setTrimError('Nem sikerült betölteni az audiót.');
        });

        return state.waveSurfer;
    }

    function prepareTrimModal(fileName, filePath) {
        state.currentAudioFilePath = filePath || '';
        state.currentAudioFileName = fileName || '';
        state.audioTrimInProgress = false;
        setTrimError('');
        setTrimStatus('Audió betöltése folyamatban...');
        toggleTrimSavingSpinner(false);
        if (state.elements.startSlider) {
            state.elements.startSlider.disabled = true;
            state.elements.startSlider.value = '0';
        }
        if (state.elements.endSlider) {
            state.elements.endSlider.disabled = true;
            state.elements.endSlider.value = '0';
        }
        state.waveSurferRegion = null;
        updateTrimSelectionDisplay();
    }

    async function saveTrimSelection() {
        const region = state.waveSurferRegion;
        if (!region) {
            setTrimError('Előbb jelöld ki a menteni kívánt hangrészletet.');
            return;
        }
        if (!state.currentAudioFilePath) {
            setTrimError('Hiányzik az audió fájl elérési útja.');
            return;
        }
        const start = Math.max(0, region.start || 0);
        const end = Math.max(0, region.end || 0);
        if (end <= start) {
            setTrimError('A kijelölés vége nagyobbnak kell lennie a kezdetnél.');
            return;
        }
        if ((end - start) < state.options.minTrimDurationSeconds) {
            setTrimError(`A kijelölésnek legalább ${state.options.minTrimDurationSeconds.toFixed(2)} másodperc hosszúnak kell lennie.`);
            return;
        }
        if (!state.projectName) {
            setTrimError('Hiányzó projektnév.');
            return;
        }

        state.audioTrimInProgress = true;
        setTrimError('');
        setTrimStatus('Mentés folyamatban...');
        setTrimSaveEnabled(false);
        toggleTrimSavingSpinner(true);

        let response = null;
        let result = {};
        try {
            response = await fetch('/api/project-audio/trim', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    projectName: state.projectName,
                    filePath: state.currentAudioFilePath,
                    start: start,
                    end: end,
                    outputName: ''
                })
            });
            try {
                result = await response.json();
            } catch (error) {
                result = {};
            }
            if (!response.ok || !result.success) {
                throw new Error((result && result.error) || 'Nem sikerült elmenteni a kijelölt hangrészletet.');
            }
            const savedPath = result.saved_path || state.currentAudioFilePath;
            const savedName = result.saved_name || state.currentAudioFileName;
            setTrimStatus('Mentés sikeresen befejezve.');
            refreshFileBrowserAfterTrim(savedPath);
            if (state.waveSurferRegions && typeof state.waveSurferRegions.clearRegions === 'function') {
                state.waveSurferRegions.clearRegions();
            } else {
                clearTrimSelection(true);
            }
            const waveSurfer = initWaveSurfer();
            if (waveSurfer) {
                toggleWaveformLoading(true, 'Betöltés...');
                setTrimStatus('Frissített audió betöltése...');
                try {
                    waveSurfer.stop();
                } catch (error) {
                    // ignore
                }
                if (state.buildWorkdirUrl) {
                    waveSurfer.load(state.buildWorkdirUrl(savedPath));
                } else {
                    waveSurfer.load(savedPath);
                }
            }
            state.currentAudioFilePath = savedPath;
            state.currentAudioFileName = savedName;
        } catch (error) {
            console.error('Trim mentés hiba:', error);
            setTrimError(error.message || 'Nem sikerült elmenteni a kijelölt hangrészletet.');
            setTrimStatus('');
        } finally {
            state.audioTrimInProgress = false;
            toggleTrimSavingSpinner(false);
            updateTrimSelectionDisplay();
        }
    }

    function getModalInstance() {
        if (!state.modalElement) {
            return null;
        }
        if (!state.modalInstance) {
            if (!global.bootstrap || !global.bootstrap.Modal) {
                console.warn('Bootstrap Modal nem érhető el.');
                return null;
            }
            state.modalInstance = new global.bootstrap.Modal(state.modalElement);
        }
        return state.modalInstance;
    }

    function handleModalHidden() {
        if (state.waveSurfer && typeof state.waveSurfer.stop === 'function') {
            state.waveSurfer.stop();
        }
        resetTrimState();
        if (state.elements.title) {
            state.elements.title.textContent = '';
        }
    }

    function showAudioPreview(fileName, fileUrl, filePath) {
        ensureInitialized();
        const modalElement = state.modalElement;
        if (!modalElement) {
            global.open(fileUrl, '_blank');
            return;
        }
        const modal = getModalInstance();
        if (!modal) {
            global.open(fileUrl, '_blank');
            return;
        }
        if (state.elements.title) {
            state.elements.title.textContent = fileName || 'Audió szerkesztése';
        }
        prepareTrimModal(fileName, filePath);

        const loadAudioIntoWaveSurfer = () => {
            const waveSurfer = initWaveSurfer();
            if (!waveSurfer) {
                setTrimError('A hullámforma lejátszó nem érhető el ebben a böngészőben.');
                toggleWaveformLoading(true, 'Hullámforma nem érhető el.');
                return;
            }
            if (state.waveSurferRegions && typeof state.waveSurferRegions.clearRegions === 'function') {
                state.waveSurferRegions.clearRegions();
            } else {
                clearTrimSelection(true);
            }
            toggleWaveformLoading(true, 'Betöltés...');
            setTrimStatus('Audió betöltése...');
            try {
                waveSurfer.stop();
            } catch (error) {
                // ignore
            }
            waveSurfer.load(fileUrl);
        };

        if (state.modalElement && state.modalElement.classList.contains('show')) {
            loadAudioIntoWaveSurfer();
        } else if (state.modalElement) {
            const handler = () => {
                state.modalElement.removeEventListener('shown.bs.modal', handler);
                loadAudioIntoWaveSurfer();
            };
            state.modalElement.addEventListener('shown.bs.modal', handler);
        } else {
            loadAudioIntoWaveSurfer();
        }

        modal.show();
    }

    const api = {
        init,
        showPreview: showAudioPreview
    };

    if (typeof module !== 'undefined' && module.exports) {
        module.exports = api;
    } else {
        global.AudioTrimmer = api;
    }
})(typeof window !== 'undefined' ? window : globalThis);
