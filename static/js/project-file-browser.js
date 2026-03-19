(function() {
    function createProjectFileBrowser(ctx) {
        let videoPreviewModal = null;
        let jsonPreviewModal = null;
        let activeVideoPreviewState = null;

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
                    label.textContent = entry.name || ctx.t('files.unnamed_folder', {}, '(unnamed folder)');
                    const actions = document.createElement('div');
                    actions.className = 'file-browser-actions';

                    const uploadBtn = document.createElement('button');
                    uploadBtn.type = 'button';
                    uploadBtn.className = 'btn btn-sm btn-outline-secondary file-upload-trigger';
                    uploadBtn.dataset.path = entry.path || '';
                    uploadBtn.title = ctx.t('file_browser.upload_title');
                    uploadBtn.textContent = ctx.t('file_browser.upload_button');
                    actions.append(uploadBtn);

                    const clearBtn = document.createElement('button');
                    clearBtn.type = 'button';
                    clearBtn.className = 'btn btn-sm btn-outline-danger file-directory-clear-btn';
                    clearBtn.dataset.path = entry.path || '';
                    clearBtn.title = ctx.t('file_browser.clear_title');
                    clearBtn.textContent = ctx.t('file_browser.clear_button');
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
                    const extension = ctx.getFileExtension(entry.name || '');
                    fileRow.dataset.fileExtension = extension;
                    if (entry.enable_failed_move) {
                        fileRow.dataset.enableFailedMove = 'true';
                    }

                    const link = document.createElement('a');
                    link.className = 'file-browser-link';
                    link.href = ctx.buildWorkdirUrl(entry.path || '');
                    link.target = '_blank';
                    link.rel = 'noopener';
                    link.textContent = entry.name || ctx.t('files.unnamed_file', {}, '(unnamed file)');
                    fileRow.appendChild(link);

                    const actions = document.createElement('div');
                    actions.className = 'file-browser-actions';
                    const deleteBtn = document.createElement('button');
                    deleteBtn.type = 'button';
                    deleteBtn.className = 'btn btn-sm btn-outline-danger file-delete-btn';
                    deleteBtn.title = ctx.t('file_browser.delete_title');
                    deleteBtn.textContent = ctx.t('file_browser.delete_button');
                    actions.append(deleteBtn);

                    if (entry.enable_failed_move && extension === '.wav') {
                        const moveBtn = document.createElement('button');
                        moveBtn.type = 'button';
                        moveBtn.className = 'btn btn-sm btn-outline-primary file-move-to-translated-btn';
                        moveBtn.title = ctx.t('file_browser.move_title');
                        moveBtn.textContent = ctx.t('file_browser.move_button');
                        actions.append(moveBtn);
                    }

                    fileRow.appendChild(actions);
                    const metadataElement = ctx.createMetadataElement(entry);
                    if (metadataElement) {
                        fileRow.insertBefore(metadataElement, actions);
                    }
                    const noteElement = ctx.createFailedOriginalTextElement(entry);
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
            legend.classList.toggle('d-none', !visible);
        }

        function capturePageScroll() {
            return {
                x: window.scrollX || window.pageXOffset || 0,
                y: window.scrollY || window.pageYOffset || 0
            };
        }

        function restorePageScroll(position) {
            if (!position) {
                return;
            }
            requestAnimationFrame(() => {
                window.scrollTo(position.x, position.y);
                requestAnimationFrame(() => {
                    window.scrollTo(position.x, position.y);
                });
            });
        }

        async function refreshDirectory(detailsElement) {
            const childrenContainer = detailsElement.querySelector('.file-browser-children');
            if (!childrenContainer) {
                return;
            }
            const scrollPosition = capturePageScroll();
            const relativePath = detailsElement.dataset.path || '';
            childrenContainer.innerHTML = `<div class="small text-muted">${ctx.t('files.loading', {}, 'Loading...')}</div>`;
            try {
                const url = `/api/project-tree/${encodeURIComponent(ctx.projectName)}?path=${encodeURIComponent(relativePath)}`;
                const response = await fetch(url);
                const result = await response.json();
                if (!response.ok || !result.success) {
                    throw new Error(result.error || ctx.t('files.load_failed'));
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
                    empty.textContent = ctx.t('files.empty_folder', {}, 'Folder is empty.');
                    childrenContainer.appendChild(empty);
                } else {
                    childrenContainer.appendChild(createFileTreeList(entries));
                }
            } catch (error) {
                console.error('Directory refresh failed:', error);
                childrenContainer.innerHTML = `<div class="text-danger small">${ctx.t('files.load_failed', {}, 'Unable to load folder contents.')}</div>`;
            } finally {
                restorePageScroll(scrollPosition);
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
            const scrollPosition = capturePageScroll();
            const normalized = (targetPath || '').replace(/\\/g, '/').replace(/^\/+|\/+$/g, '');
            if (!normalized) {
                browserRoot.innerHTML = `<div class="small text-muted">${ctx.t('files.refreshing', {}, 'Refreshing...')}</div>`;
                try {
                    const response = await fetch(`/api/project-tree/${encodeURIComponent(ctx.projectName)}?path=`);
                    const result = await response.json();
                    if (!response.ok || !result.success) {
                        throw new Error(result.error || ctx.t('files.load_failed'));
                    }
                    const entries = result.entries || [];
                    browserRoot.innerHTML = '';
                    if (!entries.length) {
                        const empty = document.createElement('div');
                        empty.className = 'small text-muted';
                        empty.textContent = ctx.t('files.empty_folder', {}, 'Folder is empty.');
                        browserRoot.appendChild(empty);
                    } else {
                        browserRoot.appendChild(createFileTreeList(entries));
                    }
                    if (Object.prototype.hasOwnProperty.call(result, 'has_highlights')) {
                        updateFailedLegend(Boolean(result.has_highlights));
                    }
                } catch (error) {
                    console.error('Root refresh failed:', error);
                    browserRoot.innerHTML = `<div class="text-danger small">${ctx.t('files.refresh_failed', {}, 'Unable to refresh file list.')}</div>`;
                } finally {
                    restorePageScroll(scrollPosition);
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
            formData.append('projectName', ctx.projectName);
            formData.append('targetPath', targetPath || '');
            formData.append('file', file);
            const response = await fetch('/api/project-file/upload', { method: 'POST', body: formData });
            let result = {};
            try {
                result = await response.json();
            } catch (error) {}
            if (!response.ok || !result.success) {
                throw new Error((result && result.error) || ctx.t('js.errors.upload_failed', {}, 'Failed to upload file.'));
            }
            return result;
        }

        async function deleteProjectFile(filePath, { ignoreMissing = false } = {}) {
            const response = await fetch(`/api/project-file/${encodeURIComponent(ctx.projectName)}`, {
                method: 'DELETE',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ path: filePath })
            });
            if (response.status === 404 && ignoreMissing) {
                return { success: true, skipped: true };
            }
            let result = {};
            try { result = await response.json(); } catch (error) {}
            if (!response.ok || !result.success) {
                throw new Error((result && result.error) || ctx.t('js.errors.delete_failed', {}, 'Failed to delete file.'));
            }
            return result;
        }

        async function clearProjectDirectory(directoryPath) {
            const response = await fetch(`/api/project-directory/${encodeURIComponent(ctx.projectName)}`, {
                method: 'DELETE',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ path: directoryPath })
            });
            let result = {};
            try { result = await response.json(); } catch (error) {}
            if (!response.ok || !result.success) {
                throw new Error((result && result.error) || ctx.t('js.errors.clear_directory_failed', {}, 'Failed to clear directory contents.'));
            }
            return result;
        }

        async function moveFailedGenerationFileRequest(filePath) {
            const response = await fetch('/api/project-file/move-failed', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ projectName: ctx.projectName, sourcePath: filePath })
            });
            let result = {};
            try { result = await response.json(); } catch (error) {}
            if (!response.ok || !result.success) {
                throw new Error((result && result.error) || ctx.t('js.errors.move_failed', {}, 'Failed to move file.'));
            }
            return result;
        }

        function createTtsChunkUploadId(prefix = 'tts') {
            if (window.crypto && window.crypto.randomUUID) {
                return `${prefix}-${crypto.randomUUID()}`;
            }
            return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
        }

        function getTtsChunkSizeBytes(fileSize) {
            if (!fileSize) {
                return ctx.defaultChunkSizeBytes;
            }
            return Math.min(ctx.defaultChunkSizeBytes, fileSize);
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
                window.__updateTtsProgress(0, ctx.t('tts.upload_prepare', { fileKey }, `Preparing upload (${fileKey})...`));
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
                const response = await fetch('/api/tts-upload', { method: 'POST', body: formData });
                let result = {};
                try { result = await response.json(); } catch (error) {}
                if (!response.ok || !result.success) {
                    throw new Error((result && result.error) || ctx.t('js.errors.tts_upload_failed', {}, 'Failed to upload file to the TTS directory.'));
                }
                uploadedBytes = end;
                const chunkPercent = Math.round((uploadedBytes / fileObj.size) * 100);
                const overallPercent = Math.round(((filesDone + (uploadedBytes / fileObj.size)) / totalFiles) * 100);
                if (window.__updateTtsProgress) {
                    window.__updateTtsProgress(
                        overallPercent,
                        ctx.t('tts.chunk_status', {
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
                            ctx.t('tts.upload_complete', { fileKey }, `File (${fileKey}) upload complete.`),
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
                    ctx.t('tts.upload_complete', { fileKey }, `File (${fileKey}) upload complete.`),
                    false,
                    filesDone + 1 === totalFiles
                );
            }
            return { success: true };
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
            if (!ctx.projectName) {
                alert(ctx.t('backup.project_missing'));
                return;
            }
            const backupBtn = document.getElementById('projectBackupBtn');
            if (backupBtn) {
                backupBtn.dataset.originalLabel = backupBtn.dataset.originalLabel || backupBtn.textContent;
                backupBtn.disabled = true;
                backupBtn.textContent = ctx.t('backup.working');
            }
            try {
                const response = await fetch('/api/project-backup', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ projectName: ctx.projectName })
                });
                const contentType = response.headers.get('content-type') || '';
                if (!response.ok) {
                    let errorMessage = ctx.t('backup.create_failed');
                    if (contentType.includes('application/json')) {
                        try {
                            const result = await response.json();
                            if (result && result.error) {
                                errorMessage = result.error;
                            }
                        } catch (error) {}
                    }
                    throw new Error(errorMessage);
                }
                const blob = await response.blob();
                const disposition = response.headers.get('content-disposition') || '';
                const fallbackName = `${ctx.projectName || 'project'}_backup.tar.gz`;
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
                alert(error.message || ctx.t('backup.create_failed'));
            } finally {
                if (backupBtn) {
                    backupBtn.disabled = false;
                    backupBtn.textContent = backupBtn.dataset.originalLabel || ctx.t('files.download_backup');
                }
            }
        }

        function buildVideoPreviewUrl(filePath, audioStreamIndex = null) {
            if (!filePath) {
                return '';
            }
            if (filePath.startsWith('/workdir/') || filePath.startsWith('http://') || filePath.startsWith('https://')) {
                return filePath;
            }
            if (audioStreamIndex === null || audioStreamIndex === undefined || audioStreamIndex === '') {
                return ctx.buildWorkdirUrl(filePath);
            }
            return `/api/video-preview/${encodeURIComponent(ctx.projectName)}?path=${encodeURIComponent(filePath)}&audio_stream=${encodeURIComponent(audioStreamIndex)}`;
        }

        async function fetchVideoAudioTracks(filePath) {
            const response = await fetch(`/api/video-audio-tracks/${encodeURIComponent(ctx.projectName)}?path=${encodeURIComponent(filePath)}`);
            const result = await response.json();
            if (!response.ok || !result.success) {
                throw new Error(result.error || 'Failed to load video audio tracks.');
            }
            return result;
        }

        function formatVideoTrackLabel(track, orderIndex) {
            const parts = [`Track ${orderIndex + 1}`];
            if (track.is_default) {
                parts.push('default');
            }
            if (track.language) {
                parts.push(track.language);
            }
            if (track.title) {
                parts.push(track.title);
            }
            if (track.codec) {
                parts.push(track.codec);
            }
            if (track.channels) {
                parts.push(`${track.channels}ch`);
            }
            return parts.join(' • ');
        }

        function resetVideoTrackControls() {
            const controls = document.getElementById('videoPreviewTrackControls');
            const select = document.getElementById('videoPreviewTrackSelect');
            const hint = document.getElementById('videoPreviewTrackHint');
            if (controls) {
                controls.classList.add('d-none');
            }
            if (select) {
                select.innerHTML = '';
                select.disabled = true;
            }
            if (hint) {
                hint.textContent = 'Select which audio track should be used during playback.';
            }
        }

        function setVideoPreviewSource(filePath, audioStreamIndex = null, options = {}) {
            const videoElement = document.getElementById('videoPreviewPlayer');
            if (!videoElement || !filePath) {
                return;
            }
            const resumeTime = Number.isFinite(options.resumeTime) ? options.resumeTime : 0;
            const shouldAutoplay = options.autoplay !== false;
            const nextUrl = buildVideoPreviewUrl(filePath, audioStreamIndex);
            if (!nextUrl) {
                return;
            }
            videoElement.pause();
            videoElement.src = nextUrl;
            videoElement.load();

            const restorePlayback = () => {
                if (resumeTime > 0) {
                    try {
                        const duration = Number.isFinite(videoElement.duration) ? videoElement.duration : null;
                        videoElement.currentTime = duration ? Math.min(resumeTime, Math.max(duration - 0.25, 0)) : resumeTime;
                    } catch (error) {}
                }
                if (shouldAutoplay) {
                    videoElement.play().catch(() => {});
                }
            };

            if (videoElement.readyState >= 1) {
                restorePlayback();
            } else {
                videoElement.addEventListener('loadedmetadata', restorePlayback, { once: true });
            }

            activeVideoPreviewState = {
                ...(activeVideoPreviewState || {}),
                filePath,
                audioStreamIndex,
            };
        }

        function applyVideoTrackSelection(audioStreamIndex) {
            if (!activeVideoPreviewState || !activeVideoPreviewState.filePath) {
                return;
            }
            const videoElement = document.getElementById('videoPreviewPlayer');
            const resumeTime = videoElement && Number.isFinite(videoElement.currentTime) ? videoElement.currentTime : 0;
            setVideoPreviewSource(activeVideoPreviewState.filePath, audioStreamIndex, {
                resumeTime,
                autoplay: true
            });
        }

        function updateVideoTrackControls(trackPayload) {
            const controls = document.getElementById('videoPreviewTrackControls');
            const select = document.getElementById('videoPreviewTrackSelect');
            const hint = document.getElementById('videoPreviewTrackHint');
            if (!controls || !select) {
                return;
            }

            const tracks = Array.isArray(trackPayload?.tracks) ? trackPayload.tracks : [];
            select.innerHTML = '';

            if (tracks.length <= 1) {
                controls.classList.add('d-none');
                select.disabled = true;
                if (hint) {
                    hint.textContent = tracks.length === 1
                        ? 'This video contains a single audio track.'
                        : 'Audio track selection is not available for this video.';
                }
                return;
            }

            tracks.forEach((track, index) => {
                const option = document.createElement('option');
                option.value = String(track.stream_index);
                option.textContent = formatVideoTrackLabel(track, index);
                if (trackPayload.default_stream_index === track.stream_index) {
                    option.selected = true;
                }
                select.appendChild(option);
            });
            select.disabled = false;
            controls.classList.remove('d-none');
            if (hint) {
                hint.textContent = 'Select which audio track should be used during playback.';
            }
        }

        async function showVideoPreview(fileName, fileUrl, filePath) {
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
                titleElement.textContent = fileName || ctx.t('video_modal.default_title', {}, 'Play video');
            }
            const videoElement = document.getElementById('videoPreviewPlayer');
            if (!videoElement) {
                window.open(fileUrl, '_blank');
                return;
            }
            activeVideoPreviewState = {
                fileName,
                filePath,
                fileUrl,
                audioStreamIndex: null,
            };
            resetVideoTrackControls();
            setVideoPreviewSource(filePath || fileUrl, null, { autoplay: false });
            videoPreviewModal.show();
            try {
                if (!filePath) {
                    throw new Error('Missing file path for track selection.');
                }
                const trackPayload = await fetchVideoAudioTracks(filePath);
                updateVideoTrackControls(trackPayload);
                if (trackPayload && trackPayload.default_stream_index !== undefined && trackPayload.default_stream_index !== null) {
                    setVideoPreviewSource(filePath, trackPayload.default_stream_index, { autoplay: true });
                } else {
                    setVideoPreviewSource(filePath || fileUrl, null, { autoplay: true });
                }
            } catch (error) {
                console.warn('Video track metadata unavailable:', error);
                const hint = document.getElementById('videoPreviewTrackHint');
                if (hint) {
                    hint.textContent = 'Audio track selection is unavailable for this file.';
                }
                setVideoPreviewSource(filePath || fileUrl, null, { autoplay: true });
            }
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
                titleElement.textContent = fileName || ctx.t('json_modal.default_title', {}, 'File preview');
            }
            const contentElement = document.getElementById('jsonPreviewContent');
            if (contentElement) {
                contentElement.textContent = ctx.t('json_modal.loading', {}, 'Loading...');
            }
            jsonPreviewModal.show();
            try {
                const response = await fetch(fileUrl, { headers: { 'Accept': 'application/json, text/plain;q=0.9, */*;q=0.8' } });
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}`);
                }
                const text = await response.text();
                let formatted = text;
                if (extension === '.json') {
                    try {
                        formatted = JSON.stringify(JSON.parse(text), null, 2);
                    } catch (error) {}
                }
                if (contentElement) {
                    contentElement.textContent = formatted || ctx.t('json_modal.empty', {}, 'Empty file.');
                }
            } catch (error) {
                console.error('Text preview failed:', error);
                if (contentElement) {
                    contentElement.textContent = ctx.t('json_modal.load_error', { error: error.message || error }, `Failed to load file.\n${error.message || error}`);
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
                const trackSelect = document.getElementById('videoPreviewTrackSelect');
                if (trackSelect) {
                    trackSelect.addEventListener('change', () => {
                        const selectedValue = trackSelect.value;
                        applyVideoTrackSelection(selectedValue ? Number(selectedValue) : null);
                    });
                }
                videoModalElement.addEventListener('hidden.bs.modal', () => {
                    const videoElement = document.getElementById('videoPreviewPlayer');
                    if (videoElement) {
                        videoElement.pause();
                        videoElement.removeAttribute('src');
                        videoElement.load();
                    }
                    activeVideoPreviewState = null;
                    resetVideoTrackControls();
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
                        alert(ctx.t('notifications.file_upload_success'));
                        await reloadFileBrowser(targetPath);
                    } catch (error) {
                        alert(error.message || ctx.t('notifications.file_upload_failed'));
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
                refreshBtn.addEventListener('click', () => reloadFileBrowser(''));
            }
            const backupBtn = document.getElementById('projectBackupBtn');
            if (backupBtn) {
                backupBtn.addEventListener('click', () => requestProjectBackup());
            }
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
                        alert(ctx.t('directories.path_missing'));
                        return;
                    }
                    const labelElement = detailsElement.querySelector('.file-browser-label');
                    const directoryName = (labelElement && labelElement.textContent) ? labelElement.textContent.trim() : directoryPath;
                    if (!confirm(ctx.t('directories.clear_confirm', { name: directoryName }))) {
                        return;
                    }
                    clearDirectoryButton.disabled = true;
                    try {
                        await clearProjectDirectory(directoryPath);
                        alert(ctx.t('directories.clear_success'));
                        await refreshDirectory(detailsElement);
                        detailsElement.setAttribute('open', '');
                    } catch (error) {
                        alert(error.message || ctx.t('directories.clear_error'));
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
                    if (!fileRow) return;
                    const filePath = fileRow.dataset.filePath || '';
                    if (!filePath) return;
                    const fileName = fileRow.dataset.fileName || filePath.split('/').pop();
                    if (!confirm(ctx.t('directories.delete_confirm', { name: fileName }))) {
                        return;
                    }
                    try {
                        await deleteProjectFile(filePath);
                        const parentPath = filePath.includes('/') ? filePath.substring(0, filePath.lastIndexOf('/')) : '';
                        alert(ctx.t('directories.delete_success'));
                        await reloadFileBrowser(parentPath);
                    } catch (error) {
                        alert(error.message || ctx.t('directories.delete_error'));
                    }
                    return;
                }
                const moveButton = event.target.closest('.file-move-to-translated-btn');
                if (moveButton && browserRoot.contains(moveButton)) {
                    event.preventDefault();
                    event.stopPropagation();
                    const fileRow = moveButton.closest('.file-browser-item');
                    if (!fileRow) return;
                    const filePath = fileRow.dataset.filePath || '';
                    if (!filePath) return;
                    const fileName = fileRow.dataset.fileName || filePath.split('/').pop();
                    if (!confirm(ctx.t('directories.move_confirm', { name: fileName }))) {
                        return;
                    }
                    moveButton.disabled = true;
                    try {
                        await moveFailedGenerationFileRequest(filePath);
                        const parentPath = filePath.includes('/') ? filePath.substring(0, filePath.lastIndexOf('/')) : '';
                        alert(ctx.t('directories.move_success'));
                        await reloadFileBrowser(parentPath);
                    } catch (error) {
                        alert(error.message || ctx.t('directories.move_error'));
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
                    if (!fileRow) return;
                    const fileName = fileRow.dataset.fileName || '';
                    const filePath = fileRow.dataset.filePath || '';
                    const extension = (fileRow.dataset.fileExtension || '').toLowerCase();
                    if (ctx.audioExtensions.has(extension)) {
                        event.preventDefault();
                        event.stopPropagation();
                        if (window.AudioTrimmer && typeof AudioTrimmer.showPreview === 'function') {
                            AudioTrimmer.showPreview(fileName, ctx.buildWorkdirUrl(filePath), filePath);
                        } else {
                            window.open(ctx.buildWorkdirUrl(filePath), '_blank');
                        }
                        return;
                    }
                    if (ctx.videoExtensions.has(extension)) {
                        event.preventDefault();
                        event.stopPropagation();
                        showVideoPreview(fileName, ctx.buildWorkdirUrl(filePath), filePath);
                        return;
                    }
                    if (ctx.textPreviewExtensions.has(extension)) {
                        event.preventDefault();
                        event.stopPropagation();
                        await showTextPreview(fileName, ctx.buildWorkdirUrl(filePath), extension);
                    }
                    return;
                }
                const summary = event.target.closest('summary.file-browser-summary');
                if (!summary || !browserRoot.contains(summary)) {
                    return;
                }
                const details = summary.closest('details.file-browser-directory');
                if (!details || event.target.closest('.file-browser-actions')) {
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

        return {
            createFileTreeList,
            refreshDirectory,
            cssEscape,
            reloadFileBrowser,
            openUploadDialog,
            uploadFilesToTtsRequest,
            initPreviewModals,
            initProjectFileActions,
            initFileBrowser,
            requestProjectBackup,
            showVideoPreview,
            showTextPreview
        };
    }

    window.ProjectFileBrowser = { createProjectFileBrowser };
})();
