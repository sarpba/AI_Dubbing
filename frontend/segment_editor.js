document.addEventListener('DOMContentLoaded', () => {
    const projectSelect = document.getElementById('project-select');
    const segmentListDiv = document.getElementById('segment-list');
    const loadingMessageDiv = document.getElementById('loading-message');
    const errorMessageDiv = document.getElementById('error-message');

    const API_BASE_URL = '/api/editor'; // Assuming the backend runs on the same host/port

    function showError(message) {
        errorMessageDiv.textContent = message;
        errorMessageDiv.style.display = 'block';
        loadingMessageDiv.style.display = 'none';
    }

    function showLoading(message = 'Betöltés...') {
        loadingMessageDiv.textContent = message;
        loadingMessageDiv.style.display = 'block';
        errorMessageDiv.style.display = 'none';
    }

    function hideMessages() {
        loadingMessageDiv.style.display = 'none';
        errorMessageDiv.style.display = 'none';
    }

    async function fetchProjects() {
        showLoading('Projektek betöltése...');
        try {
            const response = await fetch(`${API_BASE_URL}/projects`);
            if (!response.ok) {
                throw new Error(`Hiba a projektek lekérésekor: ${response.statusText}`);
            }
            const data = await response.json();
            projectSelect.innerHTML = '<option value="">Válassz egy projektet...</option>'; // Reset
            data.projects.forEach(project => {
                const option = document.createElement('option');
                option.value = project;
                option.textContent = project;
                projectSelect.appendChild(option);
            });
            hideMessages();
        } catch (error) {
            showError(`Projektek betöltése sikertelen: ${error.message}`);
            console.error(error);
        }
    }

    let lastEditedSegmentId = null; // Variable to store the ID of the last edited segment

    async function fetchSegments(projectName, scrollToSegmentId = null) {
        if (!projectName) {
            segmentListDiv.innerHTML = '';
            return;
        }
        showLoading(`Szegmensek betöltése (${projectName})...`);
        // Don't clear segmentListDiv here if we want to avoid flicker before new content,
        // but renderSegments will clear it. If fetching takes time, this is okay.
        // segmentListDiv.innerHTML = ''; 

        try {
            const response = await fetch(`${API_BASE_URL}/projects/${projectName}/segments`);
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ detail: response.statusText }));
                throw new Error(`Hiba a szegmensek lekérésekor: ${errorData.detail || response.statusText}`);
            }
            const data = await response.json();

            if (data.error) {
                showError(data.error);
                return;
            }
            if (!data.segments || data.segments.length === 0) {
                segmentListDiv.innerHTML = '<p>Nincsenek megjeleníthető szegmensek ebben a projektben.</p>';
                hideMessages();
                return;
            }

            renderSegments(data.segments, projectName);
            hideMessages();

            if (scrollToSegmentId) {
                const elementToScrollTo = document.getElementById(`segment-item-${scrollToSegmentId.replace(/[^a-zA-Z0-9-_]/g, '_')}`);
                if (elementToScrollTo) {
                    elementToScrollTo.scrollIntoView({ behavior: 'smooth', block: 'center' });
                }
                lastEditedSegmentId = null; // Reset after scrolling
            }

        } catch (error) {
            showError(`Szegmensek betöltése sikertelen: ${error.message}`);
            console.error(error);
        }
    }

    function renderSegments(segments, projectName) {
        segmentListDiv.innerHTML = ''; // Clear previous
        const epsilon = 0.001; // Small value to prevent exact overlap

        segments.forEach((segment, index) => {
            const segmentItem = document.createElement('div');
            segmentItem.classList.add('segment-item');
            // Sanitize segment.id for use in HTML id attribute
            const sanitizedSegmentId = segment.id.replace(/[^a-zA-Z0-9-_]/g, '_');
            segmentItem.id = `segment-item-${sanitizedSegmentId}`;
            segmentItem.dataset.segmentId = segment.id; // Keep original id in dataset for API calls

            const title = document.createElement('h3');
            title.textContent = `Szegmens: ${segment.id} (Beszélő: ${segment.speaker})`;
            segmentItem.appendChild(title);

            const originalTimes = document.createElement('div');
            originalTimes.classList.add('info');
            originalTimes.innerHTML = `Időtartam: ${(segment.original_end_time_sec - segment.original_start_time_sec).toFixed(3)}s`;
            segmentItem.appendChild(originalTimes);
            
            const audio = document.createElement('audio');
            audio.controls = true;
            audio.src = segment.audio_url; // URL from backend
            segmentItem.appendChild(audio);

            const waveformContainer = document.createElement('div');
            waveformContainer.classList.add('waveform-container');
            waveformContainer.id = `waveform-${sanitizedSegmentId}`;
            // Add a placeholder or message until waveform is loaded
            waveformContainer.innerHTML = `<small style="display:block; text-align:center; padding-top:30px;">Kattints a lejátszásra a hullámforma megjelenítéséhez.</small>`;
            segmentItem.appendChild(waveformContainer);

            const transcriptTextarea = document.createElement('textarea');
            transcriptTextarea.classList.add('transcript-edit');
            transcriptTextarea.value = segment.text;
            transcriptTextarea.rows = 3; // Adjust as needed
            transcriptTextarea.style.width = '100%';
            transcriptTextarea.style.marginTop = '10px';
            segmentItem.appendChild(transcriptTextarea);

            const controlsDiv = document.createElement('div');
            controlsDiv.classList.add('controls');

            // Calculate slider limits
            const currentSegmentDuration = segment.original_end_time_sec - segment.original_start_time_sec;
            const maxAllowedGlobalOffset = 1.0;
            const minAllowedGlobalOffset = -1.0;

            // --- Start Offset Calculations ---
            let minStartOffset = minAllowedGlobalOffset;
            // Constraint 1: Previous segment's end
            if (index > 0) {
                const prevSegment = segments[index - 1];
                const timeToPrevEnd = segment.original_start_time_sec - prevSegment.original_end_time_sec;
                // Negative offset cannot be more than this gap (minus epsilon)
                minStartOffset = Math.max(minAllowedGlobalOffset, -(timeToPrevEnd - epsilon));
            }
            // Constraint 2: Absolute start of the audio (cannot be less than 0)
            minStartOffset = Math.max(minStartOffset, -segment.original_start_time_sec);
            // Constraint 3: Cannot make segment duration negative/zero by moving start past original end
            minStartOffset = Math.max(minStartOffset, -(currentSegmentDuration - epsilon));
            
            // maxStartOffset remains maxAllowedGlobalOffset (1.0) as per simple interpretation of request
            // (start can move right by 1s, unless it conflicts with its own end, handled by backend)
            let maxStartOffset = maxAllowedGlobalOffset;


            // --- End Offset Calculations ---
            let maxEndOffset = maxAllowedGlobalOffset;
            // Constraint 1: Next segment's start
            if (index < segments.length - 1) {
                const nextSegment = segments[index + 1];
                const timeToNextStart = nextSegment.original_start_time_sec - segment.original_end_time_sec;
                // Positive offset cannot be more than this gap (minus epsilon)
                maxEndOffset = Math.min(maxAllowedGlobalOffset, timeToNextStart - epsilon);
            }
            // Constraint 2: Cannot make segment duration negative/zero by moving end before original start
            maxEndOffset = Math.min(maxEndOffset, currentSegmentDuration > epsilon ? maxAllowedGlobalOffset : (currentSegmentDuration - epsilon) + maxAllowedGlobalOffset ); // if duration is tiny, allow it to grow up to 1s
                                                                                                                                                                    // This needs to be relative to the *original* duration for the offset.
                                                                                                                                                                    // If current duration is 0.1s, max offset is 1s. New duration can be 1.1s.
                                                                                                                                                                    // If current duration is 0.1s, min offset is -1s. New duration can be -0.9s (invalid).
            
            let minEndOffset = minAllowedGlobalOffset;
            // Constraint 3 for minEndOffset: Cannot make segment duration negative/zero
            minEndOffset = Math.max(minAllowedGlobalOffset, -(currentSegmentDuration - epsilon));


            // Ensure min is not greater than max for sliders (can happen with very small gaps)
            if (minStartOffset > maxStartOffset) {
                // If calculated min is greater than max, it implies a very small or zero gap.
                // Set both to 0 or a very small valid range around 0 if possible.
                // For simplicity, if min > max, we might disable or severely restrict the slider.
                // Here, we'll try to make min slightly less than max, or both 0.
                if (maxStartOffset > minAllowedGlobalOffset + epsilon) { // Check if max is not already at its absolute min
                    minStartOffset = maxStartOffset - epsilon;
                } else { // maxStartOffset is already very small or negative
                    minStartOffset = maxStartOffset; // Make them equal, effectively locking if max is also min
                }
            }
            if (minEndOffset > maxEndOffset) {
                 if (maxEndOffset > minAllowedGlobalOffset + epsilon) {
                    minEndOffset = maxEndOffset - epsilon;
                } else {
                    minEndOffset = maxEndOffset;
                }
            }


            // Start Offset Slider
            const startOffsetContainer = createSlider(
                `start-offset-${segment.id}`, 
                'Kezdés eltolás:', 
                parseFloat(minStartOffset.toFixed(3)), // Use toFixed to avoid floating point arithmetic issues in HTML attributes
                parseFloat(maxStartOffset.toFixed(3)), 
                0.001, 
                0
            );
            controlsDiv.appendChild(startOffsetContainer.container);

            // End Offset Slider
            const endOffsetContainer = createSlider(
                `end-offset-${segment.id}`, 
                'Vég eltolás:', 
                parseFloat(minEndOffset.toFixed(3)), 
                parseFloat(maxEndOffset.toFixed(3)), 
                0.001, 
                0
            );
            controlsDiv.appendChild(endOffsetContainer.container);
            
            audio.addEventListener('play', () => {
                if (waveformContainer.dataset.waveformInitialized === 'true') {
                    return; // Already initialized
                }
                waveformContainer.innerHTML = ''; // Clear placeholder
                waveformContainer.dataset.waveformInitialized = 'true';

                const wavesurfer = WaveSurfer.create({
                    container: waveformContainer,
                    waveColor: 'rgb(200, 200, 200)',
                    progressColor: 'rgb(100, 100, 100)',
                    height: 80,
                    barWidth: 2,
                    barGap: 1,
                    cursorWidth: 2,
                    cursorColor: '#007bff',
                    responsive: true,
                    plugins: [] 
                });

                wavesurfer.load(segment.audio_url);

                wavesurfer.on('ready', () => {
                    const wsDuration = wavesurfer.getDuration();
                    wavesurfer.clearRegions();

                    wavesurfer.addRegion({
                        id: `start-marker-${sanitizedSegmentId}`,
                        start: 0, 
                        end: epsilon / 2, 
                        color: 'rgba(0, 123, 255, 0.7)',
                        drag: false,
                        resize: false,
                    });
                    wavesurfer.addRegion({
                        id: `end-marker-${sanitizedSegmentId}`,
                        start: wsDuration - (epsilon/2) < 0 ? 0 : wsDuration - (epsilon/2),
                        end: wsDuration,
                        color: 'rgba(0, 123, 255, 0.7)',
                        drag: false,
                        resize: false,
                    });

                    // Re-attach listeners or ensure they target the correct wavesurfer instance
                    // if they were defined outside this scope and relied on a single wavesurfer.
                    // In this case, they are specific to this wavesurfer instance.
                });

                startOffsetContainer.input.addEventListener('input', () => {
                    const offset = parseFloat(startOffsetContainer.input.value);
                    const wsDuration = wavesurfer.getDuration() || currentSegmentDuration;
                    let visualStartPos = 0 + offset; 
                    visualStartPos = Math.max(0, Math.min(wsDuration, visualStartPos));
                    const startRegion = wavesurfer.regions.list[`start-marker-${sanitizedSegmentId}`];
                    if (startRegion) {
                        startRegion.update({
                            start: visualStartPos - epsilon / 4 < 0 ? 0 : visualStartPos - epsilon / 4,
                            end: visualStartPos + epsilon / 4 > wsDuration ? wsDuration : visualStartPos + epsilon / 4
                        });
                    }
                });
    
                endOffsetContainer.input.addEventListener('input', () => {
                    const offset = parseFloat(endOffsetContainer.input.value);
                    const wsDuration = wavesurfer.getDuration() || currentSegmentDuration;
                    let visualEndPos = wsDuration + offset; 
                    visualEndPos = Math.max(0, Math.min(wsDuration, visualEndPos));
                    const endRegion = wavesurfer.regions.list[`end-marker-${sanitizedSegmentId}`];
                    if (endRegion) {
                         endRegion.update({
                            start: visualEndPos - epsilon / 4 < 0 ? 0 : visualEndPos - epsilon / 4,
                            end: visualEndPos + epsilon / 4 > wsDuration ? wsDuration : visualEndPos + epsilon / 4
                        });
                    }
                });
            });


            const saveButton = document.createElement('button');
            saveButton.textContent = 'Mentés és Újravágás';
            saveButton.style.marginRight = '10px'; // Add some space next to delete button
            saveButton.addEventListener('click', async () => {
                const startOffset = parseFloat(startOffsetContainer.input.value);
                const endOffset = parseFloat(endOffsetContainer.input.value);
                const newText = transcriptTextarea.value;
                
                lastEditedSegmentId = segment.id; 

                showLoading('Szegmens frissítése...');
                saveButton.disabled = true;
                const deleteButton = segmentItem.querySelector('.delete-button'); 
                if(deleteButton) deleteButton.disabled = true;
                transcriptTextarea.disabled = true;


                try {
                    const apiUrl = `${API_BASE_URL}/projects/${projectName}/segments/${segment.id}/update?start_offset_sec=${startOffset}&end_offset_sec=${endOffset}`;
                    const response = await fetch(apiUrl, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({ text: newText }), // Send new text in body
                    });
                    
                    if (!response.ok) {
                        const errorData = await response.json().catch(() => ({ detail: response.statusText }));
                        throw new Error(`Mentés sikertelen: ${errorData.detail || response.statusText}`);
                    }
                    const result = await response.json();
                    console.log('Mentés eredménye:', result);
                    hideMessages();
                    fetchSegments(projectName, result.new_segment_id || lastEditedSegmentId); 
                } catch (error) {
                    showError(`Hiba a mentés során: ${error.message}`);
                    console.error(error);
                    lastEditedSegmentId = null; 
                    saveButton.disabled = false; 
                    if(deleteButton) deleteButton.disabled = false;
                    transcriptTextarea.disabled = false;
                }
            });
            controlsDiv.appendChild(saveButton);

            const deleteButton = document.createElement('button');
            deleteButton.textContent = 'Törlés';
            deleteButton.classList.add('delete-button'); // For potential styling
            deleteButton.style.backgroundColor = '#dc3545'; // Red color for delete
            deleteButton.addEventListener('click', async () => {
                if (!confirm(`Biztosan törlöd a(z) "${segment.id}" szegmenst? A fájlok a "deleted_segments" mappába kerülnek.`)) {
                    return;
                }

                lastEditedSegmentId = segment.id; // Store for potential scroll focus later if needed
                showLoading('Szegmens törlése...');
                saveButton.disabled = true;
                deleteButton.disabled = true;

                try {
                    const response = await fetch(`${API_BASE_URL}/projects/${projectName}/segments/${segment.id}/delete`, {
                        method: 'POST',
                    });
                    if (!response.ok) {
                        const errorData = await response.json().catch(() => ({ detail: response.statusText }));
                        throw new Error(`Törlés sikertelen: ${errorData.detail || response.statusText}`);
                    }
                    const result = await response.json();
                    console.log('Törlés eredménye:', result);
                    hideMessages();
                    // Find the next segment to scroll to, or the previous one if it was the last
                    let nextSegmentToFocus = null;
                    if (segments.length > 1) {
                        if (index < segments.length - 1) { // If not the last element
                            nextSegmentToFocus = segments[index+1].id;
                        } else if (index > 0) { // If it was the last, focus previous
                             nextSegmentToFocus = segments[index-1].id;
                        }
                    }
                    fetchSegments(projectName, nextSegmentToFocus);
                } catch (error) {
                    showError(`Hiba a törlés során: ${error.message}`);
                    console.error(error);
                    saveButton.disabled = false;
                    deleteButton.disabled = false;
                }
            });
            controlsDiv.appendChild(deleteButton);

            segmentItem.appendChild(controlsDiv);
            segmentListDiv.appendChild(segmentItem);
        });
    }

    function createSlider(id, labelText, min, max, step, defaultValue) {
        const container = document.createElement('div');
        container.classList.add('slider-container');

        const label = document.createElement('label');
        label.setAttribute('for', id);
        label.textContent = labelText;

        const input = document.createElement('input');
        input.type = 'range';
        input.id = id;
        input.min = min;
        input.max = max;
        input.step = step;
        input.value = defaultValue;

        const valueSpan = document.createElement('span');
        valueSpan.textContent = `${parseFloat(defaultValue).toFixed(3)}s`;

        input.addEventListener('input', () => {
            valueSpan.textContent = `${parseFloat(input.value).toFixed(3)}s`;
        });

        container.appendChild(label);
        container.appendChild(input);
        container.appendChild(valueSpan);

        return { container, input, valueSpan };
    }

    projectSelect.addEventListener('change', (event) => {
        const selectedProject = event.target.value;
        fetchSegments(selectedProject);
    });

    // Initial load
    fetchProjects();
});
