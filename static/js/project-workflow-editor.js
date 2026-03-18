(function() {
    function createProjectWorkflowEditor(ctx) {
        function findTemplateById(templateId) {
            if (!templateId) {
                return null;
            }
            return ctx.getWorkflowTemplates().find(item => item.id === templateId) || null;
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
            placeholder.textContent = ctx.t('workflow.templates.placeholder', {}, 'Select a template...');
            select.append(placeholder);
            ctx.getWorkflowTemplates().forEach(template => {
                const option = document.createElement('option');
                option.value = template.id;
                option.textContent = template.name || template.id;
                select.append(option);
            });
            const targetValue = selectedId !== undefined ? selectedId : previousValue;
            if (targetValue && Array.from(select.options).some(option => option.value === targetValue)) {
                select.value = targetValue;
                ctx.setCurrentTemplateId(targetValue);
            } else {
                select.value = '';
                if (selectedId !== undefined) {
                    ctx.setCurrentTemplateId(null);
                }
            }
        }

        async function refreshWorkflowTemplates(selectedId = ctx.getCurrentTemplateId()) {
            try {
                const response = await fetch('/api/workflow-templates');
                const result = await response.json();
                if (!response.ok || !result.success) {
                    throw new Error(result.error || ctx.t('workflow.templates.list_failed'));
                }
                ctx.setWorkflowTemplates(result.templates || []);
                populateWorkflowTemplateSelect(selectedId);
            } catch (error) {
                console.error('Workflow sablon lista frissítési hiba:', error);
            }
        }

        async function loadWorkflowTemplateById(templateId, showMessage = true) {
            if (!templateId) {
                alert(ctx.t('workflow.templates.select_prompt'));
                return;
            }
            try {
                const response = await fetch(`/api/workflow-template/${encodeURIComponent(templateId)}`);
                const result = await response.json();
                if (!response.ok || !result.success) {
                    throw new Error(result.error || ctx.t('workflow.templates.load_failed'));
                }
                const template = result.template || {};
                ctx.setCurrentTemplateId(template.id || templateId);
                ctx.setWorkflowSteps(ctx.cloneSteps(template.steps || []));
                ctx.setDefaultWorkflow(ctx.cloneSteps(template.steps || []));
                ctx.markWorkflowDirty();
                populateWorkflowTemplateSelect(ctx.getCurrentTemplateId());
                renderWorkflowSteps();
                if (showMessage) {
                    ctx.updateInfoBox('info', ctx.t('workflow.templates.loaded', { name: template.name || ctx.getCurrentTemplateId() }));
                }
            } catch (error) {
                console.error('Workflow sablon betöltési hiba:', error);
                alert(ctx.t('workflow.templates.load_error', { error: error.message }));
            }
        }

        function showSaveWorkflowModal() {
            const modalElement = document.getElementById('workflowSaveModal');
            if (!modalElement) return;
            const nameInput = document.getElementById('workflowSaveName');
            const descriptionInput = document.getElementById('workflowSaveDescription');
            const overwriteCheckbox = document.getElementById('workflowSaveOverwrite');
            const overwriteInfo = document.getElementById('workflowSaveOverwriteInfo');
            const errorBox = document.getElementById('workflowSaveError');
            const currentTemplate = findTemplateById(ctx.getCurrentTemplateId());
            if (nameInput) nameInput.value = currentTemplate ? (currentTemplate.name || '') : '';
            if (descriptionInput) descriptionInput.value = currentTemplate ? (currentTemplate.description || '') : '';
            if (overwriteCheckbox) {
                overwriteCheckbox.checked = false;
                overwriteCheckbox.disabled = !ctx.getCurrentTemplateId();
            }
            if (overwriteInfo) {
                if (currentTemplate) {
                    overwriteInfo.textContent = ctx.t('workflow.templates.overwrite_current', {
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
            if (!ctx.getWorkflowSaveModal()) {
                ctx.setWorkflowSaveModal(new bootstrap.Modal(modalElement));
            }
            ctx.getWorkflowSaveModal().show();
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
            if (overwrite && !ctx.getCurrentTemplateId()) {
                if (errorBox) {
                    errorBox.textContent = ctx.t('workflow.templates.none_selected_overwrite');
                    errorBox.classList.remove('d-none');
                }
                return;
            }
            if (!name && !overwrite) {
                if (errorBox) {
                    errorBox.textContent = ctx.t('workflow.templates.name_required');
                    errorBox.classList.remove('d-none');
                }
                return;
            }
            const workflowState = ctx.collectWorkflowState();
            const body = { name, description, steps: workflowState, overwrite };
            if (overwrite && ctx.getCurrentTemplateId()) {
                body.template_id = ctx.getCurrentTemplateId();
            }
            try {
                const response = await fetch('/api/save-workflow-template', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(body)
                });
                const result = await response.json();
                if (!response.ok || !result.success) {
                    throw new Error(result.error || ctx.t('workflow.templates.save_failed'));
                }
                if (ctx.getWorkflowSaveModal()) {
                    ctx.getWorkflowSaveModal().hide();
                }
                const savedTemplate = result.template || {};
                ctx.setCurrentTemplateId(savedTemplate.id || ctx.getCurrentTemplateId());
                ctx.setDefaultWorkflow(ctx.cloneSteps(workflowState));
                await refreshWorkflowTemplates(ctx.getCurrentTemplateId());
                populateWorkflowTemplateSelect(ctx.getCurrentTemplateId());
                ctx.updateInfoBox('success', ctx.t('workflow.templates.saved', { name: savedTemplate.name || ctx.getCurrentTemplateId() }));
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
                        ctx.setCurrentTemplateId(null);
                        return;
                    }
                    if (value === ctx.getCurrentTemplateId()) {
                        return;
                    }
                    ctx.setCurrentTemplateId(value);
                    loadWorkflowTemplateById(ctx.getCurrentTemplateId());
                });
            }
            const saveTemplateBtn = document.getElementById('saveWorkflowBtn');
            if (saveTemplateBtn) saveTemplateBtn.addEventListener('click', showSaveWorkflowModal);
            const resetBtn = document.getElementById('resetWorkflowBtn');
            if (resetBtn) {
                resetBtn.addEventListener('click', () => {
                    if (!ctx.getDefaultWorkflow().length) {
                        alert(ctx.t('workflow.templates.default_missing'));
                        return;
                    }
                    if (!confirm(ctx.t('workflow.templates.reset_confirm'))) {
                        return;
                    }
                    ctx.setWorkflowSteps(ctx.cloneSteps(ctx.getDefaultWorkflow()));
                    ctx.markWorkflowDirty();
                    renderWorkflowSteps();
                    ctx.updateInfoBox('info', ctx.t('workflow.templates.default_loaded'));
                });
            }
            const startBtn = document.getElementById('startWorkflowBtn');
            if (startBtn) startBtn.addEventListener('click', ctx.handleStartWorkflow);
            const stopBtn = document.getElementById('stopWorkflowBtn');
            if (stopBtn) stopBtn.addEventListener('click', ctx.stopWorkflow);
            const searchInput = document.getElementById('workflowStepSearch');
            if (searchInput) {
                searchInput.addEventListener('input', event => populateWorkflowStepList(event.target.value || ''));
            }
        }

        function renderWorkflowSteps() {
            const tbody = document.getElementById('workflowStepsBody');
            if (!tbody) return;
            const workflowSteps = ctx.getWorkflowSteps();
            tbody.innerHTML = '';
            if (!workflowSteps.length) {
                const row = document.createElement('tr');
                row.className = 'text-muted';
                const cell = document.createElement('td');
                cell.colSpan = 5;
                cell.textContent = ctx.t('workflow.table_empty');
                row.append(cell);
                tbody.append(row);
                ctx.updateMissingBadge();
                const startButtonEmpty = document.getElementById('startWorkflowBtn');
                if (startButtonEmpty) startButtonEmpty.disabled = true;
                return;
            }
            workflowSteps.forEach((step, index) => {
                ctx.normalizeWorkflowStep(step);
                const isWidget = step.type === 'widget';
                const script = !isWidget ? ctx.findScriptById(step.script) : null;
                const widget = isWidget ? ctx.findWidgetById(step.widget) : null;
                const row = document.createElement('tr');
                row.classList.add('workflow-row');
                row.dataset.stepIndex = String(index);
                if (isWidget) row.classList.add('workflow-row-widget');
                if (step.enabled === false) row.classList.add('workflow-row-disabled', 'text-muted');
                const missing = (!isWidget && step.enabled !== false) ? ctx.getMissingParams(step, script) : [];
                if (missing.length) row.classList.add('workflow-row-missing');

                const enabledCell = document.createElement('td');
                const enabledWrapper = document.createElement('div');
                enabledWrapper.className = 'form-check form-switch';
                const enabledCheckbox = document.createElement('input');
                enabledCheckbox.type = 'checkbox';
                enabledCheckbox.className = 'form-check-input';
                enabledCheckbox.checked = step.enabled !== false;
                enabledCheckbox.addEventListener('change', () => {
                    step.enabled = enabledCheckbox.checked;
                    ctx.markWorkflowDirty();
                    renderWorkflowSteps();
                });
                enabledWrapper.append(enabledCheckbox);
                enabledCell.append(enabledWrapper);
                row.append(enabledCell);

                const infoCell = document.createElement('td');
                if (isWidget) {
                    const title = document.createElement('div');
                    title.className = 'fw-semibold';
                    title.textContent = widget ? widget.name : ctx.t('workflow.labels_extra.special_widget');
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
                    reviewBtn.addEventListener('click', () => window.open(`/review/${encodeURIComponent(ctx.projectName)}`, '_blank', 'noopener'));
                    const continueBtn = document.createElement('button');
                    continueBtn.type = 'button';
                    continueBtn.className = 'btn btn-sm btn-success';
                    continueBtn.textContent = widget && widget.continueLabel ? widget.continueLabel : 'Continue';
                    const hasRemainingScripts = ctx.collectScriptStepsForRun(index + 1).length > 0;
                    continueBtn.disabled = step.enabled === false || !hasRemainingScripts;
                    if (!hasRemainingScripts) continueBtn.title = ctx.t('workflow.messages.no_more_steps');
                    continueBtn.addEventListener('click', () => ctx.handleWidgetContinue(index));
                    buttonRow.append(reviewBtn, continueBtn);
                    infoCell.append(buttonRow);
                    if (widget && Array.isArray(widget.parameters) && widget.parameters.length) {
                        const summary = ctx.describeWidgetParameters(widget, step);
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
                    title.textContent = ctx.formatScriptDirectoryName(script, step);
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
                    summary.textContent = ctx.describeStepParameters(step, script, missing);
                    infoCell.append(summary);
                    if (script && script.api) {
                        const apiInfoLine = document.createElement('div');
                        apiInfoLine.className = 'small mt-1 script-api-highlight';
                        apiInfoLine.textContent = `${ctx.t('workflow.labels_extra.api_usage')}: ${String(script.api).toUpperCase()}`;
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
                    widgetBadge.textContent = ctx.t('workflow.labels_extra.widget_badge', {}, 'Widget');
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
                        ctx.markWorkflowDirty();
                        renderWorkflowSteps();
                    });
                    const haltLabel = document.createElement('label');
                    haltLabel.className = 'form-check-label small';
                    haltLabel.textContent = haltCheckbox.checked ? ctx.t('workflow.labels_extra.halt_on_fail') : ctx.t('workflow.labels_extra.continue_on_fail');
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
                    editBtn.textContent = ctx.t('workflow.buttons_extra.parameters');
                    editBtn.addEventListener('click', () => ctx.openWorkflowParams(index));
                    actionsCell.append(editBtn);
                } else if (widget && Array.isArray(widget.parameters) && widget.parameters.length) {
                    const editBtn = document.createElement('button');
                    editBtn.type = 'button';
                    editBtn.className = 'btn btn-sm btn-outline-primary me-2';
                    editBtn.textContent = ctx.t('workflow.buttons_extra.parameters');
                    editBtn.addEventListener('click', () => ctx.openWorkflowWidgetParams(index));
                    actionsCell.append(editBtn);
                }
                const upBtn = document.createElement('button');
                upBtn.type = 'button';
                upBtn.className = 'btn btn-sm btn-outline-secondary me-1';
                upBtn.innerHTML = '&uarr;';
                upBtn.disabled = index === 0;
                upBtn.addEventListener('click', () => ctx.moveWorkflowStep(index, -1));
                const downBtn = document.createElement('button');
                downBtn.type = 'button';
                downBtn.className = 'btn btn-sm btn-outline-secondary me-1';
                downBtn.innerHTML = '&darr;';
                downBtn.disabled = index === workflowSteps.length - 1;
                downBtn.addEventListener('click', () => ctx.moveWorkflowStep(index, 1));
                const removeBtn = document.createElement('button');
                removeBtn.type = 'button';
                removeBtn.className = 'btn btn-sm btn-outline-danger';
                removeBtn.innerHTML = '&times;';
                removeBtn.addEventListener('click', () => ctx.removeWorkflowStep(index));
                actionsCell.append(upBtn, downBtn, removeBtn);
                row.append(actionsCell);
                tbody.append(row);
            });
            if (ctx.getCycleState()) {
                const cycleState = ctx.getCycleState();
                const widgetIndex = cycleState.widgetIndex;
                const currentIteration = cycleState.activeIteration !== null ? cycleState.activeIteration : Math.max(0, cycleState.nextIteration - 1);
                ctx.updateCycleDisplay(widgetIndex, currentIteration, cycleState.totalIterations);
            }
            ctx.updateMissingBadge();
            const startButton = document.getElementById('startWorkflowBtn');
            if (startButton) {
                startButton.disabled = ctx.collectScriptStepsForRun(0).length === 0;
            }
            const statusText = document.getElementById('workflowStatusText');
            if (statusText && !ctx.getCurrentWorkflowJobId()) {
                const hasAnyScript = workflowSteps.some(step => step.type !== 'widget');
                const hasEnabledScripts = workflowSteps.some(step => step.type !== 'widget' && step.enabled !== false);
                const runnableSegment = ctx.collectScriptStepsForRun(0);
                if (!hasAnyScript) statusText.textContent = ctx.t('workflow.status_texts.add_steps');
                else if (!hasEnabledScripts) statusText.textContent = ctx.t('workflow.status_texts.enable_steps');
                else if (!runnableSegment.length) statusText.textContent = ctx.t('workflow.status_texts.no_steps_before_widget');
                else statusText.textContent = ctx.t('workflow.status_texts.ready');
            }
        }

        function openWorkflowStepPicker() {
            populateWorkflowStepList('');
            if (ctx.getWorkflowStepModal()) {
                ctx.getWorkflowStepModal().show();
            }
        }

        function populateWorkflowStepList(filterText) {
            const list = document.getElementById('workflowStepList');
            if (!list) return;
            list.innerHTML = '';
            const term = (filterText || '').toLowerCase();
            const scripts = ctx.getAvailableScripts().slice().sort((a, b) => (a.display_name || a.script).localeCompare(b.display_name || b.script));
            const filtered = term
                ? scripts.filter(script => (script.display_name && script.display_name.toLowerCase().includes(term)) || (script.script && script.script.toLowerCase().includes(term)))
                : scripts;
            if (!filtered.length) {
                const empty = document.createElement('div');
                empty.className = 'list-group-item text-muted';
                empty.textContent = ctx.t('workflow.search.no_results');
                list.append(empty);
                return;
            }
            filtered.forEach(script => {
                const item = document.createElement('div');
                item.className = 'list-group-item ps-4';
                const title = document.createElement('div');
                title.className = 'fw-semibold';
                title.textContent = script.display_name || script.script;
                item.append(title);
                if (script.description) {
                    const desc = document.createElement('div');
                    desc.className = 'small text-muted';
                    desc.textContent = script.description;
                    item.append(desc);
                }
                const addBtn = document.createElement('button');
                addBtn.type = 'button';
                addBtn.className = 'btn btn-sm btn-primary mt-2';
                addBtn.textContent = ctx.t('workflow.buttons.add_step');
                addBtn.addEventListener('click', () => {
                    ctx.addWorkflowStep(script.id);
                    if (ctx.getWorkflowStepModal()) {
                        ctx.getWorkflowStepModal().hide();
                    }
                });
                item.append(addBtn);
                list.append(item);
            });
        }

        return {
            findTemplateById,
            populateWorkflowTemplateSelect,
            refreshWorkflowTemplates,
            loadWorkflowTemplateById,
            showSaveWorkflowModal,
            saveWorkflowTemplateFromModal,
            initWorkflowButtons,
            renderWorkflowSteps,
            openWorkflowStepPicker,
            populateWorkflowStepList
        };
    }

    window.ProjectWorkflowEditor = { createProjectWorkflowEditor };
})();
