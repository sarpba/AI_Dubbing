(function() {
    async function persistProjectWorkflowState(projectName, payload) {
        const response = await fetch(`/api/project-workflow-state/${encodeURIComponent(projectName)}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload || {})
        });
        let result = null;
        try {
            result = await response.json();
        } catch (error) {
            // ignore JSON parsing issues; handled below
        }
        if (!response.ok || !result || !result.success) {
            const message = result && result.error ? result.error : `HTTP ${response.status}`;
            throw new Error(message);
        }
        return result.state || null;
    }

    async function loadProjectWorkflowState(projectName, t) {
        try {
            const response = await fetch(`/api/project-workflow-state/${encodeURIComponent(projectName)}`, { cache: 'no-store' });
            if (response.status === 404) {
                return null;
            }
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            const result = await response.json();
            if (!result || result.success !== true) {
                const message = result && result.error ? result.error : t('workflow.errors.unknown_error');
                throw new Error(message);
            }
            const data = result.state || {};
            const steps = Array.isArray(data.steps) ? data.steps : [];
            const templateId = data.template_id || null;
            return { steps, template_id: templateId };
        } catch (error) {
            console.warn(t('js.logs.workflow_state_load_failed'), error);
        }
        return null;
    }

    async function loadDefaultWorkflowTemplate(templates, t) {
        const list = Array.isArray(templates) ? templates : [];
        const defaultEntry = list.find(item => item && item.id === 'default');
        if (!defaultEntry) {
            return null;
        }
        try {
            const response = await fetch(`/api/workflow-template/${encodeURIComponent(defaultEntry.id)}`);
            const result = await response.json();
            if (!response.ok || !result.success) {
                throw new Error(result.error || t('workflow.errors.unknown_error'));
            }
            const template = result.template || {};
            return Array.isArray(template.steps) ? template.steps : [];
        } catch (error) {
            console.error(t('js.logs.workflow_default_load_failed'), error);
            return null;
        }
    }

    async function runWorkflowAutoSave(context) {
        context.setWorkflowAutoSaveTimer(null);
        if (!context.getWorkflowAutoSaveEnabled() || context.getWorkflowSuppressAutoSave()) {
            return;
        }
        if (!context.getWorkflowAutoSavePending()) {
            return;
        }
        if (context.getWorkflowSaving()) {
            context.scheduleWorkflowAutoSave(300);
            return;
        }

        const snapshot = context.buildWorkflowSnapshot();
        const snapshotKey = JSON.stringify(snapshot);
        if (snapshotKey === context.getWorkflowLastSavedSnapshot()) {
            context.setWorkflowAutoSavePending(false);
            return;
        }

        if (!context.hasEnabledScriptStep(snapshot.steps)) {
            context.setWorkflowLastSavedSnapshot(snapshotKey);
            context.setWorkflowAutoSavePending(false);
            return;
        }

        context.setWorkflowSaving(true);
        try {
            const payload = {
                steps: snapshot.steps,
                template_id: snapshot.template_id,
                saved_at: new Date().toISOString()
            };
            await persistProjectWorkflowState(context.projectName, payload);
            context.setWorkflowLastSavedSnapshot(snapshotKey);
            context.setWorkflowAutoSavePending(false);
        } catch (error) {
            console.error(context.t('js.logs.workflow_state_save_failed'), error);
            context.setWorkflowAutoSavePending(true);
            context.scheduleWorkflowAutoSave(2000);
        } finally {
            context.setWorkflowSaving(false);
        }
    }

    window.ProjectWorkflowState = {
        persistProjectWorkflowState,
        loadProjectWorkflowState,
        loadDefaultWorkflowTemplate,
        runWorkflowAutoSave
    };
})();
