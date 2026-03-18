(function() {
    function createWorkflowWidgets(t) {
        return [
            {
                id: 'reviewContinue',
                name: t('workflow.widgets.review.name', {}, 'Review + Continue'),
                description: t('workflow.widgets.review.description'),
                continueLabel: t('workflow.widgets.review.continue_label', {}, 'Continue'),
                reviewLabel: t('workflow.widgets.review.review_label', {}, 'Review')
            },
            {
                id: 'cycleWidget',
                name: t('workflow.widgets.cycle.name', {}, 'Cycle'),
                description: t('workflow.widgets.cycle.description'),
                continueLabel: t('workflow.widgets.cycle.continue_label', {}, 'Run cycle'),
                reviewLabel: t('workflow.widgets.cycle.review_label', {}, 'Review'),
                parameters: [
                    {
                        name: 'repeat_count',
                        label: t('workflow.widgets.cycle.parameters.repeat.label'),
                        type: 'number',
                        min: 1,
                        step: 1,
                        default: 1,
                        required: true,
                        helper: t('workflow.widgets.cycle.parameters.repeat.helper')
                    },
                    {
                        name: 'step_back',
                        label: t('workflow.widgets.cycle.parameters.step_back.label'),
                        type: 'number',
                        min: 1,
                        step: 1,
                        default: 1,
                        required: true,
                        helper: t('workflow.widgets.cycle.parameters.step_back.helper')
                    }
                ],
                help: t('workflow.widgets.cycle.help')
            },
            {
                id: 'translatedSplitLoopWidget',
                name: t('workflow.widgets.segment_loop.name', {}, 'Translated Split Loop'),
                description: t(
                    'workflow.widgets.segment_loop.description',
                    {},
                    'Checks translated split progress and reruns the post-review block until enough files are ready.'
                ),
                continueLabel: t('workflow.widgets.segment_loop.continue_label', {}, 'Check loop'),
                reviewLabel: t('workflow.widgets.segment_loop.review_label', {}, 'Review'),
                parameters: [
                    {
                        name: 'allowed_missing_segments',
                        label: t('workflow.widgets.segment_loop.parameters.allowed_missing.label'),
                        type: 'number',
                        min: 0,
                        step: 1,
                        default: 0,
                        required: true,
                        helper: t('workflow.widgets.segment_loop.parameters.allowed_missing.helper')
                    }
                ],
                help: t('workflow.widgets.segment_loop.help')
            }
        ];
    }

    function findWidgetById(workflowWidgets, widgetId) {
        if (!widgetId) {
            return null;
        }
        return workflowWidgets.find(widget => widget.id === widgetId) || null;
    }

    function normalizeWorkflowStep(step, workflowWidgets) {
        if (!step || typeof step !== 'object') {
            return;
        }
        if (!step.type) {
            if (step.widget) {
                step.type = 'widget';
            } else if (step.script) {
                step.type = 'script';
            }
        }
        if (step.type === 'widget') {
            if (!step.widget) {
                step.widget = 'unknownWidget';
            }
            if (step.enabled === undefined) {
                step.enabled = true;
            }
            if (!step.params || typeof step.params !== 'object') {
                step.params = {};
            }
            const widgetMeta = findWidgetById(workflowWidgets, step.widget);
            if (widgetMeta && Array.isArray(widgetMeta.parameters)) {
                widgetMeta.parameters.forEach(param => {
                    if (param && param.name && step.params[param.name] === undefined && param.default !== undefined) {
                        step.params[param.name] = param.default;
                    }
                });
            }
            return;
        }

        step.type = 'script';
        if (step.enabled === undefined) {
            step.enabled = true;
        }
        if (step.halt_on_fail === undefined) {
            step.halt_on_fail = true;
        }
        if (!step.params || typeof step.params !== 'object') {
            step.params = {};
        }
    }

    function normalizeWorkflowStepList(steps, workflowWidgets) {
        (steps || []).forEach(step => normalizeWorkflowStep(step, workflowWidgets));
        return steps || [];
    }

    function buildRunStepFromWorkflowStep(step, cloneObject) {
        return {
            script: step.script,
            enabled: true,
            halt_on_fail: step.halt_on_fail !== false,
            params: cloneObject(step.params)
        };
    }

    function collectWorkflowSegment(workflowSteps, cloneObject, startIndex = 0) {
        const segment = {
            startIndex,
            steps: [],
            widgetIndex: null
        };
        for (let i = startIndex; i < workflowSteps.length; i++) {
            const step = workflowSteps[i];
            if (step.type === 'widget') {
                if (step.enabled !== false) {
                    segment.widgetIndex = i;
                    break;
                }
                continue;
            }
            if (step.enabled === false) {
                continue;
            }
            segment.steps.push(buildRunStepFromWorkflowStep(step, cloneObject));
        }
        return segment;
    }

    function collectScriptStepsForRun(workflowSteps, cloneObject, startIndex = 0) {
        return collectWorkflowSegment(workflowSteps, cloneObject, startIndex).steps;
    }

    function collectWorkflowState(workflowSteps, cloneSteps) {
        return cloneSteps(workflowSteps);
    }

    function buildWorkflowSnapshot(workflowSteps, currentTemplateId, cloneSteps) {
        return {
            steps: collectWorkflowState(workflowSteps, cloneSteps),
            template_id: currentTemplateId || null
        };
    }

    function hasEnabledScriptStep(steps) {
        return (steps || []).some(step => step && step.type !== 'widget' && step.enabled !== false);
    }

    window.ProjectWorkflowWidgets = {
        createWorkflowWidgets,
        findWidgetById,
        normalizeWorkflowStep,
        normalizeWorkflowStepList,
        buildRunStepFromWorkflowStep,
        collectWorkflowSegment,
        collectScriptStepsForRun,
        collectWorkflowState,
        buildWorkflowSnapshot,
        hasEnabledScriptStep
    };
})();
