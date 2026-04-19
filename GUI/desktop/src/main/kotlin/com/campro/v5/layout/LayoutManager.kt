package com.campro.v5.layout

import androidx.compose.runtime.Composable
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import com.campro.v5.emitLayout

/**
 * Manages responsive layouts for the CamPro v5 application.
 *
 * The LayoutManager is responsible for:
 * - Detecting and adapting to different window sizes
 * - Supporting different screen densities
 * - Providing layout templates for different use cases
 * - Emitting events when layout changes occur
 *
 * Usage:
 * ```
 * // Get the singleton instance
 * val layoutManager = LayoutManager.getInstance()
 *
 * // Update window size when it changes
 * layoutManager.updateWindowSize(width, height)
 *
 * // Set a specific layout template
 * layoutManager.setTemplate(LayoutTemplate.ANALYSIS_WORKFLOW)
 *
 * // Use helper methods to adjust UI based on current layout
 * if (layoutManager.shouldUseCompactMode()) {
 *     // Use compact layout
 * }
 *
 * // In Compose, use the rememberLayoutManager() function
 * val layoutManager = rememberLayoutManager()
 * ```
 *
 * Layout Templates:
 * - DEFAULT: Balanced layout for general use
 * - COMPACT: Space-efficient layout for small screens
 * - EXPANDED: Spacious layout for large screens
 * - PRESENTATION: Optimized for demos with larger text and controls
 * - DEVELOPMENT: Shows additional debugging information
 * - DESIGN_WORKFLOW: Emphasizes parameter input and animation
 * - ANALYSIS_WORKFLOW: Emphasizes plots and data display
 * - SIMULATION_WORKFLOW: Emphasizes animation and controls
 * - REPORTING: Emphasizes data display and export options
 * - COLLABORATION: Emphasizes sharing and annotation tools
 * - TOUCH_OPTIMIZED: Designed for touch input with larger controls
 * - SINGLE_PANEL: Focuses on a single component at a time
 *
 * Events:
 * The LayoutManager emits events through the EventSystem when layout changes occur:
 * - "window_size_changed": When the window size changes
 * - "density_changed": When the screen density factor changes
 * - "template_changed": When the layout template changes
 */
class LayoutManager {
    // Window size breakpoints
    val smallWindowWidth = 800.dp
    val mediumWindowWidth = 1200.dp
    val largeWindowWidth = 1600.dp

    val smallWindowHeight = 600.dp
    val mediumWindowHeight = 900.dp
    val largeWindowHeight = 1200.dp

    // Screen density factors
    val lowDensityFactor = 0.75f
    val mediumDensityFactor = 1.0f
    val highDensityFactor = 1.5f
    val veryHighDensityFactor = 2.0f

    // Current window size and density
    private val _currentWindowWidth = mutableStateOf(1200.dp)
    private val _currentWindowHeight = mutableStateOf(800.dp)
    private val _currentDensityFactor = mutableStateOf(mediumDensityFactor)

    // Layout templates
    enum class LayoutTemplate {
        /**
         * Default layout with balanced space allocation
         */
        DEFAULT,

        /**
         * Compact layout for small screens or when space is limited
         */
        COMPACT,

        /**
         * Expanded layout for large screens with more content visible at once
         */
        EXPANDED,

        /**
         * Presentation layout optimized for demos and presentations
         */
        PRESENTATION,

        /**
         * Development layout with additional debugging information
         */
        DEVELOPMENT,

        /**
         * Design workflow layout with emphasis on parameter input and animation
         */
        DESIGN_WORKFLOW,

        /**
         * Analysis workflow layout with emphasis on plots and data display
         */
        ANALYSIS_WORKFLOW,

        /**
         * Simulation workflow layout with emphasis on animation and controls
         */
        SIMULATION_WORKFLOW,

        /**
         * Reporting layout with emphasis on data display and export options
         */
        REPORTING,

        /**
         * Collaboration layout with emphasis on sharing and annotation tools
         */
        COLLABORATION,

        /**
         * Touch-optimized layout for tablet devices
         */
        TOUCH_OPTIMIZED,

        /**
         * Single-panel layout for focused work on one component
         */
        SINGLE_PANEL,
    }

    private val _currentTemplate = mutableStateOf(LayoutTemplate.DEFAULT)

    // Getters for current values
    val currentWindowWidth: Dp
        get() = _currentWindowWidth.value

    val currentWindowHeight: Dp
        get() = _currentWindowHeight.value

    val currentDensityFactor: Float
        get() = _currentDensityFactor.value

    val currentTemplate: LayoutTemplate
        get() = _currentTemplate.value

    /**
     * Reset the layout manager state.
     * This is primarily used for testing to ensure a clean state between tests.
     */
    fun resetState() {
        _currentWindowWidth.value = 1200.dp
        _currentWindowHeight.value = 800.dp
        _currentDensityFactor.value = mediumDensityFactor
        _currentTemplate.value = LayoutTemplate.DEFAULT
    }

    /**
     * Update the window size.
     *
     * @param width The new window width
     * @param height The new window height
     */
    fun updateWindowSize(width: Dp, height: Dp) {
        val oldWidth = _currentWindowWidth.value
        val oldHeight = _currentWindowHeight.value
        val oldTemplate = _currentTemplate.value

        _currentWindowWidth.value = width
        _currentWindowHeight.value = height

        // Automatically adjust template based on window size
        val newTemplate =
            when {
                width < smallWindowWidth || height < smallWindowHeight -> LayoutTemplate.COMPACT
                width > largeWindowWidth && height > largeWindowHeight -> LayoutTemplate.EXPANDED
                // Special case for medium to small transition - ensure COMPACT is selected
                oldTemplate != LayoutTemplate.COMPACT &&
                    (width < 800.dp || height < 600.dp) -> LayoutTemplate.COMPACT
                else -> oldTemplate // Keep current template if no specific rule applies
            }

        _currentTemplate.value = newTemplate

        // Emit window size changed event
        emitLayout(
            "window_size_changed",
            mapOf(
                "width" to width.value,
                "height" to height.value,
                "oldWidth" to oldWidth.value,
                "oldHeight" to oldHeight.value,
            ),
        )

        // If template changed as a result, emit template changed event
        if (oldTemplate != _currentTemplate.value) {
            emitLayout(
                "template_changed",
                mapOf(
                    "template" to _currentTemplate.value.name,
                    "oldTemplate" to oldTemplate.name,
                ),
            )
        }
    }

    /**
     * Update the screen density factor.
     *
     * @param factor The new density factor
     */
    fun updateDensityFactor(factor: Float) {
        val oldFactor = _currentDensityFactor.value
        _currentDensityFactor.value = factor

        // Emit density factor changed event
        emitLayout(
            "density_changed",
            mapOf(
                "factor" to factor,
                "oldFactor" to oldFactor,
            ),
        )
    }

    /**
     * Set the layout template.
     *
     * @param template The new layout template
     */
    fun setTemplate(template: LayoutTemplate) {
        val oldTemplate = _currentTemplate.value
        _currentTemplate.value = template

        // Emit template changed event
        emitLayout(
            "template_changed",
            mapOf(
                "template" to template.name,
                "oldTemplate" to oldTemplate.name,
            ),
        )
    }

    /**
     * Get the appropriate padding for the current window size and density.
     *
     * @return The padding value in dp
     */
    fun getAppropriateSpacing(): Dp {
        val basePadding =
            when {
                _currentWindowWidth.value < smallWindowWidth -> 8.dp
                _currentWindowWidth.value < mediumWindowWidth -> 16.dp
                else -> 24.dp
            }

        return basePadding * _currentDensityFactor.value
    }

    /**
     * Determine if the layout should use a compact mode.
     *
     * @return True if compact mode should be used, false otherwise
     */
    fun shouldUseCompactMode(): Boolean = _currentWindowWidth.value < mediumWindowWidth ||
        _currentWindowHeight.value < mediumWindowHeight ||
        _currentTemplate.value == LayoutTemplate.COMPACT ||
        _currentTemplate.value == LayoutTemplate.TOUCH_OPTIMIZED

    /**
     * Determine if the layout should use a single column layout.
     *
     * @return True if single column layout should be used, false otherwise
     */
    fun shouldUseSingleColumn(): Boolean = _currentWindowWidth.value < smallWindowWidth ||
        _currentTemplate.value == LayoutTemplate.COMPACT ||
        _currentTemplate.value == LayoutTemplate.SINGLE_PANEL ||
        _currentTemplate.value == LayoutTemplate.TOUCH_OPTIMIZED

    /**
     * Determine if the layout should emphasize design elements.
     *
     * @return True if design elements should be emphasized, false otherwise
     */
    fun shouldEmphasizeDesign(): Boolean = _currentTemplate.value == LayoutTemplate.DESIGN_WORKFLOW

    /**
     * Determine if the layout should emphasize analysis elements.
     *
     * @return True if analysis elements should be emphasized, false otherwise
     */
    fun shouldEmphasizeAnalysis(): Boolean = _currentTemplate.value == LayoutTemplate.ANALYSIS_WORKFLOW

    /**
     * Determine if the layout should emphasize simulation elements.
     *
     * @return True if simulation elements should be emphasized, false otherwise
     */
    fun shouldEmphasizeSimulation(): Boolean = _currentTemplate.value == LayoutTemplate.SIMULATION_WORKFLOW

    /**
     * Determine if the layout should emphasize reporting elements.
     *
     * @return True if reporting elements should be emphasized, false otherwise
     */
    fun shouldEmphasizeReporting(): Boolean = _currentTemplate.value == LayoutTemplate.REPORTING

    /**
     * Determine if the layout should emphasize collaboration elements.
     *
     * @return True if collaboration elements should be emphasized, false otherwise
     */
    fun shouldEmphasizeCollaboration(): Boolean = _currentTemplate.value == LayoutTemplate.COLLABORATION

    /**
     * Determine if the layout should show additional development information.
     *
     * @return True if development information should be shown, false otherwise
     */
    fun shouldShowDevelopmentInfo(): Boolean = _currentTemplate.value == LayoutTemplate.DEVELOPMENT

    /**
     * Determine if the layout should be optimized for presentations.
     *
     * @return True if the layout should be optimized for presentations, false otherwise
     */
    fun shouldOptimizeForPresentation(): Boolean = _currentTemplate.value == LayoutTemplate.PRESENTATION

    // ========== PROPORTIONAL PANEL SCALING SYSTEM ==========

    // Panel configurations and state management
    private val _panelConfigurations = mutableStateOf<Map<String, PanelConfiguration>>(emptyMap())
    private val _panelUsageData = mutableStateOf<Map<String, PanelUsageData>>(emptyMap())
    private val _applicationState = mutableStateOf(ApplicationState.IDLE)
    private val _currentWorkflowType = mutableStateOf(WorkflowType.GENERAL_WORKFLOW)
    private val _workflowTemplates = mutableStateOf<Map<String, WorkflowTemplate>>(emptyMap())
    private val _proportionalSizingEnabled = mutableStateOf(true)

    // Getters for proportional sizing state
    val panelConfigurations: Map<String, PanelConfiguration>
        get() = _panelConfigurations.value

    val applicationState: ApplicationState
        get() = _applicationState.value

    val currentWorkflowType: WorkflowType
        get() = _currentWorkflowType.value

    val isProportionalSizingEnabled: Boolean
        get() = _proportionalSizingEnabled.value

    /**
     * Register a panel configuration for proportional sizing
     */
    fun registerPanelConfiguration(config: PanelConfiguration) {
        val currentConfigs = _panelConfigurations.value.toMutableMap()
        currentConfigs[config.id] = config
        _panelConfigurations.value = currentConfigs

        // Initialize usage data if not exists
        if (!_panelUsageData.value.containsKey(config.id)) {
            val currentUsageData = _panelUsageData.value.toMutableMap()
            currentUsageData[config.id] = PanelUsageData(config.id)
            _panelUsageData.value = currentUsageData
        }

        emitLayout("panel_registered", mapOf("panelId" to config.id))
    }

    /**
     * Update application state for context-aware sizing
     */
    fun updateApplicationState(newState: ApplicationState) {
        val oldState = _applicationState.value
        _applicationState.value = newState

        // Update workflow type based on application state
        _currentWorkflowType.value =
            when (newState) {
                ApplicationState.DESIGN_MODE -> WorkflowType.DESIGN_WORKFLOW
                ApplicationState.ANALYSIS_MODE -> WorkflowType.ANALYSIS_WORKFLOW
                ApplicationState.SIMULATION_MODE -> WorkflowType.SIMULATION_WORKFLOW
                ApplicationState.REPORTING_MODE -> WorkflowType.REPORTING_WORKFLOW
                ApplicationState.COLLABORATION_MODE -> WorkflowType.COLLABORATION_WORKFLOW
                ApplicationState.IDLE -> WorkflowType.GENERAL_WORKFLOW
            }

        emitLayout(
            "application_state_changed",
            mapOf(
                "newState" to newState.name,
                "oldState" to oldState.name,
                "workflowType" to _currentWorkflowType.value.name,
            ),
        )
    }

    /**
     * Calculate proportional sizes for all registered panels
     */
    fun calculateProportionalSizes(
        availableWidth: Dp = currentWindowWidth,
        availableHeight: Dp = currentWindowHeight,
        preservePadding: Dp = getAppropriateSpacing(),
    ): Map<String, Pair<Dp, Dp>> {
        if (!_proportionalSizingEnabled.value || _panelConfigurations.value.isEmpty()) {
            return emptyMap()
        }

        val currentTime = System.currentTimeMillis()
        val totalApplicationTime = 3600000L // Assume 1 hour for demo purposes

        // Calculate available space after padding
        val effectiveWidth = (availableWidth - (preservePadding * 2)).coerceAtLeast(100.dp)
        val effectiveHeight = (availableHeight - (preservePadding * 2)).coerceAtLeast(100.dp)

        // Calculate dynamic importance for each panel
        val panelImportances = mutableMapOf<String, Float>()
        var totalImportanceWeight = 0f

        for ((panelId, config) in _panelConfigurations.value) {
            val usageData = _panelUsageData.value[panelId] ?: PanelUsageData(panelId)
            val usageFrequency = usageData.calculateUsageFrequency(currentTime, totalApplicationTime)

            val dynamicImportance =
                config.calculateDynamicImportance(
                    _applicationState.value,
                    _currentWorkflowType.value,
                    usageFrequency,
                )

            panelImportances[panelId] = dynamicImportance
            totalImportanceWeight += dynamicImportance
        }

        // Calculate preferred sizes for each panel
        val calculatedSizes = mutableMapOf<String, Pair<Dp, Dp>>()

        for ((panelId, config) in _panelConfigurations.value) {
            val importance = panelImportances[panelId] ?: 0.5f
            val preferredSize =
                config.calculatePreferredSize(
                    effectiveWidth,
                    effectiveHeight,
                    importance,
                    totalImportanceWeight,
                )
            calculatedSizes[panelId] = preferredSize
        }

        return calculatedSizes
    }

    /**
     * Update panel usage data for dynamic priority adjustment
     */
    fun updatePanelUsage(panelId: String, interactionType: String = "interaction", currentSize: Pair<Dp, Dp>? = null) {
        val currentUsageData = _panelUsageData.value.toMutableMap()
        val existingData = currentUsageData[panelId] ?: PanelUsageData(panelId)

        val updatedData =
            when (interactionType) {
                "interaction" ->
                    existingData.copy(
                        interactionCount = existingData.interactionCount + 1,
                        lastInteractionTime = System.currentTimeMillis(),
                    )
                "resize" ->
                    existingData.copy(
                        resizeCount = existingData.resizeCount + 1,
                        averageSize = currentSize ?: existingData.averageSize,
                        lastInteractionTime = System.currentTimeMillis(),
                    )
                "visibility" ->
                    existingData.copy(
                        totalTimeVisible = existingData.totalTimeVisible + 1000L, // Add 1 second
                    )
                else -> existingData
            }

        currentUsageData[panelId] = updatedData
        _panelUsageData.value = currentUsageData
    }

    /**
     * Apply workflow template for optimized panel layouts
     */
    fun applyWorkflowTemplate(template: WorkflowTemplate) {
        // Update panel configurations from template
        val currentConfigs = _panelConfigurations.value.toMutableMap()
        for ((panelId, config) in template.panelConfigurations) {
            currentConfigs[panelId] = config
        }
        _panelConfigurations.value = currentConfigs

        // Update workflow type
        _currentWorkflowType.value = template.workflowType

        // Update layout template based on preferences
        val layoutTemplate =
            when {
                template.layoutPreferences.compactMode -> LayoutTemplate.COMPACT
                template.layoutPreferences.preferSingleColumn -> LayoutTemplate.SINGLE_PANEL
                template.workflowType == WorkflowType.DESIGN_WORKFLOW -> LayoutTemplate.DESIGN_WORKFLOW
                template.workflowType == WorkflowType.ANALYSIS_WORKFLOW -> LayoutTemplate.ANALYSIS_WORKFLOW
                template.workflowType == WorkflowType.SIMULATION_WORKFLOW -> LayoutTemplate.SIMULATION_WORKFLOW
                template.workflowType == WorkflowType.REPORTING_WORKFLOW -> LayoutTemplate.REPORTING
                template.workflowType == WorkflowType.COLLABORATION_WORKFLOW -> LayoutTemplate.COLLABORATION
                else -> LayoutTemplate.DEFAULT
            }

        setTemplate(layoutTemplate)

        emitLayout(
            "workflow_template_applied",
            mapOf(
                "templateName" to template.name,
                "workflowType" to template.workflowType.name,
            ),
        )
    }

    /**
     * Enable or disable proportional sizing
     */
    fun setProportionalSizingEnabled(enabled: Boolean) {
        val wasEnabled = _proportionalSizingEnabled.value
        _proportionalSizingEnabled.value = enabled

        if (wasEnabled != enabled) {
            emitLayout("proportional_sizing_toggled", mapOf("enabled" to enabled))
        }
    }

    /**
     * Get panel configuration by ID
     */
    fun getPanelConfiguration(panelId: String): PanelConfiguration? = _panelConfigurations.value[panelId]

    /**
     * Get panel usage data by ID
     */
    fun getPanelUsageData(panelId: String): PanelUsageData? = _panelUsageData.value[panelId]

    /**
     * Reset all panel usage data
     */
    fun resetPanelUsageData() {
        val resetData =
            _panelUsageData.value.mapValues { (panelId, _) ->
                PanelUsageData(panelId)
            }
        _panelUsageData.value = resetData

        emitLayout("panel_usage_reset", emptyMap())
    }

    companion object {
        // Singleton instance
        private var instance: LayoutManager? = null

        /**
         * Get the singleton instance of the LayoutManager.
         *
         * @return The LayoutManager instance
         */
        fun getInstance(): LayoutManager {
            if (instance == null) {
                instance = LayoutManager()
            }
            return instance as LayoutManager
        }
    }
}

/**
 * Composable function to remember a LayoutManager instance.
 *
 * @return The remembered LayoutManager instance
 */
@Composable
fun rememberLayoutManager(): LayoutManager = remember { LayoutManager.getInstance() }
