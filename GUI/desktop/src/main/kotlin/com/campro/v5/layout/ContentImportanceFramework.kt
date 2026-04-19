package com.campro.v5.layout

import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp

/**
 * Content Importance Classification Framework
 *
 * This framework provides a sophisticated content importance classification system
 * for proportional panel resizing functionality. It categorizes panel content based
 * on contextual relevance, user workflow patterns, and information density requirements.
 */

// Content importance hierarchy with five distinct levels
enum class ContentImportance(val weight: Float, val description: String) {
    CRITICAL(1.0f, "Essential content that must always be visible"),
    HIGH(0.8f, "Important content that should be prominently displayed"),
    MEDIUM(0.6f, "Standard content with moderate importance"),
    LOW(0.4f, "Secondary content that can be minimized when space is limited"),
    MINIMAL(0.2f, "Optional content that can be heavily compressed or hidden"),
}

/**
 * Content type enumeration covering different panel content categories
 */
enum class ContentType(val defaultImportance: ContentImportance, val preferredAspectRatio: Float) {
    PARAMETERS(ContentImportance.HIGH, 0.75f), // Slightly taller than wide
    ANIMATION(ContentImportance.CRITICAL, 1.0f), // Square aspect ratio
    PLOTS(ContentImportance.HIGH, 1.33f), // Wider than tall for better visualization
    DATA(ContentImportance.MEDIUM, 1.0f), // Square for data tables
    GENERAL(ContentImportance.MEDIUM, 1.0f), // Default square aspect ratio
}

/**
 * Application state enumeration for context-aware importance adjustment
 */
enum class ApplicationState {
    DESIGN_MODE,
    ANALYSIS_MODE,
    SIMULATION_MODE,
    REPORTING_MODE,
    COLLABORATION_MODE,
    IDLE,
}

/**
 * Workflow type enumeration for workflow-specific optimization
 */
enum class WorkflowType {
    DESIGN_WORKFLOW,
    ANALYSIS_WORKFLOW,
    SIMULATION_WORKFLOW,
    REPORTING_WORKFLOW,
    COLLABORATION_WORKFLOW,
    GENERAL_WORKFLOW,
}

/**
 * Panel configuration data structure that encapsulates all panel properties
 */
data class PanelConfiguration(
    val id: String,
    val title: String,
    val contentType: ContentType,
    val baseImportance: ContentImportance,
    val minWidthRatio: Float = 0.15f, // Minimum 15% of available width
    val maxWidthRatio: Float = 0.6f, // Maximum 60% of available width
    val minHeightRatio: Float = 0.2f, // Minimum 20% of available height
    val maxHeightRatio: Float = 0.8f, // Maximum 80% of available height
    val preferredWidthRatio: Float = 0.33f, // Preferred 33% of available width
    val preferredHeightRatio: Float = 0.4f, // Preferred 40% of available height
    val aspectRatioPreference: Float = contentType.preferredAspectRatio,
    val allowManualResize: Boolean = true,
    val isCollapsible: Boolean = false,
) {
    /**
     * Calculate dynamic importance based on application state and workflow
     */
    fun calculateDynamicImportance(applicationState: ApplicationState, workflowType: WorkflowType, usageFrequency: Float = 1.0f): Float {
        var adjustedWeight = baseImportance.weight

        // Adjust based on application state
        adjustedWeight *=
            when (applicationState) {
                ApplicationState.DESIGN_MODE ->
                    when (contentType) {
                        ContentType.PARAMETERS -> 1.2f
                        ContentType.ANIMATION -> 1.1f
                        ContentType.PLOTS -> 0.9f
                        ContentType.DATA -> 0.8f
                        ContentType.GENERAL -> 1.0f
                    }
                ApplicationState.ANALYSIS_MODE ->
                    when (contentType) {
                        ContentType.PLOTS -> 1.3f
                        ContentType.DATA -> 1.2f
                        ContentType.PARAMETERS -> 1.0f
                        ContentType.ANIMATION -> 0.8f
                        ContentType.GENERAL -> 1.0f
                    }
                ApplicationState.SIMULATION_MODE ->
                    when (contentType) {
                        ContentType.ANIMATION -> 1.4f
                        ContentType.PARAMETERS -> 1.1f
                        ContentType.PLOTS -> 1.0f
                        ContentType.DATA -> 0.9f
                        ContentType.GENERAL -> 1.0f
                    }
                ApplicationState.REPORTING_MODE ->
                    when (contentType) {
                        ContentType.PLOTS -> 1.2f
                        ContentType.DATA -> 1.3f
                        ContentType.PARAMETERS -> 0.9f
                        ContentType.ANIMATION -> 0.8f
                        ContentType.GENERAL -> 1.0f
                    }
                ApplicationState.COLLABORATION_MODE ->
                    when (contentType) {
                        ContentType.ANIMATION -> 1.2f
                        ContentType.PLOTS -> 1.1f
                        ContentType.PARAMETERS -> 1.0f
                        ContentType.DATA -> 1.0f
                        ContentType.GENERAL -> 1.0f
                    }
                ApplicationState.IDLE -> 1.0f
            }

        // Adjust based on workflow type
        adjustedWeight *=
            when (workflowType) {
                WorkflowType.DESIGN_WORKFLOW ->
                    when (contentType) {
                        ContentType.PARAMETERS -> 1.15f
                        ContentType.ANIMATION -> 1.1f
                        else -> 1.0f
                    }
                WorkflowType.ANALYSIS_WORKFLOW ->
                    when (contentType) {
                        ContentType.PLOTS -> 1.2f
                        ContentType.DATA -> 1.15f
                        else -> 1.0f
                    }
                WorkflowType.SIMULATION_WORKFLOW ->
                    when (contentType) {
                        ContentType.ANIMATION -> 1.25f
                        ContentType.PARAMETERS -> 1.1f
                        else -> 1.0f
                    }
                WorkflowType.REPORTING_WORKFLOW ->
                    when (contentType) {
                        ContentType.PLOTS -> 1.15f
                        ContentType.DATA -> 1.2f
                        else -> 1.0f
                    }
                WorkflowType.COLLABORATION_WORKFLOW -> 1.05f
                WorkflowType.GENERAL_WORKFLOW -> 1.0f
            }

        // Apply usage frequency multiplier
        adjustedWeight *= (0.8f + (usageFrequency * 0.4f)) // Range: 0.8 to 1.2

        return adjustedWeight.coerceIn(0.1f, 2.0f) // Clamp to reasonable range
    }

    /**
     * Calculate preferred size based on available space and importance
     */
    fun calculatePreferredSize(
        availableWidth: Dp,
        availableHeight: Dp,
        dynamicImportance: Float,
        totalImportanceWeight: Float,
    ): Pair<Dp, Dp> {
        // Calculate base size ratios based on importance
        val importanceRatio = dynamicImportance / totalImportanceWeight

        // Calculate width considering aspect ratio preference
        val baseWidthRatio = (preferredWidthRatio * importanceRatio).coerceIn(minWidthRatio, maxWidthRatio)
        val baseHeightRatio = (preferredHeightRatio * importanceRatio).coerceIn(minHeightRatio, maxHeightRatio)

        // Adjust for aspect ratio preference
        val calculatedWidth = availableWidth * baseWidthRatio
        val calculatedHeight = availableHeight * baseHeightRatio

        // Apply aspect ratio constraints
        val aspectRatioAdjustedHeight = calculatedWidth / aspectRatioPreference
        val aspectRatioAdjustedWidth = calculatedHeight * aspectRatioPreference

        // Choose the size that best fits within constraints
        val finalWidth =
            if (aspectRatioAdjustedHeight <= availableHeight * maxHeightRatio) {
                calculatedWidth
            } else {
                aspectRatioAdjustedWidth.coerceAtMost(availableWidth * maxWidthRatio)
            }

        val finalHeight =
            if (aspectRatioAdjustedWidth <= availableWidth * maxWidthRatio) {
                calculatedHeight
            } else {
                aspectRatioAdjustedHeight.coerceAtMost(availableHeight * maxHeightRatio)
            }

        return Pair(
            finalWidth.coerceIn(availableWidth * minWidthRatio, availableWidth * maxWidthRatio),
            finalHeight.coerceIn(availableHeight * minHeightRatio, availableHeight * maxHeightRatio),
        )
    }
}

/**
 * Usage tracking data for dynamic priority adjustment
 */
data class PanelUsageData(
    val panelId: String,
    val interactionCount: Int = 0,
    val totalTimeVisible: Long = 0L, // in milliseconds
    val lastInteractionTime: Long = 0L,
    val resizeCount: Int = 0,
    val averageSize: Pair<Dp, Dp> = Pair(300.dp, 200.dp),
) {
    /**
     * Calculate usage frequency score (0.0 to 1.0)
     */
    fun calculateUsageFrequency(currentTime: Long, totalApplicationTime: Long): Float {
        if (totalApplicationTime <= 0) return 0.5f

        val visibilityRatio = totalTimeVisible.toFloat() / totalApplicationTime
        val interactionFrequency = interactionCount.toFloat() / (totalApplicationTime / 60000f) // interactions per minute
        val recencyFactor = if (currentTime - lastInteractionTime < 300000) 1.2f else 1.0f // 5 minute recency bonus

        return ((visibilityRatio * 0.4f) + (interactionFrequency * 0.4f) + 0.2f) * recencyFactor
    }
}

/**
 * Workflow template configuration for optimized layouts
 */
data class WorkflowTemplate(
    val name: String,
    val workflowType: WorkflowType,
    val panelConfigurations: Map<String, PanelConfiguration>,
    val layoutPreferences: LayoutPreferences = LayoutPreferences(),
)

/**
 * Layout preferences for workflow templates
 */
data class LayoutPreferences(
    val preferSingleColumn: Boolean = false,
    val compactMode: Boolean = false,
    val emphasizeAnimation: Boolean = false,
    val emphasizePlots: Boolean = false,
    val emphasizeParameters: Boolean = false,
    val emphasizeData: Boolean = false,
    val customSpacing: Dp = 8.dp,
)
