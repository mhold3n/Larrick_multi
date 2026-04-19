package com.campro.v5.ui

import androidx.compose.runtime.*
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp

/**
 * Panel Layout Coordinator for managing interactions between multiple resizable panels.
 *
 * This class manages interactions between multiple resizable panels to prevent overlapping
 * and maintain proper spacing. It coordinates resize operations and ensures all panels
 * remain within acceptable bounds.
 */
class PanelLayoutCoordinator {
    private val panelStates = mutableMapOf<String, PanelState>()
    private val layoutConstraints = mutableMapOf<String, LayoutConstraints>()
    private var defaultSpacing: Dp = 8.dp

    /**
     * Register a panel with the coordinator
     */
    fun registerPanel(id: String, initialState: PanelState) {
        panelStates[id] = initialState
        layoutConstraints[id] =
            LayoutConstraints(
                minWidth = initialState.minWidth,
                minHeight = initialState.minHeight,
                maxWidth = initialState.maxWidth,
                maxHeight = initialState.maxHeight,
            )
    }

    /**
     * Unregister a panel from the coordinator
     */
    fun unregisterPanel(id: String) {
        panelStates.remove(id)
        layoutConstraints.remove(id)
    }

    /**
     * Update panel size and calculate necessary adjustments for other panels
     */
    fun updatePanelSize(id: String, newWidth: Dp, newHeight: Dp): List<PanelAdjustment> {
        val currentState = panelStates[id] ?: return emptyList()
        val constraints = layoutConstraints[id] ?: return emptyList()

        // Enforce size constraints
        val constrainedWidth = newWidth.coerceIn(constraints.minWidth, constraints.maxWidth)
        val constrainedHeight = newHeight.coerceIn(constraints.minHeight, constraints.maxHeight)

        // Update the panel state
        val updatedState =
            currentState.copy(
                width = constrainedWidth,
                height = constrainedHeight,
            )
        panelStates[id] = updatedState

        // Calculate adjustments for adjacent panels
        return calculateAdjustments(id, updatedState)
    }

    /**
     * Calculate adjustments needed for other panels when one panel is resized
     */
    private fun calculateAdjustments(resizedPanelId: String, newState: PanelState): List<PanelAdjustment> {
        val adjustments = mutableListOf<PanelAdjustment>()
        val resizedPanel = newState

        // Find panels that might be affected by this resize
        panelStates.forEach { (panelId, panelState) ->
            if (panelId != resizedPanelId) {
                // Check if panels overlap or need spacing adjustment
                val adjustment = calculatePanelAdjustment(resizedPanel, panelState, panelId)
                if (adjustment != null) {
                    adjustments.add(adjustment)
                }
            }
        }

        return adjustments
    }

    /**
     * Calculate adjustment for a specific panel relative to the resized panel
     */
    private fun calculatePanelAdjustment(resizedPanel: PanelState, targetPanel: PanelState, targetPanelId: String): PanelAdjustment? {
        // Simple logic: if panels would overlap, move the target panel
        val resizedRight = resizedPanel.x + resizedPanel.width
        val resizedBottom = resizedPanel.y + resizedPanel.height
        val targetRight = targetPanel.x + targetPanel.width
        val targetBottom = targetPanel.y + targetPanel.height

        var newX = targetPanel.x
        var newY = targetPanel.y
        var newWidth = targetPanel.width
        var newHeight = targetPanel.height
        var needsAdjustment = false

        // Check horizontal overlap
        if (resizedPanel.x < targetRight && resizedRight > targetPanel.x) {
            // Check vertical overlap
            if (resizedPanel.y < targetBottom && resizedBottom > targetPanel.y) {
                // Panels overlap - adjust position
                if (resizedPanel.x < targetPanel.x) {
                    // Move target panel to the right
                    newX = resizedRight + defaultSpacing
                    needsAdjustment = true
                } else {
                    // Move target panel down
                    newY = resizedBottom + defaultSpacing
                    needsAdjustment = true
                }
            }
        }

        return if (needsAdjustment) {
            PanelAdjustment(
                panelId = targetPanelId,
                newX = newX,
                newY = newY,
                newWidth = newWidth,
                newHeight = newHeight,
            )
        } else {
            null
        }
    }

    /**
     * Enforce minimum spacing between panels
     */
    fun enforceSpacing(spacing: Dp) {
        defaultSpacing = spacing

        // Recalculate all panel positions to maintain spacing
        val adjustments = mutableListOf<PanelAdjustment>()

        panelStates.forEach { (panelId, panelState) ->
            val otherPanels = panelStates.filterKeys { it != panelId }
            otherPanels.forEach { (otherId, otherState) ->
                val adjustment = calculateSpacingAdjustment(panelState, otherState, otherId, spacing)
                if (adjustment != null) {
                    adjustments.add(adjustment)
                }
            }
        }

        // Apply adjustments
        adjustments.forEach { adjustment ->
            val currentState = panelStates[adjustment.panelId]
            if (currentState != null) {
                panelStates[adjustment.panelId] =
                    currentState.copy(
                        x = adjustment.newX,
                        y = adjustment.newY,
                        width = adjustment.newWidth,
                        height = adjustment.newHeight,
                    )
            }
        }
    }

    /**
     * Calculate spacing adjustment between two panels
     */
    private fun calculateSpacingAdjustment(
        panel1: PanelState,
        panel2: PanelState,
        panel2Id: String,
        requiredSpacing: Dp,
    ): PanelAdjustment? {
        val panel1Right = panel1.x + panel1.width
        val panel1Bottom = panel1.y + panel1.height
        val panel2Right = panel2.x + panel2.width
        val panel2Bottom = panel2.y + panel2.height

        // Check if panels are too close horizontally
        val horizontalGap =
            when {
                panel1Right <= panel2.x -> panel2.x - panel1Right
                panel2Right <= panel1.x -> panel1.x - panel2Right
                else -> (-1).dp // Overlapping
            }

        // Check if panels are too close vertically
        val verticalGap =
            when {
                panel1Bottom <= panel2.y -> panel2.y - panel1Bottom
                panel2Bottom <= panel1.y -> panel1.y - panel2Bottom
                else -> (-1).dp // Overlapping
            }

        // If spacing is insufficient, adjust panel2
        if (horizontalGap >= 0.dp && horizontalGap < requiredSpacing) {
            val newX =
                if (panel1Right <= panel2.x) {
                    panel1Right + requiredSpacing
                } else {
                    panel1.x - panel2.width - requiredSpacing
                }

            return PanelAdjustment(
                panelId = panel2Id,
                newX = newX,
                newY = panel2.y,
                newWidth = panel2.width,
                newHeight = panel2.height,
            )
        }

        if (verticalGap >= 0.dp && verticalGap < requiredSpacing) {
            val newY =
                if (panel1Bottom <= panel2.y) {
                    panel1Bottom + requiredSpacing
                } else {
                    panel1.y - panel2.height - requiredSpacing
                }

            return PanelAdjustment(
                panelId = panel2Id,
                newX = panel2.x,
                newY = newY,
                newWidth = panel2.width,
                newHeight = panel2.height,
            )
        }

        return null
    }

    /**
     * Get current state of a panel
     */
    fun getPanelState(id: String): PanelState? = panelStates[id]

    /**
     * Get all registered panel IDs
     */
    fun getRegisteredPanels(): Set<String> = panelStates.keys.toSet()

    /**
     * Validate that all panels remain within container bounds
     */
    fun validateLayout(containerWidth: Dp, containerHeight: Dp): List<PanelAdjustment> {
        val adjustments = mutableListOf<PanelAdjustment>()

        panelStates.forEach { (panelId, panelState) ->
            var newX = panelState.x
            var newY = panelState.y
            var newWidth = panelState.width
            var newHeight = panelState.height
            var needsAdjustment = false

            // Ensure panel doesn't exceed container bounds
            if (panelState.x + panelState.width > containerWidth) {
                if (panelState.width <= containerWidth) {
                    newX = containerWidth - panelState.width
                } else {
                    newWidth = containerWidth
                    newX = 0.dp
                }
                needsAdjustment = true
            }

            if (panelState.y + panelState.height > containerHeight) {
                if (panelState.height <= containerHeight) {
                    newY = containerHeight - panelState.height
                } else {
                    newHeight = containerHeight
                    newY = 0.dp
                }
                needsAdjustment = true
            }

            // Ensure panel doesn't have negative coordinates
            if (panelState.x < 0.dp) {
                newX = 0.dp
                needsAdjustment = true
            }

            if (panelState.y < 0.dp) {
                newY = 0.dp
                needsAdjustment = true
            }

            if (needsAdjustment) {
                adjustments.add(
                    PanelAdjustment(
                        panelId = panelId,
                        newX = newX,
                        newY = newY,
                        newWidth = newWidth,
                        newHeight = newHeight,
                    ),
                )
            }
        }

        return adjustments
    }
}

/**
 * Data class representing the state of a panel
 */
data class PanelState(
    val x: Dp,
    val y: Dp,
    val width: Dp,
    val height: Dp,
    val minWidth: Dp,
    val minHeight: Dp,
    val maxWidth: Dp,
    val maxHeight: Dp,
)

/**
 * Data class representing layout constraints for a panel
 */
data class LayoutConstraints(val minWidth: Dp, val minHeight: Dp, val maxWidth: Dp, val maxHeight: Dp)

/**
 * Data class representing an adjustment to be made to a panel
 */
data class PanelAdjustment(val panelId: String, val newX: Dp, val newY: Dp, val newWidth: Dp, val newHeight: Dp)

/**
 * Composable function to remember a PanelLayoutCoordinator instance
 */
@Composable
fun rememberPanelLayoutCoordinator(): PanelLayoutCoordinator = remember { PanelLayoutCoordinator() }
