package com.campro.v5.ui

import androidx.compose.runtime.*
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.DpOffset
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/**
 * Docking zones for panel docking
 */
enum class DockZone {
    NONE,
    LEFT,
    RIGHT,
    TOP,
    BOTTOM,
    CENTER,
    TAB_GROUP,
}

/**
 * Panel docking state
 */
enum class PanelDockState {
    DOCKED,
    FLOATING,
    MINIMIZED,
    TABBED,
}

/**
 * Data class representing a dockable panel
 */
data class DockablePanel(
    val id: String,
    val title: String,
    val state: PanelDockState,
    val position: DpOffset,
    val size: Pair<Dp, Dp>,
    val dockZone: DockZone = DockZone.NONE,
    val tabGroupId: String? = null,
    val isMinimized: Boolean = false,
    val zIndex: Float = 0f,
)

/**
 * Data class representing a tab group
 */
data class TabGroup(
    val id: String,
    val title: String,
    val panelIds: List<String>,
    val activePanel: String,
    val position: DpOffset,
    val size: Pair<Dp, Dp>,
)

/**
 * Docking Manager for handling panel docking, undocking, and floating functionality
 */
class DockingManager {
    private val _panels = MutableStateFlow<Map<String, DockablePanel>>(emptyMap())
    val panels: StateFlow<Map<String, DockablePanel>> = _panels.asStateFlow()

    private val _tabGroups = MutableStateFlow<Map<String, TabGroup>>(emptyMap())
    val tabGroups: StateFlow<Map<String, TabGroup>> = _tabGroups.asStateFlow()

    private val _dragState = MutableStateFlow<DragState?>(null)
    val dragState: StateFlow<DragState?> = _dragState.asStateFlow()

    private val _dockZones = MutableStateFlow<Map<DockZone, Rect>>(emptyMap())
    val dockZones: StateFlow<Map<DockZone, Rect>> = _dockZones.asStateFlow()

    private var containerBounds: Rect = Rect.Zero
    private val dockThreshold = 50.dp

    /**
     * Register a panel with the docking manager
     */
    fun registerPanel(
        id: String,
        title: String,
        initialPosition: DpOffset = DpOffset.Zero,
        initialSize: Pair<Dp, Dp> = 400.dp to 300.dp,
        initialState: PanelDockState = PanelDockState.DOCKED,
    ) {
        val panel =
            DockablePanel(
                id = id,
                title = title,
                state = initialState,
                position = initialPosition,
                size = initialSize,
            )

        _panels.value = _panels.value + (id to panel)
    }

    /**
     * Unregister a panel from the docking manager
     */
    fun unregisterPanel(id: String) {
        _panels.value = _panels.value - id

        // Remove from any tab groups
        _tabGroups.value =
            _tabGroups.value
                .mapValues { (_, group) ->
                    val updatedPanelIds = group.panelIds.filter { it != id }
                    if (updatedPanelIds.isEmpty()) {
                        return@mapValues null
                    }
                    group.copy(
                        panelIds = updatedPanelIds,
                        activePanel = if (group.activePanel == id) updatedPanelIds.first() else group.activePanel,
                    )
                }.filterValues { it != null }
                .mapValues { it.value!! }
    }

    /**
     * Start dragging a panel
     */
    fun startDrag(panelId: String, startPosition: Offset) {
        val panel = _panels.value[panelId] ?: return

        _dragState.value =
            DragState(
                panelId = panelId,
                startPosition = startPosition,
                currentPosition = startPosition,
                isDragging = true,
            )

        // If panel is in a tab group, extract it
        if (panel.tabGroupId != null) {
            extractFromTabGroup(panelId)
        }
    }

    /**
     * Update drag position
     */
    fun updateDrag(newPosition: Offset) {
        val currentDrag = _dragState.value ?: return

        _dragState.value =
            currentDrag.copy(
                currentPosition = newPosition,
                hoveredZone = calculateHoveredZone(newPosition),
            )
    }

    /**
     * End dragging and dock the panel if appropriate
     */
    fun endDrag() {
        val dragState = _dragState.value ?: return
        val panel = _panels.value[dragState.panelId] ?: return

        val targetZone = dragState.hoveredZone

        when (targetZone) {
            DockZone.LEFT, DockZone.RIGHT, DockZone.TOP, DockZone.BOTTOM -> {
                dockPanel(dragState.panelId, targetZone)
            }
            DockZone.CENTER -> {
                // Keep as floating panel
                updatePanelPosition(dragState.panelId, dragState.currentPosition)
            }
            DockZone.TAB_GROUP -> {
                // Add to existing tab group or create new one
                handleTabGroupDocking(dragState.panelId, dragState.currentPosition)
            }
            DockZone.NONE -> {
                // Make floating
                makeFloating(dragState.panelId, dragState.currentPosition)
            }
        }

        _dragState.value = null
    }

    /**
     * Dock a panel to a specific zone
     */
    private fun dockPanel(panelId: String, zone: DockZone) {
        val panel = _panels.value[panelId] ?: return

        val dockedPosition = calculateDockedPosition(zone)
        val dockedSize = calculateDockedSize(zone)

        _panels.value = _panels.value + (
            panelId to
                panel.copy(
                    state = PanelDockState.DOCKED,
                    dockZone = zone,
                    position = dockedPosition,
                    size = dockedSize,
                    tabGroupId = null,
                )
            )
    }

    /**
     * Make a panel floating
     */
    fun makeFloating(panelId: String, position: Offset) {
        val panel = _panels.value[panelId] ?: return

        _panels.value = _panels.value + (
            panelId to
                panel.copy(
                    state = PanelDockState.FLOATING,
                    dockZone = DockZone.NONE,
                    position = DpOffset(position.x.dp, position.y.dp),
                    tabGroupId = null,
                )
            )
    }

    /**
     * Minimize a panel
     */
    fun minimizePanel(panelId: String) {
        val panel = _panels.value[panelId] ?: return

        _panels.value = _panels.value + (
            panelId to
                panel.copy(
                    state = PanelDockState.MINIMIZED,
                    isMinimized = true,
                )
            )
    }

    /**
     * Restore a minimized panel
     */
    fun restorePanel(panelId: String) {
        val panel = _panels.value[panelId] ?: return

        _panels.value = _panels.value + (
            panelId to
                panel.copy(
                    state = if (panel.dockZone != DockZone.NONE) PanelDockState.DOCKED else PanelDockState.FLOATING,
                    isMinimized = false,
                )
            )
    }

    /**
     * Create a tab group with multiple panels
     */
    fun createTabGroup(panelIds: List<String>, position: DpOffset, size: Pair<Dp, Dp>): String {
        val groupId = "tab_group_${System.currentTimeMillis()}"

        val tabGroup =
            TabGroup(
                id = groupId,
                title = "Tab Group",
                panelIds = panelIds,
                activePanel = panelIds.first(),
                position = position,
                size = size,
            )

        _tabGroups.value = _tabGroups.value + (groupId to tabGroup)

        // Update panels to be part of the tab group
        panelIds.forEach { panelId ->
            val panel = _panels.value[panelId]
            if (panel != null) {
                _panels.value = _panels.value + (
                    panelId to
                        panel.copy(
                            state = PanelDockState.TABBED,
                            tabGroupId = groupId,
                            position = position,
                            size = size,
                        )
                    )
            }
        }

        return groupId
    }

    /**
     * Add a panel to an existing tab group
     */
    fun addToTabGroup(panelId: String, tabGroupId: String) {
        val panel = _panels.value[panelId] ?: return
        val tabGroup = _tabGroups.value[tabGroupId] ?: return

        val updatedGroup =
            tabGroup.copy(
                panelIds = tabGroup.panelIds + panelId,
            )

        _tabGroups.value = _tabGroups.value + (tabGroupId to updatedGroup)

        _panels.value = _panels.value + (
            panelId to
                panel.copy(
                    state = PanelDockState.TABBED,
                    tabGroupId = tabGroupId,
                    position = tabGroup.position,
                    size = tabGroup.size,
                )
            )
    }

    /**
     * Extract a panel from its tab group
     */
    private fun extractFromTabGroup(panelId: String) {
        val panel = _panels.value[panelId] ?: return
        val tabGroupId = panel.tabGroupId ?: return
        val tabGroup = _tabGroups.value[tabGroupId] ?: return

        val updatedPanelIds = tabGroup.panelIds.filter { it != panelId }

        if (updatedPanelIds.isEmpty()) {
            // Remove the tab group if empty
            _tabGroups.value = _tabGroups.value - tabGroupId
        } else {
            // Update the tab group
            val updatedGroup =
                tabGroup.copy(
                    panelIds = updatedPanelIds,
                    activePanel = if (tabGroup.activePanel == panelId) updatedPanelIds.first() else tabGroup.activePanel,
                )
            _tabGroups.value = _tabGroups.value + (tabGroupId to updatedGroup)
        }

        // Update the panel to be floating
        _panels.value = _panels.value + (
            panelId to
                panel.copy(
                    state = PanelDockState.FLOATING,
                    tabGroupId = null,
                )
            )
    }

    /**
     * Set the active panel in a tab group
     */
    fun setActiveTabPanel(tabGroupId: String, panelId: String) {
        val tabGroup = _tabGroups.value[tabGroupId] ?: return

        if (panelId in tabGroup.panelIds) {
            _tabGroups.value = _tabGroups.value + (
                tabGroupId to
                    tabGroup.copy(
                        activePanel = panelId,
                    )
                )
        }
    }

    /**
     * Update container bounds for docking calculations
     */
    fun updateContainerBounds(bounds: Rect) {
        containerBounds = bounds
        updateDockZones()
    }

    /**
     * Update panel position
     */
    private fun updatePanelPosition(panelId: String, position: Offset) {
        val panel = _panels.value[panelId] ?: return

        _panels.value = _panels.value + (
            panelId to
                panel.copy(
                    position = DpOffset(position.x.dp, position.y.dp),
                )
            )
    }

    /**
     * Calculate which dock zone is being hovered
     */
    private fun calculateHoveredZone(position: Offset): DockZone {
        val threshold = dockThreshold.value

        return when {
            position.x < threshold -> DockZone.LEFT
            position.x > containerBounds.width - threshold -> DockZone.RIGHT
            position.y < threshold -> DockZone.TOP
            position.y > containerBounds.height - threshold -> DockZone.BOTTOM
            else -> DockZone.CENTER
        }
    }

    /**
     * Calculate docked position for a zone
     */
    private fun calculateDockedPosition(zone: DockZone): DpOffset = when (zone) {
        DockZone.LEFT -> DpOffset(0.dp, 0.dp)
        DockZone.RIGHT -> DpOffset((containerBounds.width * 0.7f).dp, 0.dp)
        DockZone.TOP -> DpOffset(0.dp, 0.dp)
        DockZone.BOTTOM -> DpOffset(0.dp, (containerBounds.height * 0.7f).dp)
        else -> DpOffset.Zero
    }

    /**
     * Calculate docked size for a zone
     */
    private fun calculateDockedSize(zone: DockZone): Pair<Dp, Dp> = when (zone) {
        DockZone.LEFT, DockZone.RIGHT -> (containerBounds.width * 0.3f).dp to containerBounds.height.dp
        DockZone.TOP, DockZone.BOTTOM -> containerBounds.width.dp to (containerBounds.height * 0.3f).dp
        else -> 400.dp to 300.dp
    }

    /**
     * Update dock zones based on container bounds
     */
    private fun updateDockZones() {
        val threshold = dockThreshold.value

        _dockZones.value =
            mapOf(
                DockZone.LEFT to Rect(0f, 0f, threshold, containerBounds.height),
                DockZone.RIGHT to Rect(containerBounds.width - threshold, 0f, containerBounds.width, containerBounds.height),
                DockZone.TOP to Rect(0f, 0f, containerBounds.width, threshold),
                DockZone.BOTTOM to Rect(0f, containerBounds.height - threshold, containerBounds.width, containerBounds.height),
                DockZone.CENTER to Rect(threshold, threshold, containerBounds.width - threshold, containerBounds.height - threshold),
            )
    }

    /**
     * Handle tab group docking logic
     */
    private fun handleTabGroupDocking(panelId: String, position: Offset) {
        // Find nearby panels to create a tab group with
        val nearbyPanels = findNearbyPanels(position, 100f)

        if (nearbyPanels.isNotEmpty()) {
            val tabGroupPosition = DpOffset(position.x.dp, position.y.dp)
            val tabGroupSize = 500.dp to 400.dp
            createTabGroup(listOf(panelId) + nearbyPanels, tabGroupPosition, tabGroupSize)
        } else {
            makeFloating(panelId, position)
        }
    }

    /**
     * Find panels near a given position
     */
    private fun findNearbyPanels(position: Offset, radius: Float): List<String> = _panels.value.values
        .filter { panel ->
            val panelCenter =
                Offset(
                    panel.position.x.value + panel.size.first.value / 2,
                    panel.position.y.value + panel.size.second.value / 2,
                )

            val distance = (position - panelCenter).getDistance()
            distance <= radius && panel.state == PanelDockState.FLOATING
        }.map { it.id }
}

/**
 * Data class representing the current drag state
 */
data class DragState(
    val panelId: String,
    val startPosition: Offset,
    val currentPosition: Offset,
    val isDragging: Boolean,
    val hoveredZone: DockZone = DockZone.NONE,
)

/**
 * Composable function to remember a DockingManager instance
 */
@Composable
fun rememberDockingManager(): DockingManager = remember { DockingManager() }
