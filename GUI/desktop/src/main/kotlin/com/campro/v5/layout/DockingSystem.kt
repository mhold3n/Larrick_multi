package com.campro.v5.layout

import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.layout.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.ArrowForward
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.DragHandle
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.unit.dp
import com.google.gson.Gson
import java.io.File

/**
 * Manages a dockable panel system for the CamPro v5 application.
 * This class provides drag-and-drop panel rearrangement, panel resizing and collapsing,
 * and workspace saving and loading functionality.
 */
class DockingSystem {
    // Panel definitions
    private val _panels = mutableStateOf<List<DockablePanel>>(emptyList())

    // Layout configuration
    private val _layout = mutableStateOf<DockingLayout>(DockingLayout.Grid)

    // Panel states
    private val _panelStates = mutableStateMapOf<String, PanelState>()

    // Drag state
    private val _dragState = mutableStateOf<DragState?>(null)

    // Getters for current values
    val panels: List<DockablePanel>
        get() = _panels.value

    val layout: DockingLayout
        get() = _layout.value

    val panelStates: Map<String, PanelState>
        get() = _panelStates.toMap()

    val dragState: DragState?
        get() = _dragState.value

    /**
     * Add a panel to the docking system.
     *
     * @param panel The panel to add
     */
    fun addPanel(panel: DockablePanel) {
        // Add panel to list
        _panels.value = _panels.value + panel

        // Initialize panel state
        _panelStates[panel.id] =
            PanelState(
                isVisible = true,
                isCollapsed = false,
                position = panel.defaultPosition,
                size = panel.defaultSize,
                zIndex = _panels.value.size,
            )
    }

    /**
     * Remove a panel from the docking system.
     *
     * @param panelId The ID of the panel to remove
     * @return True if the panel was removed, false if it wasn't found
     */
    fun removePanel(panelId: String): Boolean {
        val panelIndex = _panels.value.indexOfFirst { it.id == panelId }
        if (panelIndex == -1) {
            return false
        }

        // Remove panel from list
        _panels.value = _panels.value.filterNot { it.id == panelId }

        // Remove panel state
        _panelStates.remove(panelId)

        return true
    }

    /**
     * Set the visibility of a panel.
     *
     * @param panelId The ID of the panel
     * @param isVisible Whether the panel should be visible
     */
    fun setPanelVisibility(panelId: String, isVisible: Boolean) {
        _panelStates[panelId]?.let { state ->
            _panelStates[panelId] = state.copy(isVisible = isVisible)
        }
    }

    /**
     * Set the collapsed state of a panel.
     *
     * @param panelId The ID of the panel
     * @param isCollapsed Whether the panel should be collapsed
     */
    fun setPanelCollapsed(panelId: String, isCollapsed: Boolean) {
        _panelStates[panelId]?.let { state ->
            _panelStates[panelId] = state.copy(isCollapsed = isCollapsed)
        }
    }

    /**
     * Set the position of a panel.
     *
     * @param panelId The ID of the panel
     * @param position The new position of the panel
     */
    fun setPanelPosition(panelId: String, position: PanelPosition) {
        _panelStates[panelId]?.let { state ->
            _panelStates[panelId] = state.copy(position = position)
        }
    }

    /**
     * Set the size of a panel.
     *
     * @param panelId The ID of the panel
     * @param size The new size of the panel
     */
    fun setPanelSize(panelId: String, size: PanelSize) {
        _panelStates[panelId]?.let { state ->
            _panelStates[panelId] = state.copy(size = size)
        }
    }

    /**
     * Bring a panel to the front.
     *
     * @param panelId The ID of the panel
     */
    fun bringToFront(panelId: String) {
        // Get the highest z-index
        val maxZIndex = _panelStates.values.maxOfOrNull { it.zIndex } ?: 0

        // Set the panel's z-index to one higher
        _panelStates[panelId]?.let { state ->
            _panelStates[panelId] = state.copy(zIndex = maxZIndex + 1)
        }
    }

    /**
     * Start dragging a panel.
     *
     * @param panelId The ID of the panel
     * @param initialPosition The initial position of the drag
     */
    fun startDragging(panelId: String, initialPosition: Offset) {
        _dragState.value =
            DragState(
                panelId = panelId,
                initialPosition = initialPosition,
                currentPosition = initialPosition,
                isDragging = true,
            )

        // Bring the panel to the front
        bringToFront(panelId)
    }

    /**
     * Update the position of a dragged panel.
     *
     * @param currentPosition The current position of the drag
     */
    fun updateDragging(currentPosition: Offset) {
        _dragState.value?.let { state ->
            _dragState.value = state.copy(currentPosition = currentPosition)

            // Update the panel's position
            _panelStates[state.panelId]?.let { panelState ->
                val deltaX = currentPosition.x - state.initialPosition.x
                val deltaY = currentPosition.y - state.initialPosition.y

                val newPosition =
                    when (panelState.position) {
                        is PanelPosition.Absolute -> {
                            PanelPosition.Absolute(
                                x = panelState.position.x + deltaX,
                                y = panelState.position.y + deltaY,
                            )
                        }
                        else -> panelState.position // Other position types don't change during dragging
                    }

                _panelStates[state.panelId] = panelState.copy(position = newPosition)
            }
        }
    }

    /**
     * Stop dragging a panel.
     */
    fun stopDragging() {
        _dragState.value = null
    }

    /**
     * Set the layout of the docking system.
     *
     * @param layout The new layout
     */
    fun setLayout(layout: DockingLayout) {
        _layout.value = layout
    }

    /**
     * Save the current workspace configuration to a file.
     *
     * @param file The file to save to
     * @return True if the save was successful, false otherwise
     */
    fun saveWorkspace(file: File): Boolean = try {
        // Create a workspace configuration
        val config =
            WorkspaceConfig(
                layout = _layout.value,
                panelStates = _panelStates.toMap(),
            )

        // Convert to JSON
        val gson = Gson()
        val json = gson.toJson(config)

        // Write to file
        file.writeText(json)

        true
    } catch (e: Exception) {
        false
    }

    /**
     * Load a workspace configuration from a file.
     *
     * @param file The file to load from
     * @return True if the load was successful, false otherwise
     */
    fun loadWorkspace(file: File): Boolean = try {
        // Read JSON from file
        val json = file.readText()

        // Convert from JSON
        val gson = Gson()
        val config = gson.fromJson<WorkspaceConfig>(json, WorkspaceConfig::class.java)

        // Apply configuration
        _layout.value = config.layout

        // Apply panel states for existing panels only
        config.panelStates.forEach { (panelId, state) ->
            if (_panels.value.any { it.id == panelId }) {
                _panelStates[panelId] = state
            }
        }

        true
    } catch (e: Exception) {
        false
    }

    /**
     * Reset the workspace to the default configuration.
     */
    fun resetWorkspace() {
        // Reset layout
        _layout.value = DockingLayout.Grid

        // Reset panel states
        _panels.value.forEach { panel ->
            _panelStates[panel.id] =
                PanelState(
                    isVisible = true,
                    isCollapsed = false,
                    position = panel.defaultPosition,
                    size = panel.defaultSize,
                    zIndex = _panels.value.indexOf(panel),
                )
        }
    }

    companion object {
        // Singleton instance
        private var instance: DockingSystem? = null

        /**
         * Get the singleton instance of the DockingSystem.
         *
         * @return The DockingSystem instance
         */
        fun getInstance(): DockingSystem {
            if (instance == null) {
                instance = DockingSystem()
            }
            return instance!!
        }
    }
}

/**
 * A dockable panel in the docking system.
 *
 * @param id The unique ID of the panel
 * @param title The title of the panel
 * @param defaultPosition The default position of the panel
 * @param defaultSize The default size of the panel
 * @param content The content of the panel
 */
data class DockablePanel(
    val id: String,
    val title: String,
    val defaultPosition: PanelPosition,
    val defaultSize: PanelSize,
    val content: @Composable () -> Unit,
)

/**
 * The position of a panel.
 */
sealed class PanelPosition {
    /**
     * An absolute position in the docking area.
     *
     * @param x The x-coordinate
     * @param y The y-coordinate
     */
    data class Absolute(val x: Float, val y: Float) : PanelPosition()

    /**
     * A position in a grid layout.
     *
     * @param row The row index
     * @param column The column index
     */
    data class Grid(val row: Int, val column: Int) : PanelPosition()

    /**
     * A position in a dock layout.
     *
     * @param dock The dock location
     */
    data class Dock(val dock: DockLocation) : PanelPosition()

    /**
     * A position relative to another panel.
     *
     * @param relativeTo The ID of the panel to position relative to
     * @param location The location relative to the other panel
     */
    data class Relative(val relativeTo: String, val location: RelativeLocation) : PanelPosition()
}

/**
 * The size of a panel.
 *
 * @param width The width of the panel
 * @param height The height of the panel
 * @param minWidth The minimum width of the panel
 * @param minHeight The minimum height of the panel
 * @param maxWidth The maximum width of the panel
 * @param maxHeight The maximum height of the panel
 */
data class PanelSize(
    val width: Float,
    val height: Float,
    val minWidth: Float = 100f,
    val minHeight: Float = 100f,
    val maxWidth: Float = Float.MAX_VALUE,
    val maxHeight: Float = Float.MAX_VALUE,
)

/**
 * The state of a panel.
 *
 * @param isVisible Whether the panel is visible
 * @param isCollapsed Whether the panel is collapsed
 * @param position The position of the panel
 * @param size The size of the panel
 * @param zIndex The z-index of the panel
 */
data class PanelState(val isVisible: Boolean, val isCollapsed: Boolean, val position: PanelPosition, val size: PanelSize, val zIndex: Int)

/**
 * The state of a drag operation.
 *
 * @param panelId The ID of the panel being dragged
 * @param initialPosition The initial position of the drag
 * @param currentPosition The current position of the drag
 * @param isDragging Whether the drag is in progress
 */
data class DragState(val panelId: String, val initialPosition: Offset, val currentPosition: Offset, val isDragging: Boolean)

/**
 * The layout of the docking system.
 */
sealed class DockingLayout {
    /**
     * A grid layout with rows and columns.
     */
    object Grid : DockingLayout()

    /**
     * A dock layout with panels docked to the edges.
     */
    object Dock : DockingLayout()

    /**
     * A free layout with panels positioned absolutely.
     */
    object Free : DockingLayout()
}

/**
 * The location of a dock.
 */
enum class DockLocation {
    TOP,
    RIGHT,
    BOTTOM,
    LEFT,
    CENTER,
}

/**
 * The location relative to another panel.
 */
enum class RelativeLocation {
    ABOVE,
    BELOW,
    LEFT,
    RIGHT,
    TABBED,
}

/**
 * A workspace configuration.
 *
 * @param layout The layout of the docking system
 * @param panelStates The states of the panels
 */
data class WorkspaceConfig(val layout: DockingLayout, val panelStates: Map<String, PanelState>)

/**
 * Composable function to remember a DockingSystem instance.
 *
 * @return The remembered DockingSystem instance
 */
@Composable
fun rememberDockingSystem(): DockingSystem = remember { DockingSystem.getInstance() }

/**
 * Composable function to create a dockable panel.
 *
 * @param panel The panel definition
 * @param dockingSystem The DockingSystem instance
 * @param modifier The modifier for the panel
 */
@Composable
fun DockablePanel(panel: DockablePanel, dockingSystem: DockingSystem = rememberDockingSystem(), modifier: Modifier = Modifier) {
    val panelState = dockingSystem.panelStates[panel.id]

    if (panelState == null || !panelState.isVisible) {
        return
    }

    val isCollapsed = panelState.isCollapsed

    Card(
        modifier =
        modifier
            .shadow(4.dp)
            .then(
                when (panelState.position) {
                    is PanelPosition.Absolute -> {
                        Modifier.offset(
                            x = panelState.position.x.dp,
                            y = panelState.position.y.dp,
                        )
                    }
                    else -> Modifier
                },
            ).width(if (isCollapsed) 200.dp else panelState.size.width.dp)
            .height(if (isCollapsed) 40.dp else panelState.size.height.dp)
            .pointerInput(panel.id) {
                detectDragGestures(
                    onDragStart = { offset ->
                        dockingSystem.startDragging(panel.id, offset)
                    },
                    onDrag = { _, dragAmount ->
                        dockingSystem.updateDragging(dragAmount)
                    },
                    onDragEnd = {
                        dockingSystem.stopDragging()
                    },
                    onDragCancel = {
                        dockingSystem.stopDragging()
                    },
                )
            },
    ) {
        Column(
            modifier = Modifier.fillMaxSize(),
        ) {
            // Panel header
            Row(
                modifier =
                Modifier
                    .fillMaxWidth()
                    .background(MaterialTheme.colorScheme.primaryContainer)
                    .padding(8.dp),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                // Drag handle
                Icon(
                    imageVector = Icons.Default.DragHandle,
                    contentDescription = "Drag",
                    tint = MaterialTheme.colorScheme.onPrimaryContainer,
                )

                // Title
                Text(
                    text = panel.title,
                    style = MaterialTheme.typography.titleSmall,
                    color = MaterialTheme.colorScheme.onPrimaryContainer,
                )

                // Controls
                Row {
                    // Collapse/expand button
                    IconButton(
                        onClick = {
                            dockingSystem.setPanelCollapsed(panel.id, !isCollapsed)
                        },
                        modifier = Modifier.size(24.dp),
                    ) {
                        Icon(
                            imageVector = if (isCollapsed) Icons.Default.ArrowForward else Icons.Default.ArrowBack,
                            contentDescription = if (isCollapsed) "Expand" else "Collapse",
                            tint = MaterialTheme.colorScheme.onPrimaryContainer,
                        )
                    }

                    // Close button
                    IconButton(
                        onClick = {
                            dockingSystem.setPanelVisibility(panel.id, false)
                        },
                        modifier = Modifier.size(24.dp),
                    ) {
                        Icon(
                            imageVector = Icons.Default.Close,
                            contentDescription = "Close",
                            tint = MaterialTheme.colorScheme.onPrimaryContainer,
                        )
                    }
                }
            }

            // Panel content
            if (!isCollapsed) {
                Box(
                    modifier =
                    Modifier
                        .fillMaxSize()
                        .padding(8.dp),
                ) {
                    panel.content()
                }
            }
        }
    }
}

/**
 * Composable function to create a docking area.
 *
 * @param dockingSystem The DockingSystem instance
 * @param modifier The modifier for the docking area
 */
@Composable
fun DockingArea(dockingSystem: DockingSystem = rememberDockingSystem(), modifier: Modifier = Modifier) {
    Box(
        modifier =
        modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background),
    ) {
        when (dockingSystem.layout) {
            is DockingLayout.Grid -> {
                // Grid layout implementation
                // This would arrange panels in a grid based on their grid positions
            }
            is DockingLayout.Dock -> {
                // Dock layout implementation
                // This would arrange panels docked to the edges of the screen
            }
            is DockingLayout.Free -> {
                // Free layout implementation
                // This would display panels at their absolute positions
                dockingSystem.panels
                    .sortedBy { dockingSystem.panelStates[it.id]?.zIndex ?: 0 }
                    .forEach { panel ->
                        DockablePanel(
                            panel = panel,
                            dockingSystem = dockingSystem,
                        )
                    }
            }
        }
    }
}
