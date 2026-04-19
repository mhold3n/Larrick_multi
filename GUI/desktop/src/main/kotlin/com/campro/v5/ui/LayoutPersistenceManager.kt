package com.campro.v5.ui

import androidx.compose.runtime.*
import androidx.compose.ui.unit.DpOffset
import androidx.compose.ui.unit.dp
import com.google.gson.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileReader
import java.io.FileWriter
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter
import java.util.*

/**
 * Layout template types for different workflows
 */
enum class LayoutTemplate {
    DEVELOPMENT, // Larger parameter panel, smaller animation panel
    ANALYSIS, // Larger plot and data panels, smaller parameter panel
    PRESENTATION, // Larger animation panel, minimal other panels
    CUSTOM, // User-defined layout
}

/**
 * Serializable layout data classes
 */
data class SerializableLayout(
    val id: String,
    val name: String,
    val description: String,
    val template: LayoutTemplate,
    val createdAt: String,
    val modifiedAt: String,
    val panels: List<SerializablePanelState>,
    val tabGroups: List<SerializableTabGroup>,
    val containerBounds: SerializableRect,
    val version: String = "1.0",
)

data class SerializablePanelState(
    val id: String,
    val title: String,
    val state: String, // PanelDockState as string
    val x: Float,
    val y: Float,
    val width: Float,
    val height: Float,
    val minWidth: Float,
    val minHeight: Float,
    val maxWidth: Float,
    val maxHeight: Float,
    val dockZone: String, // DockZone as string
    val tabGroupId: String?,
    val isMinimized: Boolean,
    val zIndex: Float,
)

data class SerializableTabGroup(
    val id: String,
    val title: String,
    val panelIds: List<String>,
    val activePanel: String,
    val x: Float,
    val y: Float,
    val width: Float,
    val height: Float,
)

data class SerializableRect(val left: Float, val top: Float, val right: Float, val bottom: Float)

/**
 * Layout history entry for undo/redo functionality
 */
data class LayoutHistoryEntry(val id: String, val timestamp: LocalDateTime, val action: String, val layout: SerializableLayout)

/**
 * Layout Persistence Manager for saving, loading, and managing panel layouts
 */
class LayoutPersistenceManager {
    private val gson =
        GsonBuilder()
            .setPrettyPrinting()
            .create()

    private val _savedLayouts = MutableStateFlow<List<SerializableLayout>>(emptyList())
    val savedLayouts: StateFlow<List<SerializableLayout>> = _savedLayouts.asStateFlow()

    private val _layoutHistory = MutableStateFlow<List<LayoutHistoryEntry>>(emptyList())
    val layoutHistory: StateFlow<List<LayoutHistoryEntry>> = _layoutHistory.asStateFlow()

    private val _currentHistoryIndex = MutableStateFlow(-1)
    val currentHistoryIndex: StateFlow<Int> = _currentHistoryIndex.asStateFlow()

    private val maxHistorySize = 50
    private val layoutsDirectory = File("layouts")
    private val templatesDirectory = File("layouts/templates")

    init {
        // Create directories if they don't exist
        layoutsDirectory.mkdirs()
        templatesDirectory.mkdirs()

        // Load existing layouts
        loadAllLayouts()

        // Create default templates if they don't exist
        createDefaultTemplates()
    }

    /**
     * Save current layout state
     */
    suspend fun saveLayout(
        name: String,
        description: String,
        template: LayoutTemplate,
        dockingManager: DockingManager,
        containerBounds: androidx.compose.ui.geometry.Rect,
    ): Result<String> = withContext(Dispatchers.IO) {
        try {
            val layoutId = UUID.randomUUID().toString()
            val timestamp = LocalDateTime.now().format(DateTimeFormatter.ISO_LOCAL_DATE_TIME)

            val panels =
                dockingManager.panels.value.values.map { panel ->
                    SerializablePanelState(
                        id = panel.id,
                        title = panel.title,
                        state = panel.state.name,
                        x = panel.position.x.value,
                        y = panel.position.y.value,
                        width = panel.size.first.value,
                        height = panel.size.second.value,
                        minWidth = 200f, // Default minimum width
                        minHeight = 150f, // Default minimum height
                        maxWidth = 800f, // Default maximum width
                        maxHeight = 600f, // Default maximum height
                        dockZone = panel.dockZone.name,
                        tabGroupId = panel.tabGroupId,
                        isMinimized = panel.isMinimized,
                        zIndex = panel.zIndex,
                    )
                }

            val tabGroups =
                dockingManager.tabGroups.value.values.map { tabGroup ->
                    SerializableTabGroup(
                        id = tabGroup.id,
                        title = tabGroup.title,
                        panelIds = tabGroup.panelIds,
                        activePanel = tabGroup.activePanel,
                        x = tabGroup.position.x.value,
                        y = tabGroup.position.y.value,
                        width = tabGroup.size.first.value,
                        height = tabGroup.size.second.value,
                    )
                }

            val layout =
                SerializableLayout(
                    id = layoutId,
                    name = name,
                    description = description,
                    template = template,
                    createdAt = timestamp,
                    modifiedAt = timestamp,
                    panels = panels,
                    tabGroups = tabGroups,
                    containerBounds =
                    SerializableRect(
                        left = containerBounds.left,
                        top = containerBounds.top,
                        right = containerBounds.right,
                        bottom = containerBounds.bottom,
                    ),
                )

            // Save to file
            val filename = "${name.replace(" ", "_").lowercase()}_$layoutId.json"
            val file = File(layoutsDirectory, filename)
            FileWriter(file).use { writer ->
                gson.toJson(layout, writer)
            }

            // Update in-memory list
            _savedLayouts.value = _savedLayouts.value + layout

            // Add to history
            addToHistory("SAVE", layout)

            Result.success(layoutId)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /**
     * Load a layout and apply it to the docking manager
     */
    suspend fun loadLayout(layoutId: String, dockingManager: DockingManager): Result<SerializableLayout> = withContext(Dispatchers.IO) {
        try {
            val layout =
                _savedLayouts.value.find { it.id == layoutId }
                    ?: return@withContext Result.failure(Exception("Layout not found: $layoutId"))

            // Clear existing state
            dockingManager.panels.value.keys.forEach { panelId ->
                dockingManager.unregisterPanel(panelId)
            }

            // Restore panels
            layout.panels.forEach { panelData ->
                val panelState =
                    PanelState(
                        x = panelData.x.dp,
                        y = panelData.y.dp,
                        width = panelData.width.dp,
                        height = panelData.height.dp,
                        minWidth = panelData.minWidth.dp,
                        minHeight = panelData.minHeight.dp,
                        maxWidth = panelData.maxWidth.dp,
                        maxHeight = panelData.maxHeight.dp,
                    )

                dockingManager.registerPanel(
                    id = panelData.id,
                    title = panelData.title,
                    initialPosition = DpOffset(panelData.x.dp, panelData.y.dp),
                    initialSize = panelData.width.dp to panelData.height.dp,
                    initialState = PanelDockState.valueOf(panelData.state),
                )

                // Apply minimization state
                if (panelData.isMinimized) {
                    dockingManager.minimizePanel(panelData.id)
                }
            }

            // Restore tab groups
            layout.tabGroups.forEach { tabGroupData ->
                dockingManager.createTabGroup(
                    panelIds = tabGroupData.panelIds,
                    position = DpOffset(tabGroupData.x.dp, tabGroupData.y.dp),
                    size = tabGroupData.width.dp to tabGroupData.height.dp,
                )
            }

            // Update container bounds
            val containerBounds =
                androidx.compose.ui.geometry.Rect(
                    left = layout.containerBounds.left,
                    top = layout.containerBounds.top,
                    right = layout.containerBounds.right,
                    bottom = layout.containerBounds.bottom,
                )
            dockingManager.updateContainerBounds(containerBounds)

            // Add to history
            addToHistory("LOAD", layout)

            Result.success(layout)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /**
     * Delete a saved layout
     */
    suspend fun deleteLayout(layoutId: String): Result<Unit> = withContext(Dispatchers.IO) {
        try {
            val layout =
                _savedLayouts.value.find { it.id == layoutId }
                    ?: return@withContext Result.failure(Exception("Layout not found: $layoutId"))

            // Find and delete the file
            val filename = "${layout.name.replace(" ", "_").lowercase()}_$layoutId.json"
            val file = File(layoutsDirectory, filename)
            if (file.exists()) {
                file.delete()
            }

            // Update in-memory list
            _savedLayouts.value = _savedLayouts.value.filter { it.id != layoutId }

            // Add to history
            addToHistory("DELETE", layout)

            Result.success(Unit)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /**
     * Create layout templates for different workflows
     */
    private fun createDefaultTemplates() {
        val templates =
            mapOf(
                LayoutTemplate.DEVELOPMENT to createDevelopmentTemplate(),
                LayoutTemplate.ANALYSIS to createAnalysisTemplate(),
                LayoutTemplate.PRESENTATION to createPresentationTemplate(),
            )

        templates.forEach { (template, layout) ->
            val filename = "${template.name.lowercase()}_template.json"
            val file = File(templatesDirectory, filename)

            if (!file.exists()) {
                try {
                    FileWriter(file).use { writer ->
                        gson.toJson(layout, writer)
                    }
                } catch (e: Exception) {
                    println("Failed to create template ${template.name}: ${e.message}")
                }
            }
        }
    }

    /**
     * Create development layout template
     */
    private fun createDevelopmentTemplate(): SerializableLayout = SerializableLayout(
        id = "dev_template",
        name = "Development Layout",
        description = "Optimized for development with larger parameter panel",
        template = LayoutTemplate.DEVELOPMENT,
        createdAt = LocalDateTime.now().format(DateTimeFormatter.ISO_LOCAL_DATE_TIME),
        modifiedAt = LocalDateTime.now().format(DateTimeFormatter.ISO_LOCAL_DATE_TIME),
        panels =
        listOf(
            SerializablePanelState(
                id = "parameter_panel",
                title = "Parameters",
                state = "DOCKED",
                x = 0f,
                y = 0f,
                width = 600f,
                height = 500f,
                minWidth = 400f,
                minHeight = 300f,
                maxWidth = 800f,
                maxHeight = 700f,
                dockZone = "LEFT",
                tabGroupId = null,
                isMinimized = false,
                zIndex = 0f,
            ),
            SerializablePanelState(
                id = "animation_panel",
                title = "Animation",
                state = "DOCKED",
                x = 620f,
                y = 0f,
                width = 400f,
                height = 300f,
                minWidth = 300f,
                minHeight = 200f,
                maxWidth = 600f,
                maxHeight = 500f,
                dockZone = "RIGHT",
                tabGroupId = null,
                isMinimized = false,
                zIndex = 0f,
            ),
        ),
        tabGroups = emptyList(),
        containerBounds = SerializableRect(0f, 0f, 1200f, 800f),
    )

    /**
     * Create analysis layout template
     */
    private fun createAnalysisTemplate(): SerializableLayout = SerializableLayout(
        id = "analysis_template",
        name = "Analysis Layout",
        description = "Optimized for data analysis with larger plot and data panels",
        template = LayoutTemplate.ANALYSIS,
        createdAt = LocalDateTime.now().format(DateTimeFormatter.ISO_LOCAL_DATE_TIME),
        modifiedAt = LocalDateTime.now().format(DateTimeFormatter.ISO_LOCAL_DATE_TIME),
        panels =
        listOf(
            SerializablePanelState(
                id = "parameter_panel",
                title = "Parameters",
                state = "DOCKED",
                x = 0f,
                y = 0f,
                width = 300f,
                height = 400f,
                minWidth = 250f,
                minHeight = 300f,
                maxWidth = 500f,
                maxHeight = 600f,
                dockZone = "LEFT",
                tabGroupId = null,
                isMinimized = false,
                zIndex = 0f,
            ),
            SerializablePanelState(
                id = "plot_panel",
                title = "Plots",
                state = "DOCKED",
                x = 320f,
                y = 0f,
                width = 600f,
                height = 400f,
                minWidth = 400f,
                minHeight = 300f,
                maxWidth = 800f,
                maxHeight = 600f,
                dockZone = "CENTER",
                tabGroupId = null,
                isMinimized = false,
                zIndex = 0f,
            ),
            SerializablePanelState(
                id = "data_panel",
                title = "Data Display",
                state = "DOCKED",
                x = 320f,
                y = 420f,
                width = 600f,
                height = 300f,
                minWidth = 400f,
                minHeight = 200f,
                maxWidth = 800f,
                maxHeight = 500f,
                dockZone = "BOTTOM",
                tabGroupId = null,
                isMinimized = false,
                zIndex = 0f,
            ),
        ),
        tabGroups = emptyList(),
        containerBounds = SerializableRect(0f, 0f, 1200f, 800f),
    )

    /**
     * Create presentation layout template
     */
    private fun createPresentationTemplate(): SerializableLayout = SerializableLayout(
        id = "presentation_template",
        name = "Presentation Layout",
        description = "Optimized for presentations with larger animation panel",
        template = LayoutTemplate.PRESENTATION,
        createdAt = LocalDateTime.now().format(DateTimeFormatter.ISO_LOCAL_DATE_TIME),
        modifiedAt = LocalDateTime.now().format(DateTimeFormatter.ISO_LOCAL_DATE_TIME),
        panels =
        listOf(
            SerializablePanelState(
                id = "animation_panel",
                title = "Animation",
                state = "DOCKED",
                x = 0f,
                y = 0f,
                width = 800f,
                height = 600f,
                minWidth = 600f,
                minHeight = 400f,
                maxWidth = 1000f,
                maxHeight = 800f,
                dockZone = "CENTER",
                tabGroupId = null,
                isMinimized = false,
                zIndex = 0f,
            ),
            SerializablePanelState(
                id = "parameter_panel",
                title = "Parameters",
                state = "MINIMIZED",
                x = 820f,
                y = 0f,
                width = 300f,
                height = 200f,
                minWidth = 250f,
                minHeight = 150f,
                maxWidth = 400f,
                maxHeight = 400f,
                dockZone = "RIGHT",
                tabGroupId = null,
                isMinimized = true,
                zIndex = 0f,
            ),
        ),
        tabGroups = emptyList(),
        containerBounds = SerializableRect(0f, 0f, 1200f, 800f),
    )

    /**
     * Load all saved layouts from disk
     */
    private fun loadAllLayouts() {
        try {
            val layoutFiles =
                layoutsDirectory.listFiles { file ->
                    file.extension == "json" && !file.path.contains("templates")
                } ?: return

            val layouts =
                layoutFiles.mapNotNull { file ->
                    try {
                        FileReader(file).use { reader ->
                            gson.fromJson(reader, SerializableLayout::class.java)
                        }
                    } catch (e: Exception) {
                        println("Failed to load layout from ${file.name}: ${e.message}")
                        null
                    }
                }

            _savedLayouts.value = layouts
        } catch (e: Exception) {
            println("Failed to load layouts: ${e.message}")
        }
    }

    /**
     * Load a template layout
     */
    suspend fun loadTemplate(template: LayoutTemplate): Result<SerializableLayout> = withContext(Dispatchers.IO) {
        try {
            val filename = "${template.name.lowercase()}_template.json"
            val file = File(templatesDirectory, filename)

            if (!file.exists()) {
                return@withContext Result.failure(Exception("Template not found: ${template.name}"))
            }

            val layout =
                FileReader(file).use { reader ->
                    gson.fromJson(reader, SerializableLayout::class.java)
                }

            Result.success(layout)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /**
     * Add entry to layout history for undo/redo functionality
     */
    private fun addToHistory(action: String, layout: SerializableLayout) {
        val entry =
            LayoutHistoryEntry(
                id = UUID.randomUUID().toString(),
                timestamp = LocalDateTime.now(),
                action = action,
                layout = layout,
            )

        val currentHistory = _layoutHistory.value.toMutableList()
        val currentIndex = _currentHistoryIndex.value

        // Remove any entries after current index (when adding new entry after undo)
        if (currentIndex < currentHistory.size - 1) {
            currentHistory.subList(currentIndex + 1, currentHistory.size).clear()
        }

        // Add new entry
        currentHistory.add(entry)

        // Limit history size
        if (currentHistory.size > maxHistorySize) {
            currentHistory.removeAt(0)
        }

        _layoutHistory.value = currentHistory
        _currentHistoryIndex.value = currentHistory.size - 1
    }

    /**
     * Undo last layout change
     */
    suspend fun undo(dockingManager: DockingManager): Result<SerializableLayout?> {
        val currentIndex = _currentHistoryIndex.value
        val history = _layoutHistory.value

        if (currentIndex > 0) {
            val previousEntry = history[currentIndex - 1]
            _currentHistoryIndex.value = currentIndex - 1

            return loadLayout(previousEntry.layout.id, dockingManager)
                .map { previousEntry.layout }
        }

        return Result.success(null)
    }

    /**
     * Redo last undone layout change
     */
    suspend fun redo(dockingManager: DockingManager): Result<SerializableLayout?> {
        val currentIndex = _currentHistoryIndex.value
        val history = _layoutHistory.value

        if (currentIndex < history.size - 1) {
            val nextEntry = history[currentIndex + 1]
            _currentHistoryIndex.value = currentIndex + 1

            return loadLayout(nextEntry.layout.id, dockingManager)
                .map { nextEntry.layout }
        }

        return Result.success(null)
    }

    /**
     * Export layout to a specific file
     */
    suspend fun exportLayout(layoutId: String, exportFile: File): Result<Unit> = withContext(Dispatchers.IO) {
        try {
            val layout =
                _savedLayouts.value.find { it.id == layoutId }
                    ?: return@withContext Result.failure(Exception("Layout not found: $layoutId"))

            FileWriter(exportFile).use { writer ->
                gson.toJson(layout, writer)
            }

            Result.success(Unit)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /**
     * Import layout from a file
     */
    suspend fun importLayout(importFile: File): Result<String> = withContext(Dispatchers.IO) {
        try {
            val layout =
                FileReader(importFile).use { reader ->
                    gson.fromJson(reader, SerializableLayout::class.java)
                }

            // Generate new ID to avoid conflicts
            val newId = UUID.randomUUID().toString()
            val importedLayout =
                layout.copy(
                    id = newId,
                    name = "${layout.name} (Imported)",
                    modifiedAt = LocalDateTime.now().format(DateTimeFormatter.ISO_LOCAL_DATE_TIME),
                )

            // Save to layouts directory
            val filename = "${importedLayout.name.replace(" ", "_").lowercase()}_$newId.json"
            val file = File(layoutsDirectory, filename)
            FileWriter(file).use { writer ->
                gson.toJson(importedLayout, writer)
            }

            // Update in-memory list
            _savedLayouts.value = _savedLayouts.value + importedLayout

            // Add to history
            addToHistory("IMPORT", importedLayout)

            Result.success(newId)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }
}

/**
 * Composable function to remember a LayoutPersistenceManager instance
 */
@Composable
fun rememberLayoutPersistenceManager(): LayoutPersistenceManager = remember { LayoutPersistenceManager() }
