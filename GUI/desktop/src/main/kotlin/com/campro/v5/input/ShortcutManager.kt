package com.campro.v5.input

import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.input.key.*
import com.campro.v5.layout.StateManager
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.util.concurrent.ConcurrentHashMap

/**
 * Manages keyboard shortcuts for the CamPro v5 application.
 * This class provides default shortcuts for common operations,
 * shortcut customization, and shortcut discovery.
 */
class ShortcutManager {
    // Shortcut definitions
    private val shortcuts = ConcurrentHashMap<String, Shortcut>()

    // Shortcut categories
    private val categories = ConcurrentHashMap<String, MutableList<String>>()

    // Shortcut change events
    private val _shortcutEvents = MutableStateFlow<ShortcutEvent?>(null)
    val shortcutEvents: StateFlow<ShortcutEvent?> = _shortcutEvents.asStateFlow()

    // State manager for persistence
    private val stateManager = StateManager.getInstance()

    init {
        // Register default shortcuts
        registerDefaultShortcuts()

        // Load custom shortcuts from state
        loadCustomShortcuts()
    }

    /**
     * Register default shortcuts.
     */
    private fun registerDefaultShortcuts() {
        // File operations
        registerShortcut(
            Shortcut(
                id = "file.new",
                name = "New Project",
                description = "Create a new project",
                keyStrokes = listOf(KeyStroke(Key.N, ctrl = true)),
                category = "File",
            ),
        )

        registerShortcut(
            Shortcut(
                id = "file.open",
                name = "Open Project",
                description = "Open an existing project",
                keyStrokes = listOf(KeyStroke(Key.O, ctrl = true)),
                category = "File",
            ),
        )

        registerShortcut(
            Shortcut(
                id = "file.save",
                name = "Save Project",
                description = "Save the current project",
                keyStrokes = listOf(KeyStroke(Key.S, ctrl = true)),
                category = "File",
            ),
        )

        registerShortcut(
            Shortcut(
                id = "file.saveAs",
                name = "Save Project As",
                description = "Save the current project with a new name",
                keyStrokes = listOf(KeyStroke(Key.S, ctrl = true, shift = true)),
                category = "File",
            ),
        )

        registerShortcut(
            Shortcut(
                id = "file.export",
                name = "Export",
                description = "Export the current project",
                keyStrokes = listOf(KeyStroke(Key.E, ctrl = true)),
                category = "File",
            ),
        )

        // Edit operations
        registerShortcut(
            Shortcut(
                id = "edit.undo",
                name = "Undo",
                description = "Undo the last action",
                keyStrokes = listOf(KeyStroke(Key.Z, ctrl = true)),
                category = "Edit",
            ),
        )

        registerShortcut(
            Shortcut(
                id = "edit.redo",
                name = "Redo",
                description = "Redo the last undone action",
                keyStrokes =
                listOf(
                    KeyStroke(Key.Y, ctrl = true),
                    KeyStroke(Key.Z, ctrl = true, shift = true),
                ),
                category = "Edit",
            ),
        )

        registerShortcut(
            Shortcut(
                id = "edit.cut",
                name = "Cut",
                description = "Cut the selected content",
                keyStrokes = listOf(KeyStroke(Key.X, ctrl = true)),
                category = "Edit",
            ),
        )

        registerShortcut(
            Shortcut(
                id = "edit.copy",
                name = "Copy",
                description = "Copy the selected content",
                keyStrokes = listOf(KeyStroke(Key.C, ctrl = true)),
                category = "Edit",
            ),
        )

        registerShortcut(
            Shortcut(
                id = "edit.paste",
                name = "Paste",
                description = "Paste the copied content",
                keyStrokes = listOf(KeyStroke(Key.V, ctrl = true)),
                category = "Edit",
            ),
        )

        // View operations
        registerShortcut(
            Shortcut(
                id = "view.zoomIn",
                name = "Zoom In",
                description = "Zoom in on the current view",
                keyStrokes =
                listOf(
                    KeyStroke(Key.Plus, ctrl = true),
                    KeyStroke(Key.Equals, ctrl = true),
                ),
                category = "View",
            ),
        )

        registerShortcut(
            Shortcut(
                id = "view.zoomOut",
                name = "Zoom Out",
                description = "Zoom out on the current view",
                keyStrokes = listOf(KeyStroke(Key.Minus, ctrl = true)),
                category = "View",
            ),
        )

        registerShortcut(
            Shortcut(
                id = "view.resetZoom",
                name = "Reset Zoom",
                description = "Reset the zoom level",
                keyStrokes = listOf(KeyStroke(Key.Zero, ctrl = true)),
                category = "View",
            ),
        )

        registerShortcut(
            Shortcut(
                id = "view.toggleFullscreen",
                name = "Toggle Fullscreen",
                description = "Toggle fullscreen mode",
                keyStrokes = listOf(KeyStroke(Key.F11)),
                category = "View",
            ),
        )

        // Navigation operations
        registerShortcut(
            Shortcut(
                id = "navigation.home",
                name = "Go to Home",
                description = "Navigate to the home screen",
                keyStrokes = listOf(KeyStroke(Key.H, ctrl = true)),
                category = "Navigation",
            ),
        )

        registerShortcut(
            Shortcut(
                id = "navigation.parameters",
                name = "Go to Parameters",
                description = "Navigate to the parameters screen",
                keyStrokes = listOf(KeyStroke(Key.P, ctrl = true)),
                category = "Navigation",
            ),
        )

        registerShortcut(
            Shortcut(
                id = "navigation.animation",
                name = "Go to Animation",
                description = "Navigate to the animation screen",
                keyStrokes = listOf(KeyStroke(Key.A, ctrl = true)),
                category = "Navigation",
            ),
        )

        registerShortcut(
            Shortcut(
                id = "navigation.plots",
                name = "Go to Plots",
                description = "Navigate to the plots screen",
                keyStrokes = listOf(KeyStroke(Key.L, ctrl = true)),
                category = "Navigation",
            ),
        )

        registerShortcut(
            Shortcut(
                id = "navigation.data",
                name = "Go to Data",
                description = "Navigate to the data screen",
                keyStrokes = listOf(KeyStroke(Key.D, ctrl = true)),
                category = "Navigation",
            ),
        )

        registerShortcut(
            Shortcut(
                id = "navigation.settings",
                name = "Go to Settings",
                description = "Navigate to the settings screen",
                keyStrokes = listOf(KeyStroke(Key.S, ctrl = true, alt = true)),
                category = "Navigation",
            ),
        )

        registerShortcut(
            Shortcut(
                id = "navigation.help",
                name = "Go to Help",
                description = "Navigate to the help screen",
                keyStrokes = listOf(KeyStroke(Key.F1)),
                category = "Navigation",
            ),
        )

        // Analysis operations
        registerShortcut(
            Shortcut(
                id = "analysis.run",
                name = "Run Analysis",
                description = "Run the current analysis",
                keyStrokes = listOf(KeyStroke(Key.R, ctrl = true)),
                category = "Analysis",
            ),
        )

        registerShortcut(
            Shortcut(
                id = "analysis.stop",
                name = "Stop Analysis",
                description = "Stop the current analysis",
                keyStrokes = listOf(KeyStroke(Key.Escape)),
                category = "Analysis",
            ),
        )

        // Help operations
        registerShortcut(
            Shortcut(
                id = "help.showShortcuts",
                name = "Show Shortcuts",
                description = "Show the keyboard shortcuts",
                keyStrokes = listOf(KeyStroke(Key.F1, shift = true)),
                category = "Help",
            ),
        )

        registerShortcut(
            Shortcut(
                id = "help.showContextHelp",
                name = "Show Context Help",
                description = "Show context-sensitive help",
                keyStrokes = listOf(KeyStroke(Key.F1, ctrl = true)),
                category = "Help",
            ),
        )
    }

    /**
     * Load custom shortcuts from state.
     */
    private fun loadCustomShortcuts() {
        val customShortcutsJson = stateManager.getState("shortcuts.custom", "")
        if (customShortcutsJson.isNotEmpty()) {
            try {
                val customShortcuts = parseShortcutsJson(customShortcutsJson)
                customShortcuts.forEach { shortcut ->
                    // Only override existing shortcuts
                    if (shortcuts.containsKey(shortcut.id)) {
                        shortcuts[shortcut.id] = shortcut
                    }
                }
            } catch (e: Exception) {
                // Invalid format, ignore
            }
        }
    }

    /**
     * Parse shortcuts from JSON.
     *
     * @param json The JSON string
     * @return The list of shortcuts
     */
    private fun parseShortcutsJson(json: String): List<Shortcut> {
        // This is a placeholder. In a real implementation, this would use a JSON library.
        return emptyList()
    }

    /**
     * Register a shortcut.
     *
     * @param shortcut The shortcut to register
     */
    fun registerShortcut(shortcut: Shortcut) {
        shortcuts[shortcut.id] = shortcut

        // Add to category
        val categoryShortcuts = categories.getOrPut(shortcut.category) { mutableListOf() }
        if (!categoryShortcuts.contains(shortcut.id)) {
            categoryShortcuts.add(shortcut.id)
        }

        // Emit shortcut event
        _shortcutEvents.value = ShortcutEvent.ShortcutRegistered(shortcut)
    }

    /**
     * Unregister a shortcut.
     *
     * @param shortcutId The ID of the shortcut to unregister
     * @return True if the shortcut was unregistered, false if it wasn't found
     */
    fun unregisterShortcut(shortcutId: String): Boolean {
        val shortcut = shortcuts.remove(shortcutId) ?: return false

        // Remove from category
        categories[shortcut.category]?.remove(shortcutId)

        // Emit shortcut event
        _shortcutEvents.value = ShortcutEvent.ShortcutUnregistered(shortcutId)

        return true
    }

    /**
     * Get a shortcut by ID.
     *
     * @param shortcutId The ID of the shortcut
     * @return The shortcut, or null if it wasn't found
     */
    fun getShortcut(shortcutId: String): Shortcut? = shortcuts[shortcutId]

    /**
     * Get all shortcuts.
     *
     * @return A list of all shortcuts
     */
    fun getAllShortcuts(): List<Shortcut> = shortcuts.values.toList()

    /**
     * Get shortcuts by category.
     *
     * @param category The category
     * @return A list of shortcuts in the category
     */
    fun getShortcutsByCategory(category: String): List<Shortcut> {
        val shortcutIds = categories[category] ?: return emptyList()
        return shortcutIds.mapNotNull { shortcuts[it] }
    }

    /**
     * Get all categories.
     *
     * @return A list of all categories
     */
    fun getAllCategories(): List<String> = categories.keys.toList()

    /**
     * Customize a shortcut.
     *
     * @param shortcutId The ID of the shortcut to customize
     * @param keyStrokes The new key strokes
     * @return True if the shortcut was customized, false if it wasn't found
     */
    fun customizeShortcut(shortcutId: String, keyStrokes: List<KeyStroke>): Boolean {
        val shortcut = shortcuts[shortcutId] ?: return false

        // Create a new shortcut with the updated key strokes
        val customizedShortcut = shortcut.copy(keyStrokes = keyStrokes)

        // Update the shortcut
        shortcuts[shortcutId] = customizedShortcut

        // Save to state
        saveCustomShortcuts()

        // Emit shortcut event
        _shortcutEvents.value = ShortcutEvent.ShortcutCustomized(customizedShortcut)

        return true
    }

    /**
     * Reset a shortcut to its default.
     *
     * @param shortcutId The ID of the shortcut to reset
     * @return True if the shortcut was reset, false if it wasn't found
     */
    fun resetShortcut(shortcutId: String): Boolean {
        // This would require storing the default shortcuts separately
        // For now, just return false
        return false
    }

    /**
     * Reset all shortcuts to their defaults.
     */
    fun resetAllShortcuts() {
        // This would require storing the default shortcuts separately
        // For now, just clear the custom shortcuts
        stateManager.removeState("shortcuts.custom")

        // Re-register default shortcuts
        registerDefaultShortcuts()

        // Emit shortcut event
        _shortcutEvents.value = ShortcutEvent.AllShortcutsReset
    }

    /**
     * Save custom shortcuts to state.
     */
    private fun saveCustomShortcuts() {
        // This is a placeholder. In a real implementation, this would serialize the shortcuts to JSON.
        stateManager.setState("shortcuts.custom", "")
    }

    /**
     * Handle a key event.
     *
     * @param keyEvent The key event
     * @return True if the event was handled, false otherwise
     */
    fun handleKeyEvent(keyEvent: KeyEvent): Boolean {
        // Only handle key down events
        if (keyEvent.type != KeyEventType.KeyDown) {
            return false
        }

        // Create a key stroke from the event
        val keyStroke =
            KeyStroke(
                key = keyEvent.key,
                ctrl = keyEvent.isCtrlPressed,
                alt = keyEvent.isAltPressed,
                shift = keyEvent.isShiftPressed,
                meta = keyEvent.isMetaPressed,
            )

        // Find a shortcut that matches the key stroke
        val matchingShortcut =
            shortcuts.values.find { shortcut ->
                shortcut.keyStrokes.any { it == keyStroke }
            }

        if (matchingShortcut != null) {
            // Emit shortcut event
            _shortcutEvents.value = ShortcutEvent.ShortcutTriggered(matchingShortcut)
            return true
        }

        return false
    }

    companion object {
        // Singleton instance
        private var instance: ShortcutManager? = null

        /**
         * Get the singleton instance of the ShortcutManager.
         *
         * @return The ShortcutManager instance
         */
        fun getInstance(): ShortcutManager {
            if (instance == null) {
                instance = ShortcutManager()
            }
            return instance!!
        }
    }
}

/**
 * A keyboard shortcut.
 *
 * @param id The unique ID of the shortcut
 * @param name The name of the shortcut
 * @param description The description of the shortcut
 * @param keyStrokes The key strokes that trigger the shortcut
 * @param category The category of the shortcut
 */
data class Shortcut(val id: String, val name: String, val description: String, val keyStrokes: List<KeyStroke>, val category: String)

/**
 * A key stroke.
 *
 * @param key The key
 * @param ctrl Whether the Ctrl key is pressed
 * @param alt Whether the Alt key is pressed
 * @param shift Whether the Shift key is pressed
 * @param meta Whether the Meta key is pressed
 */
data class KeyStroke(
    val key: Key,
    val ctrl: Boolean = false,
    val alt: Boolean = false,
    val shift: Boolean = false,
    val meta: Boolean = false,
) {
    /**
     * Get a string representation of the key stroke.
     *
     * @return The string representation
     */
    fun toDisplayString(): String {
        val modifiers = mutableListOf<String>()
        if (ctrl) modifiers.add("Ctrl")
        if (alt) modifiers.add("Alt")
        if (shift) modifiers.add("Shift")
        if (meta) modifiers.add("Meta")

        val keyName =
            when (key) {
                Key.Equals -> "="
                Key.Minus -> "-"
                Key.Plus -> "+"
                Key.Zero -> "0"
                else -> key.keyCode.toString()
            }

        return if (modifiers.isEmpty()) {
            keyName
        } else {
            "${modifiers.joinToString("+")}+$keyName"
        }
    }
}

/**
 * Shortcut events emitted by the ShortcutManager.
 */
sealed class ShortcutEvent {
    /**
     * Event emitted when a shortcut is registered.
     *
     * @param shortcut The registered shortcut
     */
    data class ShortcutRegistered(val shortcut: Shortcut) : ShortcutEvent()

    /**
     * Event emitted when a shortcut is unregistered.
     *
     * @param shortcutId The ID of the unregistered shortcut
     */
    data class ShortcutUnregistered(val shortcutId: String) : ShortcutEvent()

    /**
     * Event emitted when a shortcut is customized.
     *
     * @param shortcut The customized shortcut
     */
    data class ShortcutCustomized(val shortcut: Shortcut) : ShortcutEvent()

    /**
     * Event emitted when a shortcut is triggered.
     *
     * @param shortcut The triggered shortcut
     */
    data class ShortcutTriggered(val shortcut: Shortcut) : ShortcutEvent()

    /**
     * Event emitted when all shortcuts are reset.
     */
    object AllShortcutsReset : ShortcutEvent()
}

/**
 * Composable function to remember a ShortcutManager instance.
 *
 * @return The remembered ShortcutManager instance
 */
@Composable
fun rememberShortcutManager(): ShortcutManager = remember { ShortcutManager.getInstance() }

/**
 * Extension function to handle key events.
 *
 * @param keyEvent The key event
 * @return True if the event was handled, false otherwise
 */
fun ShortcutManager.onKeyEvent(keyEvent: KeyEvent): Boolean = handleKeyEvent(keyEvent)
