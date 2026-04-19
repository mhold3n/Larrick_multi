package com.campro.v5.layout

import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import com.google.gson.Gson
import com.google.gson.GsonBuilder
import com.google.gson.JsonSyntaxException
import com.google.gson.reflect.TypeToken
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import java.io.File
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Manages application state persistence for the CamPro v5 application.
 * This class provides automatic state saving on exit, state restoration on startup,
 * and state versioning for backward compatibility.
 */
class StateManager {
    // State storage
    private val stateMap = ConcurrentHashMap<String, Any>()

    // State change tracking
    private val dirtyKeys = mutableSetOf<String>()
    private val isStateDirty = AtomicBoolean(false)

    // Auto-save configuration
    private var autoSaveEnabled = true
    private var autoSaveInterval = 60000L // 1 minute
    private var autoSaveThread: Thread? = null
    private val autoSaveRunning = AtomicBoolean(false)

    // State version
    private var stateVersion = 1

    // State change events
    private val _stateChangeEvents = MutableStateFlow<StateChangeEvent?>(null)
    val stateChangeEvents: StateFlow<StateChangeEvent?> = _stateChangeEvents.asStateFlow()

    // Coroutine scope for async operations
    private val scope = CoroutineScope(Dispatchers.IO)

    init {
        // Start auto-save thread if enabled
        if (autoSaveEnabled) {
            startAutoSave()
        }

        // Register shutdown hook to save state on exit
        Runtime.getRuntime().addShutdownHook(
            Thread {
                saveState()
            },
        )
    }

    /**
     * Set a state value.
     *
     * @param key The state key
     * @param value The state value
     */
    fun <T : Any> setState(key: String, value: T) {
        // Update state
        stateMap[key] = value

        // Mark as dirty
        dirtyKeys.add(key)
        isStateDirty.set(true)

        // Emit state change event
        _stateChangeEvents.value = StateChangeEvent.StateChanged(key, value)
    }

    /**
     * Get a state value.
     *
     * @param key The state key
     * @param defaultValue The default value to return if the key doesn't exist
     * @return The state value, or the default value if the key doesn't exist
     */
    @Suppress("UNCHECKED_CAST")
    fun <T : Any> getState(key: String, defaultValue: T): T = stateMap[key] as? T ?: defaultValue

    /**
     * Check if a state key exists.
     *
     * @param key The state key
     * @return True if the key exists, false otherwise
     */
    fun hasState(key: String): Boolean = stateMap.containsKey(key)

    /**
     * Remove a state value.
     *
     * @param key The state key
     * @return True if the key was removed, false if it didn't exist
     */
    fun removeState(key: String): Boolean {
        if (!stateMap.containsKey(key)) {
            return false
        }

        // Remove state
        stateMap.remove(key)

        // Mark as dirty
        dirtyKeys.add(key)
        isStateDirty.set(true)

        // Emit state change event
        _stateChangeEvents.value = StateChangeEvent.StateRemoved(key)

        return true
    }

    /**
     * Clear all state values.
     */
    fun clearState() {
        // Get keys before clearing
        val keys = stateMap.keys.toList()

        // Clear state
        stateMap.clear()

        // Mark all keys as dirty
        dirtyKeys.addAll(keys)
        isStateDirty.set(true)

        // Emit state change event
        _stateChangeEvents.value = StateChangeEvent.StateCleared
    }

    /**
     * Save the current state to a file.
     *
     * @param file The file to save to, or null to use the default file
     * @return True if the save was successful, false otherwise
     */
    fun saveState(file: File? = null): Boolean = try {
        // Create state object
        val state =
            StateData(
                version = stateVersion,
                timestamp = System.currentTimeMillis(),
                data = stateMap.toMap(),
            )

        // Convert to JSON
        val gson = GsonBuilder().setPrettyPrinting().create()
        val json = gson.toJson(state)

        // Determine file to save to
        val saveFile = file ?: getDefaultStateFile()

        // Ensure directory exists
        saveFile.parentFile?.mkdirs()

        // Write to file
        saveFile.writeText(json)

        // Reset dirty flags
        dirtyKeys.clear()
        isStateDirty.set(false)

        // Emit state change event
        _stateChangeEvents.value = StateChangeEvent.StateSaved(saveFile.absolutePath)

        true
    } catch (e: Exception) {
        // Emit error event
        _stateChangeEvents.value = StateChangeEvent.Error("Failed to save state: ${e.message}")

        false
    }

    /**
     * Load state from a file.
     *
     * @param file The file to load from, or null to use the default file
     * @return True if the load was successful, false otherwise
     */
    fun loadState(file: File? = null): Boolean {
        return try {
            // Determine file to load from
            val loadFile = file ?: getDefaultStateFile()

            // Check if file exists
            if (!loadFile.exists()) {
                return false
            }

            // Read JSON from file
            val json = loadFile.readText()

            // Convert from JSON
            val gson = Gson()
            val stateType = object : TypeToken<StateData>() {}.type
            val state = gson.fromJson<StateData>(json, stateType)

            // Check version compatibility
            if (state.version > stateVersion) {
                // Emit error event
                _stateChangeEvents.value =
                    StateChangeEvent.Error(
                        "State version ${state.version} is newer than current version $stateVersion",
                    )

                return false
            }

            // Apply state
            stateMap.clear()
            stateMap.putAll(state.data)

            // Reset dirty flags
            dirtyKeys.clear()
            isStateDirty.set(false)

            // Emit state change event
            _stateChangeEvents.value = StateChangeEvent.StateLoaded(loadFile.absolutePath)

            true
        } catch (e: JsonSyntaxException) {
            // Emit error event
            _stateChangeEvents.value = StateChangeEvent.Error("Invalid state file format: ${e.message}")

            false
        } catch (e: Exception) {
            // Emit error event
            _stateChangeEvents.value = StateChangeEvent.Error("Failed to load state: ${e.message}")

            false
        }
    }

    /**
     * Enable or disable auto-save.
     *
     * @param enabled Whether auto-save should be enabled
     * @param interval The auto-save interval in milliseconds
     */
    fun setAutoSave(enabled: Boolean, interval: Long = 60000L) {
        autoSaveEnabled = enabled
        autoSaveInterval = interval

        if (enabled) {
            startAutoSave()
        } else {
            stopAutoSave()
        }
    }

    /**
     * Start the auto-save thread.
     */
    private fun startAutoSave() {
        val currentThread = autoSaveThread
        if (currentThread != null && currentThread.isAlive) {
            return
        }

        autoSaveRunning.set(true)

        autoSaveThread =
            Thread {
                while (autoSaveRunning.get()) {
                    try {
                        Thread.sleep(autoSaveInterval)

                        // Save state if dirty
                        if (isStateDirty.get()) {
                            scope.launch {
                                saveState()
                            }
                        }
                    } catch (e: InterruptedException) {
                        // Thread was interrupted, exit
                        break
                    }
                }
            }

        val newThread = autoSaveThread
        if (newThread != null) {
            newThread.isDaemon = true
            newThread.start()
        }
    }

    /**
     * Stop the auto-save thread.
     */
    private fun stopAutoSave() {
        autoSaveRunning.set(false)
        autoSaveThread?.interrupt()
        autoSaveThread = null
    }

    /**
     * Get the default state file.
     *
     * @return The default state file
     */
    private fun getDefaultStateFile(): File {
        val appDataDir = System.getProperty("user.home") + File.separator + ".campro"
        return File(appDataDir, "state.json")
    }

    /**
     * Export the current state to a file.
     *
     * @param file The file to export to
     * @return True if the export was successful, false otherwise
     */
    fun exportState(file: File): Boolean = saveState(file)

    /**
     * Import state from a file.
     *
     * @param file The file to import from
     * @return True if the import was successful, false otherwise
     */
    fun importState(file: File): Boolean = loadState(file)

    /**
     * Create a backup of the current state.
     *
     * @return The backup file, or null if the backup failed
     */
    fun createBackup(): File? {
        try {
            val appDataDir = System.getProperty("user.home") + File.separator + ".campro"
            val backupsDir = File(appDataDir, "backups")
            backupsDir.mkdirs()

            val timestamp = System.currentTimeMillis()
            val backupFile = File(backupsDir, "state_$timestamp.json")

            if (saveState(backupFile)) {
                return backupFile
            }
        } catch (e: Exception) {
            // Emit error event
            _stateChangeEvents.value = StateChangeEvent.Error("Failed to create backup: ${e.message}")
        }

        return null
    }

    /**
     * Restore state from a backup.
     *
     * @param backupFile The backup file to restore from
     * @return True if the restore was successful, false otherwise
     */
    fun restoreBackup(backupFile: File): Boolean = loadState(backupFile)

    /**
     * List all available backups.
     *
     * @return A list of backup files
     */
    fun listBackups(): List<File> {
        val appDataDir = System.getProperty("user.home") + File.separator + ".campro"
        val backupsDir = File(appDataDir, "backups")

        if (!backupsDir.exists() || !backupsDir.isDirectory) {
            return emptyList()
        }

        return backupsDir
            .listFiles { file ->
                file.isFile && file.name.startsWith("state_") && file.name.endsWith(".json")
            }?.toList() ?: emptyList()
    }

    /**
     * Clean up resources when the manager is no longer needed.
     */
    fun shutdown() {
        // Save state if dirty
        if (isStateDirty.get()) {
            saveState()
        }

        // Stop auto-save thread
        stopAutoSave()
    }

    companion object {
        // Singleton instance
        private var instance: StateManager? = null

        /**
         * Get the singleton instance of the StateManager.
         *
         * @return The StateManager instance
         */
        fun getInstance(): StateManager {
            if (instance == null) {
                instance = StateManager()
            }
            return instance as StateManager
        }
    }
}

/**
 * State data for serialization.
 *
 * @param version The state version
 * @param timestamp The timestamp when the state was saved
 * @param data The state data
 */
data class StateData(val version: Int, val timestamp: Long, val data: Map<String, Any>)

/**
 * State change events emitted by the StateManager.
 */
sealed class StateChangeEvent {
    /**
     * Event emitted when a state value changes.
     *
     * @param key The state key
     * @param value The new state value
     */
    data class StateChanged(val key: String, val value: Any) : StateChangeEvent()

    /**
     * Event emitted when a state value is removed.
     *
     * @param key The state key
     */
    data class StateRemoved(val key: String) : StateChangeEvent()

    /**
     * Event emitted when all state values are cleared.
     */
    object StateCleared : StateChangeEvent()

    /**
     * Event emitted when state is saved to a file.
     *
     * @param filePath The path to the file
     */
    data class StateSaved(val filePath: String) : StateChangeEvent()

    /**
     * Event emitted when state is loaded from a file.
     *
     * @param filePath The path to the file
     */
    data class StateLoaded(val filePath: String) : StateChangeEvent()

    /**
     * Event emitted when an error occurs.
     *
     * @param message The error message
     */
    data class Error(val message: String) : StateChangeEvent()
}

/**
 * Composable function to remember a StateManager instance.
 *
 * @return The remembered StateManager instance
 */
@Composable
fun rememberStateManager(): StateManager = remember { StateManager.getInstance() }

/**
 * Extension function to get a state value with type inference.
 *
 * @param key The state key
 * @param defaultValue The default value to return if the key doesn't exist
 * @return The state value, or the default value if the key doesn't exist
 */
inline fun <reified T : Any> StateManager.get(key: String, defaultValue: T): T = getState(key, defaultValue)

/**
 * Extension function to set a state value with type inference.
 *
 * @param key The state key
 * @param value The state value
 */
inline fun <reified T : Any> StateManager.set(key: String, value: T) {
    setState(key, value)
}
