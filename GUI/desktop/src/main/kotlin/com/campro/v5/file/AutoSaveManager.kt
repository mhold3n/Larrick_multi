package com.campro.v5.file

import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import com.campro.v5.layout.StateManager
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Manages automatic saving of projects for the CamPro v5 application.
 * This class provides configurable auto-save intervals, recovery from crashes,
 * and backup management.
 */
class AutoSaveManager {
    // Auto-save configuration
    private var autoSaveEnabled = true
    private var autoSaveInterval = 60000L // 1 minute
    private var maxAutoSaves = 10

    // Auto-save state
    private val isAutoSaving = AtomicBoolean(false)
    private val _lastAutoSaveTime = MutableStateFlow<Long?>(null)
    val lastAutoSaveTime: StateFlow<Long?> = _lastAutoSaveTime.asStateFlow()

    // Auto-save events
    private val _autoSaveEvents = MutableStateFlow<AutoSaveEvent?>(null)
    val autoSaveEvents: StateFlow<AutoSaveEvent?> = _autoSaveEvents.asStateFlow()

    // Auto-save job
    private var autoSaveJob: Job? = null

    // State manager for persistence
    private val stateManager = StateManager.getInstance()

    // Coroutine scope for async operations
    private val scope = CoroutineScope(Dispatchers.IO + SupervisorJob())

    init {
        // Load auto-save configuration from state
        loadAutoSaveConfig()

        // Start auto-save if enabled
        if (autoSaveEnabled) {
            startAutoSave()
        }
    }

    /**
     * Load auto-save configuration from state.
     */
    private fun loadAutoSaveConfig() {
        autoSaveEnabled = stateManager.getState("autoSave.enabled", true)
        autoSaveInterval = stateManager.getState("autoSave.interval", 60000L)
        maxAutoSaves = stateManager.getState("autoSave.maxAutoSaves", 10)
    }

    /**
     * Save auto-save configuration to state.
     */
    private fun saveAutoSaveConfig() {
        stateManager.setState("autoSave.enabled", autoSaveEnabled)
        stateManager.setState("autoSave.interval", autoSaveInterval)
        stateManager.setState("autoSave.maxAutoSaves", maxAutoSaves)
    }

    /**
     * Start the auto-save process.
     */
    fun startAutoSave() {
        if (autoSaveJob != null) {
            return
        }

        autoSaveEnabled = true
        saveAutoSaveConfig()

        autoSaveJob =
            scope.launch {
                while (isActive) {
                    delay(autoSaveInterval)

                    if (autoSaveEnabled) {
                        performAutoSave()
                    }
                }
            }

        // Emit event
        _autoSaveEvents.value = AutoSaveEvent.AutoSaveStarted
    }

    /**
     * Stop the auto-save process.
     */
    fun stopAutoSave() {
        autoSaveJob?.cancel()
        autoSaveJob = null

        autoSaveEnabled = false
        saveAutoSaveConfig()

        // Emit event
        _autoSaveEvents.value = AutoSaveEvent.AutoSaveStopped
    }

    /**
     * Set the auto-save interval.
     *
     * @param interval The interval in milliseconds
     */
    fun setAutoSaveInterval(interval: Long) {
        if (interval < 5000) {
            throw IllegalArgumentException("Auto-save interval must be at least 5 seconds")
        }

        autoSaveInterval = interval
        saveAutoSaveConfig()

        // Restart auto-save if enabled
        if (autoSaveEnabled) {
            stopAutoSave()
            startAutoSave()
        }

        // Emit event
        _autoSaveEvents.value = AutoSaveEvent.AutoSaveIntervalChanged(interval)
    }

    /**
     * Set the maximum number of auto-saves to keep.
     *
     * @param maxSaves The maximum number of auto-saves
     */
    fun setMaxAutoSaves(maxSaves: Int) {
        if (maxSaves < 1) {
            throw IllegalArgumentException("Maximum auto-saves must be at least 1")
        }

        maxAutoSaves = maxSaves
        saveAutoSaveConfig()

        // Clean up old auto-saves if necessary
        cleanupAutoSaves()

        // Emit event
        _autoSaveEvents.value = AutoSaveEvent.MaxAutoSavesChanged(maxSaves)
    }

    /**
     * Check if auto-save is enabled.
     *
     * @return True if auto-save is enabled, false otherwise
     */
    fun isAutoSaveEnabled(): Boolean = autoSaveEnabled

    /**
     * Get the auto-save interval.
     *
     * @return The auto-save interval in milliseconds
     */
    fun getAutoSaveInterval(): Long = autoSaveInterval

    /**
     * Get the maximum number of auto-saves.
     *
     * @return The maximum number of auto-saves
     */
    fun getMaxAutoSaves(): Int = maxAutoSaves

    /**
     * Perform an auto-save operation.
     *
     * @return True if the auto-save was successful, false otherwise
     */
    suspend fun performAutoSave(): Boolean {
        // Check if auto-save is already in progress
        if (!isAutoSaving.compareAndSet(false, true)) {
            return false
        }

        try {
            // Get the current project from the ProjectManager
            val projectManager = ProjectManager.getInstance()
            val currentProject = projectManager.currentProject

            if (currentProject == null || !projectManager.isProjectModified()) {
                // No project or project not modified, nothing to save
                return false
            }

            // Create auto-save directory
            val autoSaveDir = getAutoSaveDirectory()
            autoSaveDir.mkdirs()

            // Create auto-save file
            val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss").format(Date())
            val autoSaveFile = File(autoSaveDir, "${currentProject.name}_$timestamp.autosave")

            // Save project to auto-save file
            val success = projectManager.saveProject(autoSaveFile)

            if (success) {
                // Update last auto-save time
                _lastAutoSaveTime.value = System.currentTimeMillis()

                // Clean up old auto-saves
                cleanupAutoSaves()

                // Emit event
                _autoSaveEvents.value = AutoSaveEvent.AutoSaveCompleted(autoSaveFile.absolutePath)
            } else {
                // Emit error event
                _autoSaveEvents.value = AutoSaveEvent.AutoSaveFailed("Failed to save project")
            }

            return success
        } catch (e: Exception) {
            // Emit error event
            _autoSaveEvents.value = AutoSaveEvent.AutoSaveFailed(e.message ?: "Unknown error")

            return false
        } finally {
            isAutoSaving.set(false)
        }
    }

    /**
     * Clean up old auto-saves, keeping only the most recent ones.
     */
    private fun cleanupAutoSaves() {
        val autoSaveDir = getAutoSaveDirectory()
        if (!autoSaveDir.exists() || !autoSaveDir.isDirectory) {
            return
        }

        // Get all auto-save files
        val autoSaveFiles =
            autoSaveDir
                .listFiles { file ->
                    file.isFile && file.name.endsWith(".autosave")
                }?.toList() ?: emptyList()

        // Sort by last modified time (newest first)
        val sortedFiles = autoSaveFiles.sortedByDescending { it.lastModified() }

        // Delete old auto-saves
        if (sortedFiles.size > maxAutoSaves) {
            sortedFiles.subList(maxAutoSaves, sortedFiles.size).forEach { file ->
                file.delete()
            }
        }
    }

    /**
     * Get the auto-save directory.
     *
     * @return The auto-save directory
     */
    private fun getAutoSaveDirectory(): File = File(System.getProperty("user.home"), ".campro/autosaves")

    /**
     * List all available auto-saves.
     *
     * @return A list of auto-save files
     */
    suspend fun listAutoSaves(): List<AutoSaveFile> = withContext(Dispatchers.IO) {
        val autoSaveDir = getAutoSaveDirectory()
        if (!autoSaveDir.exists() || !autoSaveDir.isDirectory) {
            return@withContext emptyList()
        }

        // Get all auto-save files
        val autoSaveFiles =
            autoSaveDir
                .listFiles { file ->
                    file.isFile && file.name.endsWith(".autosave")
                }?.toList() ?: emptyList()

        // Convert to AutoSaveFile objects
        return@withContext autoSaveFiles
            .map { file ->
                val nameWithoutExtension = file.nameWithoutExtension
                val projectName = nameWithoutExtension.substringBeforeLast("_")
                val timestamp = nameWithoutExtension.substringAfterLast("_")

                AutoSaveFile(
                    file = file,
                    projectName = projectName,
                    timestamp = timestamp,
                    size = file.length(),
                    lastModified = file.lastModified(),
                )
            }.sortedByDescending { it.lastModified }
    }

    /**
     * Restore a project from an auto-save.
     *
     * @param autoSaveFile The auto-save file to restore from
     * @return True if the restore was successful, false otherwise
     */
    suspend fun restoreFromAutoSave(autoSaveFile: File): Boolean {
        val projectManager = ProjectManager.getInstance()
        return projectManager.loadProject(autoSaveFile)
    }

    /**
     * Delete an auto-save file.
     *
     * @param autoSaveFile The auto-save file to delete
     * @return True if the delete was successful, false otherwise
     */
    suspend fun deleteAutoSave(autoSaveFile: File): Boolean = withContext(Dispatchers.IO) {
        val success = autoSaveFile.delete()

        if (success) {
            // Emit event
            _autoSaveEvents.value = AutoSaveEvent.AutoSaveDeleted(autoSaveFile.absolutePath)
        }

        return@withContext success
    }

    /**
     * Delete all auto-save files.
     *
     * @return True if the delete was successful, false otherwise
     */
    suspend fun deleteAllAutoSaves(): Boolean = withContext(Dispatchers.IO) {
        val autoSaveDir = getAutoSaveDirectory()
        if (!autoSaveDir.exists() || !autoSaveDir.isDirectory) {
            return@withContext true
        }

        // Get all auto-save files
        val autoSaveFiles =
            autoSaveDir
                .listFiles { file ->
                    file.isFile && file.name.endsWith(".autosave")
                }?.toList() ?: emptyList()

        // Delete all auto-save files
        var success = true
        autoSaveFiles.forEach { file ->
            if (!file.delete()) {
                success = false
            }
        }

        if (success) {
            // Emit event
            _autoSaveEvents.value = AutoSaveEvent.AllAutoSavesDeleted
        }

        return@withContext success
    }

    /**
     * Check for auto-save files that can be used for recovery.
     *
     * @param projectName The name of the project to check for
     * @return A list of auto-save files for the project
     */
    suspend fun checkForRecovery(projectName: String): List<AutoSaveFile> = withContext(Dispatchers.IO) {
        val autoSaves = listAutoSaves()
        return@withContext autoSaves.filter { it.projectName == projectName }
    }

    /**
     * Clean up resources when the manager is no longer needed.
     */
    fun shutdown() {
        // Cancel auto-save job
        autoSaveJob?.cancel()
        autoSaveJob = null

        // Cancel all coroutines
        scope.cancel()
    }

    companion object {
        // Singleton instance
        private var instance: AutoSaveManager? = null

        /**
         * Get the singleton instance of the AutoSaveManager.
         *
         * @return The AutoSaveManager instance
         */
        fun getInstance(): AutoSaveManager {
            if (instance == null) {
                instance = AutoSaveManager()
            }
            return instance!!
        }
    }
}

/**
 * An auto-save file.
 *
 * @param file The auto-save file
 * @param projectName The name of the project
 * @param timestamp The timestamp of the auto-save
 * @param size The size of the file in bytes
 * @param lastModified The time the file was last modified
 */
data class AutoSaveFile(val file: File, val projectName: String, val timestamp: String, val size: Long, val lastModified: Long) {
    /**
     * Get a formatted string of the file size.
     *
     * @return A formatted string of the file size
     */
    fun getFormattedSize(): String = when {
        size < 1024 -> "$size B"
        size < 1024 * 1024 -> "${size / 1024} KB"
        size < 1024 * 1024 * 1024 -> "${size / (1024 * 1024)} MB"
        else -> "${size / (1024 * 1024 * 1024)} GB"
    }

    /**
     * Get a formatted string of the last modified time.
     *
     * @return A formatted string of the last modified time
     */
    fun getFormattedLastModified(): String {
        val date = Date(lastModified)
        val format = SimpleDateFormat("yyyy-MM-dd HH:mm:ss")
        return format.format(date)
    }

    /**
     * Get a formatted string of the timestamp.
     *
     * @return A formatted string of the timestamp
     */
    fun getFormattedTimestamp(): String {
        try {
            val parsedDate = SimpleDateFormat("yyyyMMdd_HHmmss").parse(timestamp)
            val format = SimpleDateFormat("yyyy-MM-dd HH:mm:ss")
            return format.format(parsedDate)
        } catch (e: Exception) {
            return timestamp
        }
    }
}

/**
 * Auto-save events emitted by the AutoSaveManager.
 */
sealed class AutoSaveEvent {
    /**
     * Event emitted when auto-save is started.
     */
    object AutoSaveStarted : AutoSaveEvent()

    /**
     * Event emitted when auto-save is stopped.
     */
    object AutoSaveStopped : AutoSaveEvent()

    /**
     * Event emitted when an auto-save is completed.
     *
     * @param filePath The path to the auto-save file
     */
    data class AutoSaveCompleted(val filePath: String) : AutoSaveEvent()

    /**
     * Event emitted when an auto-save fails.
     *
     * @param message The error message
     */
    data class AutoSaveFailed(val message: String) : AutoSaveEvent()

    /**
     * Event emitted when the auto-save interval is changed.
     *
     * @param interval The new interval in milliseconds
     */
    data class AutoSaveIntervalChanged(val interval: Long) : AutoSaveEvent()

    /**
     * Event emitted when the maximum number of auto-saves is changed.
     *
     * @param maxSaves The new maximum number of auto-saves
     */
    data class MaxAutoSavesChanged(val maxSaves: Int) : AutoSaveEvent()

    /**
     * Event emitted when an auto-save file is deleted.
     *
     * @param filePath The path to the deleted file
     */
    data class AutoSaveDeleted(val filePath: String) : AutoSaveEvent()

    /**
     * Event emitted when all auto-save files are deleted.
     */
    object AllAutoSavesDeleted : AutoSaveEvent()
}

/**
 * Composable function to remember an AutoSaveManager instance.
 *
 * @return The remembered AutoSaveManager instance
 */
@Composable
fun rememberAutoSaveManager(): AutoSaveManager = remember { AutoSaveManager.getInstance() }
