package com.campro.v5.file

import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import com.campro.v5.layout.StateManager
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.io.File
import java.nio.file.Files
import java.nio.file.attribute.BasicFileAttributes
import java.text.SimpleDateFormat
import java.util.*

/**
 * Manages recently opened files for the CamPro v5 application.
 * This class provides tracking of recent files, file preview functionality,
 * and file pinning for important projects.
 */
class RecentFilesManager {
    // Maximum number of recent files to track
    private val maxRecentFiles = 20

    // Recent files list
    private val _recentFiles = MutableStateFlow<List<RecentFile>>(emptyList())
    val recentFiles: StateFlow<List<RecentFile>> = _recentFiles.asStateFlow()

    // Pinned files list
    private val _pinnedFiles = MutableStateFlow<List<RecentFile>>(emptyList())
    val pinnedFiles: StateFlow<List<RecentFile>> = _pinnedFiles.asStateFlow()

    // Recent files events
    private val _recentFilesEvents = MutableStateFlow<RecentFilesEvent?>(null)
    val recentFilesEvents: StateFlow<RecentFilesEvent?> = _recentFilesEvents.asStateFlow()

    // State manager for persistence
    private val stateManager = StateManager.getInstance()

    init {
        // Load recent files from state
        loadRecentFiles()

        // Load pinned files from state
        loadPinnedFiles()
    }

    /**
     * Load recent files from state.
     */
    private fun loadRecentFiles() {
        val recentFilesJson = stateManager.getState("recentFiles.list", "")
        if (recentFilesJson.isNotEmpty()) {
            try {
                val recentFiles = parseRecentFilesJson(recentFilesJson)

                // Filter out files that no longer exist
                val existingFiles =
                    recentFiles.filter { file ->
                        val f = File(file.path)
                        f.exists() && f.isFile
                    }

                _recentFiles.value = existingFiles
            } catch (e: Exception) {
                // Invalid format, ignore
            }
        }
    }

    /**
     * Load pinned files from state.
     */
    private fun loadPinnedFiles() {
        val pinnedFilesJson = stateManager.getState("recentFiles.pinned", "")
        if (pinnedFilesJson.isNotEmpty()) {
            try {
                val pinnedFiles = parseRecentFilesJson(pinnedFilesJson)

                // Filter out files that no longer exist
                val existingFiles =
                    pinnedFiles.filter { file ->
                        val f = File(file.path)
                        f.exists() && f.isFile
                    }

                _pinnedFiles.value = existingFiles
            } catch (e: Exception) {
                // Invalid format, ignore
            }
        }
    }

    /**
     * Parse recent files from JSON.
     *
     * @param json The JSON string
     * @return The list of recent files
     */
    private fun parseRecentFilesJson(json: String): List<RecentFile> {
        // This is a placeholder. In a real implementation, this would use a JSON library.
        // For now, we'll just return an empty list.
        return emptyList()
    }

    /**
     * Save recent files to state.
     */
    private fun saveRecentFiles() {
        // This is a placeholder. In a real implementation, this would serialize the recent files to JSON.
        stateManager.setState("recentFiles.list", "")
    }

    /**
     * Save pinned files to state.
     */
    private fun savePinnedFiles() {
        // This is a placeholder. In a real implementation, this would serialize the pinned files to JSON.
        stateManager.setState("recentFiles.pinned", "")
    }

    /**
     * Add a file to the recent files list.
     *
     * @param file The file to add
     * @param projectName The name of the project (optional)
     */
    fun addRecentFile(file: File, projectName: String? = null) {
        // Check if the file exists
        if (!file.exists() || !file.isFile) {
            return
        }

        // Get file attributes
        val attributes =
            try {
                Files.readAttributes(file.toPath(), BasicFileAttributes::class.java)
            } catch (e: Exception) {
                null
            }

        // Create recent file object
        val recentFile =
            RecentFile(
                path = file.absolutePath,
                name = projectName ?: file.nameWithoutExtension,
                lastOpened = System.currentTimeMillis(),
                fileSize = file.length(),
                creationTime = attributes?.creationTime()?.toMillis() ?: 0,
                lastModifiedTime = attributes?.lastModifiedTime()?.toMillis() ?: 0,
            )

        // Remove the file if it already exists in the list
        val currentList = _recentFiles.value.toMutableList()
        currentList.removeIf { it.path == recentFile.path }

        // Add the file to the beginning of the list
        currentList.add(0, recentFile)

        // Trim the list if it exceeds the maximum size
        if (currentList.size > maxRecentFiles) {
            currentList.removeAt(currentList.size - 1)
        }

        // Update the list
        _recentFiles.value = currentList

        // Save to state
        saveRecentFiles()

        // Emit event
        _recentFilesEvents.value = RecentFilesEvent.FileAdded(recentFile)
    }

    /**
     * Remove a file from the recent files list.
     *
     * @param filePath The path of the file to remove
     * @return True if the file was removed, false if it wasn't found
     */
    fun removeRecentFile(filePath: String): Boolean {
        val currentList = _recentFiles.value.toMutableList()
        val removed = currentList.removeIf { it.path == filePath }

        if (removed) {
            // Update the list
            _recentFiles.value = currentList

            // Save to state
            saveRecentFiles()

            // Emit event
            _recentFilesEvents.value = RecentFilesEvent.FileRemoved(filePath)
        }

        return removed
    }

    /**
     * Clear all recent files.
     */
    fun clearRecentFiles() {
        // Update the list
        _recentFiles.value = emptyList()

        // Save to state
        saveRecentFiles()

        // Emit event
        _recentFilesEvents.value = RecentFilesEvent.AllFilesCleared
    }

    /**
     * Pin a file to the pinned files list.
     *
     * @param filePath The path of the file to pin
     * @return True if the file was pinned, false if it wasn't found
     */
    fun pinFile(filePath: String): Boolean {
        // Find the file in the recent files list
        val recentFile = _recentFiles.value.find { it.path == filePath } ?: return false

        // Check if the file is already pinned
        if (_pinnedFiles.value.any { it.path == filePath }) {
            return false
        }

        // Add the file to the pinned files list
        val pinnedList = _pinnedFiles.value.toMutableList()
        pinnedList.add(recentFile)
        _pinnedFiles.value = pinnedList

        // Save to state
        savePinnedFiles()

        // Emit event
        _recentFilesEvents.value = RecentFilesEvent.FilePinned(recentFile)

        return true
    }

    /**
     * Unpin a file from the pinned files list.
     *
     * @param filePath The path of the file to unpin
     * @return True if the file was unpinned, false if it wasn't found
     */
    fun unpinFile(filePath: String): Boolean {
        val pinnedList = _pinnedFiles.value.toMutableList()
        val removed = pinnedList.removeIf { it.path == filePath }

        if (removed) {
            // Update the list
            _pinnedFiles.value = pinnedList

            // Save to state
            savePinnedFiles()

            // Emit event
            _recentFilesEvents.value = RecentFilesEvent.FileUnpinned(filePath)
        }

        return removed
    }

    /**
     * Get a recent file by path.
     *
     * @param filePath The path of the file
     * @return The recent file, or null if it wasn't found
     */
    fun getRecentFile(filePath: String): RecentFile? = _recentFiles.value.find { it.path == filePath }

    /**
     * Check if a file is pinned.
     *
     * @param filePath The path of the file
     * @return True if the file is pinned, false otherwise
     */
    fun isFilePinned(filePath: String): Boolean = _pinnedFiles.value.any { it.path == filePath }

    /**
     * Get all recent files.
     *
     * @return A list of all recent files
     */
    fun getAllRecentFiles(): List<RecentFile> = _recentFiles.value

    /**
     * Get all pinned files.
     *
     * @return A list of all pinned files
     */
    fun getAllPinnedFiles(): List<RecentFile> = _pinnedFiles.value

    /**
     * Get recent files sorted by last opened time.
     *
     * @param ascending Whether to sort in ascending order
     * @return A list of recent files sorted by last opened time
     */
    fun getRecentFilesByLastOpened(ascending: Boolean = false): List<RecentFile> = if (ascending) {
        _recentFiles.value.sortedBy { it.lastOpened }
    } else {
        _recentFiles.value.sortedByDescending { it.lastOpened }
    }

    /**
     * Get recent files sorted by name.
     *
     * @param ascending Whether to sort in ascending order
     * @return A list of recent files sorted by name
     */
    fun getRecentFilesByName(ascending: Boolean = true): List<RecentFile> = if (ascending) {
        _recentFiles.value.sortedBy { it.name }
    } else {
        _recentFiles.value.sortedByDescending { it.name }
    }

    /**
     * Get recent files sorted by file size.
     *
     * @param ascending Whether to sort in ascending order
     * @return A list of recent files sorted by file size
     */
    fun getRecentFilesBySize(ascending: Boolean = true): List<RecentFile> = if (ascending) {
        _recentFiles.value.sortedBy { it.fileSize }
    } else {
        _recentFiles.value.sortedByDescending { it.fileSize }
    }

    /**
     * Get recent files sorted by last modified time.
     *
     * @param ascending Whether to sort in ascending order
     * @return A list of recent files sorted by last modified time
     */
    fun getRecentFilesByLastModified(ascending: Boolean = false): List<RecentFile> = if (ascending) {
        _recentFiles.value.sortedBy { it.lastModifiedTime }
    } else {
        _recentFiles.value.sortedByDescending { it.lastModifiedTime }
    }

    /**
     * Get a preview of a file.
     *
     * @param filePath The path of the file
     * @return A preview of the file, or null if the file wasn't found
     */
    fun getFilePreview(filePath: String): FilePreview? {
        val file = File(filePath)
        if (!file.exists() || !file.isFile) {
            return null
        }

        try {
            // Read the first few lines of the file
            val lines = file.readLines().take(10)

            // Get file attributes
            val attributes = Files.readAttributes(file.toPath(), BasicFileAttributes::class.java)

            // Create preview object
            return FilePreview(
                path = filePath,
                name = file.name,
                size = file.length(),
                creationTime = attributes.creationTime().toMillis(),
                lastModifiedTime = attributes.lastModifiedTime().toMillis(),
                previewText = lines.joinToString("\n"),
            )
        } catch (e: Exception) {
            return null
        }
    }

    companion object {
        // Singleton instance
        private var instance: RecentFilesManager? = null

        /**
         * Get the singleton instance of the RecentFilesManager.
         *
         * @return The RecentFilesManager instance
         */
        fun getInstance(): RecentFilesManager {
            if (instance == null) {
                instance = RecentFilesManager()
            }
            return instance!!
        }
    }
}

/**
 * A recently opened file.
 *
 * @param path The path of the file
 * @param name The name of the file or project
 * @param lastOpened The time the file was last opened
 * @param fileSize The size of the file in bytes
 * @param creationTime The time the file was created
 * @param lastModifiedTime The time the file was last modified
 */
data class RecentFile(
    val path: String,
    val name: String,
    val lastOpened: Long,
    val fileSize: Long,
    val creationTime: Long,
    val lastModifiedTime: Long,
) {
    /**
     * Get a formatted string of the last opened time.
     *
     * @return A formatted string of the last opened time
     */
    fun getFormattedLastOpened(): String {
        val date = Date(lastOpened)
        val format = SimpleDateFormat("yyyy-MM-dd HH:mm:ss")
        return format.format(date)
    }

    /**
     * Get a formatted string of the file size.
     *
     * @return A formatted string of the file size
     */
    fun getFormattedFileSize(): String = when {
        fileSize < 1024 -> "$fileSize B"
        fileSize < 1024 * 1024 -> "${fileSize / 1024} KB"
        fileSize < 1024 * 1024 * 1024 -> "${fileSize / (1024 * 1024)} MB"
        else -> "${fileSize / (1024 * 1024 * 1024)} GB"
    }

    /**
     * Get a formatted string of the creation time.
     *
     * @return A formatted string of the creation time
     */
    fun getFormattedCreationTime(): String {
        val date = Date(creationTime)
        val format = SimpleDateFormat("yyyy-MM-dd HH:mm:ss")
        return format.format(date)
    }

    /**
     * Get a formatted string of the last modified time.
     *
     * @return A formatted string of the last modified time
     */
    fun getFormattedLastModifiedTime(): String {
        val date = Date(lastModifiedTime)
        val format = SimpleDateFormat("yyyy-MM-dd HH:mm:ss")
        return format.format(date)
    }
}

/**
 * A preview of a file.
 *
 * @param path The path of the file
 * @param name The name of the file
 * @param size The size of the file in bytes
 * @param creationTime The time the file was created
 * @param lastModifiedTime The time the file was last modified
 * @param previewText A preview of the file's contents
 */
data class FilePreview(
    val path: String,
    val name: String,
    val size: Long,
    val creationTime: Long,
    val lastModifiedTime: Long,
    val previewText: String,
)

/**
 * Recent files events emitted by the RecentFilesManager.
 */
sealed class RecentFilesEvent {
    /**
     * Event emitted when a file is added to the recent files list.
     *
     * @param file The added file
     */
    data class FileAdded(val file: RecentFile) : RecentFilesEvent()

    /**
     * Event emitted when a file is removed from the recent files list.
     *
     * @param filePath The path of the removed file
     */
    data class FileRemoved(val filePath: String) : RecentFilesEvent()

    /**
     * Event emitted when all files are cleared from the recent files list.
     */
    object AllFilesCleared : RecentFilesEvent()

    /**
     * Event emitted when a file is pinned.
     *
     * @param file The pinned file
     */
    data class FilePinned(val file: RecentFile) : RecentFilesEvent()

    /**
     * Event emitted when a file is unpinned.
     *
     * @param filePath The path of the unpinned file
     */
    data class FileUnpinned(val filePath: String) : RecentFilesEvent()
}

/**
 * Composable function to remember a RecentFilesManager instance.
 *
 * @return The remembered RecentFilesManager instance
 */
@Composable
fun rememberRecentFilesManager(): RecentFilesManager = remember { RecentFilesManager.getInstance() }
