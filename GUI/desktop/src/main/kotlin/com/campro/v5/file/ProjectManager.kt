package com.campro.v5.file

import androidx.compose.runtime.Composable
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import com.campro.v5.layout.StateManager
import com.google.gson.GsonBuilder
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date

/**
 * Manages projects for the CamPro v5 application.
 * This class provides project saving and loading, project templates,
 * and project metadata management.
 */
class ProjectManager {
    // Current project
    private val _currentProject = mutableStateOf<Project?>(null)
    val currentProject: Project?
        get() = _currentProject.value
    private val _isProjectModified = mutableStateOf(false)

    // Project events
    private val _projectEvents = MutableStateFlow<ProjectEvent?>(null)
    val projectEvents: StateFlow<ProjectEvent?> = _projectEvents.asStateFlow()

    // Project templates
    private val projectTemplates = mutableMapOf<String, ProjectTemplate>()

    // State manager for persistence
    private val stateManager = StateManager.getInstance()

    // Coroutine scope for async operations
    private val scope = CoroutineScope(Dispatchers.IO)

    // JSON serialization
    private val gson = GsonBuilder().setPrettyPrinting().create()

    init {
        // Register default project templates
        registerDefaultTemplates()

        // Try to load last project
        val lastProjectPath = stateManager.getState("project.lastProjectPath", "")
        if (lastProjectPath.isNotEmpty()) {
            val lastProjectFile = File(lastProjectPath)
            if (lastProjectFile.exists()) {
                scope.launch {
                    loadProject(lastProjectFile)
                }
            }
        }
    }

    /**
     * Register default project templates.
     */
    private fun registerDefaultTemplates() {
        // Empty project template
        registerTemplate(
            ProjectTemplate(
                id = "empty",
                name = "Empty Project",
                description = "Start with an empty project",
                parameters = emptyMap(),
            ),
        )

        // Basic mechanism template
        registerTemplate(
            ProjectTemplate(
                id = "basic_mechanism",
                name = "Basic Mechanism",
                description = "A simple cycloidal mechanism with default parameters",
                parameters =
                mapOf(
                    "Piston Diameter" to "70.0",
                    "Stroke" to "20.0",
                    "Rod Length" to "40.0",
                    "TDC Offset" to "40.0",
                    "Cycle Ratio" to "2.0",
                ),
            ),
        )

        // High-speed mechanism template
        registerTemplate(
            ProjectTemplate(
                id = "high_speed",
                name = "High-Speed Mechanism",
                description = "A cycloidal mechanism optimized for high speed",
                parameters =
                mapOf(
                    "Piston Diameter" to "60.0",
                    "Stroke" to "15.0",
                    "Rod Length" to "45.0",
                    "TDC Offset" to "35.0",
                    "Cycle Ratio" to "3.0",
                ),
            ),
        )

        // High-torque mechanism template
        registerTemplate(
            ProjectTemplate(
                id = "high_torque",
                name = "High-Torque Mechanism",
                description = "A cycloidal mechanism optimized for high torque",
                parameters =
                mapOf(
                    "Piston Diameter" to "80.0",
                    "Stroke" to "25.0",
                    "Rod Length" to "35.0",
                    "TDC Offset" to "45.0",
                    "Cycle Ratio" to "1.5",
                ),
            ),
        )
    }

    /**
     * Register a project template.
     *
     * @param template The template to register
     */
    fun registerTemplate(template: ProjectTemplate) {
        projectTemplates[template.id] = template
    }

    /**
     * Get a project template by ID.
     *
     * @param templateId The ID of the template
     * @return The template, or null if it wasn't found
     */
    fun getTemplate(templateId: String): ProjectTemplate? = projectTemplates[templateId]

    /**
     * Get all project templates.
     *
     * @return A list of all project templates
     */
    fun getAllTemplates(): List<ProjectTemplate> = projectTemplates.values.toList()

    /**
     * Create a new project from a template.
     *
     * @param templateId The ID of the template to use
     * @param name The name of the project
     * @return The new project
     */
    fun createProjectFromTemplate(templateId: String, name: String): Project {
        val template = projectTemplates[templateId] ?: throw IllegalArgumentException("Template not found: $templateId")

        val project =
            Project(
                name = name,
                parameters = template.parameters.toMutableMap(),
                metadata =
                ProjectMetadata(
                    createdAt = System.currentTimeMillis(),
                    modifiedAt = System.currentTimeMillis(),
                    templateId = templateId,
                ),
            )

        // Set as current project
        _currentProject.value = project
        _isProjectModified.value = true

        // Emit project event
        _projectEvents.value = ProjectEvent.ProjectCreated(project)

        return project
    }

    /**
     * Create a new empty project.
     *
     * @param name The name of the project
     * @return The new project
     */
    fun createEmptyProject(name: String): Project = createProjectFromTemplate("empty", name)

    /**
     * Check if the current project has been modified.
     *
     * @return True if the project has been modified, false otherwise
     */
    fun isProjectModified(): Boolean = _isProjectModified.value

    /**
     * Update the current project's parameters.
     *
     * @param parameters The new parameters
     */
    fun updateProjectParameters(parameters: Map<String, String>) {
        val currentProject = _currentProject.value ?: return

        // Update parameters
        currentProject.parameters.clear()
        currentProject.parameters.putAll(parameters)

        // Update modification time
        currentProject.metadata.modifiedAt = System.currentTimeMillis()

        // Mark as modified
        _isProjectModified.value = true

        // Emit project event
        _projectEvents.value = ProjectEvent.ProjectUpdated(currentProject)
    }

    /**
     * Update the current project's metadata.
     *
     * @param metadata The new metadata
     */
    fun updateProjectMetadata(metadata: ProjectMetadata) {
        val currentProject = _currentProject.value ?: return

        // Update metadata
        currentProject.metadata = metadata

        // Update modification time
        currentProject.metadata.modifiedAt = System.currentTimeMillis()

        // Mark as modified
        _isProjectModified.value = true

        // Emit project event
        _projectEvents.value = ProjectEvent.ProjectUpdated(currentProject)
    }

    /**
     * Save the current project to a file.
     *
     * @param file The file to save to
     * @return True if the save was successful, false otherwise
     */
    suspend fun saveProject(file: File): Boolean = withContext(Dispatchers.IO) {
        val currentProject = _currentProject.value ?: return@withContext false

        try {
            // Update modification time
            currentProject.metadata.modifiedAt = System.currentTimeMillis()

            // Convert to JSON
            val json = gson.toJson(currentProject)

            // Write to file
            file.writeText(json)

            // Update file path
            currentProject.filePath = file.absolutePath

            // Mark as not modified
            _isProjectModified.value = false

            // Save last project path
            stateManager.setState("project.lastProjectPath", file.absolutePath)

            // Emit project event
            _projectEvents.value = ProjectEvent.ProjectSaved(currentProject)

            return@withContext true
        } catch (e: Exception) {
            // Emit error event
            _projectEvents.value = ProjectEvent.Error("Failed to save project: ${e.message}")

            return@withContext false
        }
    }

    /**
     * Load a project from a file.
     *
     * @param file The file to load from
     * @return True if the load was successful, false otherwise
     */
    suspend fun loadProject(file: File): Boolean = withContext(Dispatchers.IO) {
        try {
            // Read JSON from file
            val json = file.readText()

            // Convert from JSON
            val project = gson.fromJson(json, Project::class.java)

            // Set file path
            project.filePath = file.absolutePath

            // Set as current project
            _currentProject.value = project
            _isProjectModified.value = false

            // Save last project path
            stateManager.setState("project.lastProjectPath", file.absolutePath)

            // Emit project event
            _projectEvents.value = ProjectEvent.ProjectLoaded(project)

            return@withContext true
        } catch (e: Exception) {
            // Emit error event
            _projectEvents.value = ProjectEvent.Error("Failed to load project: ${e.message}")

            return@withContext false
        }
    }

    /**
     * Close the current project.
     *
     * @return True if the project was closed, false if there was no project to close
     */
    fun closeProject(): Boolean {
        val currentProject = _currentProject.value ?: return false

        // Emit project event
        _projectEvents.value = ProjectEvent.ProjectClosed(currentProject)

        // Clear current project
        _currentProject.value = null
        _isProjectModified.value = false

        return true
    }

    /**
     * Export the current project to a different format.
     *
     * @param file The file to export to
     * @param format The format to export to
     * @return True if the export was successful, false otherwise
     */
    suspend fun exportProject(file: File, format: String): Boolean = withContext(Dispatchers.IO) {
        val currentProject = _currentProject.value ?: return@withContext false

        try {
            when (format.lowercase()) {
                "json" -> {
                    // Convert to JSON
                    val json = gson.toJson(currentProject)

                    // Write to file
                    file.writeText(json)
                }
                "csv" -> {
                    // Export parameters as CSV
                    val csv = StringBuilder()
                    csv.appendLine("Parameter,Value")
                    currentProject.parameters.forEach { (key, value) ->
                        csv.appendLine("$key,$value")
                    }

                    // Write to file
                    file.writeText(csv.toString())
                }
                else -> {
                    // Unsupported format
                    _projectEvents.value = ProjectEvent.Error("Unsupported export format: $format")
                    return@withContext false
                }
            }

            // Emit project event
            _projectEvents.value = ProjectEvent.ProjectExported(currentProject, file.absolutePath, format)

            return@withContext true
        } catch (e: Exception) {
            // Emit error event
            _projectEvents.value = ProjectEvent.Error("Failed to export project: ${e.message}")

            return@withContext false
        }
    }

    /**
     * Import a project from a file.
     *
     * @param file The file to import from
     * @param format The format of the file
     * @return True if the import was successful, false otherwise
     */
    suspend fun importProject(file: File, format: String): Boolean = withContext(Dispatchers.IO) {
        try {
            when (format.lowercase()) {
                "json" -> {
                    // Read JSON from file
                    val json = file.readText()

                    // Convert from JSON
                    val project = gson.fromJson(json, Project::class.java)

                    // Set as current project
                    _currentProject.value = project
                    _isProjectModified.value = true

                    // Emit project event
                    _projectEvents.value = ProjectEvent.ProjectImported(project)

                    return@withContext true
                }
                "csv" -> {
                    // Read CSV from file
                    val lines = file.readLines()

                    // Parse CSV
                    val parameters = mutableMapOf<String, String>()
                    for (i in 1 until lines.size) {
                        val parts = lines[i].split(",")
                        if (parts.size >= 2) {
                            parameters[parts[0]] = parts[1]
                        }
                    }

                    // Create new project
                    val project =
                        Project(
                            name = file.nameWithoutExtension,
                            parameters = parameters,
                            metadata =
                            ProjectMetadata(
                                createdAt = System.currentTimeMillis(),
                                modifiedAt = System.currentTimeMillis(),
                            ),
                        )

                    // Set as current project
                    _currentProject.value = project
                    _isProjectModified.value = true

                    // Emit project event
                    _projectEvents.value = ProjectEvent.ProjectImported(project)

                    return@withContext true
                }
                else -> {
                    // Unsupported format
                    _projectEvents.value = ProjectEvent.Error("Unsupported import format: $format")
                    return@withContext false
                }
            }
        } catch (e: Exception) {
            // Emit error event
            _projectEvents.value = ProjectEvent.Error("Failed to import project: ${e.message}")

            return@withContext false
        }
    }

    /**
     * Create a backup of the current project.
     *
     * @return The backup file, or null if the backup failed
     */
    suspend fun createBackup(): File? = withContext(Dispatchers.IO) {
        val currentProject = _currentProject.value ?: return@withContext null

        try {
            // Create backup directory
            val backupDir = File(System.getProperty("user.home"), ".campro/backups")
            backupDir.mkdirs()

            // Create backup file
            val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss").format(Date())
            val backupFile = File(backupDir, "${currentProject.name}_$timestamp.json")

            // Convert to JSON
            val json = gson.toJson(currentProject)

            // Write to file
            backupFile.writeText(json)

            // Emit project event
            _projectEvents.value = ProjectEvent.ProjectBackupCreated(currentProject, backupFile.absolutePath)

            return@withContext backupFile
        } catch (e: Exception) {
            // Emit error event
            _projectEvents.value = ProjectEvent.Error("Failed to create backup: ${e.message}")

            return@withContext null
        }
    }

    /**
     * Restore a project from a backup.
     *
     * @param backupFile The backup file to restore from
     * @return True if the restore was successful, false otherwise
     */
    suspend fun restoreBackup(backupFile: File): Boolean = withContext(Dispatchers.IO) {
        return@withContext loadProject(backupFile)
    }

    /**
     * List all available backups.
     *
     * @return A list of backup files
     */
    suspend fun listBackups(): List<File> = withContext(Dispatchers.IO) {
        val backupDir = File(System.getProperty("user.home"), ".campro/backups")
        if (!backupDir.exists() || !backupDir.isDirectory) {
            return@withContext emptyList()
        }

        return@withContext backupDir
            .listFiles { file ->
                file.isFile && file.name.endsWith(".json")
            }?.toList() ?: emptyList()
    }

    companion object {
        // Singleton instance
        private var instance: ProjectManager? = null

        /**
         * Get the singleton instance of the ProjectManager.
         *
         * @return The ProjectManager instance
         */
        fun getInstance(): ProjectManager {
            if (instance == null) {
                instance = ProjectManager()
            }
            return instance!!
        }
    }
}

/**
 * A project.
 *
 * @param name The name of the project
 * @param parameters The parameters of the project
 * @param metadata The metadata of the project
 * @param filePath The file path of the project
 */
data class Project(
    val name: String,
    val parameters: MutableMap<String, String>,
    var metadata: ProjectMetadata,
    var filePath: String? = null,
)

/**
 * Project metadata.
 *
 * @param createdAt The time the project was created
 * @param modifiedAt The time the project was last modified
 * @param author The author of the project
 * @param description The description of the project
 * @param tags The tags of the project
 * @param templateId The ID of the template used to create the project
 */
data class ProjectMetadata(
    val createdAt: Long,
    var modifiedAt: Long,
    val author: String = "",
    val description: String = "",
    val tags: List<String> = emptyList(),
    val templateId: String = "",
)

/**
 * A project template.
 *
 * @param id The unique ID of the template
 * @param name The name of the template
 * @param description The description of the template
 * @param parameters The default parameters of the template
 */
data class ProjectTemplate(val id: String, val name: String, val description: String, val parameters: Map<String, String>)

/**
 * Project events emitted by the ProjectManager.
 */
sealed class ProjectEvent {
    /**
     * Event emitted when a project is created.
     *
     * @param project The created project
     */
    data class ProjectCreated(val project: Project) : ProjectEvent()

    /**
     * Event emitted when a project is loaded.
     *
     * @param project The loaded project
     */
    data class ProjectLoaded(val project: Project) : ProjectEvent()

    /**
     * Event emitted when a project is saved.
     *
     * @param project The saved project
     */
    data class ProjectSaved(val project: Project) : ProjectEvent()

    /**
     * Event emitted when a project is closed.
     *
     * @param project The closed project
     */
    data class ProjectClosed(val project: Project) : ProjectEvent()

    /**
     * Event emitted when a project is updated.
     *
     * @param project The updated project
     */
    data class ProjectUpdated(val project: Project) : ProjectEvent()

    /**
     * Event emitted when a project is exported.
     *
     * @param project The exported project
     * @param filePath The path to the exported file
     * @param format The format of the exported file
     */
    data class ProjectExported(val project: Project, val filePath: String, val format: String) : ProjectEvent()

    /**
     * Event emitted when a project is imported.
     *
     * @param project The imported project
     */
    data class ProjectImported(val project: Project) : ProjectEvent()

    /**
     * Event emitted when a project backup is created.
     *
     * @param project The project
     * @param backupPath The path to the backup file
     */
    data class ProjectBackupCreated(val project: Project, val backupPath: String) : ProjectEvent()

    /**
     * Event emitted when an error occurs.
     *
     * @param message The error message
     */
    data class Error(val message: String) : ProjectEvent()
}

/**
 * Composable function to remember a ProjectManager instance.
 *
 * @return The remembered ProjectManager instance
 */
@Composable
fun rememberProjectManager(): ProjectManager = remember { ProjectManager.getInstance() }
