package com.campro.v5.mvvm

import androidx.compose.runtime.*
import com.campro.v5.file.ProjectManager
import com.campro.v5.file.Project
import com.campro.v5.file.ProjectTemplate
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import org.slf4j.LoggerFactory
import java.io.File

/**
 * ViewModel for project management.
 * 
 * Handles the state and business logic for project operations,
 * including creating, loading, saving, and managing projects.
 */
class ProjectViewModel : BaseViewModel() {
    private val logger = LoggerFactory.getLogger(ProjectViewModel::class.java)
    
    // Current project state
    private val _currentProject = MutableStateFlow<Project?>(null)
    val currentProject: StateFlow<Project?> = _currentProject.asStateFlow()
    
    // Project list state
    private val _recentProjects = MutableStateFlow<List<Project>>(emptyList())
    val recentProjects: StateFlow<List<Project>> = _recentProjects.asStateFlow()
    
    // Project templates
    private val _availableTemplates = MutableStateFlow<List<ProjectTemplate>>(emptyList())
    val availableTemplates: StateFlow<List<ProjectTemplate>> = _availableTemplates.asStateFlow()
    
    // Project manager
    private val projectManager = ProjectManager()
    
    init {
        logger.info("ProjectViewModel initialized")
        loadRecentProjects()
        loadAvailableTemplates()
    }
    
    /**
     * Create a new project
     */
    fun createNewProject(
        name: String,
        description: String = "",
        template: ProjectTemplate? = null
    ) {
        executeWithLoading(
            operation = {
                // Create new project
                val project = Project(
                    name = name,
                    description = description,
                    template = template,
                    createdDate = System.currentTimeMillis(),
                    lastModified = System.currentTimeMillis(),
                    filePath = null // Will be set when saved
                )
                
                _currentProject.value = project
                "Project '$name' created successfully"
            },
            onSuccess = { message ->
                setSuccess(message)
                logger.info("New project created: $name")
            }
        )
    }
    
    /**
     * Load a project from file
     */
    fun loadProject(filePath: String) {
        executeWithLoading(
            operation = {
                val file = File(filePath)
                if (!file.exists()) {
                    throw IllegalArgumentException("Project file does not exist: $filePath")
                }
                
                // In a real implementation, this would load the project from file
                // For now, we'll create a mock project
                kotlinx.coroutines.delay(500)
                
                val project = Project(
                    name = file.nameWithoutExtension,
                    description = "Loaded from $filePath",
                    template = null,
                    createdDate = file.lastModified(),
                    lastModified = file.lastModified(),
                    filePath = filePath
                )
                
                project
            },
            onSuccess = { project ->
                _currentProject.value = project
                addToRecentProjects(project)
                setSuccess("Project '${project.name}' loaded successfully")
                logger.info("Project loaded: ${project.name}")
            }
        )
    }
    
    /**
     * Save current project
     */
    fun saveProject(filePath: String? = null) {
        val project = _currentProject.value
        if (project == null) {
            setError("No project to save")
            return
        }
        
        executeWithLoading(
            operation = {
                val savePath = filePath ?: project.filePath
                if (savePath == null) {
                    throw IllegalArgumentException("No file path specified for saving")
                }
                
                val file = File(savePath)
                
                // In a real implementation, this would save the project to file
                // For now, we'll simulate the operation
                kotlinx.coroutines.delay(300)
                
                val updatedProject = project.copy(
                    filePath = savePath,
                    lastModified = System.currentTimeMillis()
                )
                
                updatedProject
            },
            onSuccess = { updatedProject ->
                _currentProject.value = updatedProject
                addToRecentProjects(updatedProject)
                setSuccess("Project '${updatedProject.name}' saved successfully")
                logger.info("Project saved: ${updatedProject.name}")
            }
        )
    }
    
    /**
     * Save project as new file
     */
    fun saveProjectAs(filePath: String) {
        saveProject(filePath)
    }
    
    /**
     * Close current project
     */
    fun closeProject() {
        _currentProject.value = null
        setSuccess("Project closed")
        logger.info("Project closed")
    }
    
    /**
     * Load recent projects
     */
    private fun loadRecentProjects() {
        executeWithLoading(
            operation = {
                // In a real implementation, this would load from persistent storage
                // For now, we'll create mock recent projects
                kotlinx.coroutines.delay(200)
                
                listOf(
                    Project(
                        name = "Recent Project 1",
                        description = "A recent project",
                        template = null,
                        createdDate = System.currentTimeMillis() - 86400000,
                        lastModified = System.currentTimeMillis() - 3600000,
                        filePath = "/path/to/recent1.campro"
                    ),
                    Project(
                        name = "Recent Project 2",
                        description = "Another recent project",
                        template = null,
                        createdDate = System.currentTimeMillis() - 172800000,
                        lastModified = System.currentTimeMillis() - 7200000,
                        filePath = "/path/to/recent2.campro"
                    )
                )
            },
            onSuccess = { projects ->
                _recentProjects.value = projects
                logger.debug("Loaded ${projects.size} recent projects")
            }
        )
    }
    
    /**
     * Load available project templates
     */
    private fun loadAvailableTemplates() {
        executeWithLoading(
            operation = {
                // In a real implementation, this would load from template directory
                // For now, we'll create mock templates
                kotlinx.coroutines.delay(200)
                
                listOf(
                    ProjectTemplate(
                        name = "Standard Cam Profile",
                        description = "Standard cam profile optimization template",
                        category = "Standard",
                        parameters = emptyMap()
                    ),
                    ProjectTemplate(
                        name = "High Performance",
                        description = "High performance cam profile template",
                        category = "Performance",
                        parameters = emptyMap()
                    ),
                    ProjectTemplate(
                        name = "Low Noise",
                        description = "Low noise cam profile template",
                        category = "Quiet",
                        parameters = emptyMap()
                    )
                )
            },
            onSuccess = { templates ->
                _availableTemplates.value = templates
                logger.debug("Loaded ${templates.size} project templates")
            }
        )
    }
    
    /**
     * Add project to recent projects list
     */
    private fun addToRecentProjects(project: Project) {
        val currentRecent = _recentProjects.value.toMutableList()
        
        // Remove if already exists
        currentRecent.removeAll { it.filePath == project.filePath }
        
        // Add to beginning
        currentRecent.add(0, project)
        
        // Keep only last 10 projects
        if (currentRecent.size > 10) {
            currentRecent.removeAt(currentRecent.size - 1)
        }
        
        _recentProjects.value = currentRecent
    }
    
    /**
     * Check if current project has unsaved changes
     */
    fun hasUnsavedChanges(): Boolean {
        // In a real implementation, this would check if the project has been modified
        // since last save
        return false
    }
    
    /**
     * Get project file extension
     */
    fun getProjectFileExtension(): String {
        return ".campro"
    }
    
    /**
     * Validate project name
     */
    fun validateProjectName(name: String): String? {
        return when {
            name.isBlank() -> "Project name cannot be empty"
            name.length < 3 -> "Project name must be at least 3 characters"
            name.length > 50 -> "Project name must be less than 50 characters"
            !name.matches(Regex("[a-zA-Z0-9\\s_-]+")) -> "Project name contains invalid characters"
            else -> null
        }
    }
}
