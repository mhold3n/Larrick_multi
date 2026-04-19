package com.campro.v5.collaboration

import androidx.compose.runtime.mutableStateOf
import com.campro.v5.layout.StateManager
import com.google.gson.GsonBuilder
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.awt.Color
import java.util.Date

/**
 * Manages annotation functionality for the CamPro v5 application.
 * This class provides comprehensive annotation capabilities including
 * comments, markup, reviews, collaborative feedback, and annotation management.
 */
class AnnotationManager {
    // Annotation state
    private val _annotations = MutableStateFlow<List<Annotation>>(emptyList())
    val annotations: StateFlow<List<Annotation>> = _annotations.asStateFlow()

    // Active annotation session
    private val _activeSession = mutableStateOf<AnnotationSession?>(null)
    val activeSession: AnnotationSession?
        get() = _activeSession.value
    private val _isAnnotating = mutableStateOf(false)
    val isAnnotating: Boolean
        get() = _isAnnotating.value

    // Annotation events
    private val _annotationEvents = MutableStateFlow<AnnotationEvent?>(null)
    val annotationEvents: StateFlow<AnnotationEvent?> = _annotationEvents.asStateFlow()

    // Annotation filters
    private val _annotationFilters = MutableStateFlow(AnnotationFilters())
    val annotationFilters: StateFlow<AnnotationFilters> = _annotationFilters.asStateFlow()

    // Annotation types
    private val supportedTypes =
        mapOf(
            "comment" to AnnotationType("Comment", "Text-based comment", Color.YELLOW),
            "highlight" to AnnotationType("Highlight", "Highlight important areas", Color.CYAN),
            "markup" to AnnotationType("Markup", "Visual markup and drawings", Color.RED),
            "note" to AnnotationType("Note", "Detailed notes and explanations", Color.GREEN),
            "question" to AnnotationType("Question", "Questions and inquiries", Color.ORANGE),
            "suggestion" to AnnotationType("Suggestion", "Improvement suggestions", Color.BLUE),
            "issue" to AnnotationType("Issue", "Problems and issues", Color.MAGENTA),
            "approval" to AnnotationType("Approval", "Approval and sign-off", Color.LIGHT_GRAY),
        )

    // State manager for persistence
    private val stateManager = StateManager.getInstance()

    // Coroutine scope for async operations
    private val scope = CoroutineScope(Dispatchers.IO)

    // JSON serialization
    private val gson = GsonBuilder().setPrettyPrinting().create()

    init {
        // Load existing annotations
        loadAnnotations()

        // Load annotation preferences
        loadAnnotationPreferences()
    }

    /**
     * Start an annotation session for a project.
     */
    suspend fun startAnnotationSession(
        projectId: String,
        projectName: String,
        options: AnnotationSessionOptions = AnnotationSessionOptions(),
    ): AnnotationSession = withContext(Dispatchers.IO) {
        val session =
            AnnotationSession(
                id = generateSessionId(),
                projectId = projectId,
                projectName = projectName,
                owner = getCurrentUser(),
                startTime = Date(),
                isActive = true,
                options = options,
            )

        _activeSession.value = session
        _isAnnotating.value = true

        emitEvent(AnnotationEvent.SessionStarted(session.id, projectName))

        session
    }

    /**
     * End the current annotation session.
     */
    suspend fun endAnnotationSession(): Boolean = withContext(Dispatchers.IO) {
        val session = _activeSession.value
        if (session != null) {
            val updatedSession =
                session.copy(
                    isActive = false,
                    endTime = Date(),
                )

            _activeSession.value = null
            _isAnnotating.value = false

            // Save session summary
            saveAnnotationSession(updatedSession)

            emitEvent(AnnotationEvent.SessionEnded(session.id))
            true
        } else {
            false
        }
    }

    /**
     * Add a new annotation.
     */
    suspend fun addAnnotation(
        type: String,
        content: String,
        position: AnnotationPosition,
        metadata: Map<String, Any> = emptyMap(),
    ): Annotation = withContext(Dispatchers.IO) {
        val annotation =
            Annotation(
                id = generateAnnotationId(),
                type = type,
                content = content,
                position = position,
                author = getCurrentUser(),
                timestamp = Date(),
                projectId = _activeSession.value?.projectId ?: "",
                sessionId = _activeSession.value?.id,
                metadata = metadata,
                status = AnnotationStatus.ACTIVE,
            )

        val currentAnnotations = _annotations.value.toMutableList()
        currentAnnotations.add(annotation)
        _annotations.value = currentAnnotations

        // Save to persistent storage
        saveAnnotations()

        emitEvent(AnnotationEvent.AnnotationAdded(annotation.id, type))

        annotation
    }

    /**
     * Update an existing annotation.
     */
    suspend fun updateAnnotation(
        annotationId: String,
        content: String? = null,
        position: AnnotationPosition? = null,
        metadata: Map<String, Any>? = null,
    ): Boolean = withContext(Dispatchers.IO) {
        val currentAnnotations = _annotations.value.toMutableList()
        val index = currentAnnotations.indexOfFirst { it.id == annotationId }

        if (index >= 0) {
            val existingAnnotation = currentAnnotations[index]
            val updatedAnnotation =
                existingAnnotation.copy(
                    content = content ?: existingAnnotation.content,
                    position = position ?: existingAnnotation.position,
                    metadata = metadata ?: existingAnnotation.metadata,
                    lastModified = Date(),
                )

            currentAnnotations[index] = updatedAnnotation
            _annotations.value = currentAnnotations

            saveAnnotations()
            emitEvent(AnnotationEvent.AnnotationUpdated(annotationId))

            true
        } else {
            false
        }
    }

    /**
     * Delete an annotation.
     */
    suspend fun deleteAnnotation(annotationId: String): Boolean = withContext(Dispatchers.IO) {
        val currentAnnotations = _annotations.value.toMutableList()
        val annotation = currentAnnotations.find { it.id == annotationId }

        if (annotation != null) {
            currentAnnotations.remove(annotation)
            _annotations.value = currentAnnotations

            saveAnnotations()
            emitEvent(AnnotationEvent.AnnotationDeleted(annotationId))

            true
        } else {
            false
        }
    }

    /**
     * Add a reply to an annotation.
     */
    suspend fun addReply(annotationId: String, content: String, metadata: Map<String, Any> = emptyMap()): AnnotationReply? =
        withContext(Dispatchers.IO) {
            val currentAnnotations = _annotations.value.toMutableList()
            val index = currentAnnotations.indexOfFirst { it.id == annotationId }

            if (index >= 0) {
                val reply =
                    AnnotationReply(
                        id = generateReplyId(),
                        content = content,
                        author = getCurrentUser(),
                        timestamp = Date(),
                        metadata = metadata,
                    )

                val existingAnnotation = currentAnnotations[index]
                val updatedAnnotation =
                    existingAnnotation.copy(
                        replies = existingAnnotation.replies + reply,
                        lastModified = Date(),
                    )

                currentAnnotations[index] = updatedAnnotation
                _annotations.value = currentAnnotations

                saveAnnotations()
                emitEvent(AnnotationEvent.ReplyAdded(annotationId, reply.id))

                reply
            } else {
                null
            }
        }

    /**
     * Resolve an annotation.
     */
    suspend fun resolveAnnotation(annotationId: String, resolution: String): Boolean = withContext(Dispatchers.IO) {
        val currentAnnotations = _annotations.value.toMutableList()
        val index = currentAnnotations.indexOfFirst { it.id == annotationId }

        if (index >= 0) {
            val existingAnnotation = currentAnnotations[index]
            val updatedAnnotation =
                existingAnnotation.copy(
                    status = AnnotationStatus.RESOLVED,
                    resolution = resolution,
                    resolvedBy = getCurrentUser(),
                    resolvedAt = Date(),
                    lastModified = Date(),
                )

            currentAnnotations[index] = updatedAnnotation
            _annotations.value = currentAnnotations

            saveAnnotations()
            emitEvent(AnnotationEvent.AnnotationResolved(annotationId))

            true
        } else {
            false
        }
    }

    /**
     * Archive an annotation.
     */
    suspend fun archiveAnnotation(annotationId: String): Boolean = withContext(Dispatchers.IO) {
        val currentAnnotations = _annotations.value.toMutableList()
        val index = currentAnnotations.indexOfFirst { it.id == annotationId }

        if (index >= 0) {
            val existingAnnotation = currentAnnotations[index]
            val updatedAnnotation =
                existingAnnotation.copy(
                    status = AnnotationStatus.ARCHIVED,
                    lastModified = Date(),
                )

            currentAnnotations[index] = updatedAnnotation
            _annotations.value = currentAnnotations

            saveAnnotations()
            emitEvent(AnnotationEvent.AnnotationArchived(annotationId))

            true
        } else {
            false
        }
    }

    /**
     * Get annotations for a specific project.
     */
    fun getAnnotationsForProject(projectId: String): List<Annotation> = _annotations.value.filter { it.projectId == projectId }

    /**
     * Get annotations by type.
     */
    fun getAnnotationsByType(type: String): List<Annotation> = _annotations.value.filter { it.type == type }

    /**
     * Get annotations by author.
     */
    fun getAnnotationsByAuthor(authorId: String): List<Annotation> = _annotations.value.filter { it.author.id == authorId }

    /**
     * Get annotations by status.
     */
    fun getAnnotationsByStatus(status: AnnotationStatus): List<Annotation> = _annotations.value.filter { it.status == status }

    /**
     * Search annotations by content.
     */
    fun searchAnnotations(query: String): List<Annotation> {
        val lowercaseQuery = query.lowercase()
        return _annotations.value.filter { annotation ->
            annotation.content.lowercase().contains(lowercaseQuery) ||
                annotation.replies.any { it.content.lowercase().contains(lowercaseQuery) }
        }
    }

    /**
     * Apply filters to annotations.
     */
    fun applyFilters(filters: AnnotationFilters) {
        _annotationFilters.value = filters
        emitEvent(AnnotationEvent.FiltersApplied(filters))
    }

    /**
     * Get filtered annotations.
     */
    fun getFilteredAnnotations(): List<Annotation> {
        val filters = _annotationFilters.value
        var filteredAnnotations = _annotations.value

        // Filter by project
        if (filters.projectId != null) {
            filteredAnnotations = filteredAnnotations.filter { it.projectId == filters.projectId }
        }

        // Filter by type
        if (filters.types.isNotEmpty()) {
            filteredAnnotations = filteredAnnotations.filter { it.type in filters.types }
        }

        // Filter by author
        if (filters.authorIds.isNotEmpty()) {
            filteredAnnotations = filteredAnnotations.filter { it.author.id in filters.authorIds }
        }

        // Filter by status
        if (filters.statuses.isNotEmpty()) {
            filteredAnnotations = filteredAnnotations.filter { it.status in filters.statuses }
        }

        // Filter by date range
        if (filters.dateFrom != null) {
            filteredAnnotations = filteredAnnotations.filter { it.timestamp >= filters.dateFrom }
        }
        if (filters.dateTo != null) {
            filteredAnnotations = filteredAnnotations.filter { it.timestamp <= filters.dateTo }
        }

        // Filter by search query
        if (filters.searchQuery.isNotEmpty()) {
            val query = filters.searchQuery.lowercase()
            filteredAnnotations =
                filteredAnnotations.filter { annotation ->
                    annotation.content.lowercase().contains(query) ||
                        annotation.replies.any { it.content.lowercase().contains(query) }
                }
        }

        return filteredAnnotations
    }

    /**
     * Export annotations to various formats.
     */
    suspend fun exportAnnotations(projectId: String, format: String, filePath: String): Boolean = withContext(Dispatchers.IO) {
        try {
            val annotations = getAnnotationsForProject(projectId)

            when (format.lowercase()) {
                "json" -> exportToJson(annotations, filePath)
                "csv" -> exportToCsv(annotations, filePath)
                "html" -> exportToHtml(annotations, filePath)
                "pdf" -> exportToPdf(annotations, filePath)
                else -> return@withContext false
            }

            emitEvent(AnnotationEvent.AnnotationsExported(format, filePath))
            true
        } catch (e: Exception) {
            emitEvent(AnnotationEvent.ExportFailed(format, e.message ?: "Unknown error"))
            false
        }
    }

    /**
     * Import annotations from file.
     */
    suspend fun importAnnotations(filePath: String, format: String): Boolean = withContext(Dispatchers.IO) {
        try {
            val importedAnnotations =
                when (format.lowercase()) {
                    "json" -> importFromJson(filePath)
                    "csv" -> importFromCsv(filePath)
                    else -> return@withContext false
                }

            val currentAnnotations = _annotations.value.toMutableList()
            currentAnnotations.addAll(importedAnnotations)
            _annotations.value = currentAnnotations

            saveAnnotations()
            emitEvent(AnnotationEvent.AnnotationsImported(format, importedAnnotations.size))

            true
        } catch (e: Exception) {
            emitEvent(AnnotationEvent.ImportFailed(format, e.message ?: "Unknown error"))
            false
        }
    }

    /**
     * Get annotation statistics.
     */
    fun getAnnotationStatistics(projectId: String? = null): AnnotationStatistics {
        val annotations =
            if (projectId != null) {
                getAnnotationsForProject(projectId)
            } else {
                _annotations.value
            }

        val typeCount = annotations.groupBy { it.type }.mapValues { it.value.size }
        val statusCount = annotations.groupBy { it.status }.mapValues { it.value.size }
        val authorCount = annotations.groupBy { it.author.id }.mapValues { it.value.size }

        return AnnotationStatistics(
            totalAnnotations = annotations.size,
            typeBreakdown = typeCount,
            statusBreakdown = statusCount,
            authorBreakdown = authorCount,
            averageRepliesPerAnnotation =
            if (annotations.isNotEmpty()) {
                annotations.sumOf { it.replies.size }.toDouble() / annotations.size
            } else {
                0.0
            },
            oldestAnnotation = annotations.minByOrNull { it.timestamp }?.timestamp,
            newestAnnotation = annotations.maxByOrNull { it.timestamp }?.timestamp,
        )
    }

    /**
     * Get supported annotation types.
     */
    fun getSupportedTypes(): Map<String, AnnotationType> = supportedTypes

    /**
     * Get current annotation session.
     */
    fun getCurrentSession(): AnnotationSession? = _activeSession.value

    // Helper methods
    private fun loadAnnotations() {
        val annotationsJson = stateManager.getState("annotations.data", "[]")
        try {
            val annotations = gson.fromJson(annotationsJson, Array<Annotation>::class.java).toList()
            _annotations.value = annotations
        } catch (e: Exception) {
            _annotations.value = emptyList()
        }
    }

    private fun saveAnnotations() {
        val annotationsJson = gson.toJson(_annotations.value)
        stateManager.setState("annotations.data", annotationsJson)
    }

    private fun loadAnnotationPreferences() {
        // Load user preferences for annotations
        val defaultType = stateManager.getState("annotations.defaultType", "comment")
        val showResolved = stateManager.getState("annotations.showResolved", "true").toBoolean()
    }

    private fun saveAnnotationSession(session: AnnotationSession) {
        val sessionsJson = stateManager.getState("annotations.sessions", "[]")
        try {
            val sessions = gson.fromJson(sessionsJson, Array<AnnotationSession>::class.java).toMutableList()
            sessions.add(session)

            // Keep only last 50 sessions
            if (sessions.size > 50) {
                sessions.removeAt(0)
            }

            stateManager.setState("annotations.sessions", gson.toJson(sessions))
        } catch (e: Exception) {
            stateManager.setState("annotations.sessions", gson.toJson(listOf(session)))
        }
    }

    private fun generateAnnotationId(): String = "ann_${System.currentTimeMillis()}_${(1000..9999).random()}"

    private fun generateReplyId(): String = "reply_${System.currentTimeMillis()}_${(100..999).random()}"

    private fun generateSessionId(): String = "session_${System.currentTimeMillis()}_${(1000..9999).random()}"

    private fun getCurrentUser(): User = User(
        id = stateManager.getState("user.id", "anonymous"),
        name = stateManager.getState("user.name", "Anonymous User"),
        email = stateManager.getState("user.email", ""),
    )

    private fun emitEvent(event: AnnotationEvent) {
        scope.launch {
            _annotationEvents.value = event
        }
    }

    // Export methods
    private suspend fun exportToJson(annotations: List<Annotation>, filePath: String) {
        val file = java.io.File(filePath)
        file.parentFile?.mkdirs()
        file.writeText(gson.toJson(annotations))
    }

    private suspend fun exportToCsv(annotations: List<Annotation>, filePath: String) {
        val file = java.io.File(filePath)
        file.parentFile?.mkdirs()

        val csvContent = StringBuilder()
        csvContent.append("ID,Type,Content,Author,Timestamp,Status,Project ID,Replies Count\n")

        annotations.forEach { annotation ->
            csvContent.append("\"${annotation.id}\",")
            csvContent.append("\"${annotation.type}\",")
            csvContent.append("\"${annotation.content.replace("\"", "\"\"")}\",")
            csvContent.append("\"${annotation.author.name}\",")
            csvContent.append("\"${annotation.timestamp}\",")
            csvContent.append("\"${annotation.status}\",")
            csvContent.append("\"${annotation.projectId}\",")
            csvContent.append("\"${annotation.replies.size}\"\n")
        }

        file.writeText(csvContent.toString())
    }

    private suspend fun exportToHtml(annotations: List<Annotation>, filePath: String) {
        val file = java.io.File(filePath)
        file.parentFile?.mkdirs()

        val htmlContent = StringBuilder()
        htmlContent.append("<!DOCTYPE html>\n<html>\n<head>\n<title>CamPro v5 Annotations</title>\n</head>\n<body>\n")
        htmlContent.append("<h1>Project Annotations</h1>\n")

        annotations.forEach { annotation ->
            htmlContent.append("<div style='border: 1px solid #ccc; margin: 10px; padding: 10px;'>\n")
            htmlContent.append("<h3>${annotation.type.uppercase()} - ${annotation.id}</h3>\n")
            htmlContent.append("<p><strong>Author:</strong> ${annotation.author.name}</p>\n")
            htmlContent.append("<p><strong>Date:</strong> ${annotation.timestamp}</p>\n")
            htmlContent.append("<p><strong>Status:</strong> ${annotation.status}</p>\n")
            htmlContent.append("<p><strong>Content:</strong> ${annotation.content}</p>\n")

            if (annotation.replies.isNotEmpty()) {
                htmlContent.append("<h4>Replies:</h4>\n")
                annotation.replies.forEach { reply ->
                    htmlContent.append("<div style='margin-left: 20px; border-left: 2px solid #eee; padding-left: 10px;'>\n")
                    htmlContent.append("<p><strong>${reply.author.name}</strong> (${reply.timestamp}):</p>\n")
                    htmlContent.append("<p>${reply.content}</p>\n")
                    htmlContent.append("</div>\n")
                }
            }

            htmlContent.append("</div>\n")
        }

        htmlContent.append("</body>\n</html>")
        file.writeText(htmlContent.toString())
    }

    private suspend fun exportToPdf(annotations: List<Annotation>, filePath: String) {
        // Placeholder implementation - would use a PDF library in real implementation
        val file = java.io.File(filePath)
        file.parentFile?.mkdirs()

        val pdfContent = StringBuilder()
        pdfContent.append("CamPro v5 Annotations Report\n")
        pdfContent.append("============================\n\n")

        annotations.forEach { annotation ->
            pdfContent.append("${annotation.type.uppercase()}: ${annotation.id}\n")
            pdfContent.append("Author: ${annotation.author.name}\n")
            pdfContent.append("Date: ${annotation.timestamp}\n")
            pdfContent.append("Status: ${annotation.status}\n")
            pdfContent.append("Content: ${annotation.content}\n")

            if (annotation.replies.isNotEmpty()) {
                pdfContent.append("Replies:\n")
                annotation.replies.forEach { reply ->
                    pdfContent.append("  - ${reply.author.name} (${reply.timestamp}): ${reply.content}\n")
                }
            }

            pdfContent.append("\n" + "-".repeat(50) + "\n\n")
        }

        file.writeText(pdfContent.toString())
    }

    // Import methods
    private suspend fun importFromJson(filePath: String): List<Annotation> {
        val file = java.io.File(filePath)
        val jsonContent = file.readText()
        return gson.fromJson(jsonContent, Array<Annotation>::class.java).toList()
    }

    private suspend fun importFromCsv(filePath: String): List<Annotation> {
        // Simplified CSV import - would need proper CSV parsing in real implementation
        val file = java.io.File(filePath)
        val lines = file.readLines().drop(1) // Skip header

        return lines.mapNotNull { line ->
            try {
                val parts = line.split(",")
                if (parts.size >= 6) {
                    Annotation(
                        id = parts[0].trim('"'),
                        type = parts[1].trim('"'),
                        content = parts[2].trim('"'),
                        position = AnnotationPosition(0.0, 0.0), // Default position for imported annotations
                        author = User("imported", "Imported User", ""),
                        timestamp = Date(),
                        projectId = parts[5].trim('"'),
                        sessionId = null,
                        metadata = emptyMap(),
                        status = AnnotationStatus.valueOf(parts[4].trim('"')),
                    )
                } else {
                    null
                }
            } catch (e: Exception) {
                null
            }
        }
    }

    /**
     * Reset the annotation manager state.
     * This is primarily used for testing to ensure a clean state between tests.
     */
    fun resetState() {
        _annotations.value = emptyList()
        _activeSession.value = null
        _isAnnotating.value = false
        _annotationEvents.value = null
        _annotationFilters.value = AnnotationFilters()
    }

    companion object {
        @Volatile
        private var INSTANCE: AnnotationManager? = null

        fun getInstance(): AnnotationManager = INSTANCE ?: synchronized(this) {
            INSTANCE ?: AnnotationManager().also { INSTANCE = it }
        }
    }
}

// Data classes
data class AnnotationType(val name: String, val description: String, val color: Color)

data class AnnotationSessionOptions(
    val allowCollaboration: Boolean = true,
    val autoSave: Boolean = true,
    val notifyOnChanges: Boolean = true,
)

data class AnnotationSession(
    val id: String,
    val projectId: String,
    val projectName: String,
    val owner: User,
    val startTime: Date,
    val endTime: Date? = null,
    val isActive: Boolean,
    val options: AnnotationSessionOptions,
)

data class Annotation(
    val id: String,
    val type: String,
    val content: String,
    val position: AnnotationPosition,
    val author: User,
    val timestamp: Date,
    val projectId: String,
    val sessionId: String?,
    val metadata: Map<String, Any>,
    val status: AnnotationStatus,
    val replies: List<AnnotationReply> = emptyList(),
    val lastModified: Date? = null,
    val resolution: String? = null,
    val resolvedBy: User? = null,
    val resolvedAt: Date? = null,
)

data class AnnotationPosition(
    val x: Double,
    val y: Double,
    val width: Double? = null,
    val height: Double? = null,
    val page: Int? = null,
    val component: String? = null,
)

data class AnnotationReply(
    val id: String,
    val content: String,
    val author: User,
    val timestamp: Date,
    val metadata: Map<String, Any> = emptyMap(),
)

data class AnnotationFilters(
    val projectId: String? = null,
    val types: Set<String> = emptySet(),
    val authorIds: Set<String> = emptySet(),
    val statuses: Set<AnnotationStatus> = emptySet(),
    val dateFrom: Date? = null,
    val dateTo: Date? = null,
    val searchQuery: String = "",
)

data class AnnotationStatistics(
    val totalAnnotations: Int,
    val typeBreakdown: Map<String, Int>,
    val statusBreakdown: Map<AnnotationStatus, Int>,
    val authorBreakdown: Map<String, Int>,
    val averageRepliesPerAnnotation: Double,
    val oldestAnnotation: Date?,
    val newestAnnotation: Date?,
)

enum class AnnotationStatus {
    ACTIVE,
    RESOLVED,
    ARCHIVED,
    DELETED,
}

sealed class AnnotationEvent {
    data class SessionStarted(val sessionId: String, val projectName: String) : AnnotationEvent()

    data class SessionEnded(val sessionId: String) : AnnotationEvent()

    data class AnnotationAdded(val annotationId: String, val type: String) : AnnotationEvent()

    data class AnnotationUpdated(val annotationId: String) : AnnotationEvent()

    data class AnnotationDeleted(val annotationId: String) : AnnotationEvent()

    data class AnnotationResolved(val annotationId: String) : AnnotationEvent()

    data class AnnotationArchived(val annotationId: String) : AnnotationEvent()

    data class ReplyAdded(val annotationId: String, val replyId: String) : AnnotationEvent()

    data class FiltersApplied(val filters: AnnotationFilters) : AnnotationEvent()

    data class AnnotationsExported(val format: String, val filePath: String) : AnnotationEvent()

    data class AnnotationsImported(val format: String, val count: Int) : AnnotationEvent()

    data class ExportFailed(val format: String, val error: String) : AnnotationEvent()

    data class ImportFailed(val format: String, val error: String) : AnnotationEvent()
}
