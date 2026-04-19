package com.campro.v5.collaboration

import com.google.gson.Gson
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertNotNull
import org.junit.jupiter.api.Assertions.assertNull
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.io.TempDir
import java.awt.Color
import java.io.File
import java.nio.file.Path
import java.util.Date

class AnnotationManagerTest {
    private lateinit var annotationManager: AnnotationManager
    private lateinit var testSession: AnnotationSession
    private lateinit var testPosition: AnnotationPosition

    @TempDir
    lateinit var tempDir: Path

    @BeforeEach
    fun setUp() = runBlocking {
        // Reset the annotation manager state completely
        annotationManager = AnnotationManager.getInstance()
        annotationManager.resetState()

        // Start a test annotation session
        testSession =
            annotationManager.startAnnotationSession(
                projectId = "test_project_123",
                projectName = "Test Annotation Project",
                options =
                AnnotationSessionOptions(
                    allowCollaboration = true,
                    autoSave = true,
                    notifyOnChanges = true,
                ),
            )

        // Create test position
        testPosition =
            AnnotationPosition(
                x = 100.0,
                y = 200.0,
                width = 50.0,
                height = 30.0,
                page = 1,
                component = "parameter_panel",
            )
    }

    @AfterEach
    fun tearDown() {
        runBlocking {
            // End annotation session and clean up
            annotationManager.endAnnotationSession()
        }
    }

    @Test
    fun `test start annotation session`() = runBlocking {
        val session =
            annotationManager.startAnnotationSession(
                projectId = "new_project_456",
                projectName = "New Test Project",
            )

        assertNotNull(session.id)
        assertEquals("new_project_456", session.projectId)
        assertEquals("New Test Project", session.projectName)
        assertTrue(session.isActive)
        assertNotNull(session.startTime)
        assertNull(session.endTime)

        assertTrue(annotationManager.isAnnotating())
        assertEquals(session, annotationManager.getCurrentSession())
    }

    @Test
    fun `test end annotation session`() = runBlocking {
        assertTrue(annotationManager.isAnnotating())

        val result = annotationManager.endAnnotationSession()

        assertTrue(result)
        assertFalse(annotationManager.isAnnotating())
        assertNull(annotationManager.getCurrentSession())
    }

    @Test
    fun `test add annotation`() = runBlocking {
        val annotation =
            annotationManager.addAnnotation(
                type = "comment",
                content = "This is a test comment",
                position = testPosition,
                metadata = mapOf("priority" to "high", "category" to "design"),
            )

        assertNotNull(annotation.id)
        assertEquals("comment", annotation.type)
        assertEquals("This is a test comment", annotation.content)
        assertEquals(testPosition, annotation.position)
        assertEquals("test_project_123", annotation.projectId)
        assertEquals(testSession.id, annotation.sessionId)
        assertEquals(AnnotationStatus.ACTIVE, annotation.status)
        assertEquals("high", annotation.metadata["priority"])
        assertEquals("design", annotation.metadata["category"])

        // Verify annotation is in the list
        val annotations = annotationManager.annotations.value
        assertTrue(annotations.contains(annotation))
    }

    @Test
    fun `test add different annotation types`() = runBlocking {
        val comment = annotationManager.addAnnotation("comment", "Comment text", testPosition)
        val highlight = annotationManager.addAnnotation("highlight", "Highlighted area", testPosition)
        val markup = annotationManager.addAnnotation("markup", "Markup drawing", testPosition)
        val note = annotationManager.addAnnotation("note", "Detailed note", testPosition)
        val question = annotationManager.addAnnotation("question", "What about this?", testPosition)
        val suggestion = annotationManager.addAnnotation("suggestion", "Try this instead", testPosition)
        val issue = annotationManager.addAnnotation("issue", "Problem found", testPosition)
        val approval = annotationManager.addAnnotation("approval", "Approved", testPosition)

        assertEquals("comment", comment.type)
        assertEquals("highlight", highlight.type)
        assertEquals("markup", markup.type)
        assertEquals("note", note.type)
        assertEquals("question", question.type)
        assertEquals("suggestion", suggestion.type)
        assertEquals("issue", issue.type)
        assertEquals("approval", approval.type)
    }

    @Test
    fun `test update annotation`() = runBlocking {
        val annotation =
            annotationManager.addAnnotation(
                type = "comment",
                content = "Original content",
                position = testPosition,
            )

        val newPosition = AnnotationPosition(x = 150.0, y = 250.0)
        val newMetadata = mapOf("updated" to "true")

        val result =
            annotationManager.updateAnnotation(
                annotationId = annotation.id,
                content = "Updated content",
                position = newPosition,
                metadata = newMetadata,
            )

        assertTrue(result)

        val updatedAnnotation = annotationManager.annotations.value.find { it.id == annotation.id }
        assertNotNull(updatedAnnotation)
        assertEquals("Updated content", updatedAnnotation!!.content)
        assertEquals(newPosition, updatedAnnotation.position)
        assertEquals(newMetadata, updatedAnnotation.metadata)
        assertNotNull(updatedAnnotation.lastModified)
    }

    @Test
    fun `test delete annotation`() = runBlocking {
        val annotation =
            annotationManager.addAnnotation(
                type = "comment",
                content = "To be deleted",
                position = testPosition,
            )

        val initialCount = annotationManager.annotations.value.size

        val result = annotationManager.deleteAnnotation(annotation.id)

        assertTrue(result)
        assertEquals(initialCount - 1, annotationManager.annotations.value.size)
        assertFalse(annotationManager.annotations.value.any { it.id == annotation.id })
    }

    @Test
    fun `test add reply to annotation`() = runBlocking {
        val annotation =
            annotationManager.addAnnotation(
                type = "question",
                content = "What do you think about this?",
                position = testPosition,
            )

        val reply =
            annotationManager.addReply(
                annotationId = annotation.id,
                content = "I think it looks good",
                metadata = mapOf("sentiment" to "positive"),
            )

        assertNotNull(reply)
        assertNotNull(reply!!.id)
        assertEquals("I think it looks good", reply.content)
        assertEquals("positive", reply.metadata["sentiment"])

        val updatedAnnotation = annotationManager.annotations.value.find { it.id == annotation.id }
        assertNotNull(updatedAnnotation)
        assertEquals(1, updatedAnnotation!!.replies.size)
        assertEquals(reply, updatedAnnotation.replies.first())
    }

    @Test
    fun `test resolve annotation`() = runBlocking {
        val annotation =
            annotationManager.addAnnotation(
                type = "issue",
                content = "This needs to be fixed",
                position = testPosition,
            )

        val result =
            annotationManager.resolveAnnotation(
                annotationId = annotation.id,
                resolution = "Fixed by updating the parameter value",
            )

        assertTrue(result)

        val resolvedAnnotation = annotationManager.annotations.value.find { it.id == annotation.id }
        assertNotNull(resolvedAnnotation)
        assertEquals(AnnotationStatus.RESOLVED, resolvedAnnotation!!.status)
        assertEquals("Fixed by updating the parameter value", resolvedAnnotation.resolution)
        assertNotNull(resolvedAnnotation.resolvedBy)
        assertNotNull(resolvedAnnotation.resolvedAt)
    }

    @Test
    fun `test archive annotation`() = runBlocking {
        val annotation =
            annotationManager.addAnnotation(
                type = "comment",
                content = "Old comment",
                position = testPosition,
            )

        val result = annotationManager.archiveAnnotation(annotation.id)

        assertTrue(result)

        val archivedAnnotation = annotationManager.annotations.value.find { it.id == annotation.id }
        assertNotNull(archivedAnnotation)
        assertEquals(AnnotationStatus.ARCHIVED, archivedAnnotation!!.status)
    }

    @Test
    fun `test get annotations for project`() = runBlocking {
        // Add annotations for different projects
        annotationManager.addAnnotation("comment", "Project 1 comment", testPosition)

        // Start session for different project
        annotationManager.endAnnotationSession()
        annotationManager.startAnnotationSession("project_2", "Project 2")
        annotationManager.addAnnotation("note", "Project 2 note", testPosition)

        val project1Annotations = annotationManager.getAnnotationsForProject("test_project_123")
        val project2Annotations = annotationManager.getAnnotationsForProject("project_2")

        assertEquals(1, project1Annotations.size)
        assertEquals("comment", project1Annotations.first().type)

        assertEquals(1, project2Annotations.size)
        assertEquals("note", project2Annotations.first().type)
    }

    @Test
    fun `test get annotations by type`() = runBlocking {
        annotationManager.addAnnotation("comment", "Comment 1", testPosition)
        annotationManager.addAnnotation("comment", "Comment 2", testPosition)
        annotationManager.addAnnotation("note", "Note 1", testPosition)

        val comments = annotationManager.getAnnotationsByType("comment")
        val notes = annotationManager.getAnnotationsByType("note")

        assertEquals(2, comments.size)
        assertTrue(comments.all { it.type == "comment" })

        assertEquals(1, notes.size)
        assertEquals("note", notes.first().type)
    }

    @Test
    fun `test get annotations by status`() = runBlocking {
        val annotation1 = annotationManager.addAnnotation("comment", "Active comment", testPosition)
        val annotation2 = annotationManager.addAnnotation("issue", "Issue to resolve", testPosition)

        annotationManager.resolveAnnotation(annotation2.id, "Resolved")

        val activeAnnotations = annotationManager.getAnnotationsByStatus(AnnotationStatus.ACTIVE)
        val resolvedAnnotations = annotationManager.getAnnotationsByStatus(AnnotationStatus.RESOLVED)

        assertEquals(1, activeAnnotations.size)
        assertEquals(annotation1.id, activeAnnotations.first().id)

        assertEquals(1, resolvedAnnotations.size)
        assertEquals(annotation2.id, resolvedAnnotations.first().id)
    }

    @Test
    fun `test search annotations`() = runBlocking {
        annotationManager.addAnnotation("comment", "This is about performance", testPosition)
        annotationManager.addAnnotation("note", "Performance optimization needed", testPosition)
        annotationManager.addAnnotation("comment", "UI design looks good", testPosition)

        val performanceAnnotations = annotationManager.searchAnnotations("performance")
        val designAnnotations = annotationManager.searchAnnotations("design")

        assertEquals(2, performanceAnnotations.size)
        assertTrue(
            performanceAnnotations.all {
                it.content.lowercase().contains("performance")
            },
        )

        assertEquals(1, designAnnotations.size)
        assertTrue(designAnnotations.first().content.contains("design"))
    }

    @Test
    fun `test annotation filters`() = runBlocking {
        // Add test data
        annotationManager.addAnnotation("comment", "Test comment", testPosition)
        annotationManager.addAnnotation("note", "Test note", testPosition)

        val filters =
            AnnotationFilters(
                projectId = "test_project_123",
                types = setOf("comment"),
                statuses = setOf(AnnotationStatus.ACTIVE),
                searchQuery = "test",
            )

        annotationManager.applyFilters(filters)
        val filteredAnnotations = annotationManager.getFilteredAnnotations()

        assertTrue(filteredAnnotations.all { it.projectId == "test_project_123" })
        assertTrue(filteredAnnotations.all { it.type == "comment" })
        assertTrue(filteredAnnotations.all { it.status == AnnotationStatus.ACTIVE })
        assertTrue(filteredAnnotations.all { it.content.lowercase().contains("test") })
    }

    @Test
    fun `test annotation statistics`() = runBlocking {
        // Add test annotations
        annotationManager.addAnnotation("comment", "Comment 1", testPosition)
        annotationManager.addAnnotation("comment", "Comment 2", testPosition)
        annotationManager.addAnnotation("note", "Note 1", testPosition)
        val issueAnnotation = annotationManager.addAnnotation("issue", "Issue 1", testPosition)

        // Add reply to one annotation
        annotationManager.addReply(issueAnnotation.id, "Reply to issue", emptyMap())

        // Resolve one annotation
        annotationManager.resolveAnnotation(issueAnnotation.id, "Fixed")

        val stats = annotationManager.getAnnotationStatistics("test_project_123")

        assertEquals(4, stats.totalAnnotations)
        assertEquals(2, stats.typeBreakdown["comment"])
        assertEquals(1, stats.typeBreakdown["note"])
        assertEquals(1, stats.typeBreakdown["issue"])
        assertEquals(3, stats.statusBreakdown[AnnotationStatus.ACTIVE])
        assertEquals(1, stats.statusBreakdown[AnnotationStatus.RESOLVED])
        assertEquals(0.25, stats.averageRepliesPerAnnotation) // 1 reply / 4 annotations
    }

    @Test
    fun `test export annotations to JSON`() = runBlocking {
        annotationManager.addAnnotation("comment", "Export test comment", testPosition)
        annotationManager.addAnnotation("note", "Export test note", testPosition)

        val exportPath = tempDir.resolve("annotations.json").toString()
        val result =
            annotationManager.exportAnnotations(
                projectId = "test_project_123",
                format = "json",
                filePath = exportPath,
            )

        assertTrue(result)
        assertTrue(File(exportPath).exists())

        val exportedContent = File(exportPath).readText()
        assertTrue(exportedContent.contains("Export test comment"))
        assertTrue(exportedContent.contains("Export test note"))
    }

    @Test
    fun `test export annotations to CSV`() = runBlocking {
        annotationManager.addAnnotation("comment", "CSV export test", testPosition)

        val exportPath = tempDir.resolve("annotations.csv").toString()
        val result =
            annotationManager.exportAnnotations(
                projectId = "test_project_123",
                format = "csv",
                filePath = exportPath,
            )

        assertTrue(result)
        assertTrue(File(exportPath).exists())

        val exportedContent = File(exportPath).readText()
        assertTrue(exportedContent.contains("ID,Type,Content,Author,Timestamp,Status,Project ID,Replies Count"))
        assertTrue(exportedContent.contains("CSV export test"))
    }

    @Test
    fun `test export annotations to HTML`() = runBlocking {
        annotationManager.addAnnotation("comment", "HTML export test", testPosition)

        val exportPath = tempDir.resolve("annotations.html").toString()
        val result =
            annotationManager.exportAnnotations(
                projectId = "test_project_123",
                format = "html",
                filePath = exportPath,
            )

        assertTrue(result)
        assertTrue(File(exportPath).exists())

        val exportedContent = File(exportPath).readText()
        assertTrue(exportedContent.contains("<!DOCTYPE html>"))
        assertTrue(exportedContent.contains("HTML export test"))
    }

    @Test
    fun `test import annotations from JSON`() = runBlocking {
        // Create test JSON file
        val testAnnotations =
            listOf(
                Annotation(
                    id = "imported_1",
                    type = "comment",
                    content = "Imported comment",
                    position = testPosition,
                    author = User("import_user", "Import User", "import@test.com"),
                    timestamp = Date(),
                    projectId = "imported_project",
                    sessionId = null,
                    metadata = emptyMap(),
                    status = AnnotationStatus.ACTIVE,
                ),
            )

        val importPath = tempDir.resolve("import_annotations.json").toString()
        File(importPath).writeText(Gson().toJson(testAnnotations))

        val initialCount = annotationManager.annotations.value.size
        val result = annotationManager.importAnnotations(importPath, "json")

        assertTrue(result)
        assertEquals(initialCount + 1, annotationManager.annotations.value.size)

        val importedAnnotation = annotationManager.annotations.value.find { it.id == "imported_1" }
        assertNotNull(importedAnnotation)
        assertEquals("Imported comment", importedAnnotation!!.content)
    }

    @Test
    fun `test supported annotation types`() {
        val supportedTypes = annotationManager.getSupportedTypes()

        assertTrue(supportedTypes.containsKey("comment"))
        assertTrue(supportedTypes.containsKey("highlight"))
        assertTrue(supportedTypes.containsKey("markup"))
        assertTrue(supportedTypes.containsKey("note"))
        assertTrue(supportedTypes.containsKey("question"))
        assertTrue(supportedTypes.containsKey("suggestion"))
        assertTrue(supportedTypes.containsKey("issue"))
        assertTrue(supportedTypes.containsKey("approval"))

        val commentType = supportedTypes["comment"]!!
        assertEquals("Comment", commentType.name)
        assertEquals("Text-based comment", commentType.description)
        assertEquals(Color.YELLOW, commentType.color)
    }

    @Test
    fun `test annotation events`() = runBlocking {
        val events = mutableListOf<AnnotationEvent>()

        // Collect events
        val job =
            launch {
                annotationManager.annotationEvents.collect { event ->
                    event?.let { events.add(it) }
                }
            }

        val annotation = annotationManager.addAnnotation("comment", "Event test", testPosition)
        annotationManager.addReply(annotation.id, "Reply test", emptyMap())
        annotationManager.resolveAnnotation(annotation.id, "Resolved")

        kotlinx.coroutines.delay(100) // Allow events to be processed
        job.cancel()

        assertTrue(events.any { it is AnnotationEvent.AnnotationAdded })
        assertTrue(events.any { it is AnnotationEvent.ReplyAdded })
        assertTrue(events.any { it is AnnotationEvent.AnnotationResolved })
    }

    @Test
    fun `test annotation with multiple replies`() = runBlocking {
        val annotation =
            annotationManager.addAnnotation(
                type = "question",
                content = "What's the best approach here?",
                position = testPosition,
            )

        annotationManager.addReply(annotation.id, "I think approach A is better", emptyMap())
        annotationManager.addReply(annotation.id, "Actually, approach B might work", emptyMap())
        annotationManager.addReply(annotation.id, "Let's go with approach A", emptyMap())

        val updatedAnnotation = annotationManager.annotations.value.find { it.id == annotation.id }
        assertNotNull(updatedAnnotation)
        assertEquals(3, updatedAnnotation!!.replies.size)

        val replyContents = updatedAnnotation.replies.map { it.content }
        assertTrue(replyContents.contains("I think approach A is better"))
        assertTrue(replyContents.contains("Actually, approach B might work"))
        assertTrue(replyContents.contains("Let's go with approach A"))
    }

    @Test
    fun `test annotation position tracking`() = runBlocking {
        val position1 = AnnotationPosition(x = 10.0, y = 20.0, component = "panel1")
        val position2 = AnnotationPosition(x = 100.0, y = 200.0, width = 50.0, height = 30.0, page = 2)

        val annotation1 = annotationManager.addAnnotation("comment", "Position test 1", position1)
        val annotation2 = annotationManager.addAnnotation("note", "Position test 2", position2)

        assertEquals(position1, annotation1.position)
        assertEquals(position2, annotation2.position)
        assertEquals("panel1", annotation1.position.component)
        assertEquals(2, annotation2.position.page)
    }
}
