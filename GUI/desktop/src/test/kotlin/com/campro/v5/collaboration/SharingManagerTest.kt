package com.campro.v5.collaboration

import com.campro.v5.waitForConditionOrFail
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertNotNull
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import java.util.Collections
import java.util.Date

class SharingManagerTest {
    private lateinit var sharingManager: SharingManager
    private lateinit var testProjectData: ProjectData
    private lateinit var testCollaborators: List<Collaborator>

    @BeforeEach
    fun setUp() {
        sharingManager = SharingManager.getInstance()
        sharingManager.resetState()

        // Create test project data
        testProjectData =
            ProjectData(
                metadata =
                ProjectMetadata(
                    name = "Test Sharing Project",
                    description = "A test project for sharing functionality",
                    timestamp = Date(),
                    version = "1.0",
                    author = "Test User",
                ),
                parameters =
                mapOf(
                    "Piston Diameter" to "70.0",
                    "Stroke" to "20.0",
                    "Rod Length" to "40.0",
                ),
                simulationResults =
                SimulationResults(
                    parameters = mapOf("Piston Diameter" to "70.0"),
                    metrics = mapOf("Max Force" to 1500.0),
                    visualizations = emptyList(),
                ),
            )

        // Create test collaborators
        testCollaborators =
            listOf(
                Collaborator(
                    user = User("user1", "John Doe", "john@example.com"),
                    role = CollaboratorRole.EDITOR,
                    permissions = listOf(Permission.READ, Permission.WRITE),
                ),
                Collaborator(
                    user = User("user2", "Jane Smith", "jane@example.com"),
                    role = CollaboratorRole.VIEWER,
                    permissions = listOf(Permission.READ),
                ),
            )
    }

    @AfterEach
    fun tearDown() {
        // Clean up any test data
    }

    @Test
    fun `test share project to cloud`() = runBlocking {
        val options =
            SharingOptions(
                accessLevel = AccessLevel.VIEW,
                encrypt = true,
                requireAuthentication = true,
            )

        val result =
            sharingManager.shareProject(
                testProjectData,
                "cloud",
                options,
            )

        assertTrue(result is SharingResult.Success)
        val successResult = result as SharingResult.Success
        assertTrue(successResult.shareUrl.startsWith("https://cloud.campro.com"))
        assertTrue(successResult.shareUrl.contains("token="))
    }

    @Test
    fun `test share project via email`() = runBlocking {
        val options =
            SharingOptions(
                recipients = listOf("test1@example.com", "test2@example.com"),
                includeAttachment = true,
            )

        val result =
            sharingManager.shareProject(
                testProjectData,
                "email",
                options,
            )

        assertTrue(result is SharingResult.Success)
        val successResult = result as SharingResult.Success
        assertTrue(successResult.shareUrl.contains("test1@example.com"))
        assertTrue(successResult.shareUrl.contains("test2@example.com"))
    }

    @Test
    fun `test generate share link`() = runBlocking {
        val options =
            SharingOptions(
                accessLevel = AccessLevel.EDIT,
                expirationDate = Date(System.currentTimeMillis() + 86400000), // 24 hours
            )

        val result =
            sharingManager.shareProject(
                testProjectData,
                "link",
                options,
            )

        assertTrue(result is SharingResult.Success)
        val successResult = result as SharingResult.Success
        assertTrue(successResult.shareUrl.contains("/share/"))
        assertTrue(successResult.shareUrl.startsWith("https://share.campro.com"))
    }

    @Test
    fun `test generate QR code`() = runBlocking {
        val result =
            sharingManager.shareProject(
                testProjectData,
                "qr",
            )

        assertTrue(result is SharingResult.Success)
        val successResult = result as SharingResult.Success
        assertTrue(successResult.shareUrl.contains("QR Code generated"))
    }

    @Test
    fun `test local network sharing`() = runBlocking {
        val result =
            sharingManager.shareProject(
                testProjectData,
                "local",
            )

        assertTrue(result is SharingResult.Success)
        val successResult = result as SharingResult.Success
        assertTrue(successResult.shareUrl.contains("localhost"))
        assertTrue(successResult.shareUrl.startsWith("Local sharing at: http://localhost:"))
    }

    @Test
    fun `test Git repository sharing`() = runBlocking {
        val options =
            SharingOptions(
                gitRemoteUrl = "https://github.com/user/campro-project.git",
            )

        val result =
            sharingManager.shareProject(
                testProjectData,
                "git",
                options,
            )

        assertTrue(result is SharingResult.Success)
        val successResult = result as SharingResult.Success
        assertTrue(successResult.shareUrl.contains("github.com"))
    }

    @Test
    fun `test share with unsupported platform`() = runBlocking {
        val result =
            sharingManager.shareProject(
                testProjectData,
                "unsupported_platform",
            )

        assertTrue(result is SharingResult.Error)
        val errorResult = result as SharingResult.Error
        assertTrue(errorResult.message.contains("Unsupported platform"))
    }

    @Test
    fun `test start collaboration session`() = runBlocking {
        val options =
            CollaborationOptions(
                permissions =
                mapOf(
                    "edit" to Permission.WRITE,
                    "comment" to Permission.READ,
                ),
                allowRealTimeEditing = true,
                allowComments = true,
            )

        val result =
            sharingManager.startCollaboration(
                testProjectData,
                testCollaborators,
                options,
            )

        assertTrue(result is CollaborationResult.Success)
        val successResult = result as CollaborationResult.Success

        assertEquals("Test Sharing Project", successResult.session.projectName)
        assertEquals(2, successResult.session.collaborators.size)
        assertTrue(successResult.session.isActive)
        assertNotNull(successResult.session.shareUrl)
    }

    @Test
    fun `test collaboration session management`() = runBlocking {
        val result =
            sharingManager.startCollaboration(
                testProjectData,
                testCollaborators,
            )

        assertTrue(result is CollaborationResult.Success)
        val session = (result as CollaborationResult.Success).session

        // Check that session is tracked
        val activeSessions = sharingManager.collaborationSessions.value
        assertTrue(activeSessions.any { it.id == session.id })

        // End the collaboration
        val endResult = sharingManager.endCollaboration(session.id)
        assertTrue(endResult)
    }

    @Test
    fun `test active shares tracking`() = runBlocking {
        val initialSharesCount = sharingManager.activeShares.value.size

        val result =
            sharingManager.shareProject(
                testProjectData,
                "cloud",
            )

        assertTrue(result is SharingResult.Success)

        val newSharesCount = sharingManager.activeShares.value.size
        assertEquals(initialSharesCount + 1, newSharesCount)

        val latestShare = sharingManager.activeShares.value.first()
        assertEquals("Test Sharing Project", latestShare.projectName)
        assertEquals("cloud", latestShare.platform)
        assertTrue(latestShare.isActive)
    }

    @Test
    fun `test revoke share`() = runBlocking {
        val result =
            sharingManager.shareProject(
                testProjectData,
                "cloud",
            )

        assertTrue(result is SharingResult.Success)

        val activeShare = sharingManager.activeShares.value.first()
        val revokeResult = sharingManager.revokeShare(activeShare.id)

        assertTrue(revokeResult)

        // Check that share is no longer active
        val updatedShare = sharingManager.activeShares.value.find { it.id == activeShare.id }
        assertNotNull(updatedShare)
        assertFalse(updatedShare!!.isActive)
        assertNotNull(updatedShare.revokedDate)
    }

    @Test
    fun `test sharing progress tracking`() = runBlocking {
        assertFalse(sharingManager.isSharing())
        assertEquals(0.0f, sharingManager.getSharingProgress())

        val result =
            sharingManager.shareProject(
                testProjectData,
                "cloud",
            )

        assertTrue(result is SharingResult.Success)
        assertFalse(sharingManager.isSharing())
        assertEquals(1.0f, sharingManager.getSharingProgress())
    }

    @Test
    fun `test sharing history`() = runBlocking {
        val initialHistorySize = sharingManager.getSharingHistory().size

        sharingManager.shareProject(
            testProjectData,
            "cloud",
        )

        val newHistorySize = sharingManager.getSharingHistory().size
        assertEquals(initialHistorySize + 1, newHistorySize)

        val latestEntry = sharingManager.getSharingHistory().first()
        assertEquals("Test Sharing Project", latestEntry.projectName)
        assertEquals("cloud", latestEntry.platform)
        assertTrue(latestEntry.success)
    }

    @Test
    fun `test supported platforms`() {
        val supportedPlatforms = sharingManager.getSupportedPlatforms()

        assertTrue(supportedPlatforms.containsKey("cloud"))
        assertTrue(supportedPlatforms.containsKey("email"))
        assertTrue(supportedPlatforms.containsKey("link"))
        assertTrue(supportedPlatforms.containsKey("qr"))
        assertTrue(supportedPlatforms.containsKey("local"))
        assertTrue(supportedPlatforms.containsKey("git"))

        val cloudPlatform = supportedPlatforms["cloud"]!!
        assertEquals("Cloud Storage", cloudPlatform.name)
        assertEquals("Secure cloud-based sharing", cloudPlatform.description)
        assertTrue(cloudPlatform.requiresAuthentication)
    }

    @Test
    fun `test sharing with different access levels`() = runBlocking {
        val viewOptions = SharingOptions(accessLevel = AccessLevel.VIEW)
        val editOptions = SharingOptions(accessLevel = AccessLevel.EDIT)
        val adminOptions = SharingOptions(accessLevel = AccessLevel.ADMIN)

        val viewResult = sharingManager.shareProject(testProjectData, "cloud", viewOptions)
        val editResult = sharingManager.shareProject(testProjectData, "cloud", editOptions)
        val adminResult = sharingManager.shareProject(testProjectData, "cloud", adminOptions)

        assertTrue(viewResult is SharingResult.Success)
        assertTrue(editResult is SharingResult.Success)
        assertTrue(adminResult is SharingResult.Success)

        val shares = sharingManager.activeShares.value.takeLast(3)
        assertEquals(AccessLevel.VIEW, shares[0].accessLevel)
        assertEquals(AccessLevel.EDIT, shares[1].accessLevel)
        assertEquals(AccessLevel.ADMIN, shares[2].accessLevel)
    }

    @Test
    fun `test sharing with expiration date`() = runBlocking {
        val expirationDate = Date(System.currentTimeMillis() + 3600000) // 1 hour
        val options = SharingOptions(expirationDate = expirationDate)

        val result =
            sharingManager.shareProject(
                testProjectData,
                "cloud",
                options,
            )

        assertTrue(result is SharingResult.Success)

        val share = sharingManager.activeShares.value.first()
        assertEquals(expirationDate, share.expirationDate)
    }

    @Test
    fun `test collaboration with different roles`() = runBlocking {
        val collaborators =
            listOf(
                Collaborator(
                    user = User("admin", "Admin User", "admin@example.com"),
                    role = CollaboratorRole.ADMIN,
                    permissions = listOf(Permission.READ, Permission.WRITE, Permission.DELETE, Permission.SHARE, Permission.ADMIN),
                ),
                Collaborator(
                    user = User("editor", "Editor User", "editor@example.com"),
                    role = CollaboratorRole.EDITOR,
                    permissions = listOf(Permission.READ, Permission.WRITE),
                ),
                Collaborator(
                    user = User("viewer", "Viewer User", "viewer@example.com"),
                    role = CollaboratorRole.VIEWER,
                    permissions = listOf(Permission.READ),
                ),
            )

        val result =
            sharingManager.startCollaboration(
                testProjectData,
                collaborators,
            )

        assertTrue(result is CollaborationResult.Success)
        val session = (result as CollaborationResult.Success).session

        assertEquals(3, session.collaborators.size)
        assertTrue(session.collaborators.any { it.role == CollaboratorRole.ADMIN })
        assertTrue(session.collaborators.any { it.role == CollaboratorRole.EDITOR })
        assertTrue(session.collaborators.any { it.role == CollaboratorRole.VIEWER })
    }

    @Test
    fun `test sharing events`() = runBlocking {
        val events = Collections.synchronizedList(mutableListOf<SharingEvent>())

        // Reset state to ensure clean test
        sharingManager.resetState()

        // Set up a flag to track when the collector is ready
        val collectorReady = CompletableDeferred<Unit>()

        // Collect events
        val job =
            launch {
                try {
                    // Signal that the collector is about to start
                    println("[DEBUG_LOG] Starting event collector")

                    // Collect events from the SharedFlow
                    sharingManager.sharingEvents.collect { event ->
                        events.add(event)
                        println("[DEBUG_LOG] Received event: $event")

                        // Complete the deferred when the first event is received
                        if (!collectorReady.isCompleted) {
                            collectorReady.complete(Unit)
                        }
                    }
                } catch (e: Exception) {
                    println("[DEBUG_LOG] Error in event collector: ${e.message}")
                    if (!collectorReady.isCompleted) {
                        collectorReady.completeExceptionally(e)
                    }
                } finally {
                    // Ensure the deferred is completed even if collection ends
                    if (!collectorReady.isCompleted) {
                        collectorReady.complete(Unit)
                    }
                }
            }

        // Wait a moment to ensure the collector is running
        kotlinx.coroutines.delay(100)

        // Signal that the collector is ready
        if (!collectorReady.isCompleted) {
            collectorReady.complete(Unit)
        }

        // Wait for the collector to be ready
        collectorReady.await()
        println("[DEBUG_LOG] Event collector is ready")

        // Share the project
        println("[DEBUG_LOG] Sharing project")
        val result = sharingManager.shareProject(testProjectData, "cloud")

        // Verify the result directly
        assertTrue(result is SharingResult.Success, "Sharing should succeed")
        println("[DEBUG_LOG] Sharing result: $result")

        // Wait for both events to be received with retries
        waitForConditionOrFail(
            maxAttempts = 100,
            delayMs = 100,
            message = "Did not receive both SharingStarted and SharingCompleted events",
        ) {
            val hasStarted = events.any { it is SharingEvent.SharingStarted }
            val hasCompleted = events.any { it is SharingEvent.SharingCompleted }
            println("[DEBUG_LOG] Events check - hasStarted: $hasStarted, hasCompleted: $hasCompleted, count: ${events.size}")
            hasStarted && hasCompleted
        }

        // Cancel the collection job
        job.cancel()

        // These assertions should now pass since we waited for the condition
        assertTrue(events.any { it is SharingEvent.SharingStarted }, "Should receive SharingStarted event")
        assertTrue(events.any { it is SharingEvent.SharingCompleted }, "Should receive SharingCompleted event")

        // Print all collected events for debugging
        println("[DEBUG_LOG] All collected events: $events")
    }

    @Test
    fun `test collaboration session duration`() = runBlocking {
        val sessionDuration = 3600000L // 1 hour in milliseconds
        val options = CollaborationOptions(sessionDuration = sessionDuration)

        val result =
            sharingManager.startCollaboration(
                testProjectData,
                testCollaborators,
                options,
            )

        assertTrue(result is CollaborationResult.Success)
        val session = (result as CollaborationResult.Success).session

        // Check that the associated share has the correct expiration
        val associatedShare =
            sharingManager.activeShares.value.find {
                it.shareUrl == session.shareUrl
            }
        assertNotNull(associatedShare)
        assertNotNull(associatedShare!!.expirationDate)
    }
}
