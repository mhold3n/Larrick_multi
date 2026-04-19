package com.campro.v5.collaboration

import androidx.compose.runtime.mutableStateOf
import com.campro.v5.layout.StateManager
import com.google.gson.GsonBuilder
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.security.MessageDigest
import java.util.Base64
import java.util.Date
import java.util.UUID
import javax.crypto.Cipher
import javax.crypto.KeyGenerator
import javax.crypto.SecretKey

/**
 * Manages sharing functionality for the CamPro v5 application.
 * This class provides project sharing capabilities including cloud sharing,
 * collaboration features, access control, and secure sharing mechanisms.
 */
class SharingManager {
    // Sharing state
    private val _isSharing = mutableStateOf(false)
    val isSharing: Boolean
        get() = _isSharing.value
    private val _sharingProgress = mutableStateOf(0.0f)
    val sharingProgress: Float
        get() = _sharingProgress.value

    // Sharing events
    private val _sharingEvents = MutableSharedFlow<SharingEvent>(replay = 10, extraBufferCapacity = 100)
    val sharingEvents: SharedFlow<SharingEvent> = _sharingEvents.asSharedFlow()

    // Active shares
    private val _activeShares = MutableStateFlow<List<SharedProject>>(emptyList())
    val activeShares: StateFlow<List<SharedProject>> = _activeShares.asStateFlow()

    // Collaboration sessions
    private val _collaborationSessions = MutableStateFlow<List<CollaborationSession>>(emptyList())
    val collaborationSessions: StateFlow<List<CollaborationSession>> = _collaborationSessions.asStateFlow()

    // Sharing platforms
    private val supportedPlatforms =
        mapOf(
            "cloud" to SharingPlatform("Cloud Storage", "Secure cloud-based sharing", true),
            "email" to SharingPlatform("Email", "Share via email attachment", false),
            "link" to SharingPlatform("Share Link", "Generate shareable link", true),
            "qr" to SharingPlatform("QR Code", "Share via QR code", false),
            "local" to SharingPlatform("Local Network", "Share on local network", true),
            "git" to SharingPlatform("Git Repository", "Share via Git repository", true),
        )

    // State manager for persistence
    private val stateManager = StateManager.getInstance()

    // Coroutine scope for async operations
    private val scope = CoroutineScope(Dispatchers.IO)

    // JSON serialization
    private val gson = GsonBuilder().setPrettyPrinting().create()

    // Encryption key for secure sharing
    private var encryptionKey: SecretKey? = null

    init {
        // Initialize encryption
        initializeEncryption()

        // Load sharing preferences
        loadSharingPreferences()

        // Load active shares
        loadActiveShares()
    }

    /**
     * Share a project with specified options.
     */
    suspend fun shareProject(projectData: ProjectData, platform: String, options: SharingOptions = SharingOptions()): SharingResult =
        withContext(Dispatchers.IO) {
            try {
                _isSharing.value = true
                _sharingProgress.value = 0.0f

                emitEvent(SharingEvent.SharingStarted(platform, projectData.metadata.name))

                val result =
                    when (platform.lowercase()) {
                        "cloud" -> shareToCloud(projectData, options)
                        "email" -> shareViaEmail(projectData, options)
                        "link" -> generateShareLink(projectData, options)
                        "qr" -> generateQRCode(projectData, options)
                        "local" -> shareOnLocalNetwork(projectData, options)
                        "git" -> shareViaGit(projectData, options)
                        else -> SharingResult.Error("Unsupported platform: $platform")
                    }

                _sharingProgress.value = 1.0f

                when (result) {
                    is SharingResult.Success -> {
                        val sharedProject =
                            SharedProject(
                                id = generateShareId(),
                                projectName = projectData.metadata.name,
                                platform = platform,
                                shareUrl = result.shareUrl,
                                accessLevel = options.accessLevel,
                                expirationDate = options.expirationDate,
                                createdDate = Date(),
                                isActive = true,
                            )

                        addActiveShare(sharedProject)
                        emitEvent(SharingEvent.SharingCompleted(platform, result.shareUrl, sharedProject.id))

                        // Save sharing history
                        saveSharingHistory(
                            SharingHistoryEntry(
                                timestamp = Date(),
                                projectName = projectData.metadata.name,
                                platform = platform,
                                shareId = sharedProject.id,
                                success = true,
                            ),
                        )
                    }
                    is SharingResult.Error -> {
                        emitEvent(SharingEvent.SharingFailed(platform, result.message))
                        saveSharingHistory(
                            SharingHistoryEntry(
                                timestamp = Date(),
                                projectName = projectData.metadata.name,
                                platform = platform,
                                shareId = "",
                                success = false,
                                error = result.message,
                            ),
                        )
                    }
                }

                result
            } catch (e: Exception) {
                val errorMessage = "Sharing failed: ${e.message}"
                emitEvent(SharingEvent.SharingFailed(platform, errorMessage))
                SharingResult.Error(errorMessage)
            } finally {
                _isSharing.value = false
            }
        }

    /**
     * Start a collaboration session.
     */
    suspend fun startCollaboration(
        projectData: ProjectData,
        collaborators: List<Collaborator>,
        options: CollaborationOptions = CollaborationOptions(),
    ): CollaborationResult = withContext(Dispatchers.IO) {
        try {
            val sessionId = generateSessionId()
            val session =
                CollaborationSession(
                    id = sessionId,
                    projectName = projectData.metadata.name,
                    owner = getCurrentUser(),
                    collaborators = collaborators,
                    startTime = Date(),
                    isActive = true,
                    permissions = options.permissions,
                )

            // Share project with collaborators
            val shareResult =
                shareProject(
                    projectData,
                    "cloud",
                    SharingOptions(
                        accessLevel = AccessLevel.EDIT,
                        allowCollaboration = true,
                        expirationDate = options.sessionDuration?.let { Date(System.currentTimeMillis() + it) },
                    ),
                )

            when (shareResult) {
                is SharingResult.Success -> {
                    session.shareUrl = shareResult.shareUrl
                    addCollaborationSession(session)

                    // Notify collaborators
                    notifyCollaborators(session, collaborators)

                    emitEvent(SharingEvent.CollaborationStarted(sessionId, collaborators.size))
                    CollaborationResult.Success(session)
                }
                is SharingResult.Error -> {
                    CollaborationResult.Error("Failed to start collaboration: ${shareResult.message}")
                }
            }
        } catch (e: Exception) {
            CollaborationResult.Error("Collaboration failed: ${e.message}")
        }
    }

    /**
     * Share to cloud storage.
     */
    private suspend fun shareToCloud(projectData: ProjectData, options: SharingOptions): SharingResult = withContext(Dispatchers.IO) {
        try {
            _sharingProgress.value = 0.2f

            // Encrypt data if required
            val dataToShare =
                if (options.encrypt) {
                    encryptProjectData(projectData)
                } else {
                    gson.toJson(projectData)
                }

            _sharingProgress.value = 0.5f

            // Simulate cloud upload
            val cloudUrl = uploadToCloud(dataToShare, projectData.metadata.name, options)

            _sharingProgress.value = 0.8f

            // Generate access token if needed
            val accessToken =
                if (options.requireAuthentication) {
                    generateAccessToken()
                } else {
                    null
                }

            val shareUrl =
                if (accessToken != null) {
                    "$cloudUrl?token=$accessToken"
                } else {
                    cloudUrl
                }

            SharingResult.Success(shareUrl)
        } catch (e: Exception) {
            SharingResult.Error("Cloud sharing failed: ${e.message}")
        }
    }

    /**
     * Share via email.
     */
    private suspend fun shareViaEmail(projectData: ProjectData, options: SharingOptions): SharingResult = withContext(Dispatchers.IO) {
        try {
            _sharingProgress.value = 0.3f

            // Create email content
            val emailContent = createEmailContent(projectData, options)

            _sharingProgress.value = 0.6f

            // Simulate email sending
            val emailResult =
                sendEmail(
                    to = options.recipients,
                    subject = "CamPro v5 Project: ${projectData.metadata.name}",
                    content = emailContent,
                    attachments = if (options.includeAttachment) listOf(projectData) else emptyList(),
                )

            _sharingProgress.value = 0.9f

            if (emailResult) {
                SharingResult.Success("Email sent to ${options.recipients.joinToString(", ")}")
            } else {
                SharingResult.Error("Failed to send email")
            }
        } catch (e: Exception) {
            SharingResult.Error("Email sharing failed: ${e.message}")
        }
    }

    /**
     * Generate shareable link.
     */
    private suspend fun generateShareLink(projectData: ProjectData, options: SharingOptions): SharingResult = withContext(Dispatchers.IO) {
        try {
            _sharingProgress.value = 0.4f

            val shareId = generateShareId()
            val linkData =
                ShareLinkData(
                    shareId = shareId,
                    projectData = projectData,
                    accessLevel = options.accessLevel,
                    expirationDate = options.expirationDate,
                    createdDate = Date(),
                )

            _sharingProgress.value = 0.7f

            // Store link data
            storeLinkData(shareId, linkData)

            val baseUrl = getBaseShareUrl()
            val shareUrl = "$baseUrl/share/$shareId"

            SharingResult.Success(shareUrl)
        } catch (e: Exception) {
            SharingResult.Error("Link generation failed: ${e.message}")
        }
    }

    /**
     * Generate QR code for sharing.
     */
    private suspend fun generateQRCode(projectData: ProjectData, options: SharingOptions): SharingResult = withContext(Dispatchers.IO) {
        try {
            // First generate a share link
            val linkResult = generateShareLink(projectData, options)

            when (linkResult) {
                is SharingResult.Success -> {
                    _sharingProgress.value = 0.8f

                    // Generate QR code for the link
                    val qrCodePath = generateQRCodeImage(linkResult.shareUrl, projectData.metadata.name)

                    SharingResult.Success("QR Code generated: $qrCodePath")
                }
                is SharingResult.Error -> linkResult
            }
        } catch (e: Exception) {
            SharingResult.Error("QR code generation failed: ${e.message}")
        }
    }

    /**
     * Share on local network.
     */
    private suspend fun shareOnLocalNetwork(projectData: ProjectData, options: SharingOptions): SharingResult =
        withContext(Dispatchers.IO) {
            try {
                _sharingProgress.value = 0.3f

                // Start local server
                val port = findAvailablePort()
                val serverUrl = startLocalServer(port, projectData, options)

                _sharingProgress.value = 0.7f

                // Announce on network
                announceOnNetwork(serverUrl, projectData.metadata.name)

                SharingResult.Success("Local sharing at: $serverUrl")
            } catch (e: Exception) {
                SharingResult.Error("Local network sharing failed: ${e.message}")
            }
        }

    /**
     * Share via Git repository.
     */
    private suspend fun shareViaGit(projectData: ProjectData, options: SharingOptions): SharingResult = withContext(Dispatchers.IO) {
        try {
            _sharingProgress.value = 0.2f

            // Create Git repository structure
            val repoPath = createGitRepository(projectData)

            _sharingProgress.value = 0.5f

            // Push to remote if specified
            val remoteUrl = options.gitRemoteUrl
            if (remoteUrl != null) {
                pushToRemote(repoPath, remoteUrl)
                _sharingProgress.value = 0.9f
                SharingResult.Success("Pushed to Git repository: $remoteUrl")
            } else {
                SharingResult.Success("Local Git repository created: $repoPath")
            }
        } catch (e: Exception) {
            SharingResult.Error("Git sharing failed: ${e.message}")
        }
    }

    /**
     * Revoke a shared project.
     */
    suspend fun revokeShare(shareId: String): Boolean = withContext(Dispatchers.IO) {
        try {
            val share = _activeShares.value.find { it.id == shareId }
            if (share != null) {
                // Deactivate share
                val updatedShare = share.copy(isActive = false, revokedDate = Date())
                updateActiveShare(updatedShare)

                // Remove from cloud/platform if needed
                revokeFromPlatform(share)

                emitEvent(SharingEvent.ShareRevoked(shareId))
                true
            } else {
                false
            }
        } catch (e: Exception) {
            emitEvent(SharingEvent.SharingFailed("revoke", "Failed to revoke share: ${e.message}"))
            false
        }
    }

    /**
     * End a collaboration session.
     */
    suspend fun endCollaboration(sessionId: String): Boolean = withContext(Dispatchers.IO) {
        try {
            val session = _collaborationSessions.value.find { it.id == sessionId }
            if (session != null) {
                val updatedSession = session.copy(isActive = false, endTime = Date())
                updateCollaborationSession(updatedSession)

                // Notify collaborators
                notifyCollaborationEnd(session)

                emitEvent(SharingEvent.CollaborationEnded(sessionId))
                true
            } else {
                false
            }
        } catch (e: Exception) {
            false
        }
    }

    /**
     * Get sharing history.
     */
    fun getSharingHistory(): List<SharingHistoryEntry> {
        val historyJson = stateManager.getState("sharing.history", "[]")
        return try {
            gson.fromJson(historyJson, Array<SharingHistoryEntry>::class.java).toList()
        } catch (e: Exception) {
            emptyList()
        }
    }

    /**
     * Get supported sharing platforms.
     */
    fun getSupportedPlatforms(): Map<String, SharingPlatform> = supportedPlatforms

    // Helper methods
    private fun initializeEncryption() {
        try {
            val keyGen = KeyGenerator.getInstance("AES")
            keyGen.init(256)
            encryptionKey = keyGen.generateKey()
        } catch (e: Exception) {
            // Fallback to stored key or generate new one
        }
    }

    private fun loadSharingPreferences() {
        // Load user preferences for sharing
    }

    private fun loadActiveShares() {
        val sharesJson = stateManager.getState("sharing.activeShares", "[]")
        try {
            val shares = gson.fromJson(sharesJson, Array<SharedProject>::class.java).toList()
            _activeShares.value = shares.filter { it.isActive }
        } catch (e: Exception) {
            _activeShares.value = emptyList()
        }
    }

    private fun addActiveShare(share: SharedProject) {
        val currentShares = _activeShares.value.toMutableList()
        currentShares.add(share)
        _activeShares.value = currentShares
        saveActiveShares()
    }

    private fun updateActiveShare(updatedShare: SharedProject) {
        val currentShares = _activeShares.value.toMutableList()
        val index = currentShares.indexOfFirst { it.id == updatedShare.id }
        if (index >= 0) {
            currentShares[index] = updatedShare
            _activeShares.value = currentShares
            saveActiveShares()
        }
    }

    private fun saveActiveShares() {
        val sharesJson = gson.toJson(_activeShares.value)
        stateManager.setState("sharing.activeShares", sharesJson)
    }

    private fun addCollaborationSession(session: CollaborationSession) {
        val currentSessions = _collaborationSessions.value.toMutableList()
        currentSessions.add(session)
        _collaborationSessions.value = currentSessions
    }

    private fun updateCollaborationSession(updatedSession: CollaborationSession) {
        val currentSessions = _collaborationSessions.value.toMutableList()
        val index = currentSessions.indexOfFirst { it.id == updatedSession.id }
        if (index >= 0) {
            currentSessions[index] = updatedSession
            _collaborationSessions.value = currentSessions
        }
    }

    private fun saveSharingHistory(entry: SharingHistoryEntry) {
        val history = getSharingHistory().toMutableList()
        history.add(0, entry)

        // Keep only last 100 entries
        if (history.size > 100) {
            history.removeAt(history.size - 1)
        }

        val historyJson = gson.toJson(history)
        stateManager.setState("sharing.history", historyJson)
    }

    private fun generateShareId(): String = UUID
        .randomUUID()
        .toString()
        .replace("-", "")
        .substring(0, 12)

    private fun generateSessionId(): String = "session_${System.currentTimeMillis()}_${(1000..9999).random()}"

    private fun generateAccessToken(): String {
        val token = UUID.randomUUID().toString()
        return MessageDigest
            .getInstance("SHA-256")
            .digest(token.toByteArray())
            .joinToString("") { "%02x".format(it) }
            .substring(0, 32)
    }

    private fun encryptProjectData(projectData: ProjectData): String = try {
        val cipher = Cipher.getInstance("AES")
        cipher.init(Cipher.ENCRYPT_MODE, encryptionKey)
        val encrypted = cipher.doFinal(gson.toJson(projectData).toByteArray())
        Base64.getEncoder().encodeToString(encrypted)
    } catch (e: Exception) {
        gson.toJson(projectData) // Fallback to unencrypted
    }

    private fun getCurrentUser(): User = User(
        id = stateManager.getState("user.id", "anonymous"),
        name = stateManager.getState("user.name", "Anonymous User"),
        email = stateManager.getState("user.email", ""),
    )

    /**
     * Reset the sharing manager state.
     * This is primarily used for testing to ensure a clean state between tests.
     */
    fun resetState() {
        _isSharing.value = false
        _sharingProgress.value = 0.0f
        // No need to reset _sharingEvents as it's a MutableSharedFlow
        _activeShares.value = emptyList()
        _collaborationSessions.value = emptyList()
        // Clear sharing history in StateManager
        stateManager.setState("sharing.history", "[]")
    }

    private fun emitEvent(event: SharingEvent) {
        // Use tryEmit for non-blocking emission
        // This is more efficient and avoids potential deadlocks
        if (!_sharingEvents.tryEmit(event)) {
            // If tryEmit fails (e.g., buffer is full), fall back to suspending emission
            scope.launch {
                _sharingEvents.emit(event)
            }
        }

        // Log the event for debugging
        println("[DEBUG_LOG] Emitted event: $event")
    }

    // Placeholder implementations for external services
    private suspend fun uploadToCloud(data: String, projectName: String, options: SharingOptions): String {
        // Simulate cloud upload
        return "https://cloud.campro.com/projects/${generateShareId()}"
    }

    private suspend fun sendEmail(to: List<String>, subject: String, content: String, attachments: List<ProjectData>): Boolean {
        // Simulate email sending
        return true
    }

    private fun createEmailContent(projectData: ProjectData, options: SharingOptions): String =
        """
        Hello,
        
        I'm sharing a CamPro v5 project with you: ${projectData.metadata.name}
        
        Description: ${projectData.metadata.description}
        Created: ${projectData.metadata.timestamp}
        
        ${if (options.includeAttachment) "The project file is attached to this email." else ""}
        
        Best regards,
        CamPro v5 User
        """.trimIndent()

    private fun storeLinkData(shareId: String, linkData: ShareLinkData) {
        val linkDataJson = gson.toJson(linkData)
        stateManager.setState("sharing.links.$shareId", linkDataJson)
    }

    private fun getBaseShareUrl(): String = stateManager.getState("sharing.baseUrl", "https://share.campro.com")

    private fun generateQRCodeImage(url: String, projectName: String): String {
        // Simulate QR code generation
        val qrCodePath = "${System.getProperty("user.home")}/CamPro_QR_${projectName.replace(" ", "_")}.png"
        return qrCodePath
    }

    private fun findAvailablePort(): Int = (8080..8090).random()

    private fun startLocalServer(port: Int, projectData: ProjectData, options: SharingOptions): String {
        // Simulate local server start
        return "http://localhost:$port"
    }

    private fun announceOnNetwork(serverUrl: String, projectName: String) {
        // Simulate network announcement
    }

    private fun createGitRepository(projectData: ProjectData): String {
        // Simulate Git repository creation
        return "${System.getProperty("user.home")}/CamPro_Git_${projectData.metadata.name.replace(" ", "_")}"
    }

    private fun pushToRemote(repoPath: String, remoteUrl: String) {
        // Simulate Git push
    }

    private fun revokeFromPlatform(share: SharedProject) {
        // Revoke access from the specific platform
    }

    private fun notifyCollaborators(session: CollaborationSession, collaborators: List<Collaborator>) {
        // Notify collaborators about the session
    }

    private fun notifyCollaborationEnd(session: CollaborationSession) {
        // Notify collaborators that session has ended
    }

    companion object {
        @Volatile
        private var INSTANCE: SharingManager? = null

        fun getInstance(): SharingManager = INSTANCE ?: synchronized(this) {
            INSTANCE ?: SharingManager().also { INSTANCE = it }
        }
    }
}

// Data classes
data class SharingPlatform(val name: String, val description: String, val requiresAuthentication: Boolean)

data class SharingOptions(
    val accessLevel: AccessLevel = AccessLevel.VIEW,
    val expirationDate: Date? = null,
    val encrypt: Boolean = false,
    val requireAuthentication: Boolean = false,
    val allowCollaboration: Boolean = false,
    val recipients: List<String> = emptyList(),
    val includeAttachment: Boolean = true,
    val gitRemoteUrl: String? = null,
)

data class CollaborationOptions(
    val permissions: Map<String, Permission> = emptyMap(),
    val sessionDuration: Long? = null, // milliseconds
    val allowRealTimeEditing: Boolean = true,
    val allowComments: Boolean = true,
)

data class SharedProject(
    val id: String,
    val projectName: String,
    val platform: String,
    val shareUrl: String,
    val accessLevel: AccessLevel,
    val expirationDate: Date?,
    val createdDate: Date,
    val revokedDate: Date? = null,
    val isActive: Boolean,
)

data class CollaborationSession(
    val id: String,
    val projectName: String,
    val owner: User,
    val collaborators: List<Collaborator>,
    val startTime: Date,
    val endTime: Date? = null,
    val isActive: Boolean,
    val permissions: Map<String, Permission>,
    var shareUrl: String? = null,
)

data class ShareLinkData(
    val shareId: String,
    val projectData: ProjectData,
    val accessLevel: AccessLevel,
    val expirationDate: Date?,
    val createdDate: Date,
)

data class SharingHistoryEntry(
    val timestamp: Date,
    val projectName: String,
    val platform: String,
    val shareId: String,
    val success: Boolean,
    val error: String? = null,
)

data class User(val id: String, val name: String, val email: String)

data class Collaborator(val user: User, val role: CollaboratorRole, val permissions: List<Permission>)

enum class AccessLevel {
    VIEW,
    COMMENT,
    EDIT,
    ADMIN,
}

enum class CollaboratorRole {
    VIEWER,
    COMMENTER,
    EDITOR,
    ADMIN,
}

enum class Permission {
    READ,
    WRITE,
    DELETE,
    SHARE,
    ADMIN,
}

sealed class SharingResult {
    data class Success(val shareUrl: String) : SharingResult()

    data class Error(val message: String) : SharingResult()
}

sealed class CollaborationResult {
    data class Success(val session: CollaborationSession) : CollaborationResult()

    data class Error(val message: String) : CollaborationResult()
}

sealed class SharingEvent {
    data class SharingStarted(val platform: String, val projectName: String) : SharingEvent()

    data class SharingCompleted(val platform: String, val shareUrl: String, val shareId: String) : SharingEvent()

    data class SharingFailed(val platform: String, val error: String) : SharingEvent()

    data class ShareRevoked(val shareId: String) : SharingEvent()

    data class CollaborationStarted(val sessionId: String, val collaboratorCount: Int) : SharingEvent()

    data class CollaborationEnded(val sessionId: String) : SharingEvent()
}
