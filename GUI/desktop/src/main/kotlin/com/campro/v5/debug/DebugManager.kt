package com.campro.v5.debug

import androidx.compose.runtime.mutableStateOf
import org.slf4j.LoggerFactory

/**
 * Centralized debug settings/state manager mirroring AccessibilityEnhancer.
 * Provides opt-in debug features that can alter runtime UI behavior when enabled.
 */
object DebugManager {
    private val logger = LoggerFactory.getLogger(DebugManager::class.java)

    /** Debug settings data class. */
    data class DebugSettings(
        val panelVisible: Boolean = false,
        val buttonDebug: Boolean = false,
        val componentHealth: Boolean = false,
        val interactionLogging: Boolean = false,
        val errorBoundary: Boolean = true,
        val performanceMonitoring: Boolean = false,
        val networkDebugging: Boolean = false,
    )

    /** Current debug settings. */
    private val _settings = mutableStateOf(DebugSettings())
    val settings: DebugSettings get() = _settings.value

    /** Update debug settings atomically and log the change. */
    fun updateSettings(settings: DebugSettings) {
        _settings.value = settings
        logger.info("Debug settings updated: $settings")
    }

    /** Convenience toggles mirroring common usage in UI. */
    fun setPanelVisible(visible: Boolean) = updateSettings(settings.copy(panelVisible = visible))
    fun togglePanelVisible() = setPanelVisible(!settings.panelVisible)
}
