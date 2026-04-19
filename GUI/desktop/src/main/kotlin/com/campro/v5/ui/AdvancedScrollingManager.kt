package com.campro.v5.ui

import androidx.compose.animation.core.*
import androidx.compose.foundation.ScrollState
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.unit.dp
import androidx.compose.ui.zIndex
import com.campro.v5.debug.DebugButton
import com.campro.v5.debug.DebugIconButton
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlin.math.*

/**
 * Scroll synchronization modes
 */
enum class ScrollSyncMode {
    NONE, // No synchronization
    HORIZONTAL, // Synchronize horizontal scrolling
    VERTICAL, // Synchronize vertical scrolling
    BOTH, // Synchronize both directions
}

/**
 * Momentum scrolling configuration
 */
data class MomentumScrollConfig(
    val enabled: Boolean = true,
    val friction: Float = 0.92f,
    val minVelocity: Float = 50f,
    val maxVelocity: Float = 5000f,
    val easingSpec: AnimationSpec<Float> =
        tween(
            durationMillis = 800,
            easing = FastOutSlowInEasing,
        ),
)

/**
 * Scroll position memory entry
 */
data class ScrollPositionMemory(
    val panelId: String,
    val horizontalPosition: Int,
    val verticalPosition: Int,
    val timestamp: Long = System.currentTimeMillis(),
)

/**
 * Minimap configuration
 */
data class MinimapConfig(
    val enabled: Boolean = true,
    val width: androidx.compose.ui.unit.Dp = 120.dp,
    val height: androidx.compose.ui.unit.Dp = 80.dp,
    val position: MinimapPosition = MinimapPosition.BOTTOM_RIGHT,
    val opacity: Float = 0.8f,
    val showViewport: Boolean = true,
    val autoHide: Boolean = true,
    val autoHideDelay: Long = 3000L,
)

/**
 * Minimap position options
 */
enum class MinimapPosition {
    TOP_LEFT,
    TOP_RIGHT,
    BOTTOM_LEFT,
    BOTTOM_RIGHT,
}

/**
 * Advanced Scrolling Manager for enhanced scrolling functionality
 */
class AdvancedScrollingManager(internal val scope: CoroutineScope) {
    private val _scrollStates = MutableStateFlow<Map<String, ScrollState>>(emptyMap())
    val scrollStates: StateFlow<Map<String, ScrollState>> = _scrollStates.asStateFlow()

    private val _scrollPositionMemory = MutableStateFlow<Map<String, ScrollPositionMemory>>(emptyMap())
    val scrollPositionMemory: StateFlow<Map<String, ScrollPositionMemory>> = _scrollPositionMemory.asStateFlow()

    private val _synchronizedGroups = MutableStateFlow<Map<String, Set<String>>>(emptyMap())
    val synchronizedGroups: StateFlow<Map<String, Set<String>>> = _synchronizedGroups.asStateFlow()

    private val _momentumScrollConfig = MutableStateFlow(MomentumScrollConfig())
    val momentumScrollConfig: StateFlow<MomentumScrollConfig> = _momentumScrollConfig.asStateFlow()

    private val _minimapConfigs = MutableStateFlow<Map<String, MinimapConfig>>(emptyMap())
    val minimapConfigs: StateFlow<Map<String, MinimapConfig>> = _minimapConfigs.asStateFlow()

    private val _activeScrollOperations = MutableStateFlow<Set<String>>(emptySet())
    val activeScrollOperations: StateFlow<Set<String>> = _activeScrollOperations.asStateFlow()

    /**
     * Register a scroll state for a panel
     */
    fun registerScrollState(panelId: String, scrollState: ScrollState) {
        _scrollStates.value = _scrollStates.value + (panelId to scrollState)

        // Restore saved scroll position if available
        val savedPosition = _scrollPositionMemory.value[panelId]
        if (savedPosition != null) {
            scope.launch {
                scrollState.animateScrollTo(savedPosition.verticalPosition)
            }
        }
    }

    /**
     * Unregister a scroll state
     */
    fun unregisterScrollState(panelId: String) {
        // Save current position before unregistering
        val scrollState = _scrollStates.value[panelId]
        if (scrollState != null) {
            saveScrollPosition(panelId, 0, scrollState.value)
        }

        _scrollStates.value = _scrollStates.value - panelId
        _minimapConfigs.value = _minimapConfigs.value - panelId
    }

    /**
     * Perform momentum-based scrolling
     */
    suspend fun performMomentumScroll(panelId: String, initialVelocity: Float, direction: ScrollDirection = ScrollDirection.VERTICAL) {
        val scrollState = _scrollStates.value[panelId] ?: return
        val config = _momentumScrollConfig.value

        if (!config.enabled || abs(initialVelocity) < config.minVelocity) return

        _activeScrollOperations.value = _activeScrollOperations.value + panelId

        try {
            val clampedVelocity = initialVelocity.coerceIn(-config.maxVelocity, config.maxVelocity)
            val targetPosition = scrollState.value + (clampedVelocity * 0.5f).toInt()
            val clampedTarget = targetPosition.coerceIn(0, scrollState.maxValue)

            // Animate to target position with easing
            scrollState.animateScrollTo(
                value = clampedTarget,
                animationSpec = config.easingSpec,
            )

            // Synchronize with other panels if needed
            synchronizeScroll(panelId, clampedTarget, direction)
        } finally {
            _activeScrollOperations.value = _activeScrollOperations.value - panelId
        }
    }

    /**
     * Create a synchronized scroll group
     */
    fun createSyncGroup(groupId: String, panelIds: Set<String>, syncMode: ScrollSyncMode) {
        _synchronizedGroups.value = _synchronizedGroups.value + (groupId to panelIds)

        // Set up synchronization listeners
        panelIds.forEach { panelId ->
            val scrollState = _scrollStates.value[panelId]
            if (scrollState != null) {
                // Monitor scroll changes and sync with other panels in the group
                scope.launch {
                    snapshotFlow { scrollState.value }.collect { position ->
                        if (panelId !in _activeScrollOperations.value) {
                            synchronizeScroll(panelId, position, ScrollDirection.VERTICAL)
                        }
                    }
                }
            }
        }
    }

    /**
     * Remove a synchronized scroll group
     */
    fun removeSyncGroup(groupId: String) {
        _synchronizedGroups.value = _synchronizedGroups.value - groupId
    }

    /**
     * Synchronize scroll position across panels in the same group
     */
    private suspend fun synchronizeScroll(sourcePanelId: String, position: Int, direction: ScrollDirection) {
        val groups = _synchronizedGroups.value
        val sourceGroup =
            groups.entries.find { (_, panelIds) -> sourcePanelId in panelIds }
                ?: return

        val targetPanels = sourceGroup.value - sourcePanelId

        targetPanels.forEach { targetPanelId ->
            val targetScrollState = _scrollStates.value[targetPanelId]
            if (targetScrollState != null && targetPanelId !in _activeScrollOperations.value) {
                _activeScrollOperations.value = _activeScrollOperations.value + targetPanelId

                try {
                    val targetPosition = position.coerceIn(0, targetScrollState.maxValue)
                    targetScrollState.animateScrollTo(targetPosition)
                } finally {
                    _activeScrollOperations.value = _activeScrollOperations.value - targetPanelId
                }
            }
        }
    }

    /**
     * Save scroll position to memory
     */
    fun saveScrollPosition(panelId: String, horizontalPosition: Int, verticalPosition: Int) {
        val memory =
            ScrollPositionMemory(
                panelId = panelId,
                horizontalPosition = horizontalPosition,
                verticalPosition = verticalPosition,
            )

        _scrollPositionMemory.value = _scrollPositionMemory.value + (panelId to memory)
    }

    /**
     * Restore scroll position from memory
     */
    suspend fun restoreScrollPosition(panelId: String): Boolean {
        val scrollState = _scrollStates.value[panelId] ?: return false
        val memory = _scrollPositionMemory.value[panelId] ?: return false

        scrollState.animateScrollTo(memory.verticalPosition)
        return true
    }

    /**
     * Clear scroll position memory for a panel
     */
    fun clearScrollMemory(panelId: String) {
        _scrollPositionMemory.value = _scrollPositionMemory.value - panelId
    }

    /**
     * Configure minimap for a panel
     */
    fun configureMinimap(panelId: String, config: MinimapConfig) {
        _minimapConfigs.value = _minimapConfigs.value + (panelId to config)
    }

    /**
     * Update momentum scroll configuration
     */
    fun updateMomentumConfig(config: MomentumScrollConfig) {
        _momentumScrollConfig.value = config
    }

    /**
     * Get scroll progress for a panel (0.0 to 1.0)
     */
    fun getScrollProgress(panelId: String): Float {
        val scrollState = _scrollStates.value[panelId] ?: return 0f
        return if (scrollState.maxValue > 0) {
            scrollState.value.toFloat() / scrollState.maxValue.toFloat()
        } else {
            0f
        }
    }

    /**
     * Scroll to a specific progress (0.0 to 1.0)
     */
    suspend fun scrollToProgress(panelId: String, progress: Float) {
        val scrollState = _scrollStates.value[panelId] ?: return
        val targetPosition = (progress * scrollState.maxValue).toInt()
        scrollState.animateScrollTo(targetPosition)
    }
}

/**
 * Scroll direction enum
 */
enum class ScrollDirection {
    HORIZONTAL,
    VERTICAL,
}

/**
 * Enhanced scrollable content with momentum scrolling
 */
@Composable
fun AdvancedScrollableContent(
    panelId: String,
    scrollingManager: AdvancedScrollingManager,
    modifier: Modifier = Modifier,
    enableMomentumScroll: Boolean = true,
    enableMinimap: Boolean = false,
    minimapConfig: MinimapConfig = MinimapConfig(),
    content: @Composable ColumnScope.() -> Unit,
) {
    val scrollState = rememberScrollState()
    val density = LocalDensity.current

    // Register scroll state
    LaunchedEffect(panelId) {
        scrollingManager.registerScrollState(panelId, scrollState)
        if (enableMinimap) {
            scrollingManager.configureMinimap(panelId, minimapConfig)
        }
    }

    // Cleanup on disposal
    DisposableEffect(panelId) {
        onDispose {
            scrollingManager.unregisterScrollState(panelId)
        }
    }

    var lastScrollTime by remember { mutableStateOf(0L) }
    var scrollVelocity by remember { mutableStateOf(0f) }

    Box(modifier = modifier.fillMaxSize()) {
        // Main scrollable content
        Column(
            modifier =
            Modifier
                .fillMaxSize()
                .verticalScroll(scrollState)
                .pointerInput(panelId) {
                    if (enableMomentumScroll) {
                        detectDragGestures(
                            onDragEnd = {
                                val currentTime = System.currentTimeMillis()
                                val timeDelta = currentTime - lastScrollTime

                                if (timeDelta > 0 && abs(scrollVelocity) > 50f) {
                                    scrollingManager.scope.launch {
                                        scrollingManager.performMomentumScroll(
                                            panelId = panelId,
                                            initialVelocity = scrollVelocity,
                                        )
                                    }
                                }
                            },
                        ) { _, dragAmount ->
                            val currentTime = System.currentTimeMillis()
                            val timeDelta = currentTime - lastScrollTime

                            if (timeDelta > 0) {
                                scrollVelocity = dragAmount.y / (timeDelta / 1000f)
                            }

                            lastScrollTime = currentTime
                        }
                    }
                },
        ) {
            content()
        }

        // Minimap overlay
        if (enableMinimap) {
            MinimapOverlay(
                panelId = panelId,
                scrollingManager = scrollingManager,
                config = minimapConfig,
            )
        }
    }
}

/**
 * Minimap overlay component
 */
@Composable
private fun BoxScope.MinimapOverlay(panelId: String, scrollingManager: AdvancedScrollingManager, config: MinimapConfig) {
    val scrollProgress by remember {
        derivedStateOf { scrollingManager.getScrollProgress(panelId) }
    }

    var isVisible by remember { mutableStateOf(true) }
    var lastInteractionTime by remember { mutableStateOf(System.currentTimeMillis()) }

    // Auto-hide logic
    LaunchedEffect(scrollProgress) {
        if (config.autoHide) {
            lastInteractionTime = System.currentTimeMillis()
            isVisible = true

            delay(config.autoHideDelay)
            if (System.currentTimeMillis() - lastInteractionTime >= config.autoHideDelay) {
                isVisible = false
            }
        }
    }

    val alpha by animateFloatAsState(
        targetValue = if (isVisible) config.opacity else 0f,
        animationSpec = tween(300),
    )

    if (alpha > 0f) {
        val alignment =
            when (config.position) {
                MinimapPosition.TOP_LEFT -> Alignment.TopStart
                MinimapPosition.TOP_RIGHT -> Alignment.TopEnd
                MinimapPosition.BOTTOM_LEFT -> Alignment.BottomStart
                MinimapPosition.BOTTOM_RIGHT -> Alignment.BottomEnd
            }

        Card(
            modifier =
            Modifier
                .align(alignment)
                .size(config.width, config.height)
                .padding(16.dp)
                .alpha(alpha)
                .zIndex(10f),
            elevation = CardDefaults.cardElevation(defaultElevation = 4.dp),
            colors =
            CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surface.copy(alpha = 0.9f),
            ),
        ) {
            Box(
                modifier =
                Modifier
                    .fillMaxSize()
                    .padding(4.dp),
            ) {
                // Minimap background
                Box(
                    modifier =
                    Modifier
                        .fillMaxSize()
                        .background(
                            MaterialTheme.colorScheme.surfaceVariant,
                            RoundedCornerShape(4.dp),
                        ).border(
                            1.dp,
                            MaterialTheme.colorScheme.outline.copy(alpha = 0.3f),
                            RoundedCornerShape(4.dp),
                        ),
                )

                // Viewport indicator
                if (config.showViewport) {
                    val viewportHeight = 20.dp
                    val viewportTop = (config.height - 8.dp - viewportHeight) * scrollProgress

                    Box(
                        modifier =
                        Modifier
                            .offset(y = viewportTop)
                            .fillMaxWidth()
                            .height(viewportHeight)
                            .padding(horizontal = 2.dp)
                            .background(
                                MaterialTheme.colorScheme.primary.copy(alpha = 0.6f),
                                RoundedCornerShape(2.dp),
                            ).border(
                                1.dp,
                                MaterialTheme.colorScheme.primary,
                                RoundedCornerShape(2.dp),
                            ).pointerInput(panelId) {
                                detectDragGestures { _, dragAmount ->
                                    val newProgress =
                                        (scrollProgress + dragAmount.y / size.height)
                                            .coerceIn(0f, 1f)

                                    scrollingManager.scope.launch {
                                        scrollingManager.scrollToProgress(panelId, newProgress)
                                    }

                                    lastInteractionTime = System.currentTimeMillis()
                                }
                            },
                    )
                }

                // Scroll progress indicator
                Text(
                    text = "${(scrollProgress * 100).toInt()}%",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurface,
                    modifier = Modifier.align(Alignment.BottomCenter),
                )
            }
        }
    }
}

/**
 * Scroll synchronization controls
 */
@Composable
fun ScrollSyncControls(scrollingManager: AdvancedScrollingManager, availablePanels: List<String>, modifier: Modifier = Modifier) {
    val synchronizedGroups by scrollingManager.synchronizedGroups.collectAsState()
    var selectedPanels by remember { mutableStateOf(setOf<String>()) }
    var syncMode by remember { mutableStateOf(ScrollSyncMode.VERTICAL) }

    Card(
        modifier = modifier.padding(8.dp),
        elevation = CardDefaults.cardElevation(defaultElevation = 2.dp),
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Text(
                text = "Scroll Synchronization",
                style = MaterialTheme.typography.titleMedium,
            )

            // Panel selection
            Text(
                text = "Select panels to synchronize:",
                style = MaterialTheme.typography.bodySmall,
            )

            availablePanels.forEach { panelId ->
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Checkbox(
                        checked = panelId in selectedPanels,
                        onCheckedChange = { checked ->
                            selectedPanels =
                                if (checked) {
                                    selectedPanels + panelId
                                } else {
                                    selectedPanels - panelId
                                }
                        },
                    )
                    Text(
                        text = panelId,
                        style = MaterialTheme.typography.bodyMedium,
                        modifier = Modifier.padding(start = 8.dp),
                    )
                }
            }

            // Sync mode selection
            Text(
                text = "Synchronization mode:",
                style = MaterialTheme.typography.bodySmall,
            )

            ScrollSyncMode.values().forEach { mode ->
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    RadioButton(
                        selected = syncMode == mode,
                        onClick = { syncMode = mode },
                    )
                    Text(
                        text = mode.name,
                        style = MaterialTheme.typography.bodyMedium,
                        modifier = Modifier.padding(start = 8.dp),
                    )
                }
            }

            // Create sync group button
            DebugButton(
                buttonId = "create-sync-group",
                onClick = {
                    if (selectedPanels.size >= 2) {
                        val groupId = "sync_group_${System.currentTimeMillis()}"
                        scrollingManager.createSyncGroup(groupId, selectedPanels, syncMode)
                        selectedPanels = emptySet()
                    }
                },
                enabled = selectedPanels.size >= 2,
            ) {
                Text("Create Sync Group")
            }

            // Active sync groups
            if (synchronizedGroups.isNotEmpty()) {
                Text(
                    text = "Active sync groups:",
                    style = MaterialTheme.typography.bodySmall,
                    modifier = Modifier.padding(top = 8.dp),
                )

                synchronizedGroups.forEach { (groupId, panelIds) ->
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Text(
                            text = panelIds.joinToString(", "),
                            style = MaterialTheme.typography.bodySmall,
                            modifier = Modifier.weight(1f),
                        )

                        DebugIconButton(
                            buttonId = "remove-sync-group",
                            onClick = { scrollingManager.removeSyncGroup(groupId) },
                        ) {
                            Icon(
                                imageVector = Icons.Default.Delete,
                                contentDescription = "Remove sync group",
                            )
                        }
                    }
                }
            }
        }
    }
}

/**
 * Composable function to remember an AdvancedScrollingManager instance
 */
@Composable
fun rememberAdvancedScrollingManager(): AdvancedScrollingManager {
    val scope = rememberCoroutineScope()
    return remember { AdvancedScrollingManager(scope) }
}
