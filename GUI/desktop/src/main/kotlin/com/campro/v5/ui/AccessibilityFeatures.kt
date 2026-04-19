package com.campro.v5.ui

import androidx.compose.foundation.focusable
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.ExperimentalComposeUiApi
import androidx.compose.ui.Modifier
import androidx.compose.ui.focus.FocusRequester
import androidx.compose.ui.focus.focusRequester
import androidx.compose.ui.focus.onFocusChanged
import androidx.compose.ui.input.key.*
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.semantics.*
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.DpSize
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.delay

/**
 * Accessible resize handle with keyboard navigation and screen reader support
 */
@OptIn(ExperimentalComposeUiApi::class)
@Composable
fun AccessibleResizeHandle(
    modifier: Modifier = Modifier,
    onResize: (deltaX: Float, deltaY: Float) -> Unit,
    resizeDirection: ResizeDirection = ResizeDirection.BOTTOM_RIGHT,
    currentWidth: Dp,
    currentHeight: Dp,
    minWidth: Dp = 200.dp,
    minHeight: Dp = 150.dp,
    maxWidth: Dp = 800.dp,
    maxHeight: Dp = 600.dp,
) {
    val focusRequester = remember { FocusRequester() }
    var isFocused by remember { mutableStateOf(false) }
    var isResizing by remember { mutableStateOf(false) }

    val density = LocalDensity.current
    val keyboardResizeStep = 10.dp

    // Haptic feedback simulation (would use actual haptic API in real implementation)
    fun triggerHapticFeedback() {
        // Placeholder for haptic feedback
        // In a real implementation, this would use platform-specific haptic APIs
    }

    Box(
        modifier =
        modifier
            .size(12.dp) // Larger size for better accessibility
            .focusRequester(focusRequester)
            .focusable()
            .onFocusChanged { isFocused = it.isFocused }
            .onKeyEvent { keyEvent ->
                if (keyEvent.type == KeyEventType.KeyDown && isFocused) {
                    val isCtrlPressed = keyEvent.isCtrlPressed
                    val step = if (isCtrlPressed) keyboardResizeStep * 2 else keyboardResizeStep

                    when (keyEvent.key) {
                        Key.DirectionRight -> {
                            if (resizeDirection in
                                setOf(ResizeDirection.RIGHT, ResizeDirection.BOTTOM_RIGHT, ResizeDirection.TOP_RIGHT)
                            ) {
                                with(density) { onResize(step.toPx(), 0f) }
                                triggerHapticFeedback()
                                true
                            } else {
                                false
                            }
                        }
                        Key.DirectionLeft -> {
                            if (resizeDirection in setOf(ResizeDirection.LEFT, ResizeDirection.BOTTOM_LEFT, ResizeDirection.TOP_LEFT)) {
                                with(density) { onResize(-step.toPx(), 0f) }
                                triggerHapticFeedback()
                                true
                            } else {
                                false
                            }
                        }
                        Key.DirectionDown -> {
                            if (resizeDirection in
                                setOf(ResizeDirection.BOTTOM, ResizeDirection.BOTTOM_RIGHT, ResizeDirection.BOTTOM_LEFT)
                            ) {
                                with(density) { onResize(0f, step.toPx()) }
                                triggerHapticFeedback()
                                true
                            } else {
                                false
                            }
                        }
                        Key.DirectionUp -> {
                            if (resizeDirection in setOf(ResizeDirection.TOP, ResizeDirection.TOP_LEFT, ResizeDirection.TOP_RIGHT)) {
                                with(density) { onResize(0f, -step.toPx()) }
                                triggerHapticFeedback()
                                true
                            } else {
                                false
                            }
                        }
                        Key.Enter, Key.Spacebar -> {
                            // Auto-fit functionality
                            autoFitContent(onResize, resizeDirection, currentWidth, currentHeight, minWidth, minHeight)
                            triggerHapticFeedback()
                            true
                        }
                        else -> false
                    }
                } else {
                    false
                }
            }.pointerInput(Unit) {
                detectDragGestures(
                    onDragStart = {
                        isResizing = true
                        triggerHapticFeedback()
                    },
                    onDragEnd = {
                        isResizing = false
                    },
                ) { change, _ ->
                    onResize(change.position.x, change.position.y)
                }
            }.semantics {
                role = Role.Button
                contentDescription = "Resize handle for ${resizeDirection.name.lowercase().replace('_', ' ')} direction. " +
                    "Current size: ${currentWidth.value.toInt()}x${currentHeight.value.toInt()}dp. " +
                    "Use arrow keys to resize, Ctrl+arrow for larger steps, Enter or Space to auto-fit."

                // Custom actions for screen readers
                customActions =
                    listOf(
                        CustomAccessibilityAction("Resize larger") {
                            val step = keyboardResizeStep
                            when (resizeDirection) {
                                ResizeDirection.BOTTOM_RIGHT ->
                                    with(density) {
                                        onResize(step.toPx(), step.toPx())
                                    }
                                ResizeDirection.RIGHT ->
                                    with(density) {
                                        onResize(step.toPx(), 0f)
                                    }
                                ResizeDirection.BOTTOM ->
                                    with(density) {
                                        onResize(0f, step.toPx())
                                    }
                                else -> {}
                            }
                            true
                        },
                        CustomAccessibilityAction("Resize smaller") {
                            val step = keyboardResizeStep
                            when (resizeDirection) {
                                ResizeDirection.BOTTOM_RIGHT ->
                                    with(density) {
                                        onResize(-step.toPx(), -step.toPx())
                                    }
                                ResizeDirection.RIGHT ->
                                    with(density) {
                                        onResize(-step.toPx(), 0f)
                                    }
                                ResizeDirection.BOTTOM ->
                                    with(density) {
                                        onResize(0f, -step.toPx())
                                    }
                                else -> {}
                            }
                            true
                        },
                        CustomAccessibilityAction("Auto-fit content") {
                            autoFitContent(onResize, resizeDirection, currentWidth, currentHeight, minWidth, minHeight)
                            true
                        },
                    )
            },
    ) {
        Surface(
            modifier = Modifier.fillMaxSize(),
            shape = RoundedCornerShape(4.dp),
            color =
            when {
                isResizing -> MaterialTheme.colorScheme.primary
                isFocused -> MaterialTheme.colorScheme.primary.copy(alpha = 0.7f)
                else -> MaterialTheme.colorScheme.primary.copy(alpha = 0.3f)
            },
            border =
            if (isFocused) {
                androidx.compose.foundation.BorderStroke(2.dp, MaterialTheme.colorScheme.primary)
            } else {
                null
            },
        ) {
            // Visual indicator for resize direction
            Box(
                modifier = Modifier.fillMaxSize(),
                contentAlignment = Alignment.Center,
            ) {
                Text(
                    text =
                    when (resizeDirection) {
                        ResizeDirection.BOTTOM -> "↓"
                        ResizeDirection.RIGHT -> "→"
                        ResizeDirection.BOTTOM_RIGHT -> "↘"
                        ResizeDirection.LEFT -> "←"
                        ResizeDirection.TOP -> "↑"
                        ResizeDirection.TOP_LEFT -> "↖"
                        ResizeDirection.TOP_RIGHT -> "↗"
                        ResizeDirection.BOTTOM_LEFT -> "↙"
                    },
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onPrimary,
                )
            }
        }
    }
}

/**
 * Auto-fit functionality that adjusts container size to optimal dimensions
 */
private fun autoFitContent(
    onResize: (deltaX: Float, deltaY: Float) -> Unit,
    resizeDirection: ResizeDirection,
    currentWidth: Dp,
    currentHeight: Dp,
    minWidth: Dp,
    minHeight: Dp,
) {
    // Simple auto-fit logic - in a real implementation, this would analyze content
    val optimalWidth = 400.dp
    val optimalHeight = 300.dp

    val deltaWidth = optimalWidth - currentWidth
    val deltaHeight = optimalHeight - currentHeight

    when (resizeDirection) {
        ResizeDirection.BOTTOM_RIGHT -> {
            // Convert Dp to pixels (approximation)
            onResize(deltaWidth.value * 3f, deltaHeight.value * 3f)
        }
        ResizeDirection.RIGHT -> {
            onResize(deltaWidth.value * 3f, 0f)
        }
        ResizeDirection.BOTTOM -> {
            onResize(0f, deltaHeight.value * 3f)
        }
        else -> {
            // Handle other directions as needed
        }
    }
}

/**
 * Accessible container with enhanced keyboard navigation and announcements
 */
@Composable
fun AccessibleResizableContainer(
    modifier: Modifier = Modifier,
    title: String = "",
    initialWidth: Dp = Dp.Unspecified,
    initialHeight: Dp = 300.dp,
    minWidth: Dp = 200.dp,
    minHeight: Dp = 150.dp,
    maxWidth: Dp = Dp.Unspecified,
    maxHeight: Dp = Dp.Unspecified,
    preservePadding: Dp = 16.dp,
    enabledDirections: Set<ResizeDirection> = setOf(ResizeDirection.BOTTOM, ResizeDirection.RIGHT, ResizeDirection.BOTTOM_RIGHT),
    onSizeChanged: (width: Dp, height: Dp) -> Unit = { _, _ -> },
    content: @Composable BoxScope.() -> Unit,
) {
    var height by remember { mutableStateOf(initialHeight) }
    var width by remember {
        mutableStateOf(if (initialWidth != Dp.Unspecified) initialWidth else 400.dp)
    }
    var lastAnnouncedSize by remember { mutableStateOf("") }

    // Announce size changes to screen readers
    LaunchedEffect(width, height) {
        delay(500) // Debounce announcements
        val sizeAnnouncement = "Container resized to ${width.value.toInt()} by ${height.value.toInt()} pixels"
        if (sizeAnnouncement != lastAnnouncedSize) {
            lastAnnouncedSize = sizeAnnouncement
            // In a real implementation, this would use platform-specific TTS or screen reader APIs
        }
    }

    val density = LocalDensity.current

    // Calculate available space considering siblings and padding
    val availableSpace =
        remember(density) {
            DpSize(
                width = if (maxWidth != Dp.Unspecified) maxWidth else 1200.dp,
                height = if (maxHeight != Dp.Unspecified) maxHeight else 800.dp,
            )
        }

    Card(
        modifier =
        modifier
            .padding(preservePadding)
            .then(
                if (initialWidth != Dp.Unspecified) {
                    Modifier.size(
                        width.coerceAtMost(availableSpace.width - preservePadding * 2),
                        height.coerceAtMost(availableSpace.height - preservePadding * 2),
                    )
                } else {
                    Modifier
                        .fillMaxWidth()
                        .height(height.coerceAtMost(availableSpace.height - preservePadding * 2))
                },
            ).semantics {
                contentDescription =
                    if (title.isNotEmpty()) {
                        "Resizable container: $title. Current size: ${width.value.toInt()} by ${height.value.toInt()} pixels"
                    } else {
                        "Resizable container. Current size: ${width.value.toInt()} by ${height.value.toInt()} pixels"
                    }
            },
        elevation = CardDefaults.cardElevation(defaultElevation = 4.dp),
    ) {
        Column {
            // Title bar
            if (title.isNotEmpty()) {
                Surface(
                    modifier =
                    Modifier
                        .fillMaxWidth()
                        .semantics {
                            heading()
                            contentDescription = "Container title: $title"
                        },
                    color = MaterialTheme.colorScheme.primaryContainer,
                ) {
                    Text(
                        text = title,
                        modifier = Modifier.padding(12.dp),
                        style = MaterialTheme.typography.titleMedium,
                        color = MaterialTheme.colorScheme.onPrimaryContainer,
                    )
                }
            }

            // Content area with accessible resize handles
            Box(modifier = Modifier.weight(1f)) {
                // Scrollable content
                EnhancedScrollableContent(
                    modifier = Modifier.fillMaxSize(),
                    content = content,
                )

                // Accessible resize handles for enabled directions
                enabledDirections.forEach { direction ->
                    val handleModifier =
                        when (direction) {
                            ResizeDirection.BOTTOM -> Modifier.align(Alignment.BottomCenter)
                            ResizeDirection.RIGHT -> Modifier.align(Alignment.CenterEnd)
                            ResizeDirection.BOTTOM_RIGHT -> Modifier.align(Alignment.BottomEnd)
                            ResizeDirection.LEFT -> Modifier.align(Alignment.CenterStart)
                            ResizeDirection.TOP -> Modifier.align(Alignment.TopCenter)
                            ResizeDirection.TOP_LEFT -> Modifier.align(Alignment.TopStart)
                            ResizeDirection.TOP_RIGHT -> Modifier.align(Alignment.TopEnd)
                            ResizeDirection.BOTTOM_LEFT -> Modifier.align(Alignment.BottomStart)
                        }

                    AccessibleResizeHandle(
                        modifier = handleModifier,
                        resizeDirection = direction,
                        currentWidth = width,
                        currentHeight = height,
                        minWidth = minWidth,
                        minHeight = minHeight,
                        maxWidth = if (maxWidth != Dp.Unspecified) maxWidth else availableSpace.width,
                        maxHeight = if (maxHeight != Dp.Unspecified) maxHeight else availableSpace.height,
                        onResize = { deltaX, deltaY ->
                            val newWidth =
                                when (direction) {
                                    ResizeDirection.RIGHT, ResizeDirection.BOTTOM_RIGHT, ResizeDirection.TOP_RIGHT ->
                                        (width + deltaX.dp).coerceIn(minWidth, availableSpace.width - preservePadding * 2)
                                    ResizeDirection.LEFT, ResizeDirection.BOTTOM_LEFT, ResizeDirection.TOP_LEFT ->
                                        (width - deltaX.dp).coerceIn(minWidth, availableSpace.width - preservePadding * 2)
                                    else -> width
                                }

                            val newHeight =
                                when (direction) {
                                    ResizeDirection.BOTTOM, ResizeDirection.BOTTOM_RIGHT, ResizeDirection.BOTTOM_LEFT ->
                                        (height + deltaY.dp).coerceIn(minHeight, availableSpace.height - preservePadding * 2)
                                    ResizeDirection.TOP, ResizeDirection.TOP_LEFT, ResizeDirection.TOP_RIGHT ->
                                        (height - deltaY.dp).coerceIn(minHeight, availableSpace.height - preservePadding * 2)
                                    else -> height
                                }

                            if (newWidth != width || newHeight != height) {
                                width = newWidth
                                height = newHeight
                                onSizeChanged(width, height)
                            }
                        },
                    )
                }
            }
        }
    }
}
