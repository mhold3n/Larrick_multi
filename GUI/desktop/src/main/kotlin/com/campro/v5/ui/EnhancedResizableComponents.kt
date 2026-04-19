package com.campro.v5.ui

import androidx.compose.animation.*
import androidx.compose.foundation.HorizontalScrollbar
import androidx.compose.foundation.LocalScrollbarStyle
import androidx.compose.foundation.VerticalScrollbar
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.hoverable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.interaction.collectIsHoveredAsState
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.rememberScrollbarAdapter
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowDownward
import androidx.compose.material.icons.filled.ArrowUpward
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.input.pointer.PointerIcon
import androidx.compose.ui.input.pointer.pointerHoverIcon
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.DpSize
import androidx.compose.ui.unit.dp
import androidx.compose.ui.zIndex
import com.campro.v5.debug.DebugIconButton
import java.awt.Cursor
import kotlin.math.max

/**
 * Enhanced resize directions supporting all edges and corners
 */
enum class ResizeDirection {
    BOTTOM,
    RIGHT,
    BOTTOM_RIGHT,
    LEFT,
    TOP,
    TOP_LEFT,
    TOP_RIGHT,
    BOTTOM_LEFT,
}

/**
 * Multi-directional resize handle with proper cursor feedback and enhanced visibility
 */
@Composable
fun MultiDirectionalResizeHandle(
    modifier: Modifier = Modifier,
    onResize: (deltaX: Float, deltaY: Float) -> Unit,
    resizeDirection: ResizeDirection = ResizeDirection.BOTTOM_RIGHT,
) {
    val cursor =
        when (resizeDirection) {
            ResizeDirection.BOTTOM -> Cursor.S_RESIZE_CURSOR
            ResizeDirection.RIGHT -> Cursor.E_RESIZE_CURSOR
            ResizeDirection.BOTTOM_RIGHT -> Cursor.SE_RESIZE_CURSOR
            ResizeDirection.LEFT -> Cursor.W_RESIZE_CURSOR
            ResizeDirection.TOP -> Cursor.N_RESIZE_CURSOR
            ResizeDirection.TOP_LEFT -> Cursor.NW_RESIZE_CURSOR
            ResizeDirection.TOP_RIGHT -> Cursor.NE_RESIZE_CURSOR
            ResizeDirection.BOTTOM_LEFT -> Cursor.SW_RESIZE_CURSOR
        }

    // Create interaction source for hover state
    val interactionSource = remember { MutableInteractionSource() }
    val isHovered by interactionSource.collectIsHoveredAsState()

    // Determine size and appearance based on hover state
    val handleSize = if (isHovered) 14.dp else 12.dp
    val handleAlpha = if (isHovered) 0.7f else 0.5f
    val shadowElevation = if (isHovered) 4.dp else 2.dp

    Box(
        modifier =
        modifier
            .size(handleSize)
            .shadow(shadowElevation, RoundedCornerShape(3.dp))
            .hoverable(interactionSource)
            .pointerHoverIcon(PointerIcon(Cursor.getPredefinedCursor(cursor)))
            .pointerInput(Unit) {
                detectDragGestures { change, _ ->
                    onResize(change.position.x, change.position.y)
                }
            }.background(
                MaterialTheme.colorScheme.primary.copy(alpha = handleAlpha),
                RoundedCornerShape(3.dp),
            ).border(
                width = 1.dp,
                color = MaterialTheme.colorScheme.onPrimary.copy(alpha = 0.3f),
                shape = RoundedCornerShape(3.dp),
            ),
    )
}

/**
 * Enhanced resizable container with multi-directional resizing,
 * overlap prevention, and padding preservation
 */
@Composable
fun ResizableContainer(
    modifier: Modifier = Modifier,
    title: String = "",
    initialWidth: Dp = Dp.Unspecified,
    initialHeight: Dp = 300.dp,
    minWidth: Dp = 200.dp,
    minHeight: Dp = 150.dp,
    maxWidth: Dp = Dp.Unspecified,
    maxHeight: Dp = Dp.Unspecified,
    preservePadding: Dp = 8.dp, // Reduced padding to maximize space
    enabledDirections: Set<ResizeDirection> = setOf(ResizeDirection.BOTTOM, ResizeDirection.RIGHT, ResizeDirection.BOTTOM_RIGHT),
    onSizeChanged: (width: Dp, height: Dp) -> Unit = { _, _ -> },
    content: @Composable BoxScope.() -> Unit,
) {
    var height by remember { mutableStateOf(initialHeight) }
    var width by remember {
        mutableStateOf(if (initialWidth != Dp.Unspecified) initialWidth else 400.dp)
    }

    val density = LocalDensity.current

    // Calculate available space considering siblings and padding
    val availableSpace =
        remember(density) {
            // This would be enhanced with actual parent constraint checking
            // For now, using reasonable defaults
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
            ),
        elevation = CardDefaults.cardElevation(defaultElevation = 4.dp),
    ) {
        Column {
            // Title bar
            if (title.isNotEmpty()) {
                Surface(
                    modifier = Modifier.fillMaxWidth(),
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

            // Content area with resize handles
            Box(modifier = Modifier.weight(1f)) {
                // Scrollable content
                EnhancedScrollableContent(
                    modifier = Modifier.fillMaxSize(),
                    content = content,
                )

                // Resize handles for enabled directions
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

                    MultiDirectionalResizeHandle(
                        modifier = handleModifier,
                        resizeDirection = direction,
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

/**
 * Enhanced scrollable content with visual feedback and scroll indicators
 */
@Composable
fun EnhancedScrollableContent(
    modifier: Modifier = Modifier,
    enableVerticalScroll: Boolean = true,
    enableHorizontalScroll: Boolean = false,
    showScrollIndicators: Boolean = true,
    content: @Composable BoxScope.() -> Unit,
) {
    val verticalScrollState = rememberScrollState()
    val horizontalScrollState = rememberScrollState()
    val isScrolling by remember {
        derivedStateOf {
            verticalScrollState.isScrollInProgress || horizontalScrollState.isScrollInProgress
        }
    }

    Box(modifier = modifier.fillMaxSize()) {
        var contentModifier =
            Modifier
                .fillMaxSize()
                .padding(8.dp)

        if (enableVerticalScroll) {
            contentModifier = contentModifier.verticalScroll(verticalScrollState)
        }
        if (enableHorizontalScroll) {
            contentModifier = contentModifier.horizontalScroll(horizontalScrollState)
        }

        Box(modifier = contentModifier) {
            content()
        }

        // Enhanced scrollbar with fade animation
        if (showScrollIndicators && enableVerticalScroll) {
            AnimatedVisibility(
                visible = verticalScrollState.maxValue > 0 || isScrolling,
                modifier = Modifier.align(Alignment.CenterEnd),
                enter = fadeIn(),
                exit = fadeOut(),
            ) {
                VerticalScrollbar(
                    modifier =
                    Modifier
                        .fillMaxHeight()
                        .padding(2.dp),
                    adapter = rememberScrollbarAdapter(verticalScrollState),
                    style =
                    LocalScrollbarStyle.current.copy(
                        unhoverColor = MaterialTheme.colorScheme.outline.copy(alpha = 0.5f),
                        hoverColor = MaterialTheme.colorScheme.primary,
                    ),
                )
            }
        }

        if (showScrollIndicators && enableHorizontalScroll) {
            AnimatedVisibility(
                visible = horizontalScrollState.maxValue > 0 || isScrolling,
                modifier = Modifier.align(Alignment.BottomCenter),
                enter = fadeIn(),
                exit = fadeOut(),
            ) {
                HorizontalScrollbar(
                    modifier =
                    Modifier
                        .fillMaxWidth()
                        .padding(2.dp),
                    adapter = rememberScrollbarAdapter(horizontalScrollState),
                    style =
                    LocalScrollbarStyle.current.copy(
                        unhoverColor = MaterialTheme.colorScheme.outline.copy(alpha = 0.5f),
                        hoverColor = MaterialTheme.colorScheme.primary,
                    ),
                )
            }
        }

        // Scroll position indicator
        if (isScrolling && verticalScrollState.maxValue > 0) {
            Surface(
                modifier =
                Modifier
                    .align(Alignment.TopEnd)
                    .padding(16.dp),
                shape = RoundedCornerShape(4.dp),
                color = MaterialTheme.colorScheme.inverseSurface.copy(alpha = 0.8f),
            ) {
                Text(
                    text = "${(verticalScrollState.value.toFloat() / verticalScrollState.maxValue * 100).toInt()}%",
                    modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp),
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.inverseOnSurface,
                )
            }
        }
    }
}

/**
 * Horizontal split pane with draggable divider
 * Modified to have a more subtle divider
 */
@Composable
fun HorizontalSplitPane(
    modifier: Modifier = Modifier,
    splitRatio: Float = 0.5f,
    minRatio: Float = 0.2f,
    maxRatio: Float = 0.8f,
    dividerWidth: Dp = 2.dp, // Reduced from 4.dp to 2.dp
    leftContent: @Composable BoxScope.() -> Unit,
    rightContent: @Composable BoxScope.() -> Unit,
) {
    var currentRatio by remember { mutableStateOf(splitRatio.coerceIn(minRatio, maxRatio)) }

    Row(modifier = modifier.fillMaxSize()) {
        // Left panel
        Box(modifier = Modifier.weight(currentRatio)) {
            leftContent()
        }

        // Draggable divider - modified to be less prominent
        Box(
            modifier =
            Modifier
                .width(dividerWidth)
                .fillMaxHeight()
                .pointerInput(Unit) {
                    detectDragGestures { change, _ ->
                        val newRatio =
                            (currentRatio + change.position.x / size.width)
                                .coerceIn(minRatio, maxRatio)
                        currentRatio = newRatio
                    }
                }.pointerHoverIcon(PointerIcon(Cursor.getPredefinedCursor(Cursor.E_RESIZE_CURSOR)))
                // Use a more subtle color with transparency
                .background(MaterialTheme.colorScheme.outline.copy(alpha = 0.3f)),
        )

        // Right panel
        Box(modifier = Modifier.weight(1f - currentRatio)) {
            rightContent()
        }
    }
}

/**
 * Vertical split pane with draggable divider
 */
@Composable
fun VerticalSplitPane(
    modifier: Modifier = Modifier,
    splitRatio: Float = 0.5f,
    minRatio: Float = 0.2f,
    maxRatio: Float = 0.8f,
    dividerHeight: Dp = 4.dp,
    topContent: @Composable BoxScope.() -> Unit,
    bottomContent: @Composable BoxScope.() -> Unit,
) {
    var currentRatio by remember { mutableStateOf(splitRatio.coerceIn(minRatio, maxRatio)) }

    Column(modifier = modifier.fillMaxSize()) {
        // Top panel
        Box(modifier = Modifier.weight(currentRatio)) {
            topContent()
        }

        // Draggable divider
        Box(
            modifier =
            Modifier
                .height(dividerHeight)
                .fillMaxWidth()
                .pointerInput(Unit) {
                    detectDragGestures { change, _ ->
                        val newRatio =
                            (currentRatio + change.position.y / size.height)
                                .coerceIn(minRatio, maxRatio)
                        currentRatio = newRatio
                    }
                }.pointerHoverIcon(PointerIcon(Cursor.getPredefinedCursor(Cursor.S_RESIZE_CURSOR)))
                .background(MaterialTheme.colorScheme.outline),
        )

        // Bottom panel
        Box(modifier = Modifier.weight(1f - currentRatio)) {
            bottomContent()
        }
    }
}

/**
 * Draggable and resizable panel that can be freely positioned anywhere within its parent container.
 * Supports stacking (z-index) and brings panel to front when dragged.
 */
@Composable
fun DraggableResizablePanel(
    modifier: Modifier = Modifier,
    title: String = "",
    initialX: Dp = 0.dp,
    initialY: Dp = 0.dp,
    initialWidth: Dp = 400.dp,
    initialHeight: Dp = 300.dp,
    minWidth: Dp = 200.dp,
    minHeight: Dp = 150.dp,
    zIndex: Float = 0f,
    onZIndexChange: (Float) -> Unit = {},
    onPositionChange: (x: Dp, y: Dp) -> Unit = { _, _ -> },
    onSizeChange: (width: Dp, height: Dp) -> Unit = { _, _ -> },
    content: @Composable BoxScope.() -> Unit,
) {
    // State for position and size
    var x by remember { mutableStateOf(initialX) }
    var y by remember { mutableStateOf(initialY) }
    var width by remember { mutableStateOf(initialWidth) }
    var height by remember { mutableStateOf(initialHeight) }

    // State for dragging
    var isDragging by remember { mutableStateOf(false) }

    // Update parent when position or size changes
    LaunchedEffect(x, y) {
        onPositionChange(x, y)
    }

    LaunchedEffect(width, height) {
        onSizeChange(width, height)
    }

    Box(
        modifier =
        modifier
            .zIndex(zIndex)
            .offset(x = x, y = y)
            .size(width = width, height = height),
    ) {
        Card(
            modifier =
            Modifier
                .fillMaxSize()
                .shadow(
                    elevation = if (isDragging) 16.dp else 4.dp,
                    shape = RoundedCornerShape(8.dp),
                ),
            elevation =
            CardDefaults.cardElevation(
                defaultElevation = if (isDragging) 8.dp else 4.dp,
            ),
        ) {
            Column {
                // Title bar with drag handle
                Surface(
                    modifier = Modifier.fillMaxWidth(),
                    color = MaterialTheme.colorScheme.primaryContainer,
                ) {
                    Row(
                        modifier =
                        Modifier
                            .fillMaxWidth()
                            .pointerInput(Unit) {
                                detectDragGestures(
                                    onDragStart = {
                                        isDragging = true
                                        // Bring panel to front when dragging starts
                                        onZIndexChange(zIndex + 1f)
                                    },
                                    onDragEnd = { isDragging = false },
                                    onDragCancel = { isDragging = false },
                                    onDrag = { change, dragAmount ->
                                        change.consume()
                                        x += dragAmount.x.toDp()
                                        y += dragAmount.y.toDp()
                                    },
                                )
                            }.pointerHoverIcon(PointerIcon(Cursor.getPredefinedCursor(Cursor.MOVE_CURSOR)))
                            .padding(horizontal = 12.dp, vertical = 8.dp),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Text(
                            text = title,
                            style = MaterialTheme.typography.titleMedium,
                            color = MaterialTheme.colorScheme.onPrimaryContainer,
                        )

                        // Z-index control buttons
                        Row(
                            horizontalArrangement = Arrangement.spacedBy(4.dp),
                        ) {
                            DebugIconButton(
                                buttonId = "resize-dock",
                                onClick = { /* TODO: Dock */ },
                                modifier = Modifier.size(20.dp),
                            ) {
                                Icon(
                                    imageVector = Icons.Default.Dock,
                                    contentDescription = "Dock",
                                    modifier = Modifier.size(14.dp),
                                )
                            }
                            DebugIconButton(
                                buttonId = "resize-undock",
                                onClick = { /* TODO: Undock */ },
                                modifier = Modifier.size(20.dp),
                            ) {
                                Icon(
                                    imageVector = Icons.Default.Undock,
                                    contentDescription = "Undock",
                                    modifier = Modifier.size(14.dp),
                                )
                            }
                            IconButton(
                                onClick = { onZIndexChange(zIndex + 1f) },
                                modifier = Modifier.size(24.dp),
                            ) {
                                Icon(
                                    imageVector = Icons.Default.ArrowUpward,
                                    contentDescription = "Bring to front",
                                    tint = MaterialTheme.colorScheme.onPrimaryContainer,
                                )
                            }
                            IconButton(
                                onClick = { onZIndexChange(max(0f, zIndex - 1f)) },
                                modifier = Modifier.size(24.dp),
                            ) {
                                Icon(
                                    imageVector = Icons.Default.ArrowDownward,
                                    contentDescription = "Send to back",
                                    tint = MaterialTheme.colorScheme.onPrimaryContainer,
                                )
                            }
                        }
                    }
                }

                // Content area
                Box(modifier = Modifier.weight(1f)) {
                    EnhancedScrollableContent(
                        modifier = Modifier.fillMaxSize(),
                        content = content,
                    )

                    // Resize handles in all directions
                    ResizeDirection.values().forEach { direction ->
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

                        MultiDirectionalResizeHandle(
                            modifier = handleModifier,
                            resizeDirection = direction,
                            onResize = { deltaX, deltaY ->
                                val newWidth =
                                    when (direction) {
                                        ResizeDirection.RIGHT, ResizeDirection.BOTTOM_RIGHT, ResizeDirection.TOP_RIGHT ->
                                            (width + deltaX.dp).coerceAtLeast(minWidth)
                                        ResizeDirection.LEFT, ResizeDirection.BOTTOM_LEFT, ResizeDirection.TOP_LEFT ->
                                            (width - deltaX.dp).coerceAtLeast(minWidth)
                                        else -> width
                                    }

                                val newHeight =
                                    when (direction) {
                                        ResizeDirection.BOTTOM, ResizeDirection.BOTTOM_RIGHT, ResizeDirection.BOTTOM_LEFT ->
                                            (height + deltaY.dp).coerceAtLeast(minHeight)
                                        ResizeDirection.TOP, ResizeDirection.TOP_LEFT, ResizeDirection.TOP_RIGHT ->
                                            (height - deltaY.dp).coerceAtLeast(minHeight)
                                        else -> height
                                    }

                                // Update position when resizing from left or top
                                if (direction == ResizeDirection.LEFT ||
                                    direction == ResizeDirection.TOP_LEFT ||
                                    direction == ResizeDirection.BOTTOM_LEFT
                                ) {
                                    x += (width - newWidth)
                                }

                                if (direction == ResizeDirection.TOP ||
                                    direction == ResizeDirection.TOP_LEFT ||
                                    direction == ResizeDirection.TOP_RIGHT
                                ) {
                                    y += (height - newHeight)
                                }

                                width = newWidth
                                height = newHeight
                            },
                        )
                    }
                }
            }
        }
    }
}
