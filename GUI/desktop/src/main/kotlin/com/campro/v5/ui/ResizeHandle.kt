package com.campro.v5.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.hoverable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.layout.*
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.input.pointer.PointerIcon
import androidx.compose.ui.input.pointer.pointerHoverIcon
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.unit.dp
import androidx.compose.ui.zIndex
import java.awt.Cursor

/**
 * Enhanced resize handle types supporting all edges and corners
 */
enum class ResizeHandleType {
    TOP,
    BOTTOM,
    LEFT,
    RIGHT,
    TOP_LEFT,
    TOP_RIGHT,
    BOTTOM_LEFT,
    BOTTOM_RIGHT,
}

/**
 * Enhanced resize handle component with improved visual feedback and accessibility
 */
@Composable
fun ResizeHandle(
    type: ResizeHandleType,
    onDrag: (Offset) -> Unit,
    onDragStart: () -> Unit = {},
    onDragEnd: () -> Unit = {},
    modifier: Modifier = Modifier,
    isEnabled: Boolean = true,
    showVisualIndicator: Boolean = true,
) {
    var isHovered by remember { mutableStateOf(false) }
    var isDragging by remember { mutableStateOf(false) }
    val interactionSource = remember { MutableInteractionSource() }

    // Determine cursor type based on handle type
    val cursorType =
        when (type) {
            ResizeHandleType.TOP, ResizeHandleType.BOTTOM -> Cursor.N_RESIZE_CURSOR
            ResizeHandleType.LEFT, ResizeHandleType.RIGHT -> Cursor.E_RESIZE_CURSOR
            ResizeHandleType.TOP_LEFT, ResizeHandleType.BOTTOM_RIGHT -> Cursor.NW_RESIZE_CURSOR
            ResizeHandleType.TOP_RIGHT, ResizeHandleType.BOTTOM_LEFT -> Cursor.NE_RESIZE_CURSOR
        }

    // Handle size based on type
    val handleModifier =
        when (type) {
            ResizeHandleType.TOP, ResizeHandleType.BOTTOM -> modifier.fillMaxWidth().height(8.dp)
            ResizeHandleType.LEFT, ResizeHandleType.RIGHT -> modifier.fillMaxHeight().width(8.dp)
            ResizeHandleType.TOP_LEFT, ResizeHandleType.TOP_RIGHT,
            ResizeHandleType.BOTTOM_LEFT, ResizeHandleType.BOTTOM_RIGHT,
            -> modifier.size(12.dp)
        }

    Box(
        modifier =
        handleModifier
            .zIndex(if (type.name.contains("_")) 2f else 1f) // Corner handles on top
            .pointerHoverIcon(
                if (isEnabled) {
                    PointerIcon(java.awt.Cursor(cursorType))
                } else {
                    PointerIcon(java.awt.Cursor(Cursor.DEFAULT_CURSOR))
                },
            ).background(
                when {
                    !isEnabled -> Color.Transparent
                    isDragging -> MaterialTheme.colorScheme.primary.copy(alpha = 0.6f)
                    isHovered -> MaterialTheme.colorScheme.primary.copy(alpha = 0.4f)
                    showVisualIndicator -> MaterialTheme.colorScheme.outline.copy(alpha = 0.2f)
                    else -> Color.Transparent
                },
            ).clip(MaterialTheme.shapes.extraSmall)
            .pointerInput(type, isEnabled) {
                if (isEnabled) {
                    detectDragGestures(
                        onDragStart = { offset ->
                            isDragging = true
                            onDragStart()
                        },
                        onDragEnd = {
                            isDragging = false
                            onDragEnd()
                        },
                        onDrag = { change, dragAmount ->
                            onDrag(dragAmount)
                        },
                    )
                }
            }.hoverable(
                interactionSource = interactionSource,
                enabled = isEnabled,
            ),
    ) {
        // Visual feedback for resize handle
        if (showVisualIndicator && (isHovered || isDragging)) {
            Box(
                modifier =
                Modifier
                    .fillMaxSize()
                    .border(
                        width = 1.dp,
                        color =
                        if (isDragging) {
                            MaterialTheme.colorScheme.primary
                        } else {
                            MaterialTheme.colorScheme.primary.copy(alpha = 0.7f)
                        },
                        shape = MaterialTheme.shapes.extraSmall,
                    ),
            )
        }

        // Handle grip pattern for better visual indication
        if (showVisualIndicator && isHovered && !isDragging) {
            HandleGripPattern(type = type)
        }
    }

    // Track hover state
    LaunchedEffect(interactionSource) {
        interactionSource.interactions.collect { interaction ->
            when (interaction) {
                is androidx.compose.foundation.interaction.HoverInteraction.Enter -> {
                    isHovered = true
                }
                is androidx.compose.foundation.interaction.HoverInteraction.Exit -> {
                    isHovered = false
                }
            }
        }
    }
}

/**
 * Visual grip pattern for resize handles
 */
@Composable
private fun BoxScope.HandleGripPattern(type: ResizeHandleType) {
    val gripColor = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f)
    val gripSize = 2.dp
    val spacing = 1.dp

    when (type) {
        ResizeHandleType.TOP, ResizeHandleType.BOTTOM -> {
            // Horizontal grip lines
            Row(
                modifier = Modifier.align(androidx.compose.ui.Alignment.Center),
                horizontalArrangement = Arrangement.spacedBy(spacing),
            ) {
                repeat(3) {
                    Box(
                        modifier =
                        Modifier
                            .size(width = 8.dp, height = gripSize)
                            .background(gripColor, MaterialTheme.shapes.extraSmall),
                    )
                }
            }
        }
        ResizeHandleType.LEFT, ResizeHandleType.RIGHT -> {
            // Vertical grip lines
            Column(
                modifier = Modifier.align(androidx.compose.ui.Alignment.Center),
                verticalArrangement = Arrangement.spacedBy(spacing),
            ) {
                repeat(3) {
                    Box(
                        modifier =
                        Modifier
                            .size(width = gripSize, height = 8.dp)
                            .background(gripColor, MaterialTheme.shapes.extraSmall),
                    )
                }
            }
        }
        ResizeHandleType.TOP_LEFT, ResizeHandleType.TOP_RIGHT,
        ResizeHandleType.BOTTOM_LEFT, ResizeHandleType.BOTTOM_RIGHT,
        -> {
            // Diagonal grip pattern for corners
            Box(
                modifier =
                Modifier
                    .align(androidx.compose.ui.Alignment.Center)
                    .size(6.dp)
                    .background(gripColor, MaterialTheme.shapes.small),
            )
        }
    }
}

/**
 * Composable function to create all resize handles for a panel
 */
@Composable
fun BoxScope.ResizeHandles(
    panelWidth: androidx.compose.ui.unit.Dp,
    panelHeight: androidx.compose.ui.unit.Dp,
    minWidth: androidx.compose.ui.unit.Dp,
    minHeight: androidx.compose.ui.unit.Dp,
    maxWidth: androidx.compose.ui.unit.Dp,
    maxHeight: androidx.compose.ui.unit.Dp,
    onResize: (androidx.compose.ui.unit.Dp, androidx.compose.ui.unit.Dp) -> Unit,
    onResizeStart: (ResizeHandleType) -> Unit = {},
    onResizeEnd: () -> Unit = {},
    enabledHandles: Set<ResizeHandleType> = ResizeHandleType.values().toSet(),
    showVisualIndicators: Boolean = true,
) {
    val density = androidx.compose.ui.platform.LocalDensity.current

    // Top edge handle
    if (ResizeHandleType.TOP in enabledHandles) {
        ResizeHandle(
            type = ResizeHandleType.TOP,
            modifier = Modifier.align(androidx.compose.ui.Alignment.TopCenter),
            showVisualIndicator = showVisualIndicators,
            onDrag = { dragAmount ->
                val newHeight =
                    (panelHeight - with(density) { dragAmount.y.toDp() })
                        .coerceIn(minHeight, maxHeight)
                onResize(panelWidth, newHeight)
            },
            onDragStart = { onResizeStart(ResizeHandleType.TOP) },
            onDragEnd = onResizeEnd,
        )
    }

    // Bottom edge handle
    if (ResizeHandleType.BOTTOM in enabledHandles) {
        ResizeHandle(
            type = ResizeHandleType.BOTTOM,
            modifier = Modifier.align(androidx.compose.ui.Alignment.BottomCenter),
            showVisualIndicator = showVisualIndicators,
            onDrag = { dragAmount ->
                val newHeight =
                    (panelHeight + with(density) { dragAmount.y.toDp() })
                        .coerceIn(minHeight, maxHeight)
                onResize(panelWidth, newHeight)
            },
            onDragStart = { onResizeStart(ResizeHandleType.BOTTOM) },
            onDragEnd = onResizeEnd,
        )
    }

    // Left edge handle
    if (ResizeHandleType.LEFT in enabledHandles) {
        ResizeHandle(
            type = ResizeHandleType.LEFT,
            modifier = Modifier.align(androidx.compose.ui.Alignment.CenterStart),
            showVisualIndicator = showVisualIndicators,
            onDrag = { dragAmount ->
                val newWidth =
                    (panelWidth - with(density) { dragAmount.x.toDp() })
                        .coerceIn(minWidth, maxWidth)
                onResize(newWidth, panelHeight)
            },
            onDragStart = { onResizeStart(ResizeHandleType.LEFT) },
            onDragEnd = onResizeEnd,
        )
    }

    // Right edge handle
    if (ResizeHandleType.RIGHT in enabledHandles) {
        ResizeHandle(
            type = ResizeHandleType.RIGHT,
            modifier = Modifier.align(androidx.compose.ui.Alignment.CenterEnd),
            showVisualIndicator = showVisualIndicators,
            onDrag = { dragAmount ->
                val newWidth =
                    (panelWidth + with(density) { dragAmount.x.toDp() })
                        .coerceIn(minWidth, maxWidth)
                onResize(newWidth, panelHeight)
            },
            onDragStart = { onResizeStart(ResizeHandleType.RIGHT) },
            onDragEnd = onResizeEnd,
        )
    }

    // Top-left corner handle
    if (ResizeHandleType.TOP_LEFT in enabledHandles) {
        ResizeHandle(
            type = ResizeHandleType.TOP_LEFT,
            modifier = Modifier.align(androidx.compose.ui.Alignment.TopStart),
            showVisualIndicator = showVisualIndicators,
            onDrag = { dragAmount ->
                val newWidth =
                    (panelWidth - with(density) { dragAmount.x.toDp() })
                        .coerceIn(minWidth, maxWidth)
                val newHeight =
                    (panelHeight - with(density) { dragAmount.y.toDp() })
                        .coerceIn(minHeight, maxHeight)
                onResize(newWidth, newHeight)
            },
            onDragStart = { onResizeStart(ResizeHandleType.TOP_LEFT) },
            onDragEnd = onResizeEnd,
        )
    }

    // Top-right corner handle
    if (ResizeHandleType.TOP_RIGHT in enabledHandles) {
        ResizeHandle(
            type = ResizeHandleType.TOP_RIGHT,
            modifier = Modifier.align(androidx.compose.ui.Alignment.TopEnd),
            showVisualIndicator = showVisualIndicators,
            onDrag = { dragAmount ->
                val newWidth =
                    (panelWidth + with(density) { dragAmount.x.toDp() })
                        .coerceIn(minWidth, maxWidth)
                val newHeight =
                    (panelHeight - with(density) { dragAmount.y.toDp() })
                        .coerceIn(minHeight, maxHeight)
                onResize(newWidth, newHeight)
            },
            onDragStart = { onResizeStart(ResizeHandleType.TOP_RIGHT) },
            onDragEnd = onResizeEnd,
        )
    }

    // Bottom-left corner handle
    if (ResizeHandleType.BOTTOM_LEFT in enabledHandles) {
        ResizeHandle(
            type = ResizeHandleType.BOTTOM_LEFT,
            modifier = Modifier.align(androidx.compose.ui.Alignment.BottomStart),
            showVisualIndicator = showVisualIndicators,
            onDrag = { dragAmount ->
                val newWidth =
                    (panelWidth - with(density) { dragAmount.x.toDp() })
                        .coerceIn(minWidth, maxWidth)
                val newHeight =
                    (panelHeight + with(density) { dragAmount.y.toDp() })
                        .coerceIn(minHeight, maxHeight)
                onResize(newWidth, newHeight)
            },
            onDragStart = { onResizeStart(ResizeHandleType.BOTTOM_LEFT) },
            onDragEnd = onResizeEnd,
        )
    }

    // Bottom-right corner handle
    if (ResizeHandleType.BOTTOM_RIGHT in enabledHandles) {
        ResizeHandle(
            type = ResizeHandleType.BOTTOM_RIGHT,
            modifier = Modifier.align(androidx.compose.ui.Alignment.BottomEnd),
            showVisualIndicator = showVisualIndicators,
            onDrag = { dragAmount ->
                val newWidth =
                    (panelWidth + with(density) { dragAmount.x.toDp() })
                        .coerceIn(minWidth, maxWidth)
                val newHeight =
                    (panelHeight + with(density) { dragAmount.y.toDp() })
                        .coerceIn(minHeight, maxHeight)
                onResize(newWidth, newHeight)
            },
            onDragStart = { onResizeStart(ResizeHandleType.BOTTOM_RIGHT) },
            onDragEnd = onResizeEnd,
        )
    }
}
