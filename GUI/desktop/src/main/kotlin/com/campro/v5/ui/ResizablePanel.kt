@file:OptIn(androidx.compose.foundation.ExperimentalFoundationApi::class)

package com.campro.v5.ui

import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.hoverable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp

/**
 * Enhanced resizable panel component with drag-to-resize functionality and scrollable content.
 *
 * Features:
 * - Comprehensive drag handles on all edges and corners
 * - Integration with PanelLayoutCoordinator for multi-panel management
 * - Scrollable content when content overflows
 * - Enhanced hover effects and visual feedback
 * - Minimum and maximum size constraints
 * - Smooth resize animations
 * - Layout coordination to prevent overlapping
 */
@Composable
fun ResizablePanel(
    panelId: String,
    modifier: Modifier = Modifier,
    initialWidth: Dp = 400.dp,
    initialHeight: Dp = 300.dp,
    initialX: Dp = 0.dp,
    initialY: Dp = 0.dp,
    minWidth: Dp = 200.dp,
    minHeight: Dp = 150.dp,
    maxWidth: Dp = 800.dp,
    maxHeight: Dp = 600.dp,
    title: String = "",
    layoutCoordinator: PanelLayoutCoordinator? = null,
    enabledHandles: Set<ResizeHandleType> = ResizeHandleType.values().toSet(),
    showVisualIndicators: Boolean = true,
    onSizeChanged: ((Dp, Dp) -> Unit)? = null,
    onPositionChanged: ((Dp, Dp) -> Unit)? = null,
    content: @Composable BoxScope.() -> Unit,
) {
    var panelWidth by remember { mutableStateOf(initialWidth) }
    var panelHeight by remember { mutableStateOf(initialHeight) }
    var panelX by remember { mutableStateOf(initialX) }
    var panelY by remember { mutableStateOf(initialY) }
    var isHovered by remember { mutableStateOf(false) }
    var isResizing by remember { mutableStateOf(false) }
    var currentResizeHandle by remember { mutableStateOf<ResizeHandleType?>(null) }

    val density = LocalDensity.current
    val interactionSource = remember { MutableInteractionSource() }

    // Register panel with layout coordinator
    LaunchedEffect(panelId, layoutCoordinator) {
        layoutCoordinator?.registerPanel(
            id = panelId,
            initialState =
            PanelState(
                x = panelX,
                y = panelY,
                width = panelWidth,
                height = panelHeight,
                minWidth = minWidth,
                minHeight = minHeight,
                maxWidth = maxWidth,
                maxHeight = maxHeight,
            ),
        )
    }

    // Handle size and position change callbacks
    LaunchedEffect(panelWidth, panelHeight) {
        onSizeChanged?.invoke(panelWidth, panelHeight)

        // Update layout coordinator
        layoutCoordinator?.let { coordinator ->
            val adjustments = coordinator.updatePanelSize(panelId, panelWidth, panelHeight)
            // Apply adjustments to other panels if needed
            // This would typically be handled by a parent composable
        }
    }

    LaunchedEffect(panelX, panelY) {
        onPositionChanged?.invoke(panelX, panelY)
    }

    Box(
        modifier =
        modifier
            .size(panelWidth, panelHeight)
            .hoverable(interactionSource),
    ) {
        // Main panel content
        Card(
            modifier =
            Modifier
                .fillMaxSize()
                .onSizeChanged { size ->
                    // Notify listeners only; avoid overriding state to prevent resize feedback loops
                    val w = with(density) { size.width.toDp() }
                    val h = with(density) { size.height.toDp() }
                    onSizeChanged?.invoke(w, h)
                },
            elevation =
            CardDefaults.cardElevation(
                defaultElevation = if (isResizing) 8.dp else 4.dp,
            ),
            colors =
            CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surface,
            ),
        ) {
            Column(
                modifier = Modifier.fillMaxSize(),
            ) {
                // Title bar if provided
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

                // Scrollable content area - using standardized EnhancedScrollableContent
                EnhancedScrollableContent(
                    modifier = Modifier.weight(1f),
                    content = content,
                )
            }
        }

        // Enhanced resize handles
        ResizeHandles(
            panelWidth = panelWidth,
            panelHeight = panelHeight,
            minWidth = minWidth,
            minHeight = minHeight,
            maxWidth = maxWidth,
            maxHeight = maxHeight,
            enabledHandles = enabledHandles,
            showVisualIndicators = showVisualIndicators,
            onResize = { newWidth, newHeight ->
                panelWidth = newWidth
                panelHeight = newHeight
            },
            onResizeStart = { handleType ->
                isResizing = true
                currentResizeHandle = handleType
            },
            onResizeEnd = {
                isResizing = false
                currentResizeHandle = null
            },
        )
    }
}
