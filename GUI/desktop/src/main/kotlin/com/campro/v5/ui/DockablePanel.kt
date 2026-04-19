package com.campro.v5.ui

import androidx.compose.animation.core.*
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.unit.DpOffset
import androidx.compose.ui.unit.dp
import androidx.compose.ui.zIndex
import com.campro.v5.debug.DebugIconButton
import kotlinx.coroutines.launch

/**
 * Enhanced dockable panel component with drag-to-dock functionality
 */
@Composable
fun DockablePanel(
    panelId: String,
    title: String,
    modifier: Modifier = Modifier,
    dockingManager: DockingManager,
    initialWidth: androidx.compose.ui.unit.Dp = 400.dp,
    initialHeight: androidx.compose.ui.unit.Dp = 300.dp,
    minWidth: androidx.compose.ui.unit.Dp = 200.dp,
    minHeight: androidx.compose.ui.unit.Dp = 150.dp,
    maxWidth: androidx.compose.ui.unit.Dp = 800.dp,
    maxHeight: androidx.compose.ui.unit.Dp = 600.dp,
    enableDocking: Boolean = true,
    enableFloating: Boolean = true,
    enableMinimization: Boolean = true,
    content: @Composable BoxScope.() -> Unit,
) {
    val density = LocalDensity.current
    val scope = rememberCoroutineScope()

    // Panel state from docking manager
    val panels by dockingManager.panels.collectAsState()
    val dragState by dockingManager.dragState.collectAsState()
    val dockZones by dockingManager.dockZones.collectAsState()

    val panel = panels[panelId]
    var isDragging by remember { mutableStateOf(false) }
    var dragOffset by remember { mutableStateOf(Offset.Zero) }

    // Animation states
    val elevation by animateDpAsState(
        targetValue = if (isDragging) 16.dp else 4.dp,
        animationSpec = tween(200),
    )

    val alpha by animateFloatAsState(
        targetValue = if (panel?.isMinimized == true) 0.7f else 1f,
        animationSpec = tween(300),
    )

    // Register panel with docking manager
    LaunchedEffect(panelId) {
        dockingManager.registerPanel(
            id = panelId,
            title = title,
            initialPosition = DpOffset.Zero,
            initialSize = initialWidth to initialHeight,
            initialState = PanelDockState.DOCKED,
        )
    }

    // Cleanup on disposal
    DisposableEffect(panelId) {
        onDispose {
            dockingManager.unregisterPanel(panelId)
        }
    }

    if (panel == null) return

    Box(
        modifier =
        modifier
            .size(
                width = with(density) { panel.size.first },
                height = with(density) { panel.size.second },
            ).offset(
                x = if (isDragging) with(density) { dragOffset.x.toDp() } else panel.position.x,
                y = if (isDragging) with(density) { dragOffset.y.toDp() } else panel.position.y,
            ).zIndex(if (isDragging) 10f else panel.zIndex)
            .alpha(alpha),
    ) {
        // Main panel card
        Card(
            modifier =
            Modifier
                .fillMaxSize()
                .shadow(elevation, RoundedCornerShape(8.dp)),
            elevation = CardDefaults.cardElevation(defaultElevation = elevation),
            colors =
            CardDefaults.cardColors(
                containerColor =
                when (panel.state) {
                    PanelDockState.FLOATING -> MaterialTheme.colorScheme.surfaceVariant
                    PanelDockState.DOCKED -> MaterialTheme.colorScheme.surface
                    PanelDockState.TABBED -> MaterialTheme.colorScheme.secondaryContainer
                    PanelDockState.MINIMIZED -> MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.7f)
                },
            ),
        ) {
            Column(modifier = Modifier.fillMaxSize()) {
                // Title bar with controls
                DockablePanelTitleBar(
                    title = title,
                    panelState = panel.state,
                    enableDocking = enableDocking,
                    enableFloating = enableFloating,
                    enableMinimization = enableMinimization,
                    onDragStart = { startPosition ->
                        if (enableDocking) {
                            isDragging = true
                            dockingManager.startDrag(panelId, startPosition)
                        }
                    },
                    onDrag = { dragAmount ->
                        if (enableDocking && isDragging) {
                            dragOffset += dragAmount
                            dockingManager.updateDrag(dragOffset)
                        }
                    },
                    onDragEnd = {
                        if (enableDocking && isDragging) {
                            isDragging = false
                            dragOffset = Offset.Zero
                            dockingManager.endDrag()
                        }
                    },
                    onMinimize = {
                        if (enableMinimization) {
                            if (panel.isMinimized) {
                                dockingManager.restorePanel(panelId)
                            } else {
                                dockingManager.minimizePanel(panelId)
                            }
                        }
                    },
                    onFloat = {
                        if (enableFloating) {
                            scope.launch {
                                dockingManager.makeFloating(panelId, dragOffset)
                            }
                        }
                    },
                    onClose = {
                        dockingManager.unregisterPanel(panelId)
                    },
                )

                // Panel content (hidden when minimized)
                if (!panel.isMinimized) {
                    Box(
                        modifier =
                        Modifier
                            .fillMaxSize()
                            .padding(8.dp),
                    ) {
                        content()
                    }
                }
            }
        }

        // Docking indicators overlay
        if (isDragging && enableDocking) {
            DockingIndicators(
                dockZones = dockZones,
                currentDragState = dragState,
                modifier = Modifier.fillMaxSize(),
            )
        }
    }
}

/**
 * Title bar for dockable panels with drag handle and controls
 */
@Composable
private fun DockablePanelTitleBar(
    title: String,
    panelState: PanelDockState,
    enableDocking: Boolean,
    enableFloating: Boolean,
    enableMinimization: Boolean,
    onDragStart: (Offset) -> Unit,
    onDrag: (Offset) -> Unit,
    onDragEnd: () -> Unit,
    onMinimize: () -> Unit,
    onFloat: () -> Unit,
    onClose: () -> Unit,
) {
    var dragStartPosition by remember { mutableStateOf(Offset.Zero) }

    Surface(
        modifier =
        Modifier
            .fillMaxWidth()
            .height(40.dp),
        color = MaterialTheme.colorScheme.primaryContainer,
        tonalElevation = 2.dp,
    ) {
        Row(
            modifier =
            Modifier
                .fillMaxSize()
                .padding(horizontal = 8.dp)
                .pointerInput(Unit) {
                    if (enableDocking) {
                        detectDragGestures(
                            onDragStart = { offset ->
                                dragStartPosition = offset
                                onDragStart(offset)
                            },
                            onDrag = { _, dragAmount ->
                                onDrag(dragAmount)
                            },
                            onDragEnd = {
                                onDragEnd()
                            },
                        )
                    }
                },
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween,
        ) {
            // Title and state indicator
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                // Drag handle indicator
                if (enableDocking) {
                    Icon(
                        imageVector = Icons.Default.DragHandle,
                        contentDescription = "Drag to move panel",
                        tint = MaterialTheme.colorScheme.onPrimaryContainer.copy(alpha = 0.7f),
                        modifier = Modifier.size(16.dp),
                    )
                }

                // Panel state indicator
                val stateIcon =
                    when (panelState) {
                        PanelDockState.FLOATING -> Icons.Default.OpenInNew
                        PanelDockState.DOCKED -> Icons.Default.Dock
                        PanelDockState.TABBED -> Icons.Default.Tab
                        PanelDockState.MINIMIZED -> Icons.Default.Minimize
                    }

                Icon(
                    imageVector = stateIcon,
                    contentDescription = "Panel state: ${panelState.name}",
                    tint = MaterialTheme.colorScheme.primary,
                    modifier = Modifier.size(14.dp),
                )

                Text(
                    text = title,
                    style = MaterialTheme.typography.titleSmall,
                    color = MaterialTheme.colorScheme.onPrimaryContainer,
                    maxLines = 1,
                )
            }

            // Control buttons
            Row(
                horizontalArrangement = Arrangement.spacedBy(4.dp),
            ) {
                // Minimize button
                if (enableMinimization) {
                    DebugIconButton(
                        buttonId = "dock-minimize-" + panelId,
                        onClick = onMinimize,
                        modifier = Modifier.size(20.dp),
                    ) {
                        Icon(
                            imageVector = Icons.Default.ExpandLess,
                            contentDescription = "Minimize",
                            modifier = Modifier.size(14.dp),
                        )
                    }
                }

                // Float button
                if (enableFloating && panelState != PanelDockState.FLOATING) {
                    IconButton(
                        onClick = onFloat,
                        modifier = Modifier.size(24.dp),
                    ) {
                        Icon(
                            imageVector = Icons.Default.OpenInNew,
                            contentDescription = "Float panel",
                            tint = MaterialTheme.colorScheme.onPrimaryContainer,
                            modifier = Modifier.size(12.dp),
                        )
                    }
                }

                // Close button
                DebugIconButton(
                    buttonId = "dock-close-" + panelId,
                    onClick = onClose,
                    modifier = Modifier.size(20.dp),
                ) {
                    Icon(
                        imageVector = Icons.Default.Close,
                        contentDescription = "Close Panel",
                        modifier = Modifier.size(14.dp),
                    )
                }
            }
        }
    }
}

/**
 * Visual indicators for docking zones during drag operations
 */
@Composable
private fun DockingIndicators(
    dockZones: Map<DockZone, androidx.compose.ui.geometry.Rect>,
    currentDragState: DragState?,
    modifier: Modifier = Modifier,
) {
    val density = LocalDensity.current

    Box(modifier = modifier) {
        dockZones.forEach { (zone, rect) ->
            val isHovered = currentDragState?.hoveredZone == zone

            val indicatorAlpha by animateFloatAsState(
                targetValue = if (isHovered) 0.6f else 0.2f,
                animationSpec = tween(200),
            )

            val indicatorColor =
                when (zone) {
                    DockZone.LEFT, DockZone.RIGHT, DockZone.TOP, DockZone.BOTTOM ->
                        MaterialTheme.colorScheme.primary
                    DockZone.CENTER -> MaterialTheme.colorScheme.secondary
                    DockZone.TAB_GROUP -> MaterialTheme.colorScheme.tertiary
                    DockZone.NONE -> Color.Transparent
                }

            if (zone != DockZone.NONE) {
                Box(
                    modifier =
                    Modifier
                        .offset(
                            x = with(density) { rect.left.toDp() },
                            y = with(density) { rect.top.toDp() },
                        ).size(
                            width = with(density) { rect.width.toDp() },
                            height = with(density) { rect.height.toDp() },
                        ).background(
                            color = indicatorColor.copy(alpha = indicatorAlpha),
                            shape = RoundedCornerShape(4.dp),
                        ).border(
                            width = if (isHovered) 2.dp else 1.dp,
                            color = indicatorColor,
                            shape = RoundedCornerShape(4.dp),
                        ).alpha(indicatorAlpha),
                ) {
                    // Zone label
                    Box(
                        modifier = Modifier.fillMaxSize(),
                        contentAlignment = Alignment.Center,
                    ) {
                        Text(
                            text = zone.name.replace("_", " "),
                            style = MaterialTheme.typography.labelMedium,
                            color = MaterialTheme.colorScheme.onPrimary,
                        )
                    }
                }
            }
        }
    }
}

/**
 * Floating panel window component
 */
@Composable
fun FloatingPanel(
    panelId: String,
    title: String,
    dockingManager: DockingManager,
    onClose: () -> Unit = {},
    content: @Composable BoxScope.() -> Unit,
) {
    val panels by dockingManager.panels.collectAsState()
    val panel = panels[panelId]

    if (panel?.state == PanelDockState.FLOATING) {
        // This would typically be rendered in a separate window
        // For now, we'll render it as an overlay
        Box(
            modifier =
            Modifier
                .offset(panel.position.x, panel.position.y)
                .size(panel.size.first, panel.size.second)
                .zIndex(20f),
        ) {
            DockablePanel(
                panelId = panelId,
                title = title,
                dockingManager = dockingManager,
                enableDocking = true,
                enableFloating = false, // Already floating
                content = content,
            )
        }
    }
}
