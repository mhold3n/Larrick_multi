package com.campro.v5.ui

import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.*
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
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
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.zIndex
import com.campro.v5.debug.DebugIconButton
import kotlinx.coroutines.launch

/**
 * Tab group panel component for managing multiple panels in a tabbed interface
 */
@Composable
fun TabGroupPanel(
    tabGroupId: String,
    dockingManager: DockingManager,
    modifier: Modifier = Modifier,
    enableDocking: Boolean = true,
    enableTabReordering: Boolean = true,
    maxTabWidth: androidx.compose.ui.unit.Dp = 200.dp,
    panelContent: Map<String, @Composable BoxScope.() -> Unit> = emptyMap(),
) {
    val density = LocalDensity.current
    val scope = rememberCoroutineScope()

    // State from docking manager
    val tabGroups by dockingManager.tabGroups.collectAsState()
    val panels by dockingManager.panels.collectAsState()
    val dragState by dockingManager.dragState.collectAsState()

    val tabGroup = tabGroups[tabGroupId]
    var isDragging by remember { mutableStateOf(false) }
    var dragOffset by remember { mutableStateOf(Offset.Zero) }

    if (tabGroup == null) return

    // Animation states
    val elevation by animateDpAsState(
        targetValue = if (isDragging) 16.dp else 4.dp,
        animationSpec = tween(200),
    )

    Box(
        modifier =
        modifier
            .size(
                width = tabGroup.size.first,
                height = tabGroup.size.second,
            ).offset(
                x = if (isDragging) with(density) { dragOffset.x.toDp() } else tabGroup.position.x,
                y = if (isDragging) with(density) { dragOffset.y.toDp() } else tabGroup.position.y,
            ).zIndex(if (isDragging) 10f else 1f),
    ) {
        Card(
            modifier =
            Modifier
                .fillMaxSize()
                .shadow(elevation, RoundedCornerShape(8.dp)),
            elevation = CardDefaults.cardElevation(defaultElevation = elevation),
            colors =
            CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.secondaryContainer,
            ),
        ) {
            Column(modifier = Modifier.fillMaxSize()) {
                // Tab bar
                TabGroupHeader(
                    tabGroup = tabGroup,
                    panels = panels,
                    maxTabWidth = maxTabWidth,
                    enableDocking = enableDocking,
                    enableTabReordering = enableTabReordering,
                    onTabSelect = { panelId ->
                        dockingManager.setActiveTabPanel(tabGroupId, panelId)
                    },
                    onTabClose = { panelId ->
                        // Extract panel from tab group
                        scope.launch {
                            dockingManager.makeFloating(panelId, Offset.Zero)
                        }
                    },
                    onDragStart = { startPosition ->
                        if (enableDocking) {
                            isDragging = true
                            dockingManager.startDrag(tabGroupId, startPosition)
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
                )

                // Active panel content
                Box(
                    modifier =
                    Modifier
                        .fillMaxSize()
                        .padding(8.dp),
                ) {
                    val activePanel = panels[tabGroup.activePanel]
                    if (activePanel != null) {
                        val content = panelContent[tabGroup.activePanel]
                        if (content != null) {
                            content()
                        } else {
                            // Default content if no specific content provided
                            DefaultTabContent(
                                panelId = tabGroup.activePanel,
                                panelTitle = activePanel.title,
                            )
                        }
                    }
                }
            }
        }
    }
}

/**
 * Header component for tab groups with tab navigation
 */
@Composable
private fun TabGroupHeader(
    tabGroup: TabGroup,
    panels: Map<String, DockablePanel>,
    maxTabWidth: androidx.compose.ui.unit.Dp,
    enableDocking: Boolean,
    enableTabReordering: Boolean,
    onTabSelect: (String) -> Unit,
    onTabClose: (String) -> Unit,
    onDragStart: (Offset) -> Unit,
    onDrag: (Offset) -> Unit,
    onDragEnd: () -> Unit,
) {
    Surface(
        modifier =
        Modifier
            .fillMaxWidth()
            .height(48.dp),
        color = MaterialTheme.colorScheme.primaryContainer,
        tonalElevation = 2.dp,
    ) {
        Row(
            modifier = Modifier.fillMaxSize(),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            // Drag handle
            if (enableDocking) {
                Box(
                    modifier =
                    Modifier
                        .width(32.dp)
                        .fillMaxHeight()
                        .pointerInput(Unit) {
                            detectDragGestures(
                                onDragStart = { offset ->
                                    onDragStart(offset)
                                },
                                onDrag = { _, dragAmount ->
                                    onDrag(dragAmount)
                                },
                                onDragEnd = {
                                    onDragEnd()
                                },
                            )
                        },
                    contentAlignment = Alignment.Center,
                ) {
                    Icon(
                        imageVector = Icons.Default.DragHandle,
                        contentDescription = "Drag to move tab group",
                        tint = MaterialTheme.colorScheme.onPrimaryContainer.copy(alpha = 0.7f),
                        modifier = Modifier.size(16.dp),
                    )
                }
            }

            // Tab list
            LazyRow(
                modifier = Modifier.weight(1f),
                horizontalArrangement = Arrangement.spacedBy(2.dp),
                contentPadding = PaddingValues(horizontal = 4.dp),
            ) {
                items(tabGroup.panelIds) { panelId ->
                    val panel = panels[panelId]
                    if (panel != null) {
                        TabItem(
                            panelId = panelId,
                            title = panel.title,
                            isActive = panelId == tabGroup.activePanel,
                            maxWidth = maxTabWidth,
                            enableReordering = enableTabReordering,
                            onSelect = { onTabSelect(panelId) },
                            onClose = { onTabClose(panelId) },
                        )
                    }
                }
            }

            // Tab group controls
            Row(
                modifier = Modifier.padding(horizontal = 8.dp),
                horizontalArrangement = Arrangement.spacedBy(4.dp),
            ) {
                // Add tab button (placeholder for future functionality)
                DebugIconButton(
                    buttonId = "tabs-add",
                    onClick = {
                        // TODO(#19): Implement add tab functionality (UI/Compose). Should open a new tab with default content.
                    },
                    modifier = Modifier.size(24.dp),
                ) {
                    Icon(
                        imageVector = Icons.Default.Add,
                        contentDescription = "Add tab",
                        tint = MaterialTheme.colorScheme.onPrimaryContainer,
                        modifier = Modifier.size(12.dp),
                    )
                }

                // Tab group menu
                DebugIconButton(
                    buttonId = "tabs-menu",
                    onClick = {
                        // TODO(#20): Implement tab group menu (rename, delete group, move tabs between groups).
                    },
                    modifier = Modifier.size(24.dp),
                ) {
                    Icon(
                        imageVector = Icons.Default.MoreVert,
                        contentDescription = "Tab group options",
                        tint = MaterialTheme.colorScheme.onPrimaryContainer,
                        modifier = Modifier.size(12.dp),
                    )
                }
            }
        }
    }
}

/**
 * Individual tab item component
 */
@Composable
private fun TabItem(
    panelId: String,
    title: String,
    isActive: Boolean,
    maxWidth: androidx.compose.ui.unit.Dp,
    enableReordering: Boolean,
    onSelect: () -> Unit,
    onClose: () -> Unit,
) {
    var isHovered by remember { mutableStateOf(false) }
    var isDragging by remember { mutableStateOf(false) }

    // Animation states
    val backgroundColor by animateColorAsState(
        targetValue =
        when {
            isActive -> MaterialTheme.colorScheme.primary
            isHovered -> MaterialTheme.colorScheme.primary.copy(alpha = 0.1f)
            else -> Color.Transparent
        },
        animationSpec = tween(200),
    )

    val textColor by animateColorAsState(
        targetValue =
        if (isActive) {
            MaterialTheme.colorScheme.onPrimary
        } else {
            MaterialTheme.colorScheme.onPrimaryContainer
        },
        animationSpec = tween(200),
    )

    val elevation by animateDpAsState(
        targetValue = if (isDragging) 8.dp else 0.dp,
        animationSpec = tween(200),
    )

    Card(
        modifier =
        Modifier
            .widthIn(min = 80.dp, max = maxWidth)
            .height(36.dp)
            .alpha(if (isDragging) 0.8f else 1f)
            .pointerInput(panelId) {
                if (enableReordering) {
                    detectDragGestures(
                        onDragStart = {
                            isDragging = true
                        },
                        onDragEnd = {
                            isDragging = false
                        },
                        onDrag = { _, _ ->
                            // TODO(#21): Implement tab reordering logic (drag-and-drop, keyboard shortcuts). Persist order in state.
                        },
                    )
                }
            },
        elevation = CardDefaults.cardElevation(defaultElevation = elevation),
        colors =
        CardDefaults.cardColors(
            containerColor = backgroundColor,
        ),
        shape = RoundedCornerShape(topStart = 8.dp, topEnd = 8.dp),
    ) {
        Row(
            modifier =
            Modifier
                .fillMaxSize()
                .clickable { onSelect() }
                .padding(horizontal = 8.dp, vertical = 4.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween,
        ) {
            // Tab title
            Text(
                text = title,
                style = MaterialTheme.typography.labelMedium,
                color = textColor,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
                modifier = Modifier.weight(1f),
            )

            // Close button
            DebugIconButton(
                buttonId = "tab-close-" + panelId,
                onClick = onClose,
                modifier = Modifier.size(16.dp),
            ) {
                Icon(
                    imageVector = Icons.Default.Close,
                    contentDescription = "Close tab",
                    tint = textColor.copy(alpha = 0.7f),
                    modifier = Modifier.size(10.dp),
                )
            }
        }
    }
}

/**
 * Default content for tabs when no specific content is provided
 */
@Composable
private fun DefaultTabContent(panelId: String, panelTitle: String) {
    Box(
        modifier = Modifier.fillMaxSize(),
        contentAlignment = Alignment.Center,
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            Icon(
                imageVector = Icons.Default.Tab,
                contentDescription = null,
                tint = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f),
                modifier = Modifier.size(48.dp),
            )

            Text(
                text = panelTitle,
                style = MaterialTheme.typography.headlineSmall,
                color = MaterialTheme.colorScheme.onSurface,
            )

            Text(
                text = "Panel ID: $panelId",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.7f),
            )

            Text(
                text = "This is a tabbed panel. Content can be customized by providing specific content composables.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.6f),
                modifier = Modifier.padding(horizontal = 32.dp),
            )
        }
    }
}

/**
 * Tab group container that manages multiple tab groups
 */
@Composable
fun TabGroupContainer(
    dockingManager: DockingManager,
    modifier: Modifier = Modifier,
    panelContent: Map<String, @Composable BoxScope.() -> Unit> = emptyMap(),
) {
    val tabGroups by dockingManager.tabGroups.collectAsState()

    Box(modifier = modifier) {
        tabGroups.forEach { (tabGroupId, _) ->
            TabGroupPanel(
                tabGroupId = tabGroupId,
                dockingManager = dockingManager,
                panelContent = panelContent,
            )
        }
    }
}

/**
 * Utility function to create a tab group from existing panels
 */
@Composable
fun CreateTabGroupFromPanels(
    panelIds: List<String>,
    dockingManager: DockingManager,
    position: androidx.compose.ui.unit.DpOffset = androidx.compose.ui.unit.DpOffset.Zero,
    size: Pair<androidx.compose.ui.unit.Dp, androidx.compose.ui.unit.Dp> = 600.dp to 400.dp,
): String? {
    var tabGroupId by remember { mutableStateOf<String?>(null) }

    LaunchedEffect(panelIds) {
        if (panelIds.isNotEmpty()) {
            tabGroupId = dockingManager.createTabGroup(panelIds, position, size)
        }
    }

    return tabGroupId
}
