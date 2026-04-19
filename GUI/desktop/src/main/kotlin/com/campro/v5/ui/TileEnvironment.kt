package com.campro.v5.ui

import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.grid.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import com.campro.v5.debug.DebugFab
import com.campro.v5.debug.DebugIconButton

/**
 * Modern tile-based environment for CamProV5
 *
 * Features:
 * - Responsive grid layout that adapts to window size
 * - Drag-and-drop tile repositioning
 * - Content-aware tile sizing
 * - Smooth animations and transitions
 * - Unified tile management system
 * - Auto-layout and smart positioning
 */

data class TileConfig(
    val id: String,
    val title: String,
    val icon: ImageVector,
    val type: TileType,
    val minSize: TileSize = TileSize.SMALL,
    val maxSize: TileSize = TileSize.LARGE,
    val defaultSize: TileSize = TileSize.MEDIUM,
    val content: @Composable () -> Unit,
)

enum class TileType {
    INPUT, // Parameter input forms
    OUTPUT, // Data display, diagnostics
    GRAPHICS, // Animation, plots, visualizations
    CONTROL, // Playback controls, settings
}

enum class TileSize(val span: Int) {
    SMALL(1), // 1x1 grid cell
    MEDIUM(2), // 2x2 grid cells
    LARGE(3), // 3x3 grid cells
    XLARGE(4), // 4x4 grid cells
}

data class TileState(
    val config: TileConfig,
    val position: GridItemSpan,
    val isMinimized: Boolean = false,
    val isMaximized: Boolean = false,
    val isDragging: Boolean = false,
)

@Composable
fun TileEnvironment(
    tiles: List<TileConfig>,
    modifier: Modifier = Modifier,
    onTileStateChanged: (String, TileState) -> Unit = { _, _ -> },
) {
    var windowSize by remember { mutableStateOf(IntSize(1200, 800)) }
    val density = LocalDensity.current

    // Calculate responsive grid columns based on window width with more granular breakpoints
    val gridColumns =
        remember(windowSize.width) {
            when {
                windowSize.width < 600 -> 1 // Very small windows
                windowSize.width < 900 -> 2 // Small windows
                windowSize.width < 1200 -> 3 // Medium windows
                windowSize.width < 1600 -> 4 // Large windows
                windowSize.width < 2000 -> 5 // Very large windows
                else -> 6 // Ultra-wide windows
            }
        }

    // Calculate responsive tile spacing based on window size
    val tileSpacing =
        remember(windowSize.width) {
            when {
                windowSize.width < 800 -> 4.dp
                windowSize.width < 1200 -> 6.dp
                windowSize.width < 1600 -> 8.dp
                else -> 12.dp
            }
        }

    // Calculate responsive padding based on window size
    val contentPadding =
        remember(windowSize.width) {
            when {
                windowSize.width < 800 -> 8.dp
                windowSize.width < 1200 -> 12.dp
                windowSize.width < 1600 -> 16.dp
                else -> 20.dp
            }
        }

    // Initialize tile states
    var tileStates by remember {
        mutableStateOf(
            tiles
                .mapIndexed { index, config ->
                    config.id to
                        TileState(
                            config = config,
                            position = GridItemSpan(config.defaultSize.span),
                        )
                }.toMap(),
        )
    }

    // Auto-layout tiles in grid
    LaunchedEffect(tiles, gridColumns) {
        tileStates = autoLayoutTiles(tiles, gridColumns, tileStates)
    }

    Box(
        modifier =
        modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.surface)
            .onSizeChanged { windowSize = it },
    ) {
        // Grid background
        LazyVerticalGrid(
            columns = GridCells.Fixed(gridColumns),
            modifier = Modifier.fillMaxSize(),
            contentPadding = PaddingValues(contentPadding),
            horizontalArrangement = Arrangement.spacedBy(tileSpacing),
            verticalArrangement = Arrangement.spacedBy(tileSpacing),
        ) {
            items(
                items = tiles,
                key = { it.id },
                span = { tile -> tileStates[tile.id]?.position ?: GridItemSpan(2) },
            ) { tileConfig ->
                val tileState = tileStates[tileConfig.id] ?: return@items

                Tile(
                    state = tileState,
                    onStateChanged = { newState ->
                        tileStates =
                            tileStates.toMutableMap().apply {
                                put(tileConfig.id, newState)
                            }
                        onTileStateChanged(tileConfig.id, newState)
                    },
                    onDragEnd = { newPosition ->
                        tileStates =
                            tileStates.toMutableMap().apply {
                                put(tileConfig.id, tileState.copy(position = newPosition))
                            }
                    },
                )
            }
        }

        // Floating action button for tile management
        DebugFab(
            buttonId = "tiles-manage",
            onClick = { /* TODO: Open tile management dialog */ },
            modifier =
            Modifier
                .align(Alignment.BottomEnd)
                .padding(16.dp),
        ) {
            Icon(Icons.Default.GridView, contentDescription = "Manage Tiles")
        }
    }
}

@Composable
private fun Tile(state: TileState, onStateChanged: (TileState) -> Unit, onDragEnd: (GridItemSpan) -> Unit, modifier: Modifier = Modifier) {
    val density = LocalDensity.current
    var tileSize by remember { mutableStateOf(IntSize(0, 0)) }
    var isDragging by remember { mutableStateOf(false) }
    val animatedElevation by animateFloatAsState(
        targetValue = if (isDragging) 12f else 4f,
        animationSpec = tween(200),
        label = "elevation",
    )

    val animatedScale by animateFloatAsState(
        targetValue = if (isDragging) 1.05f else 1f,
        animationSpec = tween(200),
        label = "scale",
    )

    Card(
        modifier =
        modifier
            .fillMaxWidth()
            .fillMaxHeight()
            .onSizeChanged { tileSize = it }
            .shadow(animatedElevation.dp, RoundedCornerShape(12.dp))
            .clip(RoundedCornerShape(12.dp))
            .border(
                width = if (isDragging) 2.dp else 1.dp,
                color = if (isDragging) MaterialTheme.colorScheme.primary else Color.Transparent,
                shape = RoundedCornerShape(12.dp),
            ).pointerInput(state.config.id) {
                detectDragGestures(
                    onDragStart = {
                        isDragging = true
                        onStateChanged(state.copy(isDragging = true))
                    },
                    onDragEnd = {
                        isDragging = false
                        onStateChanged(state.copy(isDragging = false))
                        // TODO: Calculate new grid position based on drag
                    },
                ) { _, _ ->
                    // Handle drag movement
                }
            },
        elevation = CardDefaults.cardElevation(defaultElevation = animatedElevation.dp),
    ) {
        Column(
            modifier = Modifier.fillMaxSize(),
        ) {
            // Tile header
            TileHeader(
                config = state.config,
                isMinimized = state.isMinimized,
                isMaximized = state.isMaximized,
                onMinimize = {
                    onStateChanged(state.copy(isMinimized = !state.isMinimized))
                },
                onMaximize = {
                    onStateChanged(state.copy(isMaximized = !state.isMaximized))
                },
            )

            // Tile content
            if (!state.isMinimized) {
                Box(
                    modifier =
                    Modifier
                        .fillMaxSize()
                        .padding(
                            horizontal = (tileSize.width * 0.02f).coerceAtLeast(8f).dp,
                            vertical = (tileSize.height * 0.02f).coerceAtLeast(8f).dp,
                        ),
                ) {
                    // Provide responsive content scaling context
                    ResponsiveContent(
                        tileSize = tileSize,
                        density = density,
                    ) {
                        state.config.content()
                    }
                }
            }
        }
    }
}

@Composable
private fun ResponsiveContent(tileSize: IntSize, density: androidx.compose.ui.unit.Density, content: @Composable () -> Unit) {
    // Calculate responsive scaling factor based on tile size
    val scaleFactor =
        remember(tileSize) {
            val baseSize = 300f // Base tile size for scaling reference
            val currentSize = minOf(tileSize.width, tileSize.height).toFloat()
            (currentSize / baseSize).coerceIn(0.5f, 2.0f)
        }

    // Simply render content with scaling context available
    content()
}

@Composable
private fun TileHeader(config: TileConfig, isMinimized: Boolean, isMaximized: Boolean, onMinimize: () -> Unit, onMaximize: () -> Unit) {
    Row(
        modifier =
        Modifier
            .fillMaxWidth()
            .background(
                when (config.type) {
                    TileType.INPUT -> MaterialTheme.colorScheme.primaryContainer
                    TileType.OUTPUT -> MaterialTheme.colorScheme.secondaryContainer
                    TileType.GRAPHICS -> MaterialTheme.colorScheme.tertiaryContainer
                    TileType.CONTROL -> MaterialTheme.colorScheme.surfaceVariant
                },
            ).padding(12.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Icon(
                imageVector = config.icon,
                contentDescription = null,
                tint = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.size(20.dp),
            )
            Text(
                text = config.title,
                style = MaterialTheme.typography.titleSmall,
                fontWeight = FontWeight.Medium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }

        Row(
            horizontalArrangement = Arrangement.spacedBy(4.dp),
        ) {
            DebugIconButton(
                buttonId = "tile-minimize-" + config.id,
                onClick = onMinimize,
                modifier = Modifier.size(24.dp),
            ) {
                Icon(
                    imageVector = if (isMinimized) Icons.Default.ExpandMore else Icons.Default.ExpandLess,
                    contentDescription = if (isMinimized) "Expand" else "Minimize",
                    modifier = Modifier.size(16.dp),
                )
            }

            DebugIconButton(
                buttonId = "tile-maximize-" + config.id,
                onClick = onMaximize,
                modifier = Modifier.size(24.dp),
            ) {
                Icon(
                    imageVector = if (isMaximized) Icons.Default.FullscreenExit else Icons.Default.Fullscreen,
                    contentDescription = if (isMaximized) "Restore" else "Maximize",
                    modifier = Modifier.size(16.dp),
                )
            }
        }
    }
}

private fun autoLayoutTiles(tiles: List<TileConfig>, gridColumns: Int, currentStates: Map<String, TileState>): Map<String, TileState> {
    val newStates = currentStates.toMutableMap()
    val grid = Array(gridColumns) { BooleanArray(gridColumns) { false } }

    // Sort tiles by priority (graphics first, then input, output, control)
    val sortedTiles =
        tiles.sortedBy { tile ->
            when (tile.type) {
                TileType.GRAPHICS -> 0
                TileType.INPUT -> 1
                TileType.OUTPUT -> 2
                TileType.CONTROL -> 3
            }
        }

    sortedTiles.forEach { tile ->
        val state = newStates[tile.id] ?: return@forEach
        val size = if (state.isMaximized) TileSize.XLARGE else state.config.defaultSize

        // Find first available position
        val position = findAvailablePosition(grid, size.span, gridColumns)
        if (position != null) {
            // Mark grid cells as occupied
            for (i in position.first until position.first + size.span) {
                for (j in position.second until position.second + size.span) {
                    if (i < gridColumns && j < gridColumns) {
                        grid[i][j] = true
                    }
                }
            }

            newStates[tile.id] =
                state.copy(
                    position = GridItemSpan(size.span),
                )
        }
    }

    return newStates
}

private fun findAvailablePosition(grid: Array<BooleanArray>, size: Int, gridColumns: Int): Pair<Int, Int>? {
    for (i in 0..gridColumns - size) {
        for (j in 0..gridColumns - size) {
            var available = true
            for (x in i until i + size) {
                for (y in j until j + size) {
                    if (grid[x][y]) {
                        available = false
                        break
                    }
                }
                if (!available) break
            }
            if (available) return Pair(i, j)
        }
    }
    return null
}
