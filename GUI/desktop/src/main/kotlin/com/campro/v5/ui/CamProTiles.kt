package com.campro.v5.ui

import androidx.compose.foundation.layout.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import com.campro.v5.AnimationWidget
import com.campro.v5.ParameterInputForm
import com.campro.v5.animation.MotionLawEngine
import com.campro.v5.debug.DebugButton
import com.campro.v5.layout.LayoutManager

/**
 * Tile definitions for CamProV5 application
 *
 * This file contains all the tile configurations for the tile-based environment,
 * including parameter input, animation display, plots, diagnostics, and controls.
 */

@Composable
fun createCamProTiles(
    testingMode: Boolean,
    animationStarted: Boolean,
    allParameters: Map<String, String>,
    onParametersChanged: (Map<String, String>) -> Unit,
    layoutManager: LayoutManager,
): List<TileConfig> = listOf(
    // Input Tiles
    TileConfig(
        id = "parameters",
        title = "Parameters",
        icon = Icons.Default.Settings,
        type = TileType.INPUT,
        minSize = TileSize.SMALL,
        maxSize = TileSize.LARGE,
        defaultSize = TileSize.MEDIUM,
    ) {
        ParameterInputForm(
            testingMode = testingMode,
            onParametersChanged = onParametersChanged,
            layoutManager = layoutManager,
        )
    },
    // Unified Optimization Tile
    TileConfig(
        id = "unified_optimization",
        title = "Unified Optimization",
        icon = Icons.Default.AutoAwesome,
        type = TileType.GRAPHICS,
        minSize = TileSize.LARGE,
        maxSize = TileSize.XLARGE,
        defaultSize = TileSize.LARGE,
    ) {
        UnifiedOptimizationTile(
            onResultsReceived = { result ->
                // Handle optimization results
                if (testingMode) {
                    println("EVENT:{\"type\":\"optimization_completed\",\"status\":\"${result.status}\",\"execution_time\":${result.executionTime}}")
                }
            },
        )
    },
    // Graphics Tiles
    TileConfig(
        id = "animation",
        title = "Animation",
        icon = Icons.Default.Animation,
        type = TileType.GRAPHICS,
        minSize = TileSize.MEDIUM,
        maxSize = TileSize.XLARGE,
        defaultSize = TileSize.LARGE,
    ) {
        if (animationStarted) {
            AnimationWidget(
                parameters = allParameters,
                testingMode = testingMode,
            )
        } else {
            EmptyStateWidget(
                message = "Set parameters and start animation to see the cam profile",
                icon = Icons.Default.Animation,
            )
        }
    },
    TileConfig(
        id = "plots",
        title = "Motion Profiles",
        icon = Icons.Default.BarChart,
        type = TileType.GRAPHICS,
        minSize = TileSize.SMALL,
        maxSize = TileSize.LARGE,
        defaultSize = TileSize.MEDIUM,
    ) {
        if (animationStarted) {
            PreviewsPanel(
                engine = MotionLawEngine.getInstance(),
                modifier = Modifier.fillMaxSize(),
            )
        } else {
            EmptyStateWidget(
                message = "Motion profiles will appear here after animation starts",
                icon = Icons.Default.BarChart,
            )
        }
    },
    // Output Tiles
    TileConfig(
        id = "diagnostics",
        title = "Diagnostics",
        icon = Icons.Default.Info,
        type = TileType.OUTPUT,
        minSize = TileSize.SMALL,
        maxSize = TileSize.MEDIUM,
        defaultSize = TileSize.SMALL,
    ) {
        DiagnosticsTile()
    },
    TileConfig(
        id = "performance",
        title = "Performance",
        icon = Icons.Default.Speed,
        type = TileType.OUTPUT,
        minSize = TileSize.SMALL,
        maxSize = TileSize.MEDIUM,
        defaultSize = TileSize.SMALL,
    ) {
        PerformanceTile()
    },
    // Control Tiles
    TileConfig(
        id = "playback",
        title = "Playback Controls",
        icon = Icons.Default.PlayArrow,
        type = TileType.CONTROL,
        minSize = TileSize.SMALL,
        maxSize = TileSize.SMALL,
        defaultSize = TileSize.SMALL,
    ) {
        PlaybackControlsTile(
            parameters = allParameters,
            onParametersChanged = onParametersChanged,
        )
    },
    TileConfig(
        id = "view_settings",
        title = "View Settings",
        icon = Icons.Default.Visibility,
        type = TileType.CONTROL,
        minSize = TileSize.SMALL,
        maxSize = TileSize.SMALL,
        defaultSize = TileSize.SMALL,
    ) {
        ViewSettingsTile()
    },
)

@Composable
private fun DiagnosticsTile() {
    val motion = MotionLawEngine.getInstance().getMotionLawSamples()
    val preflight =
        com.campro.v5.animation.DiagnosticsPreflight
            .validateMotionLaw(motion)

    Column(
        modifier = Modifier.fillMaxSize(),
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        Text(
            "System Status",
            style = MaterialTheme.typography.titleSmall,
            color = MaterialTheme.colorScheme.onSurface,
        )

        preflight.items.forEach { item ->
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Icon(
                    imageVector = if (item.ok) Icons.Default.CheckCircle else Icons.Default.Error,
                    contentDescription = null,
                    tint = if (item.ok) Color(0xFF4CAF50) else Color(0xFFF44336),
                    modifier = Modifier.size(16.dp),
                )
                Text(
                    text = item.name,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurface,
                )
                if (item.detail.isNotBlank()) {
                    Text(
                        text = "(${item.detail})",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
            }
        }
    }
}

@Composable
private fun PerformanceTile() {
    val fps = com.campro.v5.animation.PerfDiag.fps
    val accelMax = com.campro.v5.animation.PerfDiag.accelMaxAbs
    val jerkMax = com.campro.v5.animation.PerfDiag.jerkMaxAbs

    Column(
        modifier = Modifier.fillMaxSize(),
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        Text(
            "Performance Metrics",
            style = MaterialTheme.typography.titleSmall,
            color = MaterialTheme.colorScheme.onSurface,
        )

        PerformanceMetric(
            label = "FPS",
            value = String.format("%.1f", fps),
            unit = "fps",
        )

        PerformanceMetric(
            label = "Max Acceleration",
            value = String.format("%.2f", accelMax ?: 0.0),
            unit = "mm/ω²",
        )

        PerformanceMetric(
            label = "Max Jerk",
            value = String.format("%.2f", jerkMax ?: 0.0),
            unit = "mm/ω³",
        )
    }
}

@Composable
private fun PerformanceMetric(label: String, value: String, unit: String) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(
            text = label,
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(4.dp),
        ) {
            Text(
                text = value,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurface,
                fontWeight = androidx.compose.ui.text.font.FontWeight.Medium,
            )
            Text(
                text = unit,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

@Composable
private fun PlaybackControlsTile(parameters: Map<String, String>, onParametersChanged: (Map<String, String>) -> Unit) {
    var isPlaying by remember { mutableStateOf(parameters["animationStarted"] == "true") }
    var animationSpeed by remember { mutableStateOf(1.0f) }

    Column(
        modifier = Modifier.fillMaxSize(),
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        Text(
            "Playback",
            style = MaterialTheme.typography.titleSmall,
            color = MaterialTheme.colorScheme.onSurface,
        )

        // Play/Pause button
        DebugButton(
            buttonId = "playback-toggle",
            onClick = {
                isPlaying = !isPlaying
                onParametersChanged(parameters + ("animationStarted" to isPlaying.toString()))
            },
            modifier = Modifier.fillMaxWidth(),
        ) {
            Icon(
                imageVector = if (isPlaying) Icons.Default.Pause else Icons.Default.PlayArrow,
                contentDescription = if (isPlaying) "Pause" else "Play",
            )
            Spacer(modifier = Modifier.width(8.dp))
            Text(if (isPlaying) "Pause" else "Play")
        }

        // Speed control
        Column {
            Text(
                text = "Speed: ${String.format("%.1f", animationSpeed)}x",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurface,
            )
            Slider(
                value = animationSpeed,
                onValueChange = { animationSpeed = it },
                valueRange = 0.1f..5.0f,
                modifier = Modifier.fillMaxWidth(),
            )
        }
    }
}

@Composable
private fun ViewSettingsTile() {
    var showGrid by remember { mutableStateOf(true) }
    var showDiagnostics by remember { mutableStateOf(true) }

    Column(
        modifier = Modifier.fillMaxSize(),
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        Text(
            "View Options",
            style = MaterialTheme.typography.titleSmall,
            color = MaterialTheme.colorScheme.onSurface,
        )

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(
                text = "Show Grid",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurface,
            )
            Switch(
                checked = showGrid,
                onCheckedChange = { showGrid = it },
            )
        }

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(
                text = "Show Diagnostics",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurface,
            )
            Switch(
                checked = showDiagnostics,
                onCheckedChange = { showDiagnostics = it },
            )
        }
    }
}

@Composable
private fun EmptyStateWidget(message: String, icon: androidx.compose.ui.graphics.vector.ImageVector) {
    Column(
        modifier = Modifier.fillMaxSize(),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,
    ) {
        Icon(
            imageVector = icon,
            contentDescription = null,
            modifier = Modifier.size(48.dp),
            tint = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Spacer(modifier = Modifier.height(16.dp))
        Text(
            text = message,
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            textAlign = androidx.compose.ui.text.style.TextAlign.Center,
        )
    }
}
