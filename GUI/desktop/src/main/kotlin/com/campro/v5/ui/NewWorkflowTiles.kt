package com.campro.v5.ui

import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AutoAwesome
import androidx.compose.runtime.Composable
import com.campro.v5.models.OptimizationResult

/**
 * Tile configuration for the new unified optimization workflow.
 * Simplified version that only includes the new workflow components.
 */
@Composable
fun createNewWorkflowTiles(onResultsReceived: (OptimizationResult) -> Unit): List<TileConfig> = listOf(
    TileConfig(
        id = "unified_optimization",
        title = "Unified Optimization",
        icon = Icons.Default.AutoAwesome,
        type = TileType.GRAPHICS,
        minSize = TileSize.LARGE,
        maxSize = TileSize.XLARGE,
        defaultSize = TileSize.LARGE,
        content = {
            UnifiedOptimizationTile(
                onResultsReceived = onResultsReceived,
            )
        },
    ),
)
