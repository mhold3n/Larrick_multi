@file:OptIn(androidx.compose.foundation.ExperimentalFoundationApi::class)

package com.campro.v5.ui

import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.Layout
import androidx.compose.ui.layout.Measurable
import androidx.compose.ui.layout.MeasureResult
import androidx.compose.ui.layout.MeasureScope
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.unit.Constraints
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp

/**
 * Configuration for a resizable panel
 */
data class PanelConfig(
    val id: String,
    val title: String = "",
    val initialWidth: Dp = 400.dp,
    val initialHeight: Dp = 300.dp,
    val minWidth: Dp = 200.dp,
    val minHeight: Dp = 150.dp,
    val maxWidth: Dp = 800.dp,
    val maxHeight: Dp = 600.dp,
    val arrangement: PanelArrangement = PanelArrangement.HORIZONTAL,
    val content: @Composable BoxScope.() -> Unit,
)

/**
 * Panel arrangement options
 */
enum class PanelArrangement {
    HORIZONTAL,
    VERTICAL,
}

/**
 * Divider orientation for resize handles
 */
enum class DividerOrientation {
    HORIZONTAL,
    VERTICAL,
}

/**
 * Scope for building resizable panel layouts
 */
class ResizablePanelLayoutScope {
    internal val panels = mutableListOf<PanelConfig>()

    fun panel(
        id: String,
        title: String = "",
        initialWidth: Dp = 400.dp,
        initialHeight: Dp = 300.dp,
        minWidth: Dp = 200.dp,
        minHeight: Dp = 150.dp,
        maxWidth: Dp = 800.dp,
        maxHeight: Dp = 600.dp,
        arrangement: PanelArrangement = PanelArrangement.HORIZONTAL,
        content: @Composable BoxScope.() -> Unit,
    ) {
        panels.add(
            PanelConfig(
                id = id,
                title = title,
                initialWidth = initialWidth,
                initialHeight = initialHeight,
                minWidth = minWidth,
                minHeight = minHeight,
                maxWidth = maxWidth,
                maxHeight = maxHeight,
                arrangement = arrangement,
                content = content,
            ),
        )
    }
}

/**
 * A layout that supports multiple resizable panels with drag-to-resize functionality
 */
@Composable
fun ResizablePanelLayout(modifier: Modifier = Modifier, spacing: Dp = 8.dp, content: ResizablePanelLayoutScope.() -> Unit) {
    val scope = remember { ResizablePanelLayoutScope() }
    scope.panels.clear()
    scope.content()

    val density = LocalDensity.current
    var panelSizes by remember {
        mutableStateOf(
            scope.panels.map { it.initialWidth to it.initialHeight },
        )
    }

    Layout(
        content = {
            scope.panels.forEachIndexed { index, panelConfig ->
                val (currentWidth, currentHeight) =
                    panelSizes.getOrElse(index) {
                        panelConfig.initialWidth to panelConfig.initialHeight
                    }

                ResizablePanel(
                    panelId = panelConfig.id,
                    modifier = Modifier,
                    initialWidth = currentWidth,
                    initialHeight = currentHeight,
                    minWidth = panelConfig.minWidth,
                    minHeight = panelConfig.minHeight,
                    maxWidth = panelConfig.maxWidth,
                    maxHeight = panelConfig.maxHeight,
                    title = panelConfig.title,
                    onSizeChanged = { newWidth, newHeight ->
                        panelSizes =
                            panelSizes.toMutableList().apply {
                                set(index, newWidth to newHeight)
                            }
                    },
                    content = panelConfig.content,
                )

                // Add resize dividers between panels
                if (index < scope.panels.size - 1) {
                    ResizeDivider(
                        orientation =
                        if (panelConfig.arrangement == PanelArrangement.HORIZONTAL) {
                            DividerOrientation.VERTICAL
                        } else {
                            DividerOrientation.HORIZONTAL
                        },
                        onResize = { delta ->
                            // Adjust sizes of adjacent panels
                            val newSizes = panelSizes.toMutableList()
                            val currentSize = newSizes[index]
                            val nextSize = newSizes[index + 1]

                            if (panelConfig.arrangement == PanelArrangement.HORIZONTAL) {
                                // Horizontal resize
                                val newCurrentWidth =
                                    (currentSize.first + delta).coerceIn(
                                        panelConfig.minWidth,
                                        panelConfig.maxWidth,
                                    )
                                val widthDiff = newCurrentWidth - currentSize.first
                                val newNextWidth =
                                    (nextSize.first - widthDiff).coerceIn(
                                        scope.panels[index + 1].minWidth,
                                        scope.panels[index + 1].maxWidth,
                                    )

                                newSizes[index] = newCurrentWidth to currentSize.second
                                newSizes[index + 1] = newNextWidth to nextSize.second
                            } else {
                                // Vertical resize
                                val newCurrentHeight =
                                    (currentSize.second + delta).coerceIn(
                                        panelConfig.minHeight,
                                        panelConfig.maxHeight,
                                    )
                                val heightDiff = newCurrentHeight - currentSize.second
                                val newNextHeight =
                                    (nextSize.second - heightDiff).coerceIn(
                                        scope.panels[index + 1].minHeight,
                                        scope.panels[index + 1].maxHeight,
                                    )

                                newSizes[index] = currentSize.first to newCurrentHeight
                                newSizes[index + 1] = nextSize.first to newNextHeight
                            }

                            panelSizes = newSizes
                        },
                    )
                }
            }
        },
        modifier = modifier,
    ) { measurables, constraints ->
        // Custom layout logic for resizable panels
        layoutResizablePanels(
            measurables = measurables,
            constraints = constraints,
            panelConfigs = scope.panels,
            panelSizes = panelSizes,
            spacing = spacing,
            density = density,
        )
    }
}

/**
 * Custom layout function for resizable panels
 */
private fun MeasureScope.layoutResizablePanels(
    measurables: List<Measurable>,
    constraints: Constraints,
    panelConfigs: List<PanelConfig>,
    panelSizes: List<Pair<Dp, Dp>>,
    spacing: Dp,
    density: androidx.compose.ui.unit.Density,
): MeasureResult {
    val spacingPx = with(density) { spacing.roundToPx() }

    // Separate panels and dividers
    val panelMeasurables = measurables.filterIndexed { index, _ -> index % 2 == 0 }
    val dividerMeasurables = measurables.filterIndexed { index, _ -> index % 2 == 1 }

    // Measure panels with their specified sizes
    val panelPlaceables =
        panelMeasurables.mapIndexed { index, measurable ->
            val (width, height) =
                panelSizes.getOrElse(index) {
                    panelConfigs[index].initialWidth to panelConfigs[index].initialHeight
                }

            val panelConstraints =
                Constraints.fixed(
                    width = with(density) { width.roundToPx() },
                    height = with(density) { height.roundToPx() },
                )

            measurable.measure(panelConstraints)
        }

    // Measure dividers
    val dividerPlaceables =
        dividerMeasurables.map { measurable ->
            measurable.measure(
                Constraints.fixed(
                    width = spacingPx,
                    height = spacingPx,
                ),
            )
        }

    // Calculate layout dimensions
    val totalWidth = panelPlaceables.sumOf { it.width } + (dividerPlaceables.size * spacingPx)
    val totalHeight = panelPlaceables.maxOfOrNull { it.height } ?: 0

    return layout(
        width = totalWidth.coerceAtMost(constraints.maxWidth),
        height = totalHeight.coerceAtMost(constraints.maxHeight),
    ) {
        var xOffset = 0
        var yOffset = 0

        panelPlaceables.forEachIndexed { index, placeable ->
            placeable.placeRelative(xOffset, yOffset)

            // Place divider after panel (if not last panel)
            if (index < dividerPlaceables.size) {
                val divider = dividerPlaceables[index]
                xOffset += placeable.width
                divider.placeRelative(xOffset, yOffset)
                xOffset += divider.width
            } else {
                xOffset += placeable.width
            }
        }
    }
}

/**
 * Resize divider component for separating panels
 */
@Composable
private fun ResizeDivider(orientation: DividerOrientation, onResize: (Dp) -> Unit, modifier: Modifier = Modifier) {
    val density = LocalDensity.current
    var isDragging by remember { mutableStateOf(false) }

    Box(
        modifier =
        modifier
            .then(
                if (orientation == DividerOrientation.VERTICAL) {
                    Modifier.width(8.dp).fillMaxHeight()
                } else {
                    Modifier.height(8.dp).fillMaxWidth()
                },
            ).background(
                if (isDragging) {
                    MaterialTheme.colorScheme.primary.copy(alpha = 0.5f)
                } else {
                    MaterialTheme.colorScheme.outline.copy(alpha = 0.3f)
                },
            ).pointerInput(orientation) {
                detectDragGestures(
                    onDragStart = {
                        isDragging = true
                    },
                    onDragEnd = {
                        isDragging = false
                    },
                    onDrag = { change, dragAmount ->
                        val delta =
                            if (orientation == DividerOrientation.VERTICAL) {
                                with(density) { dragAmount.x.toDp() }
                            } else {
                                with(density) { dragAmount.y.toDp() }
                            }
                        onResize(delta)
                    },
                )
            },
    ) {
        // Visual indicator for the divider
        Box(
            modifier =
            Modifier
                .align(Alignment.Center)
                .then(
                    if (orientation == DividerOrientation.VERTICAL) {
                        Modifier.width(2.dp).height(20.dp)
                    } else {
                        Modifier.height(2.dp).width(20.dp)
                    },
                ).background(
                    MaterialTheme.colorScheme.outline,
                    MaterialTheme.shapes.small,
                ),
        )
    }
}

/**
 * Example usage of ResizablePanelLayout
 */
@Composable
fun ExampleResizablePanelLayout() {
    ResizablePanelLayout(
        modifier = Modifier.fillMaxSize(),
        spacing = 8.dp,
    ) {
        panel(
            id = "panel1",
            title = "Panel 1",
            initialWidth = 300.dp,
            initialHeight = 400.dp,
            minWidth = 200.dp,
            maxWidth = 500.dp,
        ) {
            Text(
                "Content of Panel 1",
                modifier = Modifier.padding(16.dp),
            )
        }

        panel(
            id = "panel2",
            title = "Panel 2",
            initialWidth = 400.dp,
            initialHeight = 400.dp,
            minWidth = 250.dp,
            maxWidth = 600.dp,
        ) {
            Text(
                "Content of Panel 2",
                modifier = Modifier.padding(16.dp),
            )
        }
    }
}
