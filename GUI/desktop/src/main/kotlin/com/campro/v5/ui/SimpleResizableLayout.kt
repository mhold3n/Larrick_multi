package com.campro.v5.ui

import androidx.compose.foundation.layout.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Animation
import androidx.compose.material.icons.filled.BarChart
import androidx.compose.material.icons.filled.DataArray
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import com.campro.v5.*
import com.campro.v5.layout.LayoutManager

/**
 * Data class to represent the state of a floating panel
 */
data class FloatingPanelState(val id: String, val x: Dp, val y: Dp, val width: Dp, val height: Dp, val zIndex: Float)

/**
 * Simplified resizable layout that replaces the complex ResizablePanelStandardLayout
 * Uses clean Compose-based resizing without complex management layers
 */
@Composable
fun SimpleResizableLayout(
    testingMode: Boolean,
    animationStarted: Boolean,
    allParameters: Map<String, String>,
    layoutManager: LayoutManager,
    onParametersChanged: (Map<String, String>) -> Unit,
) {
    Column(
        modifier = Modifier.fillMaxSize().padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        // Header
        Text(
            "CamProV5 - Cycloidal Animation Generator",
            style = MaterialTheme.typography.headlineMedium,
            color = MaterialTheme.colorScheme.onBackground,
        )

        // Resizable Parameter Panel
        ResizableContainer(
            title = "Parameters",
            initialHeight = 300.dp,
            minHeight = 150.dp,
            maxHeight = Dp.Unspecified,
            modifier = Modifier.weight(0.3f),
            enabledDirections = setOf(ResizeDirection.BOTTOM, ResizeDirection.RIGHT, ResizeDirection.BOTTOM_RIGHT),
        ) {
            ParameterInputForm(
                testingMode = testingMode,
                onParametersChanged = onParametersChanged,
                layoutManager = layoutManager,
            )
        }

        // Row layout for main content - always visible
        Row(
            modifier = Modifier.weight(0.5f),
            horizontalArrangement = Arrangement.spacedBy(4.dp), // Small gap between panels
        ) {
            ResizableContainer(
                title = "Animation",
                modifier = Modifier.weight(0.6f), // Initial weight ratio (60%)
                // Add vertical resize directions (TOP, BOTTOM, TOP_RIGHT, BOTTOM_RIGHT)
                enabledDirections =
                setOf(
                    ResizeDirection.RIGHT,
                    ResizeDirection.BOTTOM,
                    ResizeDirection.TOP,
                    ResizeDirection.BOTTOM_RIGHT,
                    ResizeDirection.TOP_RIGHT,
                ),
            ) {
                // Show placeholder or empty state when no animation data
                if (!animationStarted) {
                    EmptyStateWidget(
                        message = "Animation will appear here after parameters are set",
                        icon = Icons.Default.Animation,
                    )
                } else {
                    AnimationWidget(
                        parameters = allParameters,
                        testingMode = testingMode,
                    )
                }
            }

            ResizableContainer(
                title = "Plots",
                modifier = Modifier.weight(0.4f), // Initial weight ratio (40%)
                // Add vertical resize directions (TOP, BOTTOM, TOP_LEFT, BOTTOM_LEFT)
                enabledDirections =
                setOf(
                    ResizeDirection.LEFT,
                    ResizeDirection.BOTTOM,
                    ResizeDirection.TOP,
                    ResizeDirection.BOTTOM_LEFT,
                    ResizeDirection.TOP_LEFT,
                ),
            ) {
                // Show placeholder or empty state when no plot data
                if (!animationStarted) {
                    EmptyStateWidget(
                        message = "Plots will appear here after parameters are set",
                        icon = Icons.Default.BarChart,
                    )
                } else {
                    PreviewsPanel(
                        engine =
                        com.campro.v5.animation.MotionLawEngine
                            .getInstance(),
                        modifier = Modifier.fillMaxSize(),
                    )
                }
            }
        }

        // Bottom data panel - always visible
        ResizableContainer(
            title = "Data Display",
            initialHeight = 250.dp,
            minHeight = 150.dp,
            maxHeight = Dp.Unspecified,
            modifier = Modifier.weight(0.2f),
            enabledDirections = setOf(ResizeDirection.TOP, ResizeDirection.RIGHT, ResizeDirection.TOP_RIGHT),
        ) {
            // Show placeholder or empty state when no data
            if (!animationStarted) {
                EmptyStateWidget(
                    message = "Data will appear here after parameters are set",
                    icon = Icons.Default.DataArray,
                )
            } else {
                DataDisplayPanel(
                    parameters = allParameters,
                    testingMode = testingMode,
                )
            }
        }
    }
}

/**
 * Floating panels layout with freely draggable and resizable panels
 * that can be positioned anywhere and stacked on top of each other
 */
@Composable
fun FloatingPanelsLayout(
    testingMode: Boolean,
    animationStarted: Boolean,
    allParameters: Map<String, String>,
    layoutManager: LayoutManager,
    onParametersChanged: (Map<String, String>) -> Unit,
) {
    // Default panel states
    val defaultPanelStates =
        mapOf(
            "parameters" to
                FloatingPanelState(
                    id = "parameters",
                    x = 20.dp,
                    y = 60.dp,
                    width = 400.dp,
                    height = 300.dp,
                    zIndex = 0f,
                ),
            "animation" to
                FloatingPanelState(
                    id = "animation",
                    x = 20.dp,
                    y = 380.dp,
                    width = 500.dp,
                    height = 350.dp,
                    zIndex = 1f,
                ),
            "plots" to
                FloatingPanelState(
                    id = "plots",
                    x = 540.dp,
                    y = 60.dp,
                    width = 450.dp,
                    height = 350.dp,
                    zIndex = 2f,
                ),
            "data" to
                FloatingPanelState(
                    id = "data",
                    x = 540.dp,
                    y = 430.dp,
                    width = 450.dp,
                    height = 300.dp,
                    zIndex = 3f,
                ),
        )

    // State for panel positions, sizes, and z-indices
    val panelStates = remember { mutableStateMapOf<String, FloatingPanelState>() }

    // Initialize panel states with default values if not already set
    LaunchedEffect(Unit) {
        defaultPanelStates.forEach { (id, defaultState) ->
            if (!panelStates.containsKey(id)) {
                panelStates[id] = defaultState
            }
        }
    }

    // Function to update panel state
    fun updatePanelState(id: String, updater: (FloatingPanelState) -> FloatingPanelState) {
        panelStates[id]?.let { currentState ->
            panelStates[id] = updater(currentState)
        }
    }

    Box(
        modifier = Modifier.fillMaxSize().padding(16.dp),
    ) {
        // Header
        Text(
            "CamProV5 - Cycloidal Animation Generator",
            style = MaterialTheme.typography.headlineMedium,
            color = MaterialTheme.colorScheme.onBackground,
            modifier = Modifier.padding(bottom = 8.dp),
        )

        // Parameters Panel
        panelStates["parameters"]?.let { state ->
            DraggableResizablePanel(
                title = "Parameters",
                initialX = state.x,
                initialY = state.y,
                initialWidth = state.width,
                initialHeight = state.height,
                zIndex = state.zIndex,
                onZIndexChange = { newZIndex ->
                    updatePanelState("parameters") { it.copy(zIndex = newZIndex) }
                },
                onPositionChange = { newX, newY ->
                    updatePanelState("parameters") { it.copy(x = newX, y = newY) }
                },
                onSizeChange = { newWidth, newHeight ->
                    updatePanelState("parameters") { it.copy(width = newWidth, height = newHeight) }
                },
            ) {
                ParameterInputForm(
                    testingMode = testingMode,
                    onParametersChanged = onParametersChanged,
                    layoutManager = layoutManager,
                )
            }
        }

        // Animation Panel
        panelStates["animation"]?.let { state ->
            DraggableResizablePanel(
                title = "Animation",
                initialX = state.x,
                initialY = state.y,
                initialWidth = state.width,
                initialHeight = state.height,
                zIndex = state.zIndex,
                onZIndexChange = { newZIndex ->
                    updatePanelState("animation") { it.copy(zIndex = newZIndex) }
                },
                onPositionChange = { newX, newY ->
                    updatePanelState("animation") { it.copy(x = newX, y = newY) }
                },
                onSizeChange = { newWidth, newHeight ->
                    updatePanelState("animation") { it.copy(width = newWidth, height = newHeight) }
                },
            ) {
                if (!animationStarted) {
                    EmptyStateWidget(
                        message = "Animation will appear here after parameters are set",
                        icon = Icons.Default.Animation,
                    )
                } else {
                    AnimationWidget(
                        parameters = allParameters,
                        testingMode = testingMode,
                    )
                }
            }
        }

        // Plots Panel
        panelStates["plots"]?.let { state ->
            DraggableResizablePanel(
                title = "Plots",
                initialX = state.x,
                initialY = state.y,
                initialWidth = state.width,
                initialHeight = state.height,
                zIndex = state.zIndex,
                onZIndexChange = { newZIndex ->
                    updatePanelState("plots") { it.copy(zIndex = newZIndex) }
                },
                onPositionChange = { newX, newY ->
                    updatePanelState("plots") { it.copy(x = newX, y = newY) }
                },
                onSizeChange = { newWidth, newHeight ->
                    updatePanelState("plots") { it.copy(width = newWidth, height = newHeight) }
                },
            ) {
                if (!animationStarted) {
                    EmptyStateWidget(
                        message = "Plots will appear here after parameters are set",
                        icon = Icons.Default.BarChart,
                    )
                } else {
                    PlotCarouselWidget(
                        parameters = allParameters,
                        testingMode = testingMode,
                    )
                }
            }
        }

        // Data Display Panel
        panelStates["data"]?.let { state ->
            DraggableResizablePanel(
                title = "Data Display",
                initialX = state.x,
                initialY = state.y,
                initialWidth = state.width,
                initialHeight = state.height,
                zIndex = state.zIndex,
                onZIndexChange = { newZIndex ->
                    updatePanelState("data") { it.copy(zIndex = newZIndex) }
                },
                onPositionChange = { newX, newY ->
                    updatePanelState("data") { it.copy(x = newX, y = newY) }
                },
                onSizeChange = { newWidth, newHeight ->
                    updatePanelState("data") { it.copy(width = newWidth, height = newHeight) }
                },
            ) {
                if (!animationStarted) {
                    EmptyStateWidget(
                        message = "Data will appear here after parameters are set",
                        icon = Icons.Default.DataArray,
                    )
                } else {
                    DataDisplayPanel(
                        parameters = allParameters,
                        testingMode = testingMode,
                    )
                }
            }
        }
    }
}

/**
 * Responsive layout that adapts to different screen sizes
 */
@Composable
fun ResponsiveLayout(
    testingMode: Boolean,
    animationStarted: Boolean,
    allParameters: Map<String, String>,
    layoutManager: LayoutManager,
    onParametersChanged: (Map<String, String>) -> Unit,
) {
    if (layoutManager.shouldUseSingleColumn()) {
        // Single column layout for small screens
        SingleColumnSimpleLayout(
            testingMode = testingMode,
            animationStarted = animationStarted,
            allParameters = allParameters,
            layoutManager = layoutManager,
            onParametersChanged = onParametersChanged,
        )
    } else {
        // Floating panels layout for larger screens
        FloatingPanelsLayout(
            testingMode = testingMode,
            animationStarted = animationStarted,
            allParameters = allParameters,
            layoutManager = layoutManager,
            onParametersChanged = onParametersChanged,
        )
    }
}

/**
 * Single column layout for smaller screens
 */
@Composable
fun SingleColumnSimpleLayout(
    testingMode: Boolean,
    animationStarted: Boolean,
    allParameters: Map<String, String>,
    layoutManager: LayoutManager,
    onParametersChanged: (Map<String, String>) -> Unit,
) {
    Column(
        modifier = Modifier.fillMaxSize().padding(8.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        // Header
        Text(
            "CamProV5",
            style = MaterialTheme.typography.headlineSmall,
            color = MaterialTheme.colorScheme.onBackground,
        )

        // Parameter Panel - Always visible
        ResizableContainer(
            title = "Parameters",
            initialHeight = 250.dp,
            minHeight = 150.dp,
            maxHeight = Dp.Unspecified,
            modifier = Modifier.weight(0.3f),
            enabledDirections = setOf(ResizeDirection.BOTTOM),
        ) {
            ParameterInputForm(
                testingMode = testingMode,
                onParametersChanged = onParametersChanged,
                layoutManager = layoutManager,
            )
        }

        // Calculate dynamic split ratio based on content importance
        val proportionalSizes = layoutManager.calculateProportionalSizes()
        val animationSize = proportionalSizes["animation_panel"]
        val plotsSize = proportionalSizes["plot_panel"]

        // Calculate split ratio based on proportional heights, fallback to content-aware ratio
        val dynamicSplitRatio =
            when {
                animationSize != null && plotsSize != null -> {
                    val totalHeight = animationSize.second + plotsSize.second
                    if (totalHeight.value > 0) {
                        (animationSize.second / totalHeight).coerceIn(0.3f, 0.7f)
                    } else {
                        0.6f // Animation gets more space by default
                    }
                }
                testingMode -> 0.4f // Less space for animation in testing mode
                else -> 0.6f // More space for animation in normal mode
            }

        // Column layout for animation content - always visible
        Column(
            modifier = Modifier.weight(0.7f),
            verticalArrangement = Arrangement.spacedBy(4.dp), // Small gap between panels
        ) {
            ResizableContainer(
                title = "Animation",
                modifier = Modifier.weight(dynamicSplitRatio), // Dynamic weight based on content
                enabledDirections = setOf(ResizeDirection.BOTTOM),
            ) {
                if (!animationStarted) {
                    EmptyStateWidget(
                        message = "Animation will appear here after parameters are set",
                        icon = Icons.Default.Animation,
                    )
                } else {
                    AnimationWidget(
                        parameters = allParameters,
                        testingMode = testingMode,
                    )
                }
            }

            ResizableContainer(
                title = "Plots & Data",
                modifier = Modifier.weight(1f - dynamicSplitRatio), // Remaining space
                enabledDirections = setOf(ResizeDirection.TOP),
            ) {
                if (!animationStarted) {
                    EmptyStateWidget(
                        message = "Plots and data will appear here after parameters are set",
                        icon = Icons.Default.BarChart,
                    )
                } else {
                    // Tabbed interface for plots and data in single column
                    TabbedContent(
                        parameters = allParameters,
                        testingMode = testingMode,
                    )
                }
            }
        }
    }
}

/**
 * Tabbed content for single column layout
 */
@Composable
fun TabbedContent(parameters: Map<String, String>, testingMode: Boolean) {
    var selectedTab by remember { mutableStateOf(0) }
    val tabs = listOf("Plots", "Data")

    Column {
        TabRow(selectedTabIndex = selectedTab) {
            tabs.forEachIndexed { index, title ->
                Tab(
                    selected = selectedTab == index,
                    onClick = { selectedTab = index },
                    text = { Text(title) },
                )
            }
        }

        Box(modifier = Modifier.weight(1f)) {
            when (selectedTab) {
                0 ->
                    PlotCarouselWidget(
                        parameters = parameters,
                        testingMode = testingMode,
                    )
                1 ->
                    DataDisplayPanel(
                        parameters = parameters,
                        testingMode = testingMode,
                    )
            }
        }
    }
}

/**
 * Compact widget layout for very small screens
 */
@Composable
fun CompactSimpleLayout(testingMode: Boolean, allParameters: Map<String, String>, spacing: Dp = 4.dp) {
    Column(
        modifier = Modifier.fillMaxSize().padding(spacing),
        verticalArrangement = Arrangement.spacedBy(spacing),
    ) {
        // Minimal header
        Text(
            "CamProV5",
            style = MaterialTheme.typography.titleMedium,
            color = MaterialTheme.colorScheme.onBackground,
        )

        // Compact parameter input
        ResizableContainer(
            title = "Parameters",
            initialHeight = 200.dp,
            minHeight = 150.dp,
            maxHeight = 300.dp,
            enabledDirections = setOf(ResizeDirection.BOTTOM),
        ) {
            // Simplified parameter form for compact view
            Text(
                "Compact parameter view",
                style = MaterialTheme.typography.bodyMedium,
                modifier = Modifier.padding(8.dp),
            )
        }

        // Compact animation view
        ResizableContainer(
            title = "Animation",
            initialHeight = 200.dp,
            minHeight = 150.dp,
            enabledDirections = setOf(ResizeDirection.BOTTOM),
        ) {
            AnimationWidget(
                parameters = allParameters,
                testingMode = testingMode,
            )
        }
    }
}
