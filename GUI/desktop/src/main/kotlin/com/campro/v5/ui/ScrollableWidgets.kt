package com.campro.v5.ui

import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.campro.v5.AnimationWidget
import com.campro.v5.DataDisplayPanel
import com.campro.v5.ParameterInputForm
import com.campro.v5.PlotCarouselWidget
import com.campro.v5.layout.LayoutManager

/**
 * Scrollable wrapper for AnimationWidget
 */
@Composable
fun ScrollableAnimationWidget(parameters: Map<String, String>, testingMode: Boolean = false, modifier: Modifier = Modifier) {
    val verticalScrollState = rememberScrollState()
    val horizontalScrollState = rememberScrollState()

    Box(
        modifier = modifier.fillMaxSize(),
    ) {
        // Scrollable content
        Box(
            modifier =
            Modifier
                .verticalScroll(verticalScrollState)
                .horizontalScroll(horizontalScrollState)
                .padding(8.dp),
        ) {
            // Use fillMaxSize instead of fixed size
            Box(
                modifier = Modifier.fillMaxSize(),
            ) {
                AnimationWidget(
                    parameters = parameters,
                    testingMode = testingMode,
                )
            }
        }
    }
}

/**
 * Scrollable wrapper for PlotCarouselWidget
 */
@Composable
fun ScrollablePlotCarouselWidget(parameters: Map<String, String>, testingMode: Boolean = false, modifier: Modifier = Modifier) {
    val verticalScrollState = rememberScrollState()
    val horizontalScrollState = rememberScrollState()

    Box(
        modifier = modifier.fillMaxSize(),
    ) {
        // Scrollable content
        Box(
            modifier =
            Modifier
                .verticalScroll(verticalScrollState)
                .horizontalScroll(horizontalScrollState)
                .padding(8.dp),
        ) {
            // Use fillMaxSize instead of fixed size
            Box(
                modifier = Modifier.fillMaxSize(),
            ) {
                PlotCarouselWidget(
                    parameters = parameters,
                    testingMode = testingMode,
                )
            }
        }
    }
}

/**
 * Scrollable wrapper for DataDisplayPanel
 */
@Composable
fun ScrollableDataDisplayPanel(parameters: Map<String, String>, testingMode: Boolean = false, modifier: Modifier = Modifier) {
    val verticalScrollState = rememberScrollState()
    val horizontalScrollState = rememberScrollState()

    Box(
        modifier = modifier.fillMaxSize(),
    ) {
        // Scrollable content
        Box(
            modifier =
            Modifier
                .verticalScroll(verticalScrollState)
                .horizontalScroll(horizontalScrollState)
                .padding(8.dp),
        ) {
            // Use fillMaxSize instead of fixed size
            Box(
                modifier = Modifier.fillMaxSize(),
            ) {
                DataDisplayPanel(
                    parameters = parameters,
                    testingMode = testingMode,
                )
            }
        }
    }
}

/**
 * Scrollable wrapper for ParameterInputForm
 * Note: ParameterInputForm already has scrolling implemented, but this provides additional scrolling if needed
 */
@Composable
fun ScrollableParameterInputForm(
    testingMode: Boolean = false,
    onParametersChanged: (Map<String, String>) -> Unit = {},
    layoutManager: LayoutManager,
    modifier: Modifier = Modifier,
) {
    val verticalScrollState = rememberScrollState()

    Box(
        modifier = modifier.fillMaxSize(),
    ) {
        // Scrollable content
        Box(
            modifier =
            Modifier
                .verticalScroll(verticalScrollState)
                .padding(8.dp),
        ) {
            // The parameter form has its own internal scrolling, but this provides outer scrolling
            Box(
                modifier = Modifier.fillMaxWidth(),
            ) {
                ParameterInputForm(
                    testingMode = testingMode,
                    onParametersChanged = onParametersChanged,
                    layoutManager = layoutManager,
                )
            }
        }
    }
}

@Composable
fun ScrollableStaticProfilesPanel(parameters: Map<String, String>, testingMode: Boolean = false, modifier: Modifier = Modifier) {
    val verticalScrollState = rememberScrollState()
    val horizontalScrollState = rememberScrollState()

    Box(modifier = modifier.fillMaxSize()) {
        Box(
            modifier =
            Modifier
                .verticalScroll(verticalScrollState)
                .horizontalScroll(horizontalScrollState)
                .padding(8.dp),
        ) {
            Box(Modifier.fillMaxSize()) {
                StaticProfilesPanel(
                    parameters = parameters,
                    testingMode = testingMode,
                )
            }
        }
    }
}

/**
 * Generic scrollable content wrapper
 * Enhanced to ensure resize gestures have higher priority than scroll gestures
 */
@Composable
fun ScrollableContent(
    modifier: Modifier = Modifier,
    enableVerticalScroll: Boolean = true,
    enableHorizontalScroll: Boolean = false,
    content: @Composable BoxScope.() -> Unit,
) {
    val scrollState = rememberScrollState()

    Box(
        modifier = modifier.fillMaxSize(),
    ) {
        // Scrollable content with extra top padding to compensate for removed titles
        Box(
            modifier =
            Modifier
                .fillMaxSize()
                .then(
                    if (enableVerticalScroll) {
                        Modifier.verticalScroll(scrollState)
                    } else {
                        Modifier
                    },
                ).then(
                    if (enableHorizontalScroll) {
                        Modifier.horizontalScroll(rememberScrollState())
                    } else {
                        Modifier
                    },
                )
                // Add extra top padding to compensate for removed titles
                .padding(top = 16.dp, start = 8.dp, end = 8.dp, bottom = 8.dp),
        ) {
            content()
        }
    }
}
