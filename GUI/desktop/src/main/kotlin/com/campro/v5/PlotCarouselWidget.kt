@file:OptIn(androidx.compose.material3.ExperimentalMaterial3Api::class)

package com.campro.v5

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectTransformGestures
import androidx.compose.foundation.layout.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Download
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Save
import androidx.compose.material.icons.filled.ZoomIn
import androidx.compose.material.icons.filled.ZoomOut
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.scale
import androidx.compose.ui.graphics.drawscope.translate
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.unit.dp
import kotlin.math.*

/**
 * A widget that displays a carousel of different plots related to the cycloidal animation.
 *
 * @param parameters Map of parameter names to values
 * @param testingMode Whether the widget is in testing mode
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun PlotCarouselWidget(parameters: Map<String, String>, testingMode: Boolean = false) {
    // Extract key parameters with defaults if not present
    val pistonDiameter = parameters["Piston Diameter"]?.toFloatOrNull() ?: 70f
    val stroke = parameters["Stroke"]?.toFloatOrNull() ?: 20f
    val rodLength = parameters["Rod Length"]?.toFloatOrNull() ?: 40f
    val tdcOffset = parameters["TDC Offset"]?.toFloatOrNull() ?: 40f
    val cycleRatio = parameters["Cycle Ratio"]?.toFloatOrNull() ?: 2f

    // Plot state
    var selectedPlotIndex by remember { mutableStateOf(0) }
    var scale by remember { mutableStateOf(1f) }
    var offset by remember { mutableStateOf(Offset.Zero) }

    // Define plot types
    val plotTypes =
        listOf(
            "Displacement",
            "Velocity",
            "Acceleration",
            "Force",
            "Stress",
        )

    Column(
        modifier = Modifier.fillMaxSize().padding(16.dp),
    ) {
        // Plot type selector
        TabRow(selectedTabIndex = selectedPlotIndex) {
            plotTypes.forEachIndexed { index, plotType ->
                Tab(
                    selected = selectedPlotIndex == index,
                    onClick = {
                        selectedPlotIndex = index
                        if (testingMode) {
                            println("EVENT:{\"type\":\"tab_selected\",\"component\":\"PlotTypeTab\",\"value\":\"$plotType\"}")
                        }
                    },
                    text = { Text(plotType) },
                )
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        // Plot controls
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            // Zoom controls
            Row(
                verticalAlignment = Alignment.CenterVertically,
            ) {
                IconButton(
                    onClick = {
                        scale = (scale * 0.8f).coerceAtLeast(0.1f)
                        if (testingMode) {
                            println("EVENT:{\"type\":\"button_clicked\",\"component\":\"PlotZoomOutButton\"}")
                        }
                    },
                ) {
                    Icon(
                        imageVector = Icons.Filled.ZoomOut,
                        contentDescription = "Zoom Out",
                    )
                }

                Text("${(scale * 100).toInt()}%")

                IconButton(
                    onClick = {
                        scale = (scale * 1.2f).coerceAtMost(5f)
                        if (testingMode) {
                            println("EVENT:{\"type\":\"button_clicked\",\"component\":\"PlotZoomInButton\"}")
                        }
                    },
                ) {
                    Icon(
                        imageVector = Icons.Filled.ZoomIn,
                        contentDescription = "Zoom In",
                    )
                }
            }

            // Reset view button
            IconButton(
                onClick = {
                    scale = 1f
                    offset = Offset.Zero
                    if (testingMode) {
                        println("EVENT:{\"type\":\"button_clicked\",\"component\":\"PlotResetViewButton\"}")
                    }
                },
            ) {
                Icon(
                    imageVector = Icons.Filled.Refresh,
                    contentDescription = "Reset View",
                )
            }

            // Export button
            IconButton(
                onClick = {
                    // Export functionality to be implemented
                    if (testingMode) {
                        println("EVENT:{\"type\":\"button_clicked\",\"component\":\"PlotExportButton\"}")
                    }
                },
            ) {
                Icon(
                    imageVector = Icons.Filled.Save,
                    contentDescription = "Export",
                )
            }

            // Data export button
            IconButton(
                onClick = {
                    // Data export functionality to be implemented
                    if (testingMode) {
                        println("EVENT:{\"type\":\"button_clicked\",\"component\":\"DataExportButton\"}")
                    }
                },
            ) {
                Icon(
                    imageVector = Icons.Filled.Download,
                    contentDescription = "Export Data",
                )
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        // Plot canvas
        Box(
            modifier =
            Modifier
                .fillMaxSize()
                .clip(MaterialTheme.shapes.medium)
                .background(MaterialTheme.colorScheme.surface)
                .pointerInput(Unit) {
                    detectTransformGestures { _, pan, zoom, _ ->
                        scale = (scale * zoom).coerceIn(0.1f, 5f)
                        offset += pan
                        if (testingMode) {
                            println(
                                "EVENT:{\"type\":\"gesture\",\"component\":\"PlotCanvas\",\"action\":\"pan_zoom\",\"scale\":\"$scale\",\"offset\":\"$offset\"}",
                            )
                        }
                    }
                },
        ) {
            Canvas(
                modifier = Modifier.fillMaxSize(),
            ) {
                val canvasWidth = size.width
                val canvasHeight = size.height
                val centerX = canvasWidth / 2
                val centerY = canvasHeight / 2

                // Apply pan and zoom transformations
                translate(offset.x, offset.y) {
                    scale(scale) {
                        // Draw axes
                        drawLine(
                            color = Color.Gray,
                            start = Offset(50f, canvasHeight - 50f),
                            end = Offset(canvasWidth - 50f, canvasHeight - 50f),
                            strokeWidth = 2f,
                            cap = StrokeCap.Round,
                        )
                        drawLine(
                            color = Color.Gray,
                            start = Offset(50f, 50f),
                            end = Offset(50f, canvasHeight - 50f),
                            strokeWidth = 2f,
                            cap = StrokeCap.Round,
                        )

                        // Draw plot based on selected type
                        when (selectedPlotIndex) {
                            0 -> drawDisplacementPlot(canvasWidth, canvasHeight, stroke, cycleRatio)
                            1 -> drawVelocityPlot(canvasWidth, canvasHeight, stroke, cycleRatio)
                            2 -> drawAccelerationPlot(canvasWidth, canvasHeight, stroke, cycleRatio)
                            3 -> drawForcePlot(canvasWidth, canvasHeight, stroke, cycleRatio, pistonDiameter)
                            4 -> drawStressPlot(canvasWidth, canvasHeight, stroke, cycleRatio, rodLength)
                        }
                    }
                }
            }

            // Overlay information
            Column(
                modifier = Modifier.align(Alignment.TopStart).padding(16.dp),
                horizontalAlignment = Alignment.Start,
            ) {
                Text(
                    "Plot: ${plotTypes[selectedPlotIndex]}",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurface,
                )

                // Display relevant parameters based on plot type
                when (selectedPlotIndex) {
                    0 -> { // Displacement
                        Text(
                            "Stroke: $stroke mm",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurface,
                        )
                        Text(
                            "Cycle Ratio: $cycleRatio",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurface,
                        )
                    }
                    1 -> { // Velocity
                        Text(
                            "Stroke: $stroke mm",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurface,
                        )
                        Text(
                            "Cycle Ratio: $cycleRatio",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurface,
                        )
                    }
                    2 -> { // Acceleration
                        Text(
                            "Stroke: $stroke mm",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurface,
                        )
                        Text(
                            "Cycle Ratio: $cycleRatio",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurface,
                        )
                    }
                    3 -> { // Force
                        Text(
                            "Piston Diameter: $pistonDiameter mm",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurface,
                        )
                        Text(
                            "Stroke: $stroke mm",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurface,
                        )
                    }
                    4 -> { // Stress
                        Text(
                            "Rod Length: $rodLength mm",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurface,
                        )
                        Text(
                            "Stroke: $stroke mm",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurface,
                        )
                    }
                }
            }

            // Data point inspection overlay (to be implemented)
        }
    }
}

// No custom extension function needed as we're directly importing androidx.compose.foundation.background

// Draw displacement plot
private fun androidx.compose.ui.graphics.drawscope.DrawScope.drawDisplacementPlot(
    canvasWidth: Float,
    canvasHeight: Float,
    stroke: Float,
    cycleRatio: Float,
) {
    val path = Path()
    val plotWidth = canvasWidth - 100f
    val plotHeight = canvasHeight - 100f
    val startX = 50f
    val startY = canvasHeight - 50f

    path.moveTo(startX, startY - (plotHeight / 2))

    // Generate displacement curve (simple sine wave for demonstration)
    for (i in 0..360) {
        val x = startX + (i / 360f) * plotWidth
        val angle = i * PI.toFloat() / 180f
        val displacement = sin(angle * cycleRatio) * (stroke / 2)
        val y = startY - (plotHeight / 2) - displacement * (plotHeight / stroke)

        path.lineTo(x, y)
    }

    // Draw the path
    drawPath(
        path = path,
        color = Color.Blue,
        style = Stroke(width = 3f),
    )

    // Draw axis labels
    drawLine(
        color = Color.Gray,
        start = Offset(startX, startY - plotHeight / 2),
        end = Offset(canvasWidth - 50f, startY - plotHeight / 2),
        strokeWidth = 1f,
        pathEffect =
        androidx.compose.ui.graphics.PathEffect
            .dashPathEffect(floatArrayOf(5f, 5f)),
    )
}

// Draw velocity plot
private fun androidx.compose.ui.graphics.drawscope.DrawScope.drawVelocityPlot(
    canvasWidth: Float,
    canvasHeight: Float,
    stroke: Float,
    cycleRatio: Float,
) {
    val path = Path()
    val plotWidth = canvasWidth - 100f
    val plotHeight = canvasHeight - 100f
    val startX = 50f
    val startY = canvasHeight - 50f

    path.moveTo(startX, startY - (plotHeight / 2))

    // Generate velocity curve (cosine wave for demonstration)
    for (i in 0..360) {
        val x = startX + (i / 360f) * plotWidth
        val angle = i * PI.toFloat() / 180f
        val velocity = cos(angle * cycleRatio) * (stroke / 2) * cycleRatio
        val y = startY - (plotHeight / 2) - velocity * (plotHeight / (stroke * cycleRatio))

        path.lineTo(x, y)
    }

    // Draw the path
    drawPath(
        path = path,
        color = Color.Red,
        style = Stroke(width = 3f),
    )

    // Draw axis labels
    drawLine(
        color = Color.Gray,
        start = Offset(startX, startY - plotHeight / 2),
        end = Offset(canvasWidth - 50f, startY - plotHeight / 2),
        strokeWidth = 1f,
        pathEffect =
        androidx.compose.ui.graphics.PathEffect
            .dashPathEffect(floatArrayOf(5f, 5f)),
    )
}

// Draw acceleration plot
private fun androidx.compose.ui.graphics.drawscope.DrawScope.drawAccelerationPlot(
    canvasWidth: Float,
    canvasHeight: Float,
    stroke: Float,
    cycleRatio: Float,
) {
    val path = Path()
    val plotWidth = canvasWidth - 100f
    val plotHeight = canvasHeight - 100f
    val startX = 50f
    val startY = canvasHeight - 50f

    path.moveTo(startX, startY - (plotHeight / 2))

    // Generate acceleration curve (negative sine wave for demonstration)
    for (i in 0..360) {
        val x = startX + (i / 360f) * plotWidth
        val angle = i * PI.toFloat() / 180f
        val acceleration = -sin(angle * cycleRatio) * (stroke / 2) * cycleRatio * cycleRatio
        val y = startY - (plotHeight / 2) - acceleration * (plotHeight / (stroke * cycleRatio * cycleRatio))

        path.lineTo(x, y)
    }

    // Draw the path
    drawPath(
        path = path,
        color = Color.Green,
        style = Stroke(width = 3f),
    )

    // Draw axis labels
    drawLine(
        color = Color.Gray,
        start = Offset(startX, startY - plotHeight / 2),
        end = Offset(canvasWidth - 50f, startY - plotHeight / 2),
        strokeWidth = 1f,
        pathEffect =
        androidx.compose.ui.graphics.PathEffect
            .dashPathEffect(floatArrayOf(5f, 5f)),
    )
}

// Draw force plot
private fun androidx.compose.ui.graphics.drawscope.DrawScope.drawForcePlot(
    canvasWidth: Float,
    canvasHeight: Float,
    stroke: Float,
    cycleRatio: Float,
    pistonDiameter: Float,
) {
    val path = Path()
    val plotWidth = canvasWidth - 100f
    val plotHeight = canvasHeight - 100f
    val startX = 50f
    val startY = canvasHeight - 50f

    path.moveTo(startX, startY - (plotHeight / 2))

    // Generate force curve (modified sine wave for demonstration)
    for (i in 0..360) {
        val x = startX + (i / 360f) * plotWidth
        val angle = i * PI.toFloat() / 180f
        // Simulate force with a more complex pattern
        val force = sin(angle * cycleRatio) * (1 + 0.5f * sin(angle * 2)) * (pistonDiameter * pistonDiameter * 0.01f)
        val y = startY - (plotHeight / 2) - force * (plotHeight / (pistonDiameter * pistonDiameter * 0.01f))

        path.lineTo(x, y)
    }

    // Draw the path
    drawPath(
        path = path,
        color = Color.Magenta,
        style = Stroke(width = 3f),
    )

    // Draw axis labels
    drawLine(
        color = Color.Gray,
        start = Offset(startX, startY - plotHeight / 2),
        end = Offset(canvasWidth - 50f, startY - plotHeight / 2),
        strokeWidth = 1f,
        pathEffect =
        androidx.compose.ui.graphics.PathEffect
            .dashPathEffect(floatArrayOf(5f, 5f)),
    )
}

// Draw stress plot
private fun androidx.compose.ui.graphics.drawscope.DrawScope.drawStressPlot(
    canvasWidth: Float,
    canvasHeight: Float,
    stroke: Float,
    cycleRatio: Float,
    rodLength: Float,
) {
    val path = Path()
    val plotWidth = canvasWidth - 100f
    val plotHeight = canvasHeight - 100f
    val startX = 50f
    val startY = canvasHeight - 50f

    path.moveTo(startX, startY - (plotHeight / 2))

    // Generate stress curve (complex pattern for demonstration)
    for (i in 0..360) {
        val x = startX + (i / 360f) * plotWidth
        val angle = i * PI.toFloat() / 180f
        // Simulate stress with a more complex pattern
        val stress = abs(sin(angle * cycleRatio)) * (1 + 0.3f * sin(angle * 3)) * (stroke / rodLength) * 100
        val y = startY - (plotHeight / 2) - stress * (plotHeight / 100)

        path.lineTo(x, y)
    }

    // Draw the path
    drawPath(
        path = path,
        color = Color.DarkGray,
        style = Stroke(width = 3f),
    )

    // Draw axis labels
    drawLine(
        color = Color.Gray,
        start = Offset(startX, startY - plotHeight / 2),
        end = Offset(canvasWidth - 50f, startY - plotHeight / 2),
        strokeWidth = 1f,
        pathEffect =
        androidx.compose.ui.graphics.PathEffect
            .dashPathEffect(floatArrayOf(5f, 5f)),
    )
}
