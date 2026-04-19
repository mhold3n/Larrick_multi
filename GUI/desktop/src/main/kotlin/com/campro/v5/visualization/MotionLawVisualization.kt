package com.campro.v5.visualization

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.*
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.campro.v5.debug.DebugIconButton
import com.campro.v5.models.MotionLawData
import kotlin.math.*

/**
 * Motion law visualization component using Compose Canvas.
 *
 * This component displays displacement, velocity, and acceleration curves
 * for the motion law data from optimization results.
 */
@Composable
fun MotionLawVisualization(
    motionLaw: MotionLawData,
    modifier: Modifier = Modifier,
    showDisplacement: Boolean = true,
    showVelocity: Boolean = true,
    showAcceleration: Boolean = true,
) {
    val density = LocalDensity.current

    Card(
        modifier = modifier.fillMaxSize(),
        elevation = CardDefaults.cardElevation(defaultElevation = 2.dp),
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            // Header
            Text(
                text = "Motion Law Analysis",
                style = MaterialTheme.typography.titleLarge,
                color = MaterialTheme.colorScheme.onSurface,
            )

            // Legend
            MotionLawLegend(
                showDisplacement = showDisplacement,
                showVelocity = showVelocity,
                showAcceleration = showAcceleration,
                onToggleDisplacement = { /* TODO: Implement toggle */ },
                onToggleVelocity = { /* TODO: Implement toggle */ },
                onToggleAcceleration = { /* TODO: Implement toggle */ },
            )

            // Main chart
            Box(
                modifier = Modifier
                    .weight(1f)
                    .fillMaxWidth()
                    .background(
                        color = MaterialTheme.colorScheme.surfaceVariant,
                        shape = MaterialTheme.shapes.medium,
                    )
                    .padding(16.dp),
            ) {
                MotionLawChart(
                    motionLaw = motionLaw,
                    showDisplacement = showDisplacement,
                    showVelocity = showVelocity,
                    showAcceleration = showAcceleration,
                    modifier = Modifier.fillMaxSize(),
                )
            }

            // Statistics
            MotionLawStatistics(motionLaw = motionLaw)
        }
    }
}

@Composable
private fun MotionLawLegend(
    showDisplacement: Boolean,
    showVelocity: Boolean,
    showAcceleration: Boolean,
    onToggleDisplacement: () -> Unit,
    onToggleVelocity: () -> Unit,
    onToggleAcceleration: () -> Unit,
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(16.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        LegendItem(
            color = Color(0xFF2196F3), // Blue
            label = "Displacement",
            isVisible = showDisplacement,
            onToggle = onToggleDisplacement,
        )

        LegendItem(
            color = Color(0xFF4CAF50), // Green
            label = "Velocity",
            isVisible = showVelocity,
            onToggle = onToggleVelocity,
        )

        LegendItem(
            color = Color(0xFFFF9800), // Orange
            label = "Acceleration",
            isVisible = showAcceleration,
            onToggle = onToggleAcceleration,
        )
    }
}

@Composable
private fun LegendItem(color: Color, label: String, isVisible: Boolean, onToggle: () -> Unit) {
    Row(
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        Box(
            modifier = Modifier
                .size(16.dp)
                .background(
                    color = if (isVisible) color else color.copy(alpha = 0.3f),
                    shape = androidx.compose.foundation.shape.CircleShape,
                ),
        )

        Text(
            text = label,
            style = MaterialTheme.typography.bodyMedium,
            color = if (isVisible) MaterialTheme.colorScheme.onSurface else MaterialTheme.colorScheme.onSurfaceVariant,
        )

        DebugIconButton(
            buttonId = "motion-legend-toggle-" + label.lowercase().replace(" ", "-"),
            onClick = onToggle,
            modifier = Modifier.size(20.dp),
        ) {
            Icon(
                imageVector = if (isVisible) Icons.Default.Visibility else Icons.Default.VisibilityOff,
                contentDescription = if (isVisible) "Hide $label" else "Show $label",
                modifier = Modifier.size(16.dp),
                tint = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

@Composable
private fun MotionLawChart(
    motionLaw: MotionLawData,
    showDisplacement: Boolean,
    showVelocity: Boolean,
    showAcceleration: Boolean,
    modifier: Modifier = Modifier,
) {
    Canvas(modifier = modifier) {
        val canvasWidth = size.width
        val canvasHeight = size.height

        // Calculate margins
        val marginLeft = 60.dp.toPx()
        val marginRight = 20.dp.toPx()
        val marginTop = 20.dp.toPx()
        val marginBottom = 60.dp.toPx()

        val chartWidth = canvasWidth - marginLeft - marginRight
        val chartHeight = canvasHeight - marginTop - marginBottom

        // Draw axes
        drawAxes(
            chartWidth = chartWidth,
            chartHeight = chartHeight,
            marginLeft = marginLeft,
            marginTop = marginTop,
        )

        // Draw grid
        drawGrid(
            chartWidth = chartWidth,
            chartHeight = chartHeight,
            marginLeft = marginLeft,
            marginTop = marginTop,
        )

        // Draw curves
        if (showDisplacement) {
            drawMotionLawCurve(
                data = motionLaw.displacement,
                theta = motionLaw.thetaDeg,
                color = Color(0xFF2196F3),
                chartWidth = chartWidth,
                chartHeight = chartHeight,
                marginLeft = marginLeft,
                marginTop = marginTop,
                label = "Displacement (mm)",
            )
        }

        if (showVelocity) {
            drawMotionLawCurve(
                data = motionLaw.velocity,
                theta = motionLaw.thetaDeg,
                color = Color(0xFF4CAF50),
                chartWidth = chartWidth,
                chartHeight = chartHeight,
                marginLeft = marginLeft,
                marginTop = marginTop,
                label = "Velocity (mm/ω)",
            )
        }

        if (showAcceleration) {
            drawMotionLawCurve(
                data = motionLaw.acceleration,
                theta = motionLaw.thetaDeg,
                color = Color(0xFFFF9800),
                chartWidth = chartWidth,
                chartHeight = chartHeight,
                marginLeft = marginLeft,
                marginTop = marginTop,
                label = "Acceleration (mm/ω²)",
            )
        }
    }
}

private fun DrawScope.drawAxes(chartWidth: Float, chartHeight: Float, marginLeft: Float, marginTop: Float) {
    val strokeWidth = 2.dp.toPx()
    val axisColor = Color.Gray

    // X-axis (theta)
    drawLine(
        start = Offset(marginLeft, marginTop + chartHeight),
        end = Offset(marginLeft + chartWidth, marginTop + chartHeight),
        color = axisColor,
        strokeWidth = strokeWidth,
    )

    // Y-axis
    drawLine(
        start = Offset(marginLeft, marginTop),
        end = Offset(marginLeft, marginTop + chartHeight),
        color = axisColor,
        strokeWidth = strokeWidth,
    )
}

private fun DrawScope.drawGrid(chartWidth: Float, chartHeight: Float, marginLeft: Float, marginTop: Float) {
    val gridColor = Color.Gray.copy(alpha = 0.3f)
    val strokeWidth = 1.dp.toPx()

    // Vertical grid lines (every 30 degrees)
    for (i in 0..12) {
        val x = marginLeft + (chartWidth * i / 12)
        drawLine(
            start = Offset(x, marginTop),
            end = Offset(x, marginTop + chartHeight),
            color = gridColor,
            strokeWidth = strokeWidth,
        )
    }

    // Horizontal grid lines
    for (i in 0..10) {
        val y = marginTop + (chartHeight * i / 10)
        drawLine(
            start = Offset(marginLeft, y),
            end = Offset(marginLeft + chartWidth, y),
            color = gridColor,
            strokeWidth = strokeWidth,
        )
    }
}

private fun DrawScope.drawMotionLawCurve(
    data: DoubleArray,
    theta: DoubleArray,
    color: Color,
    chartWidth: Float,
    chartHeight: Float,
    marginLeft: Float,
    marginTop: Float,
    label: String,
) {
    if (data.isEmpty() || theta.isEmpty()) return

    // Find data range for scaling
    val dataMin = data.minOrNull() ?: 0.0
    val dataMax = data.maxOrNull() ?: 0.0
    val dataRange = dataMax - dataMin

    // Find theta range
    val thetaMin = theta.minOrNull() ?: 0.0
    val thetaMax = theta.maxOrNull() ?: 0.0
    val thetaRange = thetaMax - thetaMin

    // Convert data points to canvas coordinates
    val points = data.mapIndexed { index, value ->
        val x = marginLeft + (chartWidth * (theta[index] - thetaMin) / thetaRange)
        val y = marginTop + chartHeight - (chartHeight * (value - dataMin) / dataRange)
        Offset(x.toFloat(), y.toFloat())
    }

    // Draw curve
    if (points.size > 1) {
        val path = Path()
        path.moveTo(points[0].x, points[0].y)

        for (i in 1 until points.size) {
            path.lineTo(points[i].x, points[i].y)
        }

        drawPath(
            path = path,
            color = color,
            style = Stroke(width = 3.dp.toPx()),
        )
    }

    // Draw data points
    points.forEach { point ->
        drawCircle(
            color = color,
            radius = 4.dp.toPx(),
            center = point,
        )
    }
}

@Composable
private fun MotionLawStatistics(motionLaw: MotionLawData) {
    Card(
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant,
        ),
    ) {
        Column(
            modifier = Modifier.padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Text(
                text = "Statistics",
                style = MaterialTheme.typography.titleSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
            ) {
                StatisticItem(
                    label = "Max Displacement",
                    value = String.format("%.2f", motionLaw.displacement.maxOrNull() ?: 0.0),
                    unit = "mm",
                )

                StatisticItem(
                    label = "Max Velocity",
                    value = String.format("%.2f", motionLaw.velocity.maxOrNull() ?: 0.0),
                    unit = "mm/ω",
                )

                StatisticItem(
                    label = "Max Acceleration",
                    value = String.format("%.2f", motionLaw.acceleration.maxOrNull() ?: 0.0),
                    unit = "mm/ω²",
                )
            }
        }
    }
}

@Composable
private fun StatisticItem(label: String, value: String, unit: String) {
    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Text(
            text = label,
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Text(
            text = "$value $unit",
            style = MaterialTheme.typography.titleMedium,
            fontWeight = FontWeight.Bold,
            color = MaterialTheme.colorScheme.onSurface,
        )
    }
}
