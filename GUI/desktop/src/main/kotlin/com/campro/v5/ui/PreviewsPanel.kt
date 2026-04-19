package com.campro.v5.ui

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.campro.v5.animation.MotionLawEngine
import com.campro.v5.data.litvin.MotionLawSample
import com.campro.v5.data.litvin.MotionLawSamples

/**
 * PreviewsPanel displays motion law profiles (displacement, velocity, acceleration)
 * from the MotionLawEngine samples.
 */
@Composable
fun PreviewsPanel(engine: MotionLawEngine, modifier: Modifier = Modifier) {
    // Get the current size of the container for responsive scaling
    BoxWithConstraints(modifier = modifier.fillMaxSize()) {
        val containerWidth = constraints.maxWidth.toFloat()
        val containerHeight = constraints.maxHeight.toFloat()

        // Calculate responsive scaling based on container size
        val scaleFactor =
            remember(containerWidth, containerHeight) {
                val baseWidth = 400f
                val baseHeight = 300f
                val widthScale = (containerWidth / baseWidth).coerceIn(0.5f, 2.0f)
                val heightScale = (containerHeight / baseHeight).coerceIn(0.5f, 2.0f)
                minOf(widthScale, heightScale)
            }
        val motionSamples = engine.getMotionLawSamples()

        Column(modifier = Modifier.fillMaxSize()) {
            Text(
                "Motion Law Profiles",
                style =
                MaterialTheme.typography.titleSmall.copy(
                    fontSize = (MaterialTheme.typography.titleSmall.fontSize.value * scaleFactor.toFloat()).sp,
                ),
                modifier = Modifier.padding(bottom = (8 * scaleFactor.toFloat()).dp),
            )

            if (motionSamples == null || motionSamples.samples.isEmpty()) {
                Text(
                    "No motion law samples available. Set parameters and generate animation.",
                    style =
                    MaterialTheme.typography.bodySmall.copy(
                        fontSize = (MaterialTheme.typography.bodySmall.fontSize.value * scaleFactor.toFloat()).sp,
                    ),
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            } else {
                // Simple plot of displacement, velocity, and acceleration with responsive sizing
                SimpleMotionPlot(
                    motionSamples,
                    Modifier
                        .fillMaxWidth()
                        .height((200 * scaleFactor.toFloat()).dp),
                )
            }
        }
    }
}

@Composable
private fun SimpleMotionPlot(motionSamples: MotionLawSamples, modifier: Modifier = Modifier) {
    val samples = motionSamples.samples
    if (samples.isEmpty()) return

    // Find ranges for scaling
    val xValues = samples.map { it.thetaDeg }
    val displacementValues = samples.map { it.xMm }
    val velocityValues = samples.map { it.vMmPerOmega }
    val accelerationValues = samples.map { it.aMmPerOmega2 }

    val xMin = xValues.minOrNull() ?: 0.0
    val xMax = xValues.maxOrNull() ?: 360.0
    val displacementMin = displacementValues.minOrNull() ?: 0.0
    val displacementMax = displacementValues.maxOrNull() ?: 10.0
    val velocityMin = velocityValues.minOrNull() ?: -1.0
    val velocityMax = velocityValues.maxOrNull() ?: 1.0
    val accelerationMin = accelerationValues.minOrNull() ?: -10.0
    val accelerationMax = accelerationValues.maxOrNull() ?: 10.0

    Canvas(modifier = modifier) {
        val canvasWidth = size.width
        val canvasHeight = size.height
        val plotHeight = canvasHeight / 3f
        // Responsive padding based on canvas size
        val padding = (minOf(canvasWidth, canvasHeight) * 0.05f).coerceAtLeast(10f)

        // Draw displacement plot (top third)
        drawDisplacementPlot(
            samples, displacementMin, displacementMax,
            xMin, xMax, 0f, plotHeight, padding,
        )

        // Draw velocity plot (middle third)
        drawVelocityPlot(
            samples, velocityMin, velocityMax,
            xMin, xMax, plotHeight, plotHeight, padding,
        )

        // Draw acceleration plot (bottom third)
        drawAccelerationPlot(
            samples, accelerationMin, accelerationMax,
            xMin, xMax, plotHeight * 2, plotHeight, padding,
        )

        // TODO: Add text labels when proper Canvas text drawing is implemented
    }
}

private fun androidx.compose.ui.graphics.drawscope.DrawScope.drawDisplacementPlot(
    samples: List<MotionLawSample>,
    yMin: Double,
    yMax: Double,
    xMin: Double,
    xMax: Double,
    yOffset: Float,
    height: Float,
    padding: Float,
) {
    val path = Path()
    val xScale = (size.width - 2 * padding) / (xMax - xMin).toFloat()
    val yScale = (height - 2 * padding) / (yMax - yMin).toFloat()
    val yBaseline = yOffset + height - padding

    var firstPoint = true
    for (sample in samples) {
        val x = padding + ((sample.thetaDeg - xMin) * xScale).toFloat()
        val y = yBaseline - ((sample.xMm - yMin) * yScale).toFloat()

        if (firstPoint) {
            path.moveTo(x, y)
            firstPoint = false
        } else {
            path.lineTo(x, y)
        }
    }

    // Responsive stroke width based on canvas size
    val strokeWidth = (minOf(size.width, size.height) * 0.005f).coerceAtLeast(1f)
    drawPath(path, Color.Blue, style = Stroke(width = strokeWidth))
}

private fun androidx.compose.ui.graphics.drawscope.DrawScope.drawVelocityPlot(
    samples: List<MotionLawSample>,
    yMin: Double,
    yMax: Double,
    xMin: Double,
    xMax: Double,
    yOffset: Float,
    height: Float,
    padding: Float,
) {
    val path = Path()
    val xScale = (size.width - 2 * padding) / (xMax - xMin).toFloat()
    val yScale = (height - 2 * padding) / (yMax - yMin).toFloat()
    val yBaseline = yOffset + height - padding

    var firstPoint = true
    for (sample in samples) {
        val x = padding + ((sample.thetaDeg - xMin) * xScale).toFloat()
        val y = yBaseline - ((sample.vMmPerOmega - yMin) * yScale).toFloat()

        if (firstPoint) {
            path.moveTo(x, y)
            firstPoint = false
        } else {
            path.lineTo(x, y)
        }
    }

    // Responsive stroke width based on canvas size
    val strokeWidth = (minOf(size.width, size.height) * 0.005f).coerceAtLeast(1f)
    drawPath(path, Color.Green, style = Stroke(width = strokeWidth))
}

private fun androidx.compose.ui.graphics.drawscope.DrawScope.drawAccelerationPlot(
    samples: List<MotionLawSample>,
    yMin: Double,
    yMax: Double,
    xMin: Double,
    xMax: Double,
    yOffset: Float,
    height: Float,
    padding: Float,
) {
    val path = Path()
    val xScale = (size.width - 2 * padding) / (xMax - xMin).toFloat()
    val yScale = (height - 2 * padding) / (yMax - yMin).toFloat()
    val yBaseline = yOffset + height - padding

    var firstPoint = true
    for (sample in samples) {
        val x = padding + ((sample.thetaDeg - xMin) * xScale).toFloat()
        val y = yBaseline - ((sample.aMmPerOmega2 - yMin) * yScale).toFloat()

        if (firstPoint) {
            path.moveTo(x, y)
            firstPoint = false
        } else {
            path.lineTo(x, y)
        }
    }

    // Responsive stroke width based on canvas size
    val strokeWidth = (minOf(size.width, size.height) * 0.005f).coerceAtLeast(1f)
    drawPath(path, Color.Red, style = Stroke(width = strokeWidth))
}
