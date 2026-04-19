package com.campro.v5.animation

import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.scale
import androidx.compose.ui.graphics.drawscope.translate
import kotlin.math.*

/**
 * Renderer for FEA-based animation.
 * This class contains the drawing code for the FEA-based animation.
 */
object FeaBasedAnimationRenderer {
    // Color gradient for stress visualization
    private val stressColors =
        listOf(
            Color(0xFF0000FF), // Blue (low stress)
            Color(0xFF00FFFF), // Cyan
            Color(0xFF00FF00), // Green
            Color(0xFFFFFF00), // Yellow
            Color(0xFFFF0000), // Red (high stress)
        )

    /**
     * Draw a frame of the FEA-based animation.
     *
     * @param drawScope The draw scope to use for drawing
     * @param canvasWidth The width of the canvas
     * @param canvasHeight The height of the canvas
     * @param scale The scale factor
     * @param offset The offset
     * @param angle The current angle in degrees
     * @param parameters The animation parameters
     * @param analysisData The FEA analysis data
     */
    fun drawFrame(
        drawScope: DrawScope,
        canvasWidth: Float,
        canvasHeight: Float,
        scale: Float,
        offset: Offset,
        angle: Float,
        parameters: Map<String, String>,
        analysisData: AnalysisData?,
    ) {
        // Extract key parameters with defaults if not present
        val baseCircleRadius = ParameterResolver.float(parameters, "base_circle_radius", 25f)

        // Estimate content bounds (includes mesh envelope and legend footprint)
        val meshOuter = baseCircleRadius + 20f // base + padding for mesh/displacement
        val legendW = 200f
        val legendH = 20f
        val extraPadding = 40f
        val contentWidth = meshOuter * 2 + legendW + extraPadding
        val contentHeight = meshOuter * 2 + legendH + extraPadding

        // Auto-fit scale factor with 10% margin
        val panelScaleFactor =
            minOf(
                canvasWidth / (contentWidth * 1.1f),
                canvasHeight / (contentHeight * 1.1f),
            )
        val effectiveScale = scale * panelScaleFactor

        drawScope.apply {
            val centerX = canvasWidth / 2
            val centerY = canvasHeight / 2

            if (analysisData == null) {
                // Draw placeholder using unified transform
                scale(effectiveScale, pivot = Offset(centerX, centerY)) {
                    translate(offset.x, offset.y) {
                        drawPlaceholderInternal(centerX, centerY)
                    }
                }
                return
            }

            // Apply unified transform for real data
            scale(effectiveScale, pivot = Offset(centerX, centerY)) {
                translate(offset.x, offset.y) {
                    // Draw base circle as reference
                    drawCircle(
                        color = Color.Gray.copy(alpha = 0.3f),
                        radius = baseCircleRadius,
                        center = Offset(centerX, centerY),
                        style = Stroke(width = 1f),
                    )

                    // Draw mesh with displacements and stress
                    drawMesh(
                        centerX = centerX,
                        centerY = centerY,
                        analysisData = analysisData,
                        angle = angle,
                    )

                    // Draw center point
                    drawCircle(
                        color = Color.Black,
                        radius = 5f,
                        center = Offset(centerX, centerY),
                    )

                    // Draw angle indicator
                    val angleRad = angle * PI.toFloat() / 180f
                    val x = centerX + baseCircleRadius * 1.5f * cos(angleRad)
                    val y = centerY + baseCircleRadius * 1.5f * sin(angleRad)

                    drawLine(
                        color = Color.Green,
                        start = Offset(centerX, centerY),
                        end = Offset(x, y),
                        strokeWidth = 1f,
                        alpha = 0.5f,
                    )

                    // Draw legend (position relative to center)
                    drawStressLegend(
                        x = centerX - legendW / 2,
                        y = centerY + meshOuter + 30f,
                        width = legendW,
                        height = legendH,
                    )
                }
            }
        }
    }

    /**
     * Draw a placeholder when no analysis data is available.
     */
    private fun DrawScope.drawPlaceholderInternal(centerX: Float, centerY: Float) {
        // Draw placeholder circle
        drawCircle(
            color = Color.Gray.copy(alpha = 0.3f),
            radius = 100f,
            center = Offset(centerX, centerY),
            style = Stroke(width = 2f),
        )
        // Draw cross
        drawLine(
            color = Color.Gray.copy(alpha = 0.5f),
            start = Offset(centerX - 70f, centerY - 70f),
            end = Offset(centerX + 70f, centerY + 70f),
            strokeWidth = 2f,
        )
        drawLine(
            color = Color.Gray.copy(alpha = 0.5f),
            start = Offset(centerX + 70f, centerY - 70f),
            end = Offset(centerX - 70f, centerY + 70f),
            strokeWidth = 2f,
        )
    }

    /**
     * Draw the mesh with displacements and stress.
     *
     * @param centerX The x-coordinate of the center
     * @param centerY The y-coordinate of the center
     * @param analysisData The FEA analysis data
     * @param angle The current angle in degrees
     */
    private fun DrawScope.drawMesh(centerX: Float, centerY: Float, analysisData: AnalysisData, angle: Float) {
        // Find the time step closest to the current angle
        val timeStepIndex =
            if (analysisData.timeSteps.isNotEmpty()) {
                val normalizedAngle = angle / 360f
                val closestTimeStep =
                    analysisData.timeSteps.minByOrNull {
                        abs(it - normalizedAngle)
                    } ?: 0f
                analysisData.timeSteps.indexOf(closestTimeStep).coerceAtLeast(0)
            } else {
                0
            }

        // Draw elements with stress coloring
        val maxStress = analysisData.stresses.values.maxOrNull() ?: 1f

        // Create a mesh of triangles (simplified for visualization)
        val numRadialSegments = 16
        val numCircumferentialSegments = 32

        for (i in 0 until numRadialSegments) {
            val innerRadius = 25f + 10f * i / numRadialSegments
            val outerRadius = 25f + 10f * (i + 1) / numRadialSegments

            for (j in 0 until numCircumferentialSegments) {
                val startAngle = j * 2 * PI.toFloat() / numCircumferentialSegments
                val endAngle = (j + 1) * 2 * PI.toFloat() / numCircumferentialSegments

                val innerStartX = centerX + innerRadius * cos(startAngle)
                val innerStartY = centerY + innerRadius * sin(startAngle)
                val innerEndX = centerX + innerRadius * cos(endAngle)
                val innerEndY = centerY + innerRadius * sin(endAngle)
                val outerStartX = centerX + outerRadius * cos(startAngle)
                val outerStartY = centerY + outerRadius * sin(startAngle)
                val outerEndX = centerX + outerRadius * cos(endAngle)
                val outerEndY = centerY + outerRadius * sin(endAngle)

                // Calculate a pseudo-stress value for visualization
                val elementId = i * numCircumferentialSegments + j
                val stress = analysisData.stresses[elementId] ?: (0.5f * maxStress)
                val normalizedStress = (stress / maxStress).coerceIn(0f, 1f)

                // Get color based on stress
                val color = getStressColor(normalizedStress)

                // Draw the quad as two triangles
                drawPath(
                    path =
                    androidx.compose.ui.graphics.Path().apply {
                        moveTo(innerStartX, innerStartY)
                        lineTo(innerEndX, innerEndY)
                        lineTo(outerEndX, outerEndY)
                        lineTo(outerStartX, outerStartY)
                        close()
                    },
                    color = color,
                    alpha = 0.7f,
                )

                // Draw element outline
                drawLine(
                    color = Color.Black.copy(alpha = 0.2f),
                    start = Offset(innerStartX, innerStartY),
                    end = Offset(innerEndX, innerEndY),
                    strokeWidth = 0.5f,
                )

                drawLine(
                    color = Color.Black.copy(alpha = 0.2f),
                    start = Offset(outerStartX, outerStartY),
                    end = Offset(outerEndX, outerEndY),
                    strokeWidth = 0.5f,
                )

                drawLine(
                    color = Color.Black.copy(alpha = 0.2f),
                    start = Offset(innerStartX, innerStartY),
                    end = Offset(outerStartX, outerStartY),
                    strokeWidth = 0.5f,
                )

                drawLine(
                    color = Color.Black.copy(alpha = 0.2f),
                    start = Offset(innerEndX, innerEndY),
                    end = Offset(outerEndX, outerEndY),
                    strokeWidth = 0.5f,
                )
            }
        }

        // Draw nodes with displacements
        for ((nodeId, displacement) in analysisData.displacements) {
            // Calculate original node position (simplified)
            val theta = nodeId % numCircumferentialSegments * 2 * PI.toFloat() / numCircumferentialSegments
            val radius = 25f + 10f * (nodeId / numCircumferentialSegments) / numRadialSegments

            val originalX = centerX + radius * cos(theta)
            val originalY = centerY + radius * sin(theta)

            // Apply displacement (scaled for visibility)
            val displacementScale = 10f // Scale factor for displacements
            val displacedX = originalX + displacement.x * displacementScale
            val displacedY = originalY + displacement.y * displacementScale

            // Draw displacement vector
            drawLine(
                color = Color.Black.copy(alpha = 0.5f),
                start = Offset(originalX, originalY),
                end = Offset(displacedX, displacedY),
                strokeWidth = 0.5f,
            )

            // Draw node
            drawCircle(
                color = Color.Black,
                radius = 1f,
                center = Offset(displacedX, displacedY),
            )
        }
    }

    /**
     * Draw a legend for the stress colors.
     */
    private fun DrawScope.drawStressLegend(x: Float, y: Float, width: Float, height: Float) {
        // Draw gradient bar
        val segmentWidth = width / (stressColors.size - 1)

        for (i in 0 until stressColors.size - 1) {
            val startX = x + i * segmentWidth
            val endX = x + (i + 1) * segmentWidth

            // Draw gradient segment
            drawRect(
                color = stressColors[i],
                topLeft = Offset(startX, y),
                size =
                androidx.compose.ui.geometry
                    .Size(segmentWidth, height),
            )
        }

        // Draw border
        drawRect(
            color = Color.Black,
            topLeft = Offset(x, y),
            size =
            androidx.compose.ui.geometry
                .Size(width, height),
            style = Stroke(width = 1f),
        )

        // Draw labels (would need Text composables for actual text)
        drawLine(
            color = Color.Black,
            start = Offset(x, y + height + 5f),
            end = Offset(x, y + height + 10f),
            strokeWidth = 1f,
        )

        drawLine(
            color = Color.Black,
            start = Offset(x + width, y + height + 5f),
            end = Offset(x + width, y + height + 10f),
            strokeWidth = 1f,
        )
    }

    /**
     * Get a color based on the normalized stress value.
     *
     * @param normalizedStress The normalized stress value (0.0 to 1.0)
     * @return The color corresponding to the stress value
     */
    private fun getStressColor(normalizedStress: Float): Color {
        if (normalizedStress <= 0f) return stressColors.first()
        if (normalizedStress >= 1f) return stressColors.last()

        val segmentCount = stressColors.size - 1
        val segment = (normalizedStress * segmentCount).toInt()
        val segmentFraction = (normalizedStress * segmentCount) - segment

        val startColor = stressColors[segment]
        val endColor = stressColors[segment + 1]

        return Color(
            red = lerp(startColor.red, endColor.red, segmentFraction),
            green = lerp(startColor.green, endColor.green, segmentFraction),
            blue = lerp(startColor.blue, endColor.blue, segmentFraction),
            alpha = 1f,
        )
    }

    /**
     * Linear interpolation between two values.
     */
    private fun lerp(start: Float, end: Float, fraction: Float): Float = start + (end - start) * fraction
}
