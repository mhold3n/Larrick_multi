package com.campro.v5.animation

import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.scale
import androidx.compose.ui.graphics.drawscope.translate
import kotlin.math.*

/**
 * Renderer for component-based animation.
 * This class contains the drawing code for the component-based animation.
 */
object ComponentBasedAnimationRenderer {
    /**
     * Calculate the bounds of all animation components.
     *
     * @param baseCircleRadius The radius of the base circle
     * @param maxLift The maximum lift
     * @param pistonDiameter The diameter of the piston
     * @return The bounds of all components as a Pair of width and height
     */
    private fun calculateComponentBounds(baseCircleRadius: Float, maxLift: Float, pistonDiameter: Float): Pair<Float, Float> {
        // Calculate the maximum extent of the cam profile
        val camProfileRadius = baseCircleRadius + maxLift

        // Calculate the maximum extent of the piston
        val pistonHeight = pistonDiameter * 1.5f
        val pistonWidth = pistonDiameter

        // Calculate the maximum extent of the follower
        val followerRadius = baseCircleRadius * 0.2f

        // Calculate the maximum extent of the motion path
        // The motion path extends baseCircleRadius + maxLift in all directions
        val motionPathExtent = baseCircleRadius + maxLift

        // Calculate the total width and height needed
        // We need to account for the cam profile, piston, follower, and motion path
        // The width is determined by the cam profile, piston, and motion path
        // The height is determined by the cam profile, piston height, and motion path
        val totalWidth = (camProfileRadius + pistonWidth + motionPathExtent) * 2
        val totalHeight = (camProfileRadius + pistonHeight + motionPathExtent) * 2

        return Pair(totalWidth, totalHeight)
    }

    /**
     * Draw a frame of the component-based animation.
     *
     * @param drawScope The draw scope to use for drawing
     * @param canvasWidth The width of the canvas
     * @param canvasHeight The height of the canvas
     * @param scale The scale factor
     * @param offset The offset
     * @param angle The current angle in degrees
     * @param parameters The animation parameters
     * @param componentPositions The positions of the components
     */

    /**
     * Enhanced bounds including optional eccentric ring envelope
     */
    private fun calculateComponentBoundsIncludingRing(
        baseCircleRadius: Float,
        maxLift: Float,
        pistonDiameter: Float,
        pistonHeightRatio: Float,
        ringEnabled: Boolean,
        ringRadius: Float,
        ringThickness: Float,
        eccentricity: Float,
    ): Pair<Float, Float> {
        val camProfileRadius = baseCircleRadius + maxLift
        val followerMargin = baseCircleRadius * 0.2f // approximate follower radius
        val pistonHeight = pistonDiameter * pistonHeightRatio
        val pistonWidth = pistonDiameter
        val motionPathExtent = baseCircleRadius + maxLift

        val ringOuter = if (ringEnabled) ringRadius + ringThickness / 2f + abs(eccentricity) else 0f
        val contentRadius = max(camProfileRadius + followerMargin, ringOuter)

        val totalWidth = (max(contentRadius, camProfileRadius) + pistonWidth + motionPathExtent) * 2
        val totalHeight = (max(contentRadius, camProfileRadius) + pistonHeight + motionPathExtent) * 2
        return totalWidth to totalHeight
    }

    fun drawFrame(
        drawScope: DrawScope,
        canvasWidth: Float,
        canvasHeight: Float,
        scale: Float,
        offset: Offset,
        angle: Float,
        parameters: Map<String, String>,
        assemblies: List<ComponentPositions>,
    ) {
        println("DEBUG: ComponentBasedAnimationRenderer.drawFrame called - canvas: ${canvasWidth}x$canvasHeight, scale: $scale, angle: $angle, assemblies: ${assemblies.size}")
        // Extract key parameters with defaults if not present
        val pistonDiameter = ParameterResolver.float(parameters, "piston_diameter", 70f, "Piston Diameter")
        val baseCircleRadius = ParameterResolver.float(parameters, "base_circle_radius", 25f)
        val maxLift = ParameterResolver.float(parameters, "max_lift", 10f)

        // Ring and replication parameters
        val ringEnabled = ParameterResolver.bool(parameters, "ring_enabled", true, "Show Ring")
        val stroke = ParameterResolver.float(parameters, "stroke", maxLift)
        val ringThickness = ParameterResolver.float(parameters, "ring_thickness", 8f, "Ring Thickness")
        val minRingRadius = baseCircleRadius + stroke + max(ringThickness, 3f)
        val requestedRingRadius = ParameterResolver.float(parameters, "ring_radius", minRingRadius, "Ring Radius")
        val ringRadius = max(requestedRingRadius, minRingRadius)
        val eccentricity = ParameterResolver.float(parameters, "eccentricity", 0f, "Ring Eccentricity")
        val ringAngle = ParameterResolver.float(parameters, "ring_angle", 0f, "Ring Angle")

        // Calculate the bounds including ring envelope
        val pistonHeightRatio = ParameterResolver.float(parameters, "piston_height_ratio", 0.8f, "Piston Height Ratio")

        val (contentWidth, contentHeight) =
            calculateComponentBoundsIncludingRing(
                baseCircleRadius = baseCircleRadius,
                maxLift = maxLift,
                pistonDiameter = pistonDiameter,
                pistonHeightRatio = pistonHeightRatio,
                ringEnabled = ringEnabled,
                ringRadius = ringRadius,
                ringThickness = ringThickness,
                eccentricity = eccentricity,
            )

        // Calculate scale factor to fit within panel with 10% margin
        val panelScaleFactor =
            minOf(
                canvasWidth / (contentWidth * 1.1f),
                canvasHeight / (contentHeight * 1.1f),
            )

        // Apply automatic scaling first, then user scaling
        val effectiveScale = scale * panelScaleFactor

        drawScope.apply {
            val centerX = canvasWidth / 2
            val centerY = canvasHeight / 2

            // Unified transform: scale about canvas center, then pan in screen space
            scale(effectiveScale, pivot = Offset(centerX, centerY)) {
                translate(offset.x, offset.y) {
                    // Draw ring first if enabled
                    if (ringEnabled) {
                        drawEccentricRing(centerX, centerY, ringRadius, ringThickness, eccentricity, ringAngle)
                    }

                    // Draw base circle (cam base)
                    drawCircle(
                        color = Color.Blue.copy(alpha = 0.3f),
                        radius = baseCircleRadius,
                        center = Offset(centerX, centerY),
                        style = Stroke(width = 2f),
                    )

                    // Draw cam profile
                    drawCamProfile(
                        centerX = centerX,
                        centerY = centerY,
                        baseCircleRadius = baseCircleRadius,
                        maxLift = maxLift,
                        angle = angle,
                        parameters = parameters,
                    )

                    // Draw motion path once (shared)
                    drawMotionPath(
                        centerX = centerX,
                        centerY = centerY,
                        baseCircleRadius = baseCircleRadius,
                        maxLift = maxLift,
                        parameters = parameters,
                    )

                    // Draw assemblies
                    assemblies.forEachIndexed { i, componentPositions ->
                        drawAssembly(
                            centerX = centerX,
                            centerY = centerY,
                            baseCircleRadius = baseCircleRadius,
                            pistonDiameter = pistonDiameter,
                            pistonHeightRatio = pistonHeightRatio,
                            angle = angle,
                            positions = componentPositions,
                            tintIndex = i,
                        )
                    }

                    // Draw center point
                    drawCircle(
                        color = Color.Black,
                        radius = 5f,
                        center = Offset(centerX, centerY),
                    )
                }
            }
        }
    }

    private fun DrawScope.drawEccentricRing(
        centerX: Float,
        centerY: Float,
        ringRadius: Float,
        ringThickness: Float,
        eccentricity: Float,
        ringAngleDeg: Float,
    ) {
        val angRad = Math.toRadians(ringAngleDeg.toDouble()).toFloat()
        val cx = centerX + eccentricity * cos(angRad)
        val cy = centerY + eccentricity * sin(angRad)
        val rOut = ringRadius + ringThickness / 2f
        val rIn = max(1f, ringRadius - ringThickness / 2f)

        val path =
            androidx.compose.ui.graphics.Path().apply {
                this.fillType = androidx.compose.ui.graphics.PathFillType.EvenOdd
                addOval(
                    androidx.compose.ui.geometry
                        .Rect(cx - rOut, cy - rOut, cx + rOut, cy + rOut),
                )
                addOval(
                    androidx.compose.ui.geometry
                        .Rect(cx - rIn, cy - rIn, cx + rIn, cy + rIn),
                )
            }
        drawPath(path = path, color = Color(0xFF606060).copy(alpha = 0.7f))
        drawCircle(color = Color.Black.copy(alpha = 0.5f), radius = rOut, center = Offset(cx, cy), style = Stroke(width = 0.8f))
        drawCircle(color = Color.Black.copy(alpha = 0.4f), radius = rIn, center = Offset(cx, cy), style = Stroke(width = 0.6f))
    }

    private fun DrawScope.drawAssembly(
        centerX: Float,
        centerY: Float,
        baseCircleRadius: Float,
        pistonDiameter: Float,
        pistonHeightRatio: Float,
        angle: Float,
        positions: ComponentPositions,
        tintIndex: Int,
    ) {
        val followerX = centerX + positions.rodPosition.x
        val followerY = centerY + positions.rodPosition.y
        val pistonX = centerX + positions.pistonPosition.x
        val pistonY = centerY + positions.pistonPosition.y

        drawRollerFollower(
            centerX = followerX,
            centerY = followerY,
            radius = baseCircleRadius * 0.2f,
            angle = angle,
        )
        drawConnectingRod(
            followerX = followerX,
            followerY = followerY,
            pistonX = pistonX,
            pistonY = pistonY,
            width = baseCircleRadius * 0.15f,
        )
        drawPiston(
            centerX = pistonX,
            centerY = pistonY,
            diameter = pistonDiameter,
            height = pistonDiameter * pistonHeightRatio,
        )
    }

    /**
     * Draw the cam profile.
     *
     * @param centerX The x-coordinate of the center
     * @param centerY The y-coordinate of the center
     * @param baseCircleRadius The radius of the base circle
     * @param maxLift The maximum lift
     * @param angle The current angle in degrees
     */
    private fun DrawScope.drawCamProfile(
        centerX: Float,
        centerY: Float,
        baseCircleRadius: Float,
        maxLift: Float,
        angle: Float,
        parameters: Map<String, String>,
    ) {
        // Draw cam profile using SHM + dwells parameterization similar to CamProV3
        val numPoints = 360
        val points = mutableListOf<Offset>()

        // Read V3-style parameters with sensible defaults
        val stroke = ParameterResolver.float(parameters, "stroke", maxLift)
        val tdcAngle = ParameterResolver.float(parameters, "TDC_angle", 180f)
        val bdcDwell = ParameterResolver.float(parameters, "BDC_dwell", 0f)
        val tdcDwell = ParameterResolver.float(parameters, "TDC_dwell", 0f)
        val halfBDC = bdcDwell * 0.5f
        val halfTDC = tdcDwell * 0.5f
        val riseStart = halfBDC
        val riseEnd = (tdcAngle - halfTDC).coerceAtLeast(riseStart + 1e-4f)
        val fallStart = tdcAngle + halfTDC
        val fallEnd = (360f - halfBDC).coerceAtLeast(fallStart + 1e-4f)

        fun shmDisp(theta: Float, startDeg: Float, endDeg: Float, s0: Float, s1: Float): Float {
            val range = (endDeg - startDeg).coerceAtLeast(1e-6f)
            val p = ((theta - startDeg) / range).coerceIn(0f, 1f)
            return s0 + (s1 - s0) * (1f - cos(PI.toFloat() * p)) / 2f
        }

        // Create the cam profile points
        for (i in 0 until numPoints) {
            val theta = i.toFloat() // 1 degree increments
            val thetaRad = Math.toRadians(theta.toDouble()).toFloat()

            val s =
                when {
                    theta < riseStart -> 0f
                    theta <= riseEnd -> shmDisp(theta, riseStart, riseEnd, 0f, stroke)
                    theta <= fallStart -> stroke
                    theta <= fallEnd -> shmDisp(theta, fallStart, fallEnd, stroke, 0f)
                    else -> 0f
                }

            val radius = baseCircleRadius + s
            val x = centerX + radius * cos(thetaRad)
            val y = centerY + radius * sin(thetaRad)
            points.add(Offset(x, y))
        }

        // Draw the cam profile with a filled shape and gradient for 3D effect
        val camPath =
            androidx.compose.ui.graphics
                .Path()

        // Start from the center
        camPath.moveTo(centerX, centerY)

        // Add all points of the cam profile
        for (point in points) {
            camPath.lineTo(point.x, point.y)
        }

        // Close the path back to the first point
        camPath.close()

        // Draw the filled cam with a gradient for 3D effect
        drawPath(
            path = camPath,
            brush =
            androidx.compose.ui.graphics.Brush.radialGradient(
                colors =
                listOf(
                    Color(0xFF8A8A8A), // Light metallic gray at center
                    Color(0xFF5A5A5A), // Darker gray at edges
                ),
                center = Offset(centerX, centerY),
                radius = baseCircleRadius + maxLift * 1.5f,
            ),
            alpha = 0.9f,
        )

        // Draw the cam profile outline
        for (i in 0 until numPoints) {
            val start = points[i]
            val end = points[(i + 1) % numPoints]

            drawLine(
                color = Color.Red,
                start = start,
                end = end,
                strokeWidth = 1.5f,
            )
        }

        // Draw mechanical details: keyway
        val keyWidth = baseCircleRadius * 0.2f
        val keyDepth = baseCircleRadius * 0.3f
        val keyPath =
            androidx.compose.ui.graphics
                .Path()
        keyPath.moveTo(centerX, centerY)
        keyPath.lineTo(centerX - keyWidth / 2, centerY - keyDepth)
        keyPath.lineTo(centerX + keyWidth / 2, centerY - keyDepth)
        keyPath.close()

        drawPath(
            path = keyPath,
            color = Color.DarkGray,
            alpha = 0.8f,
        )

        // Draw center hole
        drawCircle(
            color = Color.DarkGray,
            radius = baseCircleRadius * 0.15f,
            center = Offset(centerX, centerY),
            alpha = 0.8f,
        )

        // Draw mounting holes
        val mountingHoleRadius = baseCircleRadius * 0.1f
        val mountingHoleDistance = baseCircleRadius * 0.6f

        for (holeAngle in 0 until 360 step 90) {
            val holeRad = Math.toRadians(holeAngle.toDouble()).toFloat()
            val holeX = centerX + mountingHoleDistance * cos(holeRad)
            val holeY = centerY + mountingHoleDistance * sin(holeRad)

            drawCircle(
                color = Color.DarkGray,
                radius = mountingHoleRadius,
                center = Offset(holeX, holeY),
                alpha = 0.8f,
            )
        }

        // Draw a line indicating the current angle using the same SHM+dwell law
        val currentAngleRad = angle * PI.toFloat() / 180f
        val sCurrent =
            when {
                angle < riseStart -> 0f
                angle <= riseEnd -> shmDisp(angle, riseStart, riseEnd, 0f, stroke)
                angle <= fallStart -> stroke
                angle <= fallEnd -> shmDisp(angle, fallStart, fallEnd, stroke, 0f)
                else -> 0f
            }
        val currentRadius = baseCircleRadius + sCurrent
        val x = centerX + currentRadius * cos(currentAngleRad)
        val y = centerY + currentRadius * sin(currentAngleRad)

        drawLine(
            color = Color.Green,
            start = Offset(centerX, centerY),
            end = Offset(x, y),
            strokeWidth = 1f,
            alpha = 0.5f,
        )
    }

    /**
     * Draw a roller follower with mechanical details.
     *
     * @param centerX The x-coordinate of the center of the follower
     * @param centerY The y-coordinate of the center of the follower
     * @param radius The radius of the follower roller
     * @param angle The current angle in degrees
     */
    private fun DrawScope.drawRollerFollower(centerX: Float, centerY: Float, radius: Float, angle: Float) {
        // Draw the main roller body with metallic gradient
        drawCircle(
            brush =
            androidx.compose.ui.graphics.Brush.radialGradient(
                colors =
                listOf(
                    Color(0xFFD0D0D0), // Light silver at center
                    Color(0xFF909090), // Darker silver at edges
                ),
                center = Offset(centerX, centerY),
                radius = radius * 1.2f,
            ),
            radius = radius,
            center = Offset(centerX, centerY),
        )

        // Draw roller outline
        drawCircle(
            color = Color.DarkGray,
            radius = radius,
            center = Offset(centerX, centerY),
            style = Stroke(width = 1.5f),
        )

        // Draw bearing inner race
        drawCircle(
            color = Color.DarkGray,
            radius = radius * 0.3f,
            center = Offset(centerX, centerY),
            style = Stroke(width = 1f),
        )

        // Draw bearing center hole
        drawCircle(
            color = Color.DarkGray,
            radius = radius * 0.15f,
            center = Offset(centerX, centerY),
            alpha = 0.8f,
        )

        // Draw roller bearing balls
        val numBalls = 8
        val ballRadius = radius * 0.1f
        val ballDistance = radius * 0.65f

        for (i in 0 until numBalls) {
            val ballAngle = i * 2 * PI.toFloat() / numBalls
            val ballX = centerX + ballDistance * cos(ballAngle)
            val ballY = centerY + ballDistance * sin(ballAngle)

            drawCircle(
                color = Color(0xFFD0D0D0), // Silver
                radius = ballRadius,
                center = Offset(ballX, ballY),
            )
        }

        // Draw highlight to give 3D appearance
        val highlightAngle = PI.toFloat() * 0.25f
        val highlightX = centerX + radius * 0.7f * cos(highlightAngle)
        val highlightY = centerY + radius * 0.7f * sin(highlightAngle)

        drawCircle(
            color = Color.White,
            radius = radius * 0.2f,
            center = Offset(highlightX, highlightY),
            alpha = 0.3f,
        )
    }

    /**
     * Draw a connecting rod with proper mechanical details.
     *
     * @param followerX The x-coordinate of the follower end
     * @param followerY The y-coordinate of the follower end
     * @param pistonX The x-coordinate of the piston end
     * @param pistonY The y-coordinate of the piston end
     * @param width The width of the connecting rod
     */
    private fun DrawScope.drawConnectingRod(followerX: Float, followerY: Float, pistonX: Float, pistonY: Float, width: Float) {
        // Calculate rod angle and length
        val rodAngle = atan2(pistonY - followerY, pistonX - followerX)
        val rodLength = sqrt((pistonX - followerX).pow(2) + (pistonY - followerY).pow(2))

        // Create path for the rod body
        val rodPath =
            androidx.compose.ui.graphics
                .Path()

        // Calculate corner points for the rod
        val halfWidth = width / 2
        val perpAngle = rodAngle + PI.toFloat() / 2
        val perpX = cos(perpAngle)
        val perpY = sin(perpAngle)

        // Follower end is rounded (small end)
        val smallEndRadius = width * 0.8f

        // Piston end is larger (big end)
        val bigEndRadius = width * 1.2f

        // Calculate the four corners of the rod body
        val topLeftX = followerX + halfWidth * perpX
        val topLeftY = followerY + halfWidth * perpY
        val topRightX = pistonX + halfWidth * perpX
        val topRightY = pistonY + halfWidth * perpY
        val bottomRightX = pistonX - halfWidth * perpX
        val bottomRightY = pistonY - halfWidth * perpY
        val bottomLeftX = followerX - halfWidth * perpX
        val bottomLeftY = followerY - halfWidth * perpY

        // Draw the rod body
        rodPath.moveTo(topLeftX, topLeftY)
        rodPath.lineTo(topRightX, topRightY)
        rodPath.lineTo(bottomRightX, bottomRightY)
        rodPath.lineTo(bottomLeftX, bottomLeftY)
        rodPath.close()

        // Draw the rod with a gradient for 3D effect
        drawPath(
            path = rodPath,
            brush =
            androidx.compose.ui.graphics.Brush.linearGradient(
                colors =
                listOf(
                    Color(0xFF909090), // Darker gray at top
                    Color(0xFFB0B0B0), // Medium gray in middle
                    Color(0xFF707070), // Darkest gray at bottom
                ),
                start = Offset(followerX, followerY - width),
                end = Offset(followerX, followerY + width),
            ),
            alpha = 0.9f,
        )

        // Draw rod outline
        drawPath(
            path = rodPath,
            color = Color.DarkGray,
            style = Stroke(width = 1f),
        )

        // Draw small end (follower end)
        drawCircle(
            brush =
            androidx.compose.ui.graphics.Brush.radialGradient(
                colors =
                listOf(
                    Color(0xFFB0B0B0), // Light gray at center
                    Color(0xFF808080), // Darker gray at edges
                ),
                center = Offset(followerX, followerY),
                radius = smallEndRadius * 1.2f,
            ),
            radius = smallEndRadius,
            center = Offset(followerX, followerY),
        )

        drawCircle(
            color = Color.DarkGray,
            radius = smallEndRadius,
            center = Offset(followerX, followerY),
            style = Stroke(width = 1f),
        )

        // Draw big end (piston end)
        drawCircle(
            brush =
            androidx.compose.ui.graphics.Brush.radialGradient(
                colors =
                listOf(
                    Color(0xFFB0B0B0), // Light gray at center
                    Color(0xFF808080), // Darker gray at edges
                ),
                center = Offset(pistonX, pistonY),
                radius = bigEndRadius * 1.2f,
            ),
            radius = bigEndRadius,
            center = Offset(pistonX, pistonY),
        )

        drawCircle(
            color = Color.DarkGray,
            radius = bigEndRadius,
            center = Offset(pistonX, pistonY),
            style = Stroke(width = 1f),
        )

        // Draw bearing holes
        drawCircle(
            color = Color.DarkGray,
            radius = smallEndRadius * 0.4f,
            center = Offset(followerX, followerY),
            alpha = 0.8f,
        )

        drawCircle(
            color = Color.DarkGray,
            radius = bigEndRadius * 0.4f,
            center = Offset(pistonX, pistonY),
            alpha = 0.8f,
        )
    }

    /**
     * Draw a piston with realistic mechanical details.
     *
     * @param centerX The x-coordinate of the center of the piston
     * @param centerY The y-coordinate of the center of the piston
     * @param diameter The diameter of the piston
     * @param height The height of the piston
     */
    private fun DrawScope.drawPiston(centerX: Float, centerY: Float, diameter: Float, height: Float) {
        val radius = diameter / 2

        // Draw piston body (cylinder)
        val pistonPath =
            androidx.compose.ui.graphics
                .Path()

        // Top of piston (rounded)
        pistonPath.moveTo(centerX - radius, centerY - height / 2)
        pistonPath.lineTo(centerX + radius, centerY - height / 2)

        // Right side
        pistonPath.lineTo(centerX + radius, centerY + height / 2)

        // Bottom (flat)
        pistonPath.lineTo(centerX - radius, centerY + height / 2)

        // Left side back to top
        pistonPath.close()

        // Draw piston body with gradient for 3D effect
        drawPath(
            path = pistonPath,
            brush =
            androidx.compose.ui.graphics.Brush.linearGradient(
                colors =
                listOf(
                    Color(0xFFC0C0C0), // Light gray at left
                    Color(0xFFE0E0E0), // Lightest gray in middle
                    Color(0xFFA0A0A0), // Darker gray at right
                ),
                start = Offset(centerX - radius, centerY),
                end = Offset(centerX + radius, centerY),
            ),
            alpha = 0.9f,
        )

        // Draw piston outline
        drawPath(
            path = pistonPath,
            color = Color.DarkGray,
            style = Stroke(width = 1.5f),
        )

        // Draw piston rings
        val ringSpacing = height / 6
        val ringThickness = height / 30

        for (i in 1..2) {
            val ringY = centerY - height / 4 + i * ringSpacing

            // Draw ring
            drawLine(
                color = Color.DarkGray,
                start = Offset(centerX - radius, ringY),
                end = Offset(centerX + radius, ringY),
                strokeWidth = ringThickness,
            )
        }

        // Draw piston pin hole
        drawOval(
            color = Color.DarkGray,
            topLeft = Offset(centerX - radius * 0.4f, centerY - radius * 0.3f),
            size =
            androidx.compose.ui.geometry
                .Size(radius * 0.8f, radius * 0.6f),
            alpha = 0.8f,
        )

        // Draw highlight to give 3D appearance
        drawOval(
            color = Color.White,
            topLeft = Offset(centerX - radius * 0.7f, centerY - height * 0.4f),
            size =
            androidx.compose.ui.geometry
                .Size(radius * 0.3f, height * 0.2f),
            alpha = 0.2f,
        )
    }

    /**
     * Draw the motion path.
     *
     * @param centerX The x-coordinate of the center
     * @param centerY The y-coordinate of the center
     * @param baseCircleRadius The radius of the base circle
     * @param maxLift The maximum lift
     */
    private fun DrawScope.drawMotionPath(
        centerX: Float,
        centerY: Float,
        baseCircleRadius: Float,
        maxLift: Float,
        parameters: Map<String, String>,
    ) {
        // Draw the motion path using SHM + dwells
        val numPoints = 360
        val points = mutableListOf<Offset>()

        val stroke = ParameterResolver.float(parameters, "stroke", maxLift)
        val tdcAngle = ParameterResolver.float(parameters, "TDC_angle", 180f)
        val bdcDwell = ParameterResolver.float(parameters, "BDC_dwell", 0f)
        val tdcDwell = ParameterResolver.float(parameters, "TDC_dwell", 0f)
        val halfBDC = bdcDwell * 0.5f
        val halfTDC = tdcDwell * 0.5f
        val riseStart = halfBDC
        val riseEnd = (tdcAngle - halfTDC).coerceAtLeast(riseStart + 1e-4f)
        val fallStart = tdcAngle + halfTDC
        val fallEnd = (360f - halfBDC).coerceAtLeast(fallStart + 1e-4f)

        fun shmDisp(theta: Float, startDeg: Float, endDeg: Float, s0: Float, s1: Float): Float {
            val range = (endDeg - startDeg).coerceAtLeast(1e-6f)
            val p = ((theta - startDeg) / range).coerceIn(0f, 1f)
            return s0 + (s1 - s0) * (1f - cos(PI.toFloat() * p)) / 2f
        }

        for (i in 0 until numPoints) {
            val theta = i.toFloat()
            val thetaRad = Math.toRadians(theta.toDouble()).toFloat()

            val s =
                when {
                    theta < riseStart -> 0f
                    theta <= riseEnd -> shmDisp(theta, riseStart, riseEnd, 0f, stroke)
                    theta <= fallStart -> stroke
                    theta <= fallEnd -> shmDisp(theta, fallStart, fallEnd, stroke, 0f)
                    else -> 0f
                }

            // Follower path (use s as radial displacement component consistent with profile)
            val x = centerX + (baseCircleRadius + s) * cos(thetaRad)
            val y = centerY + (baseCircleRadius + s) * sin(thetaRad)

            points.add(Offset(x, y))
        }

        // Draw the motion path
        for (i in 0 until numPoints step 5) {
            val start = points[i]
            val end = points[(i + 5) % numPoints]

            drawLine(
                color = Color.Gray.copy(alpha = 0.3f),
                start = start,
                end = end,
                strokeWidth = 1f,
            )
        }
    }
}
