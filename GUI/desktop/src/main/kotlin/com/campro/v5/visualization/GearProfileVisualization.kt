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
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.campro.v5.debug.DebugIconButton
import com.campro.v5.models.GearProfileData
import kotlin.math.*

/**
 * Gear profile visualization component using Compose Canvas.
 *
 * This component displays the optimized gear profiles including sun, planet,
 * and ring gear profiles in a 2D view.
 */
@Composable
fun GearProfileVisualization(
    gearProfiles: GearProfileData,
    modifier: Modifier = Modifier,
    showSun: Boolean = true,
    showPlanet: Boolean = true,
    showRing: Boolean = true,
) {
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
                text = "Gear Profile Analysis",
                style = MaterialTheme.typography.titleLarge,
                color = MaterialTheme.colorScheme.onSurface,
            )

            // Legend
            GearProfileLegend(
                showSun = showSun,
                showPlanet = showPlanet,
                showRing = showRing,
                onToggleSun = { /* TODO: Implement toggle */ },
                onTogglePlanet = { /* TODO: Implement toggle */ },
                onToggleRing = { /* TODO: Implement toggle */ },
            )

            // Main visualization
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
                GearProfileChart(
                    gearProfiles = gearProfiles,
                    showSun = showSun,
                    showPlanet = showPlanet,
                    showRing = showRing,
                    modifier = Modifier.fillMaxSize(),
                )
            }

            // Profile information
            GearProfileInformation(gearProfiles = gearProfiles)
        }
    }
}

@Composable
private fun GearProfileLegend(
    showSun: Boolean,
    showPlanet: Boolean,
    showRing: Boolean,
    onToggleSun: () -> Unit,
    onTogglePlanet: () -> Unit,
    onToggleRing: () -> Unit,
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(16.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        LegendItem(
            color = Color(0xFF2196F3), // Blue
            label = "Sun Gear",
            isVisible = showSun,
            onToggle = onToggleSun,
        )

        LegendItem(
            color = Color(0xFF4CAF50), // Green
            label = "Planet Gear",
            isVisible = showPlanet,
            onToggle = onTogglePlanet,
        )

        LegendItem(
            color = Color(0xFFFF9800), // Orange
            label = "Ring Gear",
            isVisible = showRing,
            onToggle = onToggleRing,
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
            buttonId = "gear-legend-toggle-" + label.lowercase().replace(" ", "-"),
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
private fun GearProfileChart(
    gearProfiles: GearProfileData,
    showSun: Boolean,
    showPlanet: Boolean,
    showRing: Boolean,
    modifier: Modifier = Modifier,
) {
    Canvas(modifier = modifier) {
        val canvasWidth = size.width
        val canvasHeight = size.height

        // Calculate center and scale
        val centerX = canvasWidth / 2
        val centerY = canvasHeight / 2

        // Find maximum radius for scaling
        val maxRadius = maxOf(
            gearProfiles.rSun.maxOrNull() ?: 0.0,
            gearProfiles.rPlanet.maxOrNull() ?: 0.0,
            gearProfiles.rRingInner.maxOrNull() ?: 0.0,
        )

        val scale = minOf(canvasWidth, canvasHeight) * 0.4f / maxRadius.toFloat()

        // Draw coordinate system
        drawCoordinateSystem(centerX, centerY, scale)

        // Draw gear profiles with CORRECT planetary gearset kinematics
        if (showRing) {
            // Ring gear: centered at origin (0,0) - stationary
            drawGearProfile(
                radii = gearProfiles.rRingInner,
                color = Color(0xFFFF9800),
                centerX = centerX,
                centerY = centerY,
                scale = scale,
                label = "Ring",
            )
        }

        if (showSun) {
            // Sun gear: centered at origin (0,0)
            drawGearProfile(
                radii = gearProfiles.rSun,
                color = Color(0xFF2196F3),
                centerX = centerX,
                centerY = centerY,
                scale = scale,
                label = "Sun",
            )
        }

        if (showPlanet) {
            // Planet gear: show 2 planet gears with 180-degree offset for static assembly
            val planetCount = 2
            val planetOffset = PI.toFloat() // 180 degrees offset between planets
            
            // Calculate planet center distance from sun center
            // For proper meshing: planet center distance = r_sun + r_planet
            // This ensures the planet touches the sun gear
            val avgSunRadius = gearProfiles.rSun.average()
            val avgPlanetRadius = gearProfiles.rPlanet.average()
            val planetCenterDistance = avgSunRadius + avgPlanetRadius
            
            for (i in 0 until planetCount) {
                val orbitAngle = (i * planetOffset).toFloat()
                
                // Planet center positions (orbiting around sun)
                val planetCenterX = centerX + (planetCenterDistance * cos(orbitAngle) * scale).toFloat()
                val planetCenterY = centerY + (planetCenterDistance * sin(orbitAngle) * scale).toFloat()
                
                // Draw planet gear at this orbital position with proper rotation
                // For a 2:1 gear ratio, the planet rotates 2x the orbital angle
                val gearRatio = 2.0f
                val planetRotation = gearRatio * orbitAngle
                
                drawGearProfile(
                    radii = gearProfiles.rPlanet,
                    color = Color(0xFF4CAF50), // Solid color for static assembly
                    centerX = planetCenterX,
                    centerY = planetCenterY,
                    scale = scale,
                    label = "Planet",
                    rotationOffset = planetRotation,
                )
            }
        }
    }
}

private fun DrawScope.drawCoordinateSystem(centerX: Float, centerY: Float, scale: Float) {
    val strokeWidth = 1.dp.toPx()
    val gridColor = Color.Gray.copy(alpha = 0.3f)

    // Draw concentric circles for reference
    for (i in 1..5) {
        val radius = (i * 50.0 * scale).toFloat()
        drawCircle(
            color = gridColor,
            radius = radius,
            center = Offset(centerX, centerY),
            style = Stroke(width = strokeWidth),
        )
    }

    // Draw center point
    drawCircle(
        color = Color.Gray,
        radius = 4.dp.toPx(),
        center = Offset(centerX, centerY),
    )
}

private fun DrawScope.drawGearProfile(radii: DoubleArray, color: Color, centerX: Float, centerY: Float, scale: Float, label: String, rotationOffset: Float = 0f) {
    if (radii.isEmpty()) return

    // Draw gear profile as a series of connected points
    val points = radii.mapIndexed { index, radius ->
        val angle = (index * 2 * PI / radii.size + rotationOffset).toFloat()
        val x = centerX + (radius * cos(angle) * scale).toFloat()
        val y = centerY + (radius * sin(angle) * scale).toFloat()
        Offset(x, y)
    }

    // Draw profile curve
    if (points.size > 1) {
        val path = Path()
        path.moveTo(points[0].x, points[0].y)

        for (i in 1 until points.size) {
            path.lineTo(points[i].x, points[i].y)
        }

        // Close the path
        path.close()

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
            radius = 3.dp.toPx(),
            center = point,
        )
    }

    // Draw label
    if (points.isNotEmpty()) {
        val labelPoint = points[0]
        // Note: Text drawing removed - will be handled by Compose Text components
    }
}

@Composable
private fun GearProfileInformation(gearProfiles: GearProfileData) {
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
                text = "Profile Information",
                style = MaterialTheme.typography.titleSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
            ) {
                ProfileInfoItem(
                    label = "Gear Ratio",
                    value = String.format("%.2f", gearProfiles.gearRatio),
                    unit = "",
                )

                ProfileInfoItem(
                    label = "Optimal Method",
                    value = gearProfiles.optimalMethod,
                    unit = "",
                )

                ProfileInfoItem(
                    label = "Sun Radius",
                    value = String.format("%.1f", gearProfiles.rSun.average()),
                    unit = "mm",
                )
            }

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
            ) {
                ProfileInfoItem(
                    label = "Planet Radius",
                    value = String.format("%.1f", gearProfiles.rPlanet.average()),
                    unit = "mm",
                )

                ProfileInfoItem(
                    label = "Ring Radius",
                    value = String.format("%.1f", gearProfiles.rRingInner.average()),
                    unit = "mm",
                )

                ProfileInfoItem(
                    label = "Profile Points",
                    value = gearProfiles.rSun.size.toString(),
                    unit = "",
                )
            }
        }
    }
}

@Composable
private fun ProfileInfoItem(label: String, value: String, unit: String) {
    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Text(
            text = label,
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
        Text(
            text = if (unit.isNotEmpty()) "$value $unit" else value,
            style = MaterialTheme.typography.titleMedium,
            fontWeight = FontWeight.Bold,
            color = MaterialTheme.colorScheme.onSurface,
        )
    }
}
