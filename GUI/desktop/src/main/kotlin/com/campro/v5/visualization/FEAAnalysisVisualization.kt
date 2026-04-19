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
import com.campro.v5.models.FEAAnalysisData
import kotlin.math.*

/**
 * FEA analysis visualization component.
 *
 * This component displays FEA analysis results including stress distribution,
 * natural frequencies, and fatigue analysis.
 */
@Composable
fun FEAAnalysisVisualization(feaAnalysis: FEAAnalysisData, modifier: Modifier = Modifier) {
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
                text = "FEA Analysis Results",
                style = MaterialTheme.typography.titleLarge,
                color = MaterialTheme.colorScheme.onSurface,
            )

            // Stress visualization
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
                StressVisualization(
                    feaAnalysis = feaAnalysis,
                    modifier = Modifier.fillMaxSize(),
                )
            }

            // Analysis summary
            FEAAnalysisSummary(feaAnalysis = feaAnalysis)
        }
    }
}

@Composable
private fun StressVisualization(feaAnalysis: FEAAnalysisData, modifier: Modifier = Modifier) {
    Canvas(modifier = modifier) {
        val canvasWidth = size.width
        val canvasHeight = size.height

        // Calculate margins with minimum size constraints
        val margin = maxOf(20.dp.toPx(), 10.dp.toPx())
        val chartWidth = maxOf(canvasWidth - 2 * margin, 100.dp.toPx())
        val chartHeight = maxOf(canvasHeight - 2 * margin, 100.dp.toPx())

        // Only draw if we have valid dimensions
        if (canvasWidth > 0 && canvasHeight > 0 && chartWidth > 0 && chartHeight > 0) {
            // Draw stress contour
            drawStressContour(
                maxStress = feaAnalysis.maxStress,
                chartWidth = chartWidth,
                chartHeight = chartHeight,
                margin = margin,
            )

            // Draw stress scale
            drawStressScale(
                maxStress = feaAnalysis.maxStress,
                chartWidth = chartWidth,
                chartHeight = chartHeight,
                margin = margin,
            )
        }
    }
}

private fun DrawScope.drawStressContour(maxStress: Double, chartWidth: Float, chartHeight: Float, margin: Float) {
    // Ensure valid dimensions
    if (chartWidth <= 0 || chartHeight <= 0 || maxStress <= 0) return

    // Simulate stress distribution (in real implementation, this would come from FEA data)
    val stressLevels = 10
    val stressStep = maxStress / stressLevels

    val centerX = margin + chartWidth / 2
    val centerY = margin + chartHeight / 2

    // Ensure center is within bounds
    if (centerX < 0 || centerY < 0 || centerX > size.width || centerY > size.height) return

    for (i in 0 until stressLevels) {
        val stress = i * stressStep
        val normalizedStress = stress / maxStress

        // Create color based on stress level
        val color = when {
            normalizedStress < 0.3 -> Color(0xFF4CAF50) // Green - low stress
            normalizedStress < 0.6 -> Color(0xFFFFEB3B) // Yellow - medium stress
            normalizedStress < 0.8 -> Color(0xFFFF9800) // Orange - high stress
            else -> Color(0xFFF44336) // Red - very high stress
        }

        // Draw stress contour (simplified as concentric circles)
        val radius = maxOf((chartWidth * 0.4f * (1 - normalizedStress)).toFloat(), 1.dp.toPx())

        // Only draw if radius is valid and circle fits within bounds
        if (radius > 0 &&
            centerX - radius >= 0 &&
            centerX + radius <= size.width &&
            centerY - radius >= 0 &&
            centerY + radius <= size.height
        ) {
            drawCircle(
                color = color.copy(alpha = 0.7f),
                radius = radius,
                center = Offset(centerX, centerY),
                style = Stroke(width = 2.dp.toPx()),
            )
        }
    }

    // Draw maximum stress point
    val pointRadius = 8.dp.toPx()
    if (centerX - pointRadius >= 0 &&
        centerX + pointRadius <= size.width &&
        centerY - pointRadius >= 0 &&
        centerY + pointRadius <= size.height
    ) {
        drawCircle(
            color = Color(0xFFF44336),
            radius = pointRadius,
            center = Offset(centerX, centerY),
        )
    }
}

private fun DrawScope.drawStressScale(maxStress: Double, chartWidth: Float, chartHeight: Float, margin: Float) {
    // Ensure minimum dimensions to prevent invalid rectangles
    if (chartWidth <= 0 || chartHeight <= 0) return

    val scaleWidth = 20.dp.toPx()
    val scaleHeight = maxOf(chartHeight * 0.8f, 40.dp.toPx()) // Minimum height
    val scaleX = maxOf(margin + chartWidth - scaleWidth - 10.dp.toPx(), margin)
    val scaleY = margin + chartHeight * 0.1f

    // Ensure scale fits within canvas bounds
    val maxScaleX = size.width - scaleWidth - margin
    val maxScaleY = size.height - scaleHeight - margin
    val finalScaleX = minOf(scaleX, maxScaleX)
    val finalScaleY = minOf(scaleY, maxScaleY)

    // Only draw if we have valid dimensions
    if (scaleWidth > 0 && scaleHeight > 0 && finalScaleX >= 0 && finalScaleY >= 0) {
        // Draw scale background
        drawRect(
            color = Color.White,
            topLeft = Offset(finalScaleX, finalScaleY),
            size = androidx.compose.ui.geometry.Size(scaleWidth, scaleHeight),
        )

        // Draw scale gradient
        val steps = 10
        val stepHeight = maxOf(scaleHeight / steps, 1.dp.toPx()) // Minimum step height

        for (i in 0 until steps) {
            val normalizedStress = i.toFloat() / (steps - 1)
            val color = when {
                normalizedStress < 0.3f -> Color(0xFF4CAF50)
                normalizedStress < 0.6f -> Color(0xFFFFEB3B)
                normalizedStress < 0.8f -> Color(0xFFFF9800)
                else -> Color(0xFFF44336)
            }

            val y = finalScaleY + i * stepHeight
            val rectHeight = minOf(stepHeight, finalScaleY + scaleHeight - y)

            // Only draw if rectangle is valid
            if (rectHeight > 0 && y >= finalScaleY && y < finalScaleY + scaleHeight) {
                drawRect(
                    color = color,
                    topLeft = Offset(finalScaleX, y),
                    size = androidx.compose.ui.geometry.Size(scaleWidth, rectHeight),
                )
            }
        }
    }
}

@Composable
private fun FEAAnalysisSummary(feaAnalysis: FEAAnalysisData) {
    Column(
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        // Key metrics
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
                    text = "Key Metrics",
                    style = MaterialTheme.typography.titleSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                ) {
                    FEAMetric(
                        label = "Max Stress",
                        value = String.format("%.1f", feaAnalysis.maxStress),
                        unit = "MPa",
                        color = when {
                            feaAnalysis.maxStress > 500 -> Color(0xFFF44336) // Red - High
                            feaAnalysis.maxStress > 200 -> Color(0xFFFF9800) // Orange - Medium
                            else -> Color(0xFF4CAF50) // Green - Good
                        },
                    )

                    FEAMetric(
                        label = "Fatigue Life",
                        value = when {
                            feaAnalysis.fatigueLife >= 1000000 -> String.format("%.1fM", feaAnalysis.fatigueLife / 1000000)
                            feaAnalysis.fatigueLife >= 1000 -> String.format("%.0fK", feaAnalysis.fatigueLife / 1000)
                            else -> String.format("%.0f", feaAnalysis.fatigueLife)
                        },
                        unit = "cycles",
                        color = when {
                            feaAnalysis.fatigueLife < 10000 -> Color(0xFFF44336) // Red - Low
                            feaAnalysis.fatigueLife < 100000 -> Color(0xFFFF9800) // Orange - Medium
                            else -> Color(0xFF4CAF50) // Green - Good
                        },
                    )

                    FEAMetric(
                        label = "Natural Freq",
                        value = String.format("%.0f", feaAnalysis.naturalFrequencies.firstOrNull() ?: 0.0),
                        unit = "Hz",
                        color = when {
                            (feaAnalysis.naturalFrequencies.firstOrNull() ?: 0.0) < 50 -> Color(0xFFF44336) // Red - Low
                            (feaAnalysis.naturalFrequencies.firstOrNull() ?: 0.0) > 200 -> Color(0xFF4CAF50) // Green - High
                            else -> Color(0xFFFF9800) // Orange - Medium
                        },
                    )
                }
            }
        }

        // Detailed component analysis
        ComponentAnalysisCard(feaAnalysis = feaAnalysis)

        // Natural frequencies
        NaturalFrequenciesCard(feaAnalysis = feaAnalysis)

        // Recommendations
        RecommendationsCard(feaAnalysis = feaAnalysis)
    }
}

@Composable
private fun FEAMetric(label: String, value: String, unit: String, color: Color) {
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
            color = color,
        )
    }
}

@Composable
private fun NaturalFrequenciesCard(feaAnalysis: FEAAnalysisData) {
    Card(
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.3f),
        ),
    ) {
        Column(
            modifier = Modifier.padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Text(
                text = "Natural Frequencies",
                style = MaterialTheme.typography.titleSmall,
                color = MaterialTheme.colorScheme.onPrimaryContainer,
            )

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceEvenly,
            ) {
                feaAnalysis.naturalFrequencies.take(3).forEachIndexed { index, frequency ->
                    Column(
                        horizontalAlignment = Alignment.CenterHorizontally,
                    ) {
                        Text(
                            text = "Mode ${index + 1}",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onPrimaryContainer,
                        )
                        Text(
                            text = "${frequency.toInt()} Hz",
                            style = MaterialTheme.typography.titleSmall,
                            fontWeight = FontWeight.Bold,
                            color = MaterialTheme.colorScheme.primary,
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun ComponentAnalysisCard(feaAnalysis: FEAAnalysisData) {
    Card(
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.secondaryContainer.copy(alpha = 0.3f),
        ),
    ) {
        Column(
            modifier = Modifier.padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Text(
                text = "Component Analysis",
                style = MaterialTheme.typography.titleSmall,
                color = MaterialTheme.colorScheme.onSecondaryContainer,
            )

            // Stress analysis breakdown
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
            ) {
                Column(
                    horizontalAlignment = Alignment.CenterHorizontally,
                ) {
                    Text(
                        text = "Stress Level",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSecondaryContainer,
                    )
                    Text(
                        text = when {
                            feaAnalysis.maxStress > 500 -> "HIGH"
                            feaAnalysis.maxStress > 200 -> "MEDIUM"
                            else -> "LOW"
                        },
                        style = MaterialTheme.typography.titleSmall,
                        fontWeight = FontWeight.Bold,
                        color = when {
                            feaAnalysis.maxStress > 500 -> Color(0xFFF44336)
                            feaAnalysis.maxStress > 200 -> Color(0xFFFF9800)
                            else -> Color(0xFF4CAF50)
                        },
                    )
                }

                Column(
                    horizontalAlignment = Alignment.CenterHorizontally,
                ) {
                    Text(
                        text = "Fatigue Risk",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSecondaryContainer,
                    )
                    Text(
                        text = when {
                            feaAnalysis.fatigueLife < 10000 -> "HIGH"
                            feaAnalysis.fatigueLife < 100000 -> "MEDIUM"
                            else -> "LOW"
                        },
                        style = MaterialTheme.typography.titleSmall,
                        fontWeight = FontWeight.Bold,
                        color = when {
                            feaAnalysis.fatigueLife < 10000 -> Color(0xFFF44336)
                            feaAnalysis.fatigueLife < 100000 -> Color(0xFFFF9800)
                            else -> Color(0xFF4CAF50)
                        },
                    )
                }

                Column(
                    horizontalAlignment = Alignment.CenterHorizontally,
                ) {
                    Text(
                        text = "Resonance Risk",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSecondaryContainer,
                    )
                    Text(
                        text = when {
                            (feaAnalysis.naturalFrequencies.firstOrNull() ?: 0.0) < 50 -> "HIGH"
                            (feaAnalysis.naturalFrequencies.firstOrNull() ?: 0.0) > 200 -> "LOW"
                            else -> "MEDIUM"
                        },
                        style = MaterialTheme.typography.titleSmall,
                        fontWeight = FontWeight.Bold,
                        color = when {
                            (feaAnalysis.naturalFrequencies.firstOrNull() ?: 0.0) < 50 -> Color(0xFFF44336)
                            (feaAnalysis.naturalFrequencies.firstOrNull() ?: 0.0) > 200 -> Color(0xFF4CAF50)
                            else -> Color(0xFFFF9800)
                        },
                    )
                }
            }

            // Overall assessment
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Icon(
                    imageVector = when {
                        feaAnalysis.maxStress > 500 || feaAnalysis.fatigueLife < 10000 -> Icons.Default.Warning
                        feaAnalysis.maxStress > 200 || feaAnalysis.fatigueLife < 100000 -> Icons.Default.Info
                        else -> Icons.Default.CheckCircle
                    },
                    contentDescription = null,
                    modifier = Modifier.size(20.dp),
                    tint = when {
                        feaAnalysis.maxStress > 500 || feaAnalysis.fatigueLife < 10000 -> Color(0xFFF44336)
                        feaAnalysis.maxStress > 200 || feaAnalysis.fatigueLife < 100000 -> Color(0xFFFF9800)
                        else -> Color(0xFF4CAF50)
                    },
                )
                Text(
                    text = when {
                        feaAnalysis.maxStress > 500 || feaAnalysis.fatigueLife < 10000 -> "Design requires immediate attention"
                        feaAnalysis.maxStress > 200 || feaAnalysis.fatigueLife < 100000 -> "Design acceptable with monitoring"
                        else -> "Design meets all safety criteria"
                    },
                    style = MaterialTheme.typography.bodyMedium,
                    fontWeight = FontWeight.Medium,
                    color = MaterialTheme.colorScheme.onSecondaryContainer,
                )
            }
        }
    }
}

@Composable
private fun RecommendationsCard(feaAnalysis: FEAAnalysisData) {
    Card(
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.tertiaryContainer.copy(alpha = 0.3f),
        ),
    ) {
        Column(
            modifier = Modifier.padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Text(
                text = "Engineering Recommendations",
                style = MaterialTheme.typography.titleSmall,
                color = MaterialTheme.colorScheme.onTertiaryContainer,
            )

            if (feaAnalysis.recommendations.isNotEmpty()) {
                feaAnalysis.recommendations.take(5).forEach { recommendation ->
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                    ) {
                        Icon(
                            imageVector = Icons.Default.Lightbulb,
                            contentDescription = null,
                            modifier = Modifier.size(16.dp),
                            tint = MaterialTheme.colorScheme.tertiary,
                        )
                        Text(
                            text = recommendation,
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onTertiaryContainer,
                        )
                    }
                }
            } else {
                Text(
                    text = "No specific recommendations available",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onTertiaryContainer.copy(alpha = 0.7f),
                )
            }
        }
    }
}
