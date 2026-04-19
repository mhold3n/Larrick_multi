package com.campro.v5.ui

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.campro.v5.debug.DebugOutlinedButton
import com.campro.v5.models.OptimizationParameters
import com.campro.v5.ui.SimpleLayoutManager as LayoutManager
// Using local ParameterField to avoid cross-file dependency during build

/**
 * Comprehensive parameter input form for optimization parameters.
 *
 * This form provides organized input fields for all optimization parameters,
 * grouped by category with validation and preset support.
 */
@Composable
fun OptimizationParameterForm(
    parameters: OptimizationParameters,
    onParametersChanged: (OptimizationParameters) -> Unit,
    layoutManager: LayoutManager = LayoutManager(),
    modifier: Modifier = Modifier,
) {
    var selectedCategory by remember { mutableStateOf(0) }
    var showValidationErrors by remember { mutableStateOf(false) }

    val categories = listOf(
        "Basic",
        "Motion Law",
        "Gear Design",
        "Advanced",
    )

    val validationErrors = parameters.validate()

    Card(
        modifier = modifier.fillMaxSize(),
        elevation = CardDefaults.cardElevation(defaultElevation = 2.dp),
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(16.dp)
                .verticalScroll(rememberScrollState()),
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            // Header with validation status
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text(
                    text = "Optimization Parameters",
                    style = MaterialTheme.typography.titleLarge,
                    color = MaterialTheme.colorScheme.onSurface,
                )

                if (validationErrors.isNotEmpty()) {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(4.dp),
                    ) {
                        Icon(
                            imageVector = Icons.Default.Warning,
                            contentDescription = "Validation errors",
                            tint = MaterialTheme.colorScheme.error,
                            modifier = Modifier.size(20.dp),
                        )
                        Text(
                            text = "${validationErrors.size} errors",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.error,
                        )
                    }
                }
            }

            // Category tabs
            TabRow(selectedTabIndex = selectedCategory) {
                categories.forEachIndexed { index, category ->
                    Tab(
                        selected = selectedCategory == index,
                        onClick = { selectedCategory = index },
                        text = {
                            Text(
                                text = category,
                            )
                        },
                    )
                }
            }

            // Parameter content
            Column(
                modifier = Modifier
                    .weight(1f)
                    .verticalScroll(rememberScrollState()),
                verticalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                when (selectedCategory) {
                    0 -> BasicParametersSection(
                        parameters = parameters,
                        onParametersChanged = onParametersChanged,
                        layoutManager = layoutManager,
                    )
                    1 -> MotionLawParametersSection(
                        parameters = parameters,
                        onParametersChanged = onParametersChanged,
                        layoutManager = layoutManager,
                    )
                    2 -> GearDesignParametersSection(
                        parameters = parameters,
                        onParametersChanged = onParametersChanged,
                        layoutManager = layoutManager,
                    )
                    3 -> AdvancedParametersSection(
                        parameters = parameters,
                        onParametersChanged = onParametersChanged,
                        layoutManager = layoutManager,
                    )
                }
            }

            // Validation errors display
            if (showValidationErrors && validationErrors.isNotEmpty()) {
                Card(
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.errorContainer,
                    ),
                ) {
                    Column(
                        modifier = Modifier.padding(12.dp),
                        verticalArrangement = Arrangement.spacedBy(4.dp),
                    ) {
                        Text(
                            text = "Validation Errors",
                            style = MaterialTheme.typography.titleSmall,
                            color = MaterialTheme.colorScheme.onErrorContainer,
                        )
                        validationErrors.forEach { error ->
                            Text(
                                text = "• $error",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onErrorContainer,
                            )
                        }
                    }
                }
            }

            // Action buttons
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                DebugOutlinedButton(
                    buttonId = "toggle-validation-errors",
                    onClick = { showValidationErrors = !showValidationErrors },
                    modifier = Modifier.weight(1f),
                ) {
                    Icon(
                        imageVector = if (showValidationErrors) Icons.Default.VisibilityOff else Icons.Default.Visibility,
                        contentDescription = null,
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(if (showValidationErrors) "Hide Errors" else "Show Errors")
                }

                DebugOutlinedButton(
                    buttonId = "reset-defaults",
                    onClick = { onParametersChanged(OptimizationParameters.createDefault()) },
                    modifier = Modifier.weight(1f),
                ) {
                    Icon(
                        imageVector = Icons.Default.Restore,
                        contentDescription = null,
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("Reset to Defaults")
                }

                DebugOutlinedButton(
                    buttonId = "quick-test",
                    onClick = { onParametersChanged(OptimizationParameters.createQuickTest()) },
                    modifier = Modifier.weight(1f),
                ) {
                    Icon(
                        imageVector = Icons.Default.Speed,
                        contentDescription = null,
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("Quick Test")
                }
            }
        }
    }
}

@Composable
private fun BasicParametersSection(
    parameters: OptimizationParameters,
    onParametersChanged: (OptimizationParameters) -> Unit,
    layoutManager: LayoutManager,
) {
    Column(
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text(
            text = "Basic Parameters",
            style = MaterialTheme.typography.titleMedium,
            color = MaterialTheme.colorScheme.onSurface,
        )

        // Row 1: Sampling and stroke
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            ParameterField(
                label = "Sampling Step",
                value = parameters.samplingStepDeg,
                unit = "°",
                onValueChange = { onParametersChanged(parameters.copy(samplingStepDeg = it)) },
                modifier = Modifier.weight(1f),
            )

            ParameterField(
                label = "Stroke Length",
                value = parameters.strokeLengthMm,
                unit = "mm",
                onValueChange = { onParametersChanged(parameters.copy(strokeLengthMm = it)) },
                modifier = Modifier.weight(1f),
            )
        }

        ParameterField(
            label = "Piston Diameter",
            value = parameters.pistonDiameterMm,
            unit = "mm",
            onValueChange = { onParametersChanged(parameters.copy(pistonDiameterMm = it)) },
            modifier = Modifier.fillMaxWidth(),
        )

        // Row 2: Gear ratio and RPM
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            ParameterField(
                label = "Gear Ratio",
                value = parameters.gearRatio,
                unit = "",
                onValueChange = { onParametersChanged(parameters.copy(gearRatio = it)) },
                modifier = Modifier.weight(1f),
            )

            ParameterField(
                label = "RPM",
                value = parameters.rpm,
                unit = "",
                onValueChange = { onParametersChanged(parameters.copy(rpm = it)) },
                modifier = Modifier.weight(1f),
            )
        }

        // Row 3: Rod length and compression duration
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            ParameterField(
                label = "Rod Length",
                value = parameters.rodLength,
                unit = "mm",
                onValueChange = { onParametersChanged(parameters.copy(rodLength = it)) },
                modifier = Modifier.weight(1f),
            )

            ParameterField(
                label = "Compression Duration",
                value = parameters.compressionDurationPercent,
                unit = "%",
                onValueChange = { onParametersChanged(parameters.copy(compressionDurationPercent = it)) },
                modifier = Modifier.weight(1f),
            )
        }
    }
}

@Composable
private fun MotionLawParametersSection(
    parameters: OptimizationParameters,
    onParametersChanged: (OptimizationParameters) -> Unit,
    layoutManager: LayoutManager,
) {
    Column(
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text(
            text = "Motion Law Parameters",
            style = MaterialTheme.typography.titleMedium,
            color = MaterialTheme.colorScheme.onSurface,
        )

        // TDC parameters
        Text(
            text = "TDC (Top Dead Center)",
            style = MaterialTheme.typography.titleSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            ParameterField(
                label = "Ramp Before",
                value = parameters.rampBeforeTdcDeg,
                unit = "°",
                onValueChange = { onParametersChanged(parameters.copy(rampBeforeTdcDeg = it)) },
                modifier = Modifier.weight(1f),
            )

            ParameterField(
                label = "Dwell",
                value = parameters.dwellTdcDeg,
                unit = "°",
                onValueChange = { onParametersChanged(parameters.copy(dwellTdcDeg = it)) },
                modifier = Modifier.weight(1f),
            )
        }

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            ParameterField(
                label = "Ramp After",
                value = parameters.rampAfterTdcDeg,
                unit = "°",
                onValueChange = { onParametersChanged(parameters.copy(rampAfterTdcDeg = it)) },
                modifier = Modifier.weight(1f),
            )

            ParameterField(
                label = "Constant Velocity",
                value = parameters.constantVelocityTdcDeg,
                unit = "°",
                onValueChange = { onParametersChanged(parameters.copy(constantVelocityTdcDeg = it)) },
                modifier = Modifier.weight(1f),
            )
        }

        // BDC parameters
        Text(
            text = "BDC (Bottom Dead Center)",
            style = MaterialTheme.typography.titleSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            ParameterField(
                label = "Ramp Before",
                value = parameters.rampBeforeBdcDeg,
                unit = "°",
                onValueChange = { onParametersChanged(parameters.copy(rampBeforeBdcDeg = it)) },
                modifier = Modifier.weight(1f),
            )

            ParameterField(
                label = "Dwell",
                value = parameters.dwellBdcDeg,
                unit = "°",
                onValueChange = { onParametersChanged(parameters.copy(dwellBdcDeg = it)) },
                modifier = Modifier.weight(1f),
            )
        }

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            ParameterField(
                label = "Ramp After",
                value = parameters.rampAfterBdcDeg,
                unit = "°",
                onValueChange = { onParametersChanged(parameters.copy(rampAfterBdcDeg = it)) },
                modifier = Modifier.weight(1f),
            )

            ParameterField(
                label = "Constant Velocity",
                value = parameters.constantVelocityBdcDeg,
                unit = "°",
                onValueChange = { onParametersChanged(parameters.copy(constantVelocityBdcDeg = it)) },
                modifier = Modifier.weight(1f),
            )
        }
    }
}

@Composable
private fun GearDesignParametersSection(
    parameters: OptimizationParameters,
    onParametersChanged: (OptimizationParameters) -> Unit,
    layoutManager: LayoutManager,
) {
    Column(
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text(
            text = "Gear Design Parameters",
            style = MaterialTheme.typography.titleMedium,
            color = MaterialTheme.colorScheme.onSurface,
        )

        // Fixed planetary configuration info
        Text(
            text = "Planetary Configuration: 2 planets, 180° offset (fixed)",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )

        // Physical dimensions
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            ParameterField(
                label = "Journal Radius",
                value = parameters.journalRadius,
                unit = "mm",
                onValueChange = { onParametersChanged(parameters.copy(journalRadius = it)) },
                modifier = Modifier.weight(1f),
            )

            ParameterField(
                label = "Ring Thickness",
                value = parameters.ringThickness,
                unit = "mm",
                onValueChange = { onParametersChanged(parameters.copy(ringThickness = it)) },
                modifier = Modifier.weight(1f),
            )
        }

        // Interference buffer (carrier offset is fixed at 180° for 2 planets)
        ParameterField(
            label = "Interference Buffer",
            value = parameters.interferenceBuffer,
            unit = "mm",
            onValueChange = { onParametersChanged(parameters.copy(interferenceBuffer = it)) },
            modifier = Modifier.fillMaxWidth(),
        )

        // Ring rotation
        ParameterField(
            label = "Ring Rotation",
            value = parameters.ringRotationDeg,
            unit = "°",
            onValueChange = { onParametersChanged(parameters.copy(ringRotationDeg = it)) },
            modifier = Modifier.fillMaxWidth(),
        )
    }
}

@Composable
private fun AdvancedParametersSection(
    parameters: OptimizationParameters,
    onParametersChanged: (OptimizationParameters) -> Unit,
    layoutManager: LayoutManager,
) {
    Column(
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text(
            text = "Advanced Parameters",
            style = MaterialTheme.typography.titleMedium,
            color = MaterialTheme.colorScheme.onSurface,
        )

        // Planet radius factors
        Text(
            text = "Planet Radius Factors",
            style = MaterialTheme.typography.titleSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            ParameterField(
                label = "Base Factor",
                value = parameters.planetRadiusBaseFactor,
                unit = "",
                onValueChange = { onParametersChanged(parameters.copy(planetRadiusBaseFactor = it)) },
                modifier = Modifier.weight(1f),
            )

            ParameterField(
                label = "Variation Factor",
                value = parameters.planetRadiusVariationFactor,
                unit = "",
                onValueChange = { onParametersChanged(parameters.copy(planetRadiusVariationFactor = it)) },
                modifier = Modifier.weight(1f),
            )
        }

        // Sun radius factors
        Text(
            text = "Sun Radius Factors",
            style = MaterialTheme.typography.titleSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            ParameterField(
                label = "Base Factor",
                value = parameters.sunRadiusBaseFactor,
                unit = "",
                onValueChange = { onParametersChanged(parameters.copy(sunRadiusBaseFactor = it)) },
                modifier = Modifier.weight(1f),
            )

            ParameterField(
                label = "Variation Factor",
                value = parameters.sunRadiusVariationFactor,
                unit = "",
                onValueChange = { onParametersChanged(parameters.copy(sunRadiusVariationFactor = it)) },
                modifier = Modifier.weight(1f),
            )
        }

        // Safety and adjustment factors
        Text(
            text = "Safety & Adjustment Factors",
            style = MaterialTheme.typography.titleSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            ParameterField(
                label = "Stroke Achievable",
                value = parameters.strokeAchievableFactor,
                unit = "",
                onValueChange = { onParametersChanged(parameters.copy(strokeAchievableFactor = it)) },
                modifier = Modifier.weight(1f),
            )

            ParameterField(
                label = "Clearance Safety",
                value = parameters.clearanceSafetyMargin,
                unit = "",
                onValueChange = { onParametersChanged(parameters.copy(clearanceSafetyMargin = it)) },
                modifier = Modifier.weight(1f),
            )
        }

        ParameterField(
            label = "Adjustment Split Factor",
            value = parameters.adjustmentSplitFactor,
            unit = "",
            onValueChange = { onParametersChanged(parameters.copy(adjustmentSplitFactor = it)) },
            modifier = Modifier.fillMaxWidth(),
        )
    }
}

@Composable
private fun ParameterField(label: String, value: Double, unit: String, onValueChange: (Double) -> Unit, modifier: Modifier = Modifier) {
    OutlinedTextField(
        value = value.toString(),
        onValueChange = {
            val newValue = it.toDoubleOrNull() ?: value
            onValueChange(newValue)
        },
        label = {
            Text("$label $unit".trim())
        },
        modifier = modifier,
        singleLine = true,
        isError = false,
    )
}
