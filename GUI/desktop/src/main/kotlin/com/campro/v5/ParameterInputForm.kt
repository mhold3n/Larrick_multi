@file:OptIn(
    androidx.compose.material3.ExperimentalMaterial3Api::class,
    androidx.compose.foundation.ExperimentalFoundationApi::class,
)

package com.campro.v5

import androidx.compose.foundation.layout.*
import androidx.compose.material.DropdownMenu
import androidx.compose.material.DropdownMenuItem
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.campro.v5.layout.LayoutManager
import com.campro.v5.layout.rememberLayoutManager

/**
 * Parameter input form for CamPro v5.
 * This component allows users to input all parameters needed for the simulation.
 * Parameters are organized into categories using tabs.
 */
@Composable
fun ParameterInputForm(
    testingMode: Boolean = false,
    onParametersChanged: (Map<String, String>) -> Unit = {},
    layoutManager: LayoutManager = rememberLayoutManager(),
) {
    var selectedTabIndex by remember { mutableStateOf(0) }
    val parameterValues = remember { mutableStateMapOf<String, String>() }
    var errorMessage by remember { mutableStateOf<String?>(null) }

    // Define parameter categories
    val categories =
        listOf(
            "Cam Geometry",
            "Piston & Linkage",
            "Combustion",
            "Valve Timing",
            "Vibration",
            "Materials",
            "Analysis",
        )

    // Define parameters for each category
    val categoryParameters =
        mapOf(
            "Cam Geometry" to
                listOf(
                    Parameter("Piston Diameter", "70.0", "mm", "float"),
                    Parameter("Stroke", "20", "mm", "float"),
                    Parameter("Chamber CC", "5.0", "cc", "float"),
                    Parameter("TDC Angle", "90", "deg", "float"),
                    Parameter("BDC Dwell", "8", "deg", "float"),
                    Parameter("TDC Dwell", "12", "deg", "float"),
                    Parameter("Enable Smoothing", "1", "(0=Off, 1=On)", "int"),
                    Parameter("Cam Timestep", "1.0", "deg", "float"),
                    Parameter("Rod Length", "40", "mm", "float"),
                    Parameter("TDC Offset", "40.0", "mm", "float"),
                    Parameter("Cycle Ratio", "2", "", "float"),
                    Parameter("Envelope Wall Thickness", "10.0", "mm", "float"),
                ),
            "Piston & Linkage" to
                listOf(
                    Parameter("Piston Mass", "0.2", "kg", "float"),
                ),
            "Combustion" to
                listOf(
                    Parameter("Manifold Pressure", "101325.0", "Pa", "float"),
                    Parameter("Ignition Timing BTDC", "15.0", "deg", "float"),
                    Parameter("Ignition Duration", "1.0", "ms", "float"),
                    Parameter("Equivalence Ratio (phi)", "1.0", "", "float"),
                    Parameter("Gamma (Air)", "1.4", "", "float"),
                    Parameter("Initial Temp BDC", "300.0", "K", "float"),
                    Parameter("Fuel Type", "Diesel", "", "string", listOf("Diesel", "Gasoline", "Natural Gas", "Hydrogen")),
                ),
            "Valve Timing" to
                listOf(
                    Parameter("IVO deg ABD", "0.0", "deg", "float"),
                    Parameter("IVC deg ABD", "15.0", "deg", "float"),
                    Parameter("EVO deg BBD", "15.0", "deg", "float"),
                    Parameter("EVC deg ABD", "0.0", "deg", "float"),
                ),
            "Vibration" to
                listOf(
                    Parameter("Assembly RPM", "1000", "RPM", "float"),
                    Parameter("Mount Mass", "5.0", "kg", "float"),
                    Parameter("Mount Stiffness X", "1e6", "N/m", "float"),
                    Parameter("Mount Stiffness Y", "1e6", "N/m", "float"),
                    Parameter("Mount Damping Ratio X", "0.05", "", "float"),
                    Parameter("Mount Damping Ratio Y", "0.05", "", "float"),
                ),
            "Materials" to
                listOf(
                    Parameter("Cam Material", "Steel", "", "string", listOf("Steel", "Aluminum", "Titanium", "Brass")),
                    Parameter("Rod Material", "Steel", "", "string", listOf("Steel", "Aluminum", "Titanium", "Brass")),
                    Parameter("Piston Material", "Aluminum", "", "string", listOf("Steel", "Aluminum", "Titanium", "Brass")),
                    Parameter("Envelope Material", "Steel", "", "string", listOf("Steel", "Aluminum", "Titanium", "Brass")),
                    Parameter("Rod Mass (Override)", "0.1", "kg", "float"),
                    Parameter("Piston Mass (Override)", "0.2", "kg", "float"),
                    Parameter("Cam Thickness (Estimate)", "10.0", "mm", "float"),
                ),
            "Analysis" to
                listOf(
                    Parameter(
                        "Profile Solver",
                        "Piecewise",
                        "",
                        "string",
                        if (com.campro.v5.config.FeatureFlags.Collocation
                                .isUIVisible()
                        ) {
                            listOf("Piecewise", "Collocation")
                        } else {
                            listOf("Piecewise")
                        },
                    ),
                    Parameter("RK Analysis Revs", "5", "revs", "int"),
                    Parameter("Show Test Plot", "0", "(0=Off, 1=On)", "int"),
                    Parameter("Use Cantera", "0", "(0=No, 1=Yes)", "int"),
                ),
        )

    // Initialize parameter values if empty
    if (parameterValues.isEmpty()) {
        categoryParameters.values.flatten().forEach { parameter ->
            parameterValues[parameter.name] = parameter.defaultValue
        }
    }

    Card(
        modifier = Modifier.fillMaxWidth(),
    ) {
        Column(
            modifier = Modifier.padding(layoutManager.getAppropriateSpacing()),
        ) {
            Text(
                "Parameter Input Form",
                style = MaterialTheme.typography.titleLarge,
            )

            Spacer(modifier = Modifier.height(16.dp))

            // Tab row for parameter categories
            TabRow(selectedTabIndex = selectedTabIndex) {
                categories.forEachIndexed { index, category ->
                    Tab(
                        selected = selectedTabIndex == index,
                        onClick = { selectedTabIndex = index },
                        text = {
                            Text(
                                if (layoutManager.shouldUseCompactMode()) {
                                    category.take(8) + if (category.length > 8) "..." else ""
                                } else {
                                    category
                                },
                            )
                        },
                    )
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            // Parameters for the selected category - NOW WITH SCROLLING
            val selectedCategory = categories[selectedTabIndex]
            val parameters = categoryParameters[selectedCategory] ?: emptyList()

            Column(
                modifier =
                Modifier
                    .fillMaxWidth()
                    .fillMaxHeight(),
                // Use fillMaxHeight instead of heightIn
                verticalArrangement = Arrangement.spacedBy(4.dp),
            ) {
                parameters.forEach { parameter ->
                    ParameterInputField(
                        parameter = parameter,
                        value = parameterValues[parameter.name] ?: parameter.defaultValue,
                        onValueChange = { newValue ->
                            parameterValues[parameter.name] = newValue
                            errorMessage = validateInput(parameter, newValue)
                            onParametersChanged(parameterValues)

                            if (testingMode) {
                                println("EVENT:{\"type\":\"input_changed\",\"component\":\"${parameter.name}\",\"value\":\"$newValue\"}")
                            }
                        },
                        isError = errorMessage != null,
                        testingMode = testingMode,
                        isCompact = layoutManager.shouldUseCompactMode(),
                    )
                }
            }

            // Error message
            if (errorMessage != null) {
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    errorMessage!!,
                    color = MaterialTheme.colorScheme.error,
                    style = MaterialTheme.typography.bodySmall,
                )
            }

            Spacer(modifier = Modifier.height(16.dp))

            // Buttons for presets and import/export
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
            ) {
                Button(
                    onClick = {
                        // Load preset (to be implemented)
                        if (testingMode) {
                            println("EVENT:{\"type\":\"button_clicked\",\"component\":\"LoadPresetButton\"}")
                        }
                    },
                ) {
                    Text("Load Preset")
                }

                Button(
                    onClick = {
                        // Save preset (to be implemented)
                        if (testingMode) {
                            println("EVENT:{\"type\":\"button_clicked\",\"component\":\"SavePresetButton\"}")
                        }
                    },
                ) {
                    Text("Save Preset")
                }

                Button(
                    onClick = {
                        // Import parameters (to be implemented)
                        if (testingMode) {
                            println("EVENT:{\"type\":\"button_clicked\",\"component\":\"ImportButton\"}")
                        }
                    },
                ) {
                    Text("Import")
                }

                Button(
                    onClick = {
                        // Export parameters (to be implemented)
                        if (testingMode) {
                            println("EVENT:{\"type\":\"button_clicked\",\"component\":\"ExportButton\"}")
                        }
                    },
                ) {
                    Text("Export")
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            // Generate Animation Button
            Button(
                onClick = {
                    // Validate all inputs
                    val errors = validateAllInputs(categoryParameters.values.flatten(), parameterValues)
                    if (errors.isEmpty()) {
                        // Set animation started flag and notify parent
                        parameterValues["animationStarted"] = "true"
                        onParametersChanged(parameterValues)

                        // Report event if in testing mode
                        if (testingMode) {
                            println("EVENT:{\"type\":\"button_clicked\",\"component\":\"GenerateAnimationButton\"}")
                            println("EVENT:{\"type\":\"animation_started\",\"parameters\":$parameterValues}")
                        }
                    } else {
                        errorMessage = errors.first()
                    }
                },
                modifier = Modifier.align(Alignment.End),
                enabled = errorMessage == null,
            ) {
                Text("Generate Animation")
            }
        }
    }
}

@Composable
private fun ParameterInputField(
    parameter: Parameter,
    value: String,
    onValueChange: (String) -> Unit,
    isError: Boolean,
    testingMode: Boolean,
    isCompact: Boolean,
) {
    when (parameter.type) {
        "string" -> {
            if (parameter.options.isNotEmpty()) {
                // Dropdown implementation
                var expanded by remember { mutableStateOf(false) }

                Column(
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    if (!isCompact) {
                        Text(
                            "${parameter.name} ${parameter.units}",
                            style = MaterialTheme.typography.bodyMedium,
                        )
                    }

                    Box(modifier = Modifier.fillMaxWidth()) {
                        OutlinedButton(
                            onClick = { expanded = true },
                            modifier = Modifier.fillMaxWidth(),
                        ) {
                            Text(
                                if (isCompact) "${parameter.name}: $value" else value,
                                maxLines = 1,
                            )
                        }

                        DropdownMenu(
                            expanded = expanded,
                            onDismissRequest = { expanded = false },
                            modifier = Modifier.fillMaxWidth(0.9f),
                        ) {
                            parameter.options.forEach { option ->
                                DropdownMenuItem(
                                    onClick = {
                                        onValueChange(option)
                                        expanded = false
                                    },
                                ) {
                                    Text(option)
                                }
                            }
                        }
                    }
                }
            } else {
                // Text field for string parameters
                OutlinedTextField(
                    value = value,
                    onValueChange = onValueChange,
                    label = {
                        Text(
                            if (isCompact) parameter.name else "${parameter.name} ${parameter.units}",
                            maxLines = 1,
                        )
                    },
                    modifier = Modifier.fillMaxWidth(),
                    isError = isError,
                    singleLine = true,
                )
            }
        }
        else -> {
            // Text field for numeric parameters
            OutlinedTextField(
                value = value,
                onValueChange = onValueChange,
                label = {
                    Text(
                        if (isCompact) parameter.name else "${parameter.name} ${parameter.units}",
                        maxLines = 1,
                    )
                },
                modifier = Modifier.fillMaxWidth(),
                isError = isError,
                singleLine = true,
            )
        }
    }
}

/**
 * Data class representing a parameter.
 */
data class Parameter(
    val name: String,
    val defaultValue: String,
    val units: String,
    val type: String,
    val options: List<String> = emptyList(),
)

/**
 * Validate a single parameter input.
 */
fun validateInput(parameter: Parameter, value: String): String? {
    return try {
        when (parameter.type) {
            "float" -> {
                val floatValue = value.toDouble()
                if (floatValue < 0 && !parameter.name.contains("Offset")) {
                    return "${parameter.name} must be non-negative"
                }
            }
            "int" -> {
                val intValue = value.toInt()
                if (intValue < 0 && !parameter.name.contains("Offset")) {
                    return "${parameter.name} must be non-negative"
                }
            }
        }
        null
    } catch (e: NumberFormatException) {
        "${parameter.name} must be a valid ${parameter.type}"
    }
}

/**
 * Validate all parameter inputs.
 */
fun validateAllInputs(parameters: List<Parameter>, values: Map<String, String>): List<String> {
    val errors = mutableListOf<String>()

    parameters.forEach { parameter ->
        val value = values[parameter.name] ?: parameter.defaultValue
        val error = validateInput(parameter, value)
        if (error != null) {
            errors.add(error)
        }
    }

    return errors
}
