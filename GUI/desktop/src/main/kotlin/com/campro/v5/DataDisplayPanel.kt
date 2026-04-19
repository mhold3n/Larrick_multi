@file:OptIn(androidx.compose.material3.ExperimentalMaterial3Api::class)

package com.campro.v5

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.material.icons.Icons
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.campro.v5.debug.DebugButton
import kotlin.math.*

/**
 * A panel that displays tabular data and statistics related to the cycloidal animation.
 *
 * @param parameters Map of parameter names to values
 * @param testingMode Whether the widget is in testing mode
 */
@Composable
fun DataDisplayPanel(parameters: Map<String, String>, testingMode: Boolean = false) {
    // Tab state
    var selectedTabIndex by remember { mutableStateOf(0) }

    // Define tab types
    val tabTypes =
        listOf(
            "Summary",
            "Kinematics",
            "Forces",
            "Efficiency",
        )

    Column(
        modifier = Modifier.fillMaxSize().padding(16.dp),
    ) {
        // Tab row
        TabRow(selectedTabIndex = selectedTabIndex) {
            tabTypes.forEachIndexed { index, tabType ->
                Tab(
                    selected = selectedTabIndex == index,
                    onClick = {
                        selectedTabIndex = index
                        if (testingMode) {
                            println("EVENT:{\"type\":\"tab_selected\",\"component\":\"DataDisplayTab\",\"value\":\"$tabType\"}")
                        }
                    },
                    text = { Text(tabType) },
                )
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        // Export buttons
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.End,
        ) {
            DebugButton(
                buttonId = "data-refresh",
                onClick = { /* TODO: Refresh data */ },
            ) {
                Icon(Icons.Default.Refresh, contentDescription = null)
                Spacer(modifier = Modifier.width(8.dp))
                Text("Refresh Data")
            }

            DebugButton(
                buttonId = "data-export",
                onClick = { /* TODO: Export data */ },
            ) {
                Icon(Icons.Default.Download, contentDescription = null)
                Spacer(modifier = Modifier.width(8.dp))
                Text("Export")
            }
        }

        Spacer(modifier = Modifier.height(16.dp))

        // Content based on selected tab
        when (selectedTabIndex) {
            0 -> SummaryTab(parameters)
            1 -> KinematicsTab(parameters)
            2 -> ForcesTab(parameters)
            3 -> EfficiencyTab(parameters)
        }
    }
}

@Composable
fun SummaryTab(parameters: Map<String, String>) {
    // Extract key parameters
    val pistonDiameter = parameters["Piston Diameter"]?.toFloatOrNull() ?: 70f
    val stroke = parameters["Stroke"]?.toFloatOrNull() ?: 20f
    val rodLength = parameters["Rod Length"]?.toFloatOrNull() ?: 40f
    val tdcOffset = parameters["TDC Offset"]?.toFloatOrNull() ?: 40f
    val cycleRatio = parameters["Cycle Ratio"]?.toFloatOrNull() ?: 2f

    // Calculate some derived values
    val displacement = (PI * pistonDiameter * pistonDiameter * stroke / 4 / 1000).toFloat() // cc
    val compressionRatio = (PI * pistonDiameter * pistonDiameter * (stroke + 2 * tdcOffset) / 4 / 1000 / displacement).toFloat()
    val maxVelocity = (PI * stroke * cycleRatio / 30).toFloat() // mm/s at 1 RPM

    // Create summary data
    val summaryData =
        listOf(
            Pair("Displacement", String.format("%.2f cc", displacement)),
            Pair("Compression Ratio", String.format("%.2f:1", compressionRatio)),
            Pair("Max Piston Velocity", String.format("%.2f mm/s at 1 RPM", maxVelocity)),
            Pair("Rod/Stroke Ratio", String.format("%.2f", rodLength / stroke)),
            Pair("Cycle Type", if (cycleRatio % 1 == 0f) "Integer" else "Fractional"),
            Pair("Cam Lobes", if (cycleRatio % 1 == 0f) cycleRatio.toInt().toString() else "Variable"),
        )

    // Display summary data in a card
    Card(
        modifier = Modifier.fillMaxWidth(),
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
        ) {
            Text(
                "Key Metrics",
                style = MaterialTheme.typography.titleMedium,
            )

            Spacer(modifier = Modifier.height(8.dp))

            // Display each metric
            summaryData.forEach { (label, value) ->
                Row(
                    modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp),
                    horizontalArrangement = Arrangement.SpaceBetween,
                ) {
                    Text(
                        label,
                        style = MaterialTheme.typography.bodyMedium,
                    )
                    Text(
                        value,
                        style = MaterialTheme.typography.bodyMedium,
                    )
                }
                Divider()
            }
        }
    }
}

@Composable
fun KinematicsTab(parameters: Map<String, String>) {
    // Extract key parameters
    val stroke = parameters["Stroke"]?.toFloatOrNull() ?: 20f
    val cycleRatio = parameters["Cycle Ratio"]?.toFloatOrNull() ?: 2f

    // Generate sample kinematics data
    val kinematicsData =
        (0..360 step 30).map { angle ->
            val angleRad = angle * PI.toFloat() / 180f
            val displacement = sin(angleRad * cycleRatio) * (stroke / 2)
            val velocity = cos(angleRad * cycleRatio) * (stroke / 2) * cycleRatio
            val acceleration = -sin(angleRad * cycleRatio) * (stroke / 2) * cycleRatio * cycleRatio

            KinematicsData(
                angle = angle,
                displacement = displacement,
                velocity = velocity,
                acceleration = acceleration,
            )
        }

    // Display kinematics data in a table
    Column {
        // Table header
        Row(
            modifier = Modifier.fillMaxWidth().background(MaterialTheme.colorScheme.surfaceVariant).padding(8.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
        ) {
            Text(
                "Angle (°)",
                style = MaterialTheme.typography.titleSmall,
                modifier = Modifier.weight(1f),
            )
            Text(
                "Displacement (mm)",
                style = MaterialTheme.typography.titleSmall,
                modifier = Modifier.weight(1f),
            )
            Text(
                "Velocity (mm/rad)",
                style = MaterialTheme.typography.titleSmall,
                modifier = Modifier.weight(1f),
            )
            Text(
                "Acceleration (mm/rad²)",
                style = MaterialTheme.typography.titleSmall,
                modifier = Modifier.weight(1f),
            )
        }

        // Table rows
        Column {
            kinematicsData.forEach { data ->
                Row(
                    modifier = Modifier.fillMaxWidth().padding(8.dp),
                    horizontalArrangement = Arrangement.SpaceBetween,
                ) {
                    Text(
                        data.angle.toString(),
                        style = MaterialTheme.typography.bodyMedium,
                        modifier = Modifier.weight(1f),
                    )
                    Text(
                        String.format("%.2f", data.displacement),
                        style = MaterialTheme.typography.bodyMedium,
                        modifier = Modifier.weight(1f),
                    )
                    Text(
                        String.format("%.2f", data.velocity),
                        style = MaterialTheme.typography.bodyMedium,
                        modifier = Modifier.weight(1f),
                    )
                    Text(
                        String.format("%.2f", data.acceleration),
                        style = MaterialTheme.typography.bodyMedium,
                        modifier = Modifier.weight(1f),
                    )
                }
                Divider()
            }
        }
    }
}

@Composable
fun ForcesTab(parameters: Map<String, String>) {
    // Extract key parameters
    val pistonDiameter = parameters["Piston Diameter"]?.toFloatOrNull() ?: 70f
    val stroke = parameters["Stroke"]?.toFloatOrNull() ?: 20f

    // Generate sample force data
    val forceData =
        (0..360 step 30).map { angle ->
            val angleRad = angle * PI.toFloat() / 180f
            val pistonArea = PI.toFloat() * pistonDiameter * pistonDiameter / 4

            // Simulate combustion pressure (simplified model)
            val pressure =
                if (angle in 0..180) {
                    101325f + 900000f * sin(angleRad) // Pa
                } else {
                    101325f // Atmospheric pressure
                }

            val pistonForce = pressure * pistonArea / 1000 // N
            val rodForce = pistonForce * cos(angleRad) // N (simplified)
            val sideForce = pistonForce * sin(angleRad) // N (simplified)

            ForceData(
                angle = angle,
                pressure = pressure / 1000, // kPa
                pistonForce = pistonForce,
                rodForce = rodForce,
                sideForce = sideForce,
            )
        }

    // Display force data in a table
    Column {
        // Table header
        Row(
            modifier = Modifier.fillMaxWidth().background(MaterialTheme.colorScheme.surfaceVariant).padding(8.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
        ) {
            Text(
                "Angle (°)",
                style = MaterialTheme.typography.titleSmall,
                modifier = Modifier.weight(1f),
            )
            Text(
                "Pressure (kPa)",
                style = MaterialTheme.typography.titleSmall,
                modifier = Modifier.weight(1f),
            )
            Text(
                "Piston Force (N)",
                style = MaterialTheme.typography.titleSmall,
                modifier = Modifier.weight(1f),
            )
            Text(
                "Rod Force (N)",
                style = MaterialTheme.typography.titleSmall,
                modifier = Modifier.weight(1f),
            )
            Text(
                "Side Force (N)",
                style = MaterialTheme.typography.titleSmall,
                modifier = Modifier.weight(1f),
            )
        }

        // Table rows
        Column {
            forceData.forEach { data ->
                Row(
                    modifier = Modifier.fillMaxWidth().padding(8.dp),
                    horizontalArrangement = Arrangement.SpaceBetween,
                ) {
                    Text(
                        data.angle.toString(),
                        style = MaterialTheme.typography.bodyMedium,
                        modifier = Modifier.weight(1f),
                    )
                    Text(
                        String.format("%.2f", data.pressure),
                        style = MaterialTheme.typography.bodyMedium,
                        modifier = Modifier.weight(1f),
                    )
                    Text(
                        String.format("%.2f", data.pistonForce),
                        style = MaterialTheme.typography.bodyMedium,
                        modifier = Modifier.weight(1f),
                    )
                    Text(
                        String.format("%.2f", data.rodForce),
                        style = MaterialTheme.typography.bodyMedium,
                        modifier = Modifier.weight(1f),
                    )
                    Text(
                        String.format("%.2f", data.sideForce),
                        style = MaterialTheme.typography.bodyMedium,
                        modifier = Modifier.weight(1f),
                    )
                }
                Divider()
            }
        }
    }
}

@Composable
fun EfficiencyTab(parameters: Map<String, String>) {
    // Extract key parameters
    val pistonDiameter = parameters["Piston Diameter"]?.toFloatOrNull() ?: 70f
    val stroke = parameters["Stroke"]?.toFloatOrNull() ?: 20f
    val cycleRatio = parameters["Cycle Ratio"]?.toFloatOrNull() ?: 2f

    // Calculate some efficiency metrics
    val displacement = (PI * pistonDiameter * pistonDiameter * stroke / 4 / 1000).toFloat() // cc
    val power = displacement * 0.08f * cycleRatio // Estimated power in kW at 1000 RPM
    val torque = power * 9.5488f // Nm at 1000 RPM
    val efficiency = 0.32f + (cycleRatio - 1) * 0.05f // Simplified thermal efficiency model

    // Create efficiency data
    val efficiencyData =
        listOf(
            Pair("Estimated Power", String.format("%.2f kW at 1000 RPM", power)),
            Pair("Estimated Torque", String.format("%.2f Nm at 1000 RPM", torque)),
            Pair("Thermal Efficiency", String.format("%.1f%%", efficiency * 100)),
            Pair("Specific Fuel Consumption", String.format("%.1f g/kWh", 270f / efficiency)),
            Pair("Power Density", String.format("%.2f kW/L", power / (displacement / 1000))),
            Pair("Mechanical Efficiency", "78% (estimated)"),
        )

    // Display efficiency data in a card
    Card(
        modifier = Modifier.fillMaxWidth(),
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
        ) {
            Text(
                "Efficiency Metrics",
                style = MaterialTheme.typography.titleMedium,
            )

            Spacer(modifier = Modifier.height(8.dp))

            // Display each metric
            efficiencyData.forEach { (label, value) ->
                Row(
                    modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp),
                    horizontalArrangement = Arrangement.SpaceBetween,
                ) {
                    Text(
                        label,
                        style = MaterialTheme.typography.bodyMedium,
                    )
                    Text(
                        value,
                        style = MaterialTheme.typography.bodyMedium,
                    )
                }
                Divider()
            }
        }
    }
}

// Data classes for tables
data class KinematicsData(val angle: Int, val displacement: Float, val velocity: Float, val acceleration: Float)

data class ForceData(val angle: Int, val pressure: Float, val pistonForce: Float, val rodForce: Float, val sideForce: Float)

// No custom extension function needed as we're directly importing androidx.compose.foundation.background
