package com.campro.v5.debug

import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp

/**
 * Debug panel composable mirroring the AccessibilitySettingsPanel pattern.
 * It exposes toggles for core debug capabilities and updates DebugManager.
 */
@Composable
fun DebugPanel(onSettingsChanged: (DebugManager.DebugSettings) -> Unit, modifier: Modifier = Modifier) {
    var settings by remember { mutableStateOf(DebugManager.settings) }

    Card(
        modifier = modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant,
        ),
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            Text(
                text = "Debug Panel",
                style = MaterialTheme.typography.titleMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                fontWeight = FontWeight.Bold,
            )

            // Button Debug toggle
            LabeledSwitch(
                title = "Button Debug",
                subtitle = "Show overlays and log interactions",
                checked = settings.buttonDebug,
                onCheckedChange = {
                    settings = settings.copy(buttonDebug = it)
                    onSettingsChanged(settings)
                },
            )

            // Component Health toggle
            LabeledSwitch(
                title = "Component Health Monitoring",
                subtitle = "Detect rendering and dependency issues",
                checked = settings.componentHealth,
                onCheckedChange = {
                    settings = settings.copy(componentHealth = it)
                    onSettingsChanged(settings)
                },
            )

            // Interaction Logging toggle
            LabeledSwitch(
                title = "Interaction Logging",
                subtitle = "Log user actions with context",
                checked = settings.interactionLogging,
                onCheckedChange = {
                    settings = settings.copy(interactionLogging = it)
                    onSettingsChanged(settings)
                },
            )

            // Error Boundary toggle
            LabeledSwitch(
                title = "Error Boundary",
                subtitle = "Catch and report UI errors",
                checked = settings.errorBoundary,
                onCheckedChange = {
                    settings = settings.copy(errorBoundary = it)
                    onSettingsChanged(settings)
                },
            )

            // Performance Monitoring toggle
            LabeledSwitch(
                title = "Performance Monitoring",
                subtitle = "Track rendering and operation times",
                checked = settings.performanceMonitoring,
                onCheckedChange = {
                    settings = settings.copy(performanceMonitoring = it)
                    onSettingsChanged(settings)
                },
            )

            // Network Debugging toggle
            LabeledSwitch(
                title = "Network Debugging",
                subtitle = "Monitor bridge/API calls",
                checked = settings.networkDebugging,
                onCheckedChange = {
                    settings = settings.copy(networkDebugging = it)
                    onSettingsChanged(settings)
                },
            )
        }
    }
}

@Composable
private fun LabeledSwitch(title: String, subtitle: String, checked: Boolean, onCheckedChange: (Boolean) -> Unit) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Column {
            Text(
                text = title,
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Text(
                text = subtitle,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.7f),
            )
        }
        Switch(checked = checked, onCheckedChange = onCheckedChange)
    }
}
