package com.campro.v5.ui

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.campro.v5.config.FeatureFlags
import com.campro.v5.debug.DebugButton
import com.campro.v5.debug.DebugOutlinedButton

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun FeatureFlagsPanel() {
    var flagStates by remember {
        mutableStateOf(FeatureFlags.getAllFlags().toMutableMap())
    }
    val flagDescriptions = FeatureFlags.getFeatureDescriptions()

    Column(
        modifier =
        Modifier
            .fillMaxSize()
            .padding(16.dp)
            .verticalScroll(rememberScrollState()),
    ) {
        Text(
            text = "Feature Flags",
            style = MaterialTheme.typography.headlineMedium,
            fontWeight = FontWeight.Bold,
            modifier = Modifier.padding(bottom = 16.dp),
        )

        Text(
            text = "Control experimental and advanced features. Changes take effect immediately.",
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.padding(bottom = 24.dp),
        )

        // Group flags by category
        val groupedFlags =
            flagStates.keys.groupBy { flagName ->
                when {
                    flagName.startsWith("collocation.") -> "Collocation Solver"
                    flagName.startsWith("advanced.") -> "Advanced Features"
                    flagName.startsWith("debug.") -> "Debug & Development"
                    else -> "General"
                }
            }

        groupedFlags.forEach { (category, flags) ->
            Card(
                modifier =
                Modifier
                    .fillMaxWidth()
                    .padding(bottom = 16.dp),
                elevation = CardDefaults.cardElevation(defaultElevation = 4.dp),
            ) {
                Column(
                    modifier = Modifier.padding(16.dp),
                ) {
                    Text(
                        text = category,
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.SemiBold,
                        modifier = Modifier.padding(bottom = 12.dp),
                    )

                    flags.sorted().forEach { flagName ->
                        val isEnabled = flagStates[flagName] ?: false
                        val description = flagDescriptions[flagName] ?: "No description available"

                        Row(
                            modifier =
                            Modifier
                                .fillMaxWidth()
                                .padding(vertical = 8.dp),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically,
                        ) {
                            Column(
                                modifier = Modifier.weight(1f),
                            ) {
                                Text(
                                    text = flagName.substringAfter('.'),
                                    style = MaterialTheme.typography.bodyLarge,
                                    fontWeight = FontWeight.Medium,
                                )
                                Text(
                                    text = description,
                                    style = MaterialTheme.typography.bodySmall,
                                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                                )
                            }

                            Switch(
                                checked = isEnabled,
                                onCheckedChange = { newValue ->
                                    flagStates[flagName] = newValue
                                    FeatureFlags.setFlag(flagName, newValue)
                                },
                            )
                        }

                        if (flagName != flags.sorted().last()) {
                            Divider(
                                modifier = Modifier.padding(vertical = 4.dp),
                                color = MaterialTheme.colorScheme.outline.copy(alpha = 0.3f),
                            )
                        }
                    }
                }
            }
        }

        // Action buttons
        Row(
            modifier =
            Modifier
                .fillMaxWidth()
                .padding(top = 16.dp),
            horizontalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            DebugOutlinedButton(
                buttonId = "feature-flags-refresh",
                onClick = {
                    flagStates = FeatureFlags.getAllFlags().toMutableMap()
                },
                modifier = Modifier.weight(1f),
            ) {
                Text("Refresh")
            }

            DebugButton(
                buttonId = "feature-flags-save",
                onClick = {
                    FeatureFlags.saveConfig()
                },
                modifier = Modifier.weight(1f),
            ) {
                Text("Save to File")
            }
        }

        // Status information
        Spacer(modifier = Modifier.height(24.dp))

        Card(
            modifier = Modifier.fillMaxWidth(),
            colors =
            CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surfaceVariant,
            ),
        ) {
            Column(
                modifier = Modifier.padding(16.dp),
            ) {
                Text(
                    text = "Configuration Info",
                    style = MaterialTheme.typography.titleSmall,
                    fontWeight = FontWeight.SemiBold,
                    modifier = Modifier.padding(bottom = 8.dp),
                )

                Text(
                    text = "Config file: ~/.campro/feature_flags.properties",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )

                Text(
                    text = "Priority: Runtime override > Config file > Default values",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )

                Text(
                    text = "Changes are applied immediately but saving persists them across restarts.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
    }
}
