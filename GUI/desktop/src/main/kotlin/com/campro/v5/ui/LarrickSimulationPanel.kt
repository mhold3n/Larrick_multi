package com.campro.v5.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.campro.v5.pipeline.SimulationBackendProvider
import com.campro.v5.pipeline.SimulationRequest
import kotlinx.coroutines.launch

/**
 * Minimal dashboard panel for simulation bridge interaction.
 */
@Composable
fun LarrickSimulationPanel(
    modifier: Modifier = Modifier,
) {
    val scope = rememberCoroutineScope()
    val simulationPort = remember { SimulationBackendProvider.createSimulationPort() }
    var lastStatus by remember { mutableStateOf("idle") }
    var lastSummary by remember { mutableStateOf("No simulation run yet.") }

    Card(modifier = modifier.fillMaxWidth()) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Text("Larrick Simulation", style = MaterialTheme.typography.titleMedium)
            Text("Status: $lastStatus", style = MaterialTheme.typography.bodyMedium)
            Text(lastSummary, style = MaterialTheme.typography.bodySmall)
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Button(
                    onClick = {
                        scope.launch {
                            val response = simulationPort.simulate(
                                SimulationRequest(
                                    payload = mapOf(
                                        "rpm" to 3000.0,
                                        "fidelity" to 1,
                                        "case" to "dashboard-smoke",
                                    ),
                                ),
                            )
                            lastStatus = response.status
                            lastSummary = response.payload["payload"].toString()
                        }
                    },
                ) {
                    Text("Run Simulation")
                }
            }
        }
    }
}
