package com.campro.v5

import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Window
import androidx.compose.ui.window.application
import androidx.compose.ui.window.rememberWindowState
import com.campro.v5.ui.UnifiedOptimizationTile
import org.slf4j.LoggerFactory

/**
 * Simple GUI test to verify the UnifiedOptimizationTile displays correctly.
 */
fun main() = application {
    val logger = LoggerFactory.getLogger("SimpleGUITest")
    logger.info("Starting Simple GUI Test")

    Window(
        onCloseRequest = ::exitApplication,
        title = "CamProV5 - Unified Optimization Workflow",
        state = rememberWindowState(
            width = 1200.dp,
            height = 800.dp,
            position = androidx.compose.ui.window.WindowPosition(100.dp, 100.dp),
        ),
        alwaysOnTop = true,
        resizable = true,
        undecorated = false,
        focusable = true,
    ) {
        logger.info("Window created and content is rendering")
        MaterialTheme {
            Column(
                modifier = Modifier.fillMaxSize().padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(16.dp),
            ) {
                Text(
                    text = "CamProV5 - Unified Optimization Workflow",
                    style = MaterialTheme.typography.headlineMedium,
                )

                Text(
                    text = "The UnifiedOptimizationTile below should show parameter input fields and optimization controls.",
                    style = MaterialTheme.typography.bodyMedium,
                )

                Card(
                    modifier = Modifier.fillMaxSize(),
                    elevation = CardDefaults.cardElevation(defaultElevation = 4.dp),
                ) {
                    // Test the actual UnifiedOptimizationTile
                    UnifiedOptimizationTile()
                }
            }
        }
    }
}
