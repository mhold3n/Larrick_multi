package com.campro.v5

import androidx.compose.desktop.ui.tooling.preview.Preview
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Window
import androidx.compose.ui.window.application
import androidx.compose.ui.window.rememberWindowState
import com.campro.v5.models.OptimizationParameters
import com.campro.v5.ui.ModernTileLayout
import org.slf4j.LoggerFactory

/**
 * Main entry point for the new unified optimization workflow.
 *
 * This replaces the old DesktopMain.kt with a clean implementation
 * that focuses on the new workflow components.
 */
fun main() = application {
    val logger = LoggerFactory.getLogger("NewWorkflowMain")
    logger.info("Starting CamProV5 with new unified optimization workflow GUI")

    Window(
        onCloseRequest = ::exitApplication,
        title = "CamProV5 - Unified Optimization Workflow",
        state = rememberWindowState(width = 1200.dp, height = 800.dp),
    ) {
        MaterialTheme {
            ModernTileLayout(
                testingMode = false,
                animationStarted = false,
                allParameters = OptimizationParameters.createDefault(),
                layoutManager = null,
                onParametersChanged = { parameters ->
                    logger.info("Parameters changed: ${parameters.gearRatio}")
                },
            )
        }
    }
}

@Preview
@Composable
fun NewWorkflowMainPreview() {
    MaterialTheme {
        ModernTileLayout(
            testingMode = true,
            animationStarted = false,
            allParameters = OptimizationParameters.createDefault(),
            layoutManager = null,
            onParametersChanged = { },
        )
    }
}
