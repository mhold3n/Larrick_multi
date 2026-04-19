package com.campro.v5

import androidx.compose.desktop.Window
import androidx.compose.foundation.layout.*
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import com.campro.v5.config.FeatureFlags
import com.campro.v5.legacy.LegacyComponentWrapper
import org.slf4j.LoggerFactory

/**
 * Simplified main entry point for testing the new unified optimization workflow.
 *
 * This is a minimal version that can compile and run our tests.
 */
fun main() {
    val logger = LoggerFactory.getLogger("CamProV5.SimpleNewWorkflowMain")

    // Log feature flag status
    logger.info("Starting CamProV5 with new unified optimization workflow")
    logger.info("Feature flags status:")
    logger.info("  Old workflow components: ${if (FeatureFlags.hasOldFeaturesEnabled()) "ENABLED" else "DISABLED"}")
    logger.info("  New workflow components: ${if (FeatureFlags.hasAllNewFeaturesEnabled()) "ENABLED" else "PARTIALLY ENABLED"}")

    // Log legacy component status
    LegacyComponentWrapper.logLegacyComponentStatus()

    // Show warning if old features are still enabled
    if (FeatureFlags.hasOldFeaturesEnabled()) {
        logger.warn("Some old workflow features are still enabled. Consider migrating to new workflow.")
    }

    // Start the new workflow application
    Window(
        title = "CamProV5 - Unified Optimization Pipeline (Test Mode)",
        size = IntSize(800, 600),
    ) {
        MaterialTheme {
            SimpleNewWorkflowApp()
        }
    }
}

/**
 * Simplified application composable for testing.
 */
@Composable
fun SimpleNewWorkflowApp() {
    Column(
        modifier = Modifier.fillMaxSize(),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center,
    ) {
        Text(
            text = "CamProV5 - New Unified Optimization Workflow",
            style = MaterialTheme.typography.headlineMedium,
        )

        Spacer(modifier = Modifier.height(16.dp))

        Text(
            text = "Feature Flags Status:",
            style = MaterialTheme.typography.bodyLarge,
        )

        Text(
            text = "Old Workflow: ${if (FeatureFlags.hasOldFeaturesEnabled()) "ENABLED" else "DISABLED"}",
            style = MaterialTheme.typography.bodyMedium,
        )

        Text(
            text = "New Workflow: ${if (FeatureFlags.hasAllNewFeaturesEnabled()) "ENABLED" else "PARTIALLY ENABLED"}",
            style = MaterialTheme.typography.bodyMedium,
        )

        Spacer(modifier = Modifier.height(16.dp))

        Text(
            text = "This is a simplified test version of the new workflow.",
            style = MaterialTheme.typography.bodySmall,
        )

        Text(
            text = "The full implementation is ready and waiting for API fixes.",
            style = MaterialTheme.typography.bodySmall,
        )
    }
}
