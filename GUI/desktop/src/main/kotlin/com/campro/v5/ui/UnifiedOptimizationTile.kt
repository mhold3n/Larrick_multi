package com.campro.v5.ui

import androidx.compose.foundation.layout.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.material3.ScrollableTabRow
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import com.campro.v5.models.ConvergenceStatus
import com.campro.v5.models.OptimizationParameters
import com.campro.v5.models.OptimizationResult
import com.campro.v5.optimization.OptimizationState
import com.campro.v5.optimization.OptimizationStateManager
import com.campro.v5.pipeline.OptimizationBackendProvider
import com.campro.v5.pipeline.OptimizationPort
import com.campro.v5.visualization.MotionLawVisualization
import com.campro.v5.visualization.GearProfileVisualization
import com.campro.v5.visualization.EfficiencyAnalysisVisualization
import com.campro.v5.visualization.FEAAnalysisVisualization
// import com.campro.v5.ui.AdvancedFeaturesPanel // Temporarily excluded
import com.campro.v5.performance.PerformanceOptimizer
import com.campro.v5.error.ErrorHandler
import com.campro.v5.accessibility.AccessibilityEnhancer
import com.campro.v5.accessibility.AccessibilitySettingsPanel
import com.campro.v5.debug.DebugManager
import com.campro.v5.debug.DebugPanel
import com.campro.v5.debug.DebugButton
import com.campro.v5.debug.DebugOutlinedButton
import com.campro.v5.debug.DebugIconButton
import kotlinx.coroutines.launch
import java.nio.file.Paths
import org.slf4j.LoggerFactory

/**
 * Unified optimization tile for the CamProV5 application.
 *
 * This tile provides a complete interface for running the unified optimization
 * pipeline, including parameter input, progress tracking, and result display.
 */
@Composable
fun UnifiedOptimizationTile(
    optimizationPort: OptimizationPort = OptimizationBackendProvider.createOptimizationPort(),
    onResultsReceived: (OptimizationResult) -> Unit = {},
    modifier: Modifier = Modifier,
) {
    val logger = LoggerFactory.getLogger("UnifiedOptimizationTile")
    val scope = rememberCoroutineScope()

    // State management
    var parameters by remember { mutableStateOf(OptimizationParameters.createDefault()) }
    var outputDir by remember { mutableStateOf(Paths.get("./output")) }
    var currentResult by remember { mutableStateOf<OptimizationResult?>(null) }

    // Performance and UX state
    val errorHandler = remember { ErrorHandler() }
    val performanceOptimizer = remember { PerformanceOptimizer }
    var showAccessibilitySettings by remember { mutableStateOf(false) }
    var showDebugPanel by remember { mutableStateOf(DebugManager.settings.panelVisible) }

    // Create state manager
    val stateManager = remember(optimizationPort) { OptimizationStateManager(optimizationPort) }
    val optimizationState by stateManager.optimizationState.collectAsState()

    // Handle results and errors
    LaunchedEffect(optimizationState) {
        when (val state = optimizationState) {
            is OptimizationState.Completed -> {
                val result = state.result
                currentResult = result
                onResultsReceived(result)
                logger.info("Optimization results received: ${result.status}")

                // Track performance
                performanceOptimizer.updateMetrics()
            }
            is OptimizationState.Failed -> {
                errorHandler.reportError(
                    message = state.error.message ?: "Optimization failed",
                    severity = ErrorHandler.ErrorSeverity.ERROR,
                    context = "Optimization",
                    recoveryAction = ErrorHandler.RecoveryAction(
                        label = "Retry",
                        action = {
                            scope.launch {
                                stateManager.runOptimization(parameters, outputDir)
                            }
                        },
                        canRetry = true,
                    ),
                    technicalDetails = state.error.stackTraceToString(),
                )
            }
            else -> {}
        }
    }

    Card(
        modifier = modifier.fillMaxSize(),
        elevation = CardDefaults.cardElevation(defaultElevation = 4.dp),
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            // Header with accessibility & debug settings
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text(
                    text = "Unified Optimization",
                    style = MaterialTheme.typography.headlineSmall,
                    color = MaterialTheme.colorScheme.onSurface,
                )

                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    DebugIconButton(
                        buttonId = "toggle-debug-panel",
                        onClick = {
                            showDebugPanel = !showDebugPanel
                            DebugManager.setPanelVisible(showDebugPanel)
                        },
                    ) {
                        Icon(
                            imageVector = Icons.Default.BugReport,
                            contentDescription = "Debug Panel",
                        )
                    }
                    DebugIconButton(
                        buttonId = "toggle-accessibility-panel",
                        onClick = { showAccessibilitySettings = !showAccessibilitySettings },
                    ) {
                        Icon(
                            imageVector = Icons.Default.Accessibility,
                            contentDescription = "Accessibility Settings",
                        )
                    }
                }
            }

            // Debug panel
            if (showDebugPanel) {
                DebugPanel(
                    onSettingsChanged = { DebugManager.updateSettings(it) },
                    modifier = Modifier.fillMaxWidth(),
                )
            }

            // Accessibility settings panel
            if (showAccessibilitySettings) {
                AccessibilitySettingsPanel(
                    onSettingsChanged = { AccessibilityEnhancer.updateSettings(it) },
                    modifier = Modifier.fillMaxWidth(),
                )
            }

            // Main content area with proper space allocation
            Box(
                modifier = Modifier.weight(1f),
            ) {
                when (optimizationState) {
                    is OptimizationState.Idle, is OptimizationState.Running, is OptimizationState.Failed -> {
                        // Show parameter input when not completed
                        Column(
                            modifier = Modifier.fillMaxSize(),
                            verticalArrangement = Arrangement.spacedBy(16.dp),
                        ) {
                            // Parameter input section
                            OptimizationParameterForm(
                                parameters = parameters,
                                onParametersChanged = { parameters = it },
                                modifier = Modifier.weight(1f),
                            )

                            // Error display
                            errorHandler.currentError?.let { error ->
                                Text(
                                    text = "Error Display (Temporarily Disabled): ${error.message}",
                                    modifier = Modifier.fillMaxWidth(),
                                )
                            }

                            // Loading state
                            if (optimizationState is OptimizationState.Running) {
                                LinearProgressIndicator(
                                    progress = (optimizationState as OptimizationState.Running).progress.toFloat(),
                                    modifier = Modifier.fillMaxWidth(),
                                )
                            }

                            // Progress and status
                            OptimizationStatus(
                                optimizationState = optimizationState,
                            )
                        }
                    }
                    is OptimizationState.Completed -> {
                        // Show results when completed
                        val result = (optimizationState as OptimizationState.Completed).result
                        OptimizationResultsVisualization(
                            result = result,
                            modifier = Modifier.fillMaxSize(),
                        )
                    }
                }
            }

            // Larrick dashboard bridge section.
            Column(
                modifier = Modifier.fillMaxWidth(),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Text(
                    text = "Larrick Dashboard",
                    style = MaterialTheme.typography.titleMedium,
                    color = MaterialTheme.colorScheme.onSurface,
                )
                LarrickOrchestrationPanel(modifier = Modifier.fillMaxWidth())
                LarrickSimulationPanel(modifier = Modifier.fillMaxWidth())
            }

            // Control buttons (always visible, adapts to state)
            OptimizationControls(
                optimizationState = optimizationState,
                onStartOptimization = {
                    scope.launch {
                        stateManager.runOptimization(parameters, outputDir)
                    }
                },
                onCancelOptimization = {
                    stateManager.cancelOptimization()
                },
                onResetState = {
                    stateManager.resetState()
                },
            )
        }
    }
}

@Composable
private fun OptimizationControls(
    optimizationState: OptimizationState,
    onStartOptimization: () -> Unit,
    onCancelOptimization: () -> Unit,
    onResetState: () -> Unit,
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        when (optimizationState) {
            is OptimizationState.Idle -> {
                DebugButton(
                    buttonId = "start-optimization",
                    onClick = onStartOptimization,
                    modifier = Modifier.weight(1f),
                ) {
                    Icon(
                        imageVector = Icons.Default.PlayArrow,
                        contentDescription = "Start optimization",
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("Start Optimization")
                }
            }

            is OptimizationState.Running -> {
                DebugButton(
                    buttonId = "cancel-optimization",
                    onClick = onCancelOptimization,
                    modifier = Modifier.weight(1f),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = MaterialTheme.colorScheme.error,
                    ),
                ) {
                    Icon(
                        imageVector = Icons.Default.Stop,
                        contentDescription = "Cancel optimization",
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("Cancel")
                }
            }

            is OptimizationState.Completed, is OptimizationState.Failed -> {
                DebugButton(
                    buttonId = "run-again",
                    onClick = onStartOptimization,
                    modifier = Modifier.weight(1f),
                ) {
                    Icon(
                        imageVector = Icons.Default.Refresh,
                        contentDescription = "Run again",
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("Run Again")
                }

                DebugOutlinedButton(
                    buttonId = "reset-state",
                    onClick = onResetState,
                    modifier = Modifier.weight(1f),
                ) {
                    Icon(
                        imageVector = Icons.Default.Clear,
                        contentDescription = "Reset",
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("Reset")
                }
            }
        }
    }
}

@Composable
private fun OptimizationStatus(optimizationState: OptimizationState) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = when (optimizationState) {
                is OptimizationState.Idle -> MaterialTheme.colorScheme.surfaceVariant
                is OptimizationState.Running -> MaterialTheme.colorScheme.primaryContainer
                is OptimizationState.Completed -> MaterialTheme.colorScheme.tertiaryContainer
                is OptimizationState.Failed -> MaterialTheme.colorScheme.errorContainer
            },
        ),
    ) {
        Column(
            modifier = Modifier.padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Icon(
                    imageVector = when (optimizationState) {
                        is OptimizationState.Idle -> Icons.Default.Pause
                        is OptimizationState.Running -> Icons.Default.PlayArrow
                        is OptimizationState.Completed -> Icons.Default.CheckCircle
                        is OptimizationState.Failed -> Icons.Default.Error
                    },
                    contentDescription = null,
                    tint = when (optimizationState) {
                        is OptimizationState.Idle -> MaterialTheme.colorScheme.onSurfaceVariant
                        is OptimizationState.Running -> MaterialTheme.colorScheme.primary
                        is OptimizationState.Completed -> MaterialTheme.colorScheme.tertiary
                        is OptimizationState.Failed -> MaterialTheme.colorScheme.error
                    },
                )

                Text(
                    text = when (optimizationState) {
                        is OptimizationState.Idle -> "Ready to optimize"
                        is OptimizationState.Running -> "Optimization in progress..."
                        is OptimizationState.Completed -> "Optimization completed successfully"
                        is OptimizationState.Failed -> "Optimization failed"
                    },
                    style = MaterialTheme.typography.titleSmall,
                    color = when (optimizationState) {
                        is OptimizationState.Idle -> MaterialTheme.colorScheme.onSurfaceVariant
                        is OptimizationState.Running -> MaterialTheme.colorScheme.onPrimaryContainer
                        is OptimizationState.Completed -> MaterialTheme.colorScheme.onTertiaryContainer
                        is OptimizationState.Failed -> MaterialTheme.colorScheme.onErrorContainer
                    },
                )
            }

            // Progress bar for running state
            if (optimizationState is OptimizationState.Running) {
                LinearProgressIndicator(
                    progress = (optimizationState as OptimizationState.Running).progress.toFloat(),
                    modifier = Modifier.fillMaxWidth(),
                )
            }

            // Error message for failed state
            if (optimizationState is OptimizationState.Failed) {
                Text(
                    text = (optimizationState as OptimizationState.Failed).error.message ?: "Unknown error occurred",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onErrorContainer,
                )
            }
        }
    }
}

@Composable
private fun OptimizationResultsVisualization(result: OptimizationResult, modifier: Modifier = Modifier) {
    var selectedTabIndex by rememberSaveable(result.status) { mutableStateOf(0) }

    val tabItems = remember(result) {
        listOf(
            ResultTab("Motion Law") {
                MotionLawVisualization(
                    motionLaw = result.motionLaw,
                    modifier = Modifier.fillMaxSize(),
                )
            },
            ResultTab("Gear Profiles") {
                GearProfileVisualization(
                    gearProfiles = result.optimalProfiles,
                    modifier = Modifier.fillMaxSize(),
                )
            },
            ResultTab("Efficiency") {
                EfficiencyAnalysisVisualization(
                    gearProfiles = result.optimalProfiles,
                    modifier = Modifier.fillMaxSize(),
                )
            },
            ResultTab("FEA Analysis") {
                FEAAnalysisVisualization(
                    feaAnalysis = result.feaAnalysis,
                    modifier = Modifier.fillMaxSize(),
                )
            },
            ResultTab("Advanced") {
                Box(
                    modifier = Modifier.fillMaxSize(),
                    contentAlignment = Alignment.Center,
                ) {
                    Text(
                        text = "Advanced Features Panel (Temporarily Disabled)",
                        style = MaterialTheme.typography.bodyLarge,
                    )
                }
            },
        )
    }

    LaunchedEffect(tabItems.size) {
        if (selectedTabIndex >= tabItems.size) {
            selectedTabIndex = 0
        }
    }

    Column(
        modifier = modifier.fillMaxSize(),
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        ResultSummaryHeader(result)

        ResultTabRow(
            tabs = tabItems,
            selectedIndex = selectedTabIndex,
            onTabSelected = { selectedTabIndex = it },
        )

        ResultTabContent(
            tabs = tabItems,
            selectedIndex = selectedTabIndex,
            modifier = Modifier.weight(1f),
        )
    }
}

private data class ResultTab(
    val label: String,
    val content: @Composable () -> Unit,
)

@Composable
private fun ResultSummaryHeader(result: OptimizationResult, modifier: Modifier = Modifier) {
    val convergence = result.convergence
    val statusColor = if (result.isSuccess()) Color(0xFF4CAF50) else Color(0xFFF44336)

    Card(
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.tertiaryContainer,
        ),
        modifier = modifier.fillMaxWidth(),
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    Icon(
                        imageVector = if (result.isSuccess()) Icons.Default.CheckCircle else Icons.Default.Error,
                        contentDescription = null,
                        tint = statusColor,
                    )
                    Text(
                        text = "Optimization Results",
                        style = MaterialTheme.typography.titleMedium,
                        color = MaterialTheme.colorScheme.onTertiaryContainer,
                    )
                }

                Text(
                    text = "Execution time: ${String.format("%.2f", result.executionTime)}s",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onTertiaryContainer,
                )
            }

            result.getErrorMessage()?.let { errorMessage ->
                Text(
                    text = errorMessage,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.error,
                )
            }

            convergence?.let { DiagnosticsMetricsRow(it) }
        }
    }
}

@Composable
private fun ResultTabRow(
    tabs: List<ResultTab>,
    selectedIndex: Int,
    onTabSelected: (Int) -> Unit,
    modifier: Modifier = Modifier,
) {
    ScrollableTabRow(
        selectedTabIndex = selectedIndex,
        modifier = modifier.height(48.dp),
        edgePadding = 0.dp,
    ) {
        tabs.forEachIndexed { index, tab ->
            Tab(
                selected = selectedIndex == index,
                onClick = { onTabSelected(index) },
                modifier = Modifier.height(48.dp),
                text = {
                    Text(
                        text = tab.label,
                        style = MaterialTheme.typography.bodyMedium,
                    )
                },
            )
        }
    }
}

@Composable
private fun ResultTabContent(
    tabs: List<ResultTab>,
    selectedIndex: Int,
    modifier: Modifier = Modifier,
) {
    Card(
        modifier = modifier.fillMaxWidth(),
        elevation = CardDefaults.cardElevation(defaultElevation = 2.dp),
    ) {
        Box(modifier = Modifier.fillMaxSize()) {
            tabs.getOrNull(selectedIndex)?.content?.invoke()
        }
    }
}

@Composable
private fun DiagnosticsMetricsRow(convergence: ConvergenceStatus, modifier: Modifier = Modifier) {
    Row(
        modifier = modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(12.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        MetricChip(
            label = "Converged",
            value = if (convergence.converged) "Yes" else "No",
            tint = if (convergence.converged) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.error,
        )
        convergence.kktError?.let { MetricChip(label = "KKT", value = it.asDisplayString()) }
        convergence.constraintTotalViolation?.let { MetricChip(label = "Constraints", value = it.asDisplayString()) }
        convergence.iterations?.let { MetricChip(label = "Iterations", value = it.toString()) }
        convergence.solverSuccess?.let { MetricChip(label = "Solver", value = if (it) "Success" else "Fallback") }
    }
}

@Composable
private fun MetricChip(label: String, value: String, tint: Color = MaterialTheme.colorScheme.secondary) {
    Surface(
        color = tint.copy(alpha = 0.15f),
        contentColor = MaterialTheme.colorScheme.onSurface,
        shape = MaterialTheme.shapes.small,
        tonalElevation = 0.dp,
        shadowElevation = 0.dp,
    ) {
        Text(
            text = "$label: $value",
            style = MaterialTheme.typography.bodySmall,
            modifier = Modifier.padding(horizontal = 12.dp, vertical = 6.dp),
        )
    }
}

private fun Double.asDisplayString(): String =
    when {
        kotlin.math.abs(this) >= 1.0 -> String.format("%.2f", this)
        else -> String.format("%.2e", this)
    }
