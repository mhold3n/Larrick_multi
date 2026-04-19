package com.campro.v5.ui

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import com.campro.v5.batch.BatchProcessor
import com.campro.v5.batch.BatchProgress
import com.campro.v5.batch.BatchResult
import com.campro.v5.batch.BatchResultStatus
import com.campro.v5.batch.BatchState
import com.campro.v5.debug.DebugButton
import com.campro.v5.debug.DebugIconButton
import com.campro.v5.io.ResultExporter
import com.campro.v5.models.OptimizationParameters
import com.campro.v5.models.OptimizationResult
import com.campro.v5.presets.PresetManager
import kotlinx.coroutines.launch
import org.slf4j.LoggerFactory
import java.nio.file.Paths

/**
 * Advanced features panel for export/import, presets, and batch processing.
 */
@Composable
fun AdvancedFeaturesPanel(
    currentParameters: OptimizationParameters,
    currentResult: OptimizationResult?,
    onParametersChanged: (OptimizationParameters) -> Unit,
    modifier: Modifier = Modifier,
) {
    val logger = LoggerFactory.getLogger(AdvancedFeaturesPanel::class.java)
    val scope = rememberCoroutineScope()

    var selectedTab by remember { mutableStateOf(0) }

    val tabs = listOf("Presets", "Export/Import", "Batch Processing")

    Card(
        modifier = modifier.fillMaxSize(),
        elevation = CardDefaults.cardElevation(defaultElevation = 2.dp),
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            // Header
            Text(
                text = "Advanced Features",
                style = MaterialTheme.typography.titleLarge,
                color = MaterialTheme.colorScheme.onSurface,
            )

            // Tab row
            TabRow(selectedTabIndex = selectedTab) {
                tabs.forEachIndexed { index, tab ->
                    Tab(
                        selected = selectedTab == index,
                        onClick = { selectedTab = index },
                        text = { Text(tab) },
                    )
                }
            }

            // Tab content
            Box(
                modifier = Modifier.weight(1f),
            ) {
                when (selectedTab) {
                    0 -> PresetsPanel(
                        currentParameters = currentParameters,
                        onParametersChanged = onParametersChanged,
                        modifier = Modifier.fillMaxSize(),
                    )
                    1 -> ExportImportPanel(
                        currentParameters = currentParameters,
                        currentResult = currentResult,
                        modifier = Modifier.fillMaxSize(),
                    )
                    2 -> BatchProcessingPanel(
                        currentParameters = currentParameters,
                        modifier = Modifier.fillMaxSize(),
                    )
                }
            }
        }
    }
}

@Composable
private fun PresetsPanel(
    currentParameters: OptimizationParameters,
    onParametersChanged: (OptimizationParameters) -> Unit,
    modifier: Modifier = Modifier,
) {
    val presetManager = remember { PresetManager() }
    var availablePresets by remember { mutableStateOf<List<PresetManager.PresetInfo>>(emptyList()) }
    var showSaveDialog by remember { mutableStateOf(false) }
    var showLoadDialog by remember { mutableStateOf(false) }

    // Load presets on first composition
    LaunchedEffect(Unit) {
        availablePresets = presetManager.getAvailablePresets()
    }

    Column(
        modifier = modifier,
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        // Action buttons
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            DebugButton(
                buttonId = "presets-save",
                onClick = { showSaveDialog = true },
                modifier = Modifier.weight(1f),
            ) {
                Icon(
                    imageVector = Icons.Default.Save,
                    contentDescription = null,
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text("Save Preset")
            }

            DebugButton(
                buttonId = "presets-load",
                onClick = { showLoadDialog = true },
                modifier = Modifier.weight(1f),
            ) {
                Icon(
                    imageVector = Icons.Default.FolderOpen,
                    contentDescription = null,
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text("Load Preset")
            }
        }

        // Presets list
        Card(
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surfaceVariant,
            ),
        ) {
            Column(
                modifier = Modifier.padding(12.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Text(
                    text = "Available Presets",
                    style = MaterialTheme.typography.titleSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )

                if (availablePresets.isEmpty()) {
                    Text(
                        text = "No presets available",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                } else {
                    LazyColumn(
                        verticalArrangement = Arrangement.spacedBy(4.dp),
                    ) {
                        items(availablePresets) { preset ->
                            PresetItem(
                                preset = preset,
                                onLoad = {
                                    val loadedPreset = presetManager.loadPreset(preset.name)
                                    loadedPreset?.let { onParametersChanged(it.parameters) }
                                },
                                onDelete = {
                                    presetManager.deletePreset(preset.name)
                                    availablePresets = presetManager.getAvailablePresets()
                                },
                            )
                        }
                    }
                }
            }
        }
    }

    // Save preset dialog
    if (showSaveDialog) {
        SavePresetDialog(
            onDismiss = { showSaveDialog = false },
            onSave = { name, description ->
                val preset = presetManager.createPreset(
                    name = name,
                    description = description,
                    parameters = currentParameters,
                )
                presetManager.savePreset(preset)
                availablePresets = presetManager.getAvailablePresets()
                showSaveDialog = false
            },
        )
    }

    // Load preset dialog
    if (showLoadDialog) {
        LoadPresetDialog(
            presets = availablePresets,
            onDismiss = { showLoadDialog = false },
            onLoad = { presetName ->
                val preset = presetManager.loadPreset(presetName)
                preset?.let { onParametersChanged(it.parameters) }
                showLoadDialog = false
            },
        )
    }
}

@Composable
private fun PresetItem(preset: PresetManager.PresetInfo, onLoad: () -> Unit, onDelete: () -> Unit) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surface,
        ),
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(12.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Column(
                modifier = Modifier.weight(1f),
            ) {
                Text(
                    text = preset.name,
                    style = MaterialTheme.typography.titleSmall,
                    color = MaterialTheme.colorScheme.onSurface,
                )
                if (preset.description.isNotEmpty()) {
                    Text(
                        text = preset.description,
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                }
                Text(
                    text = "Modified: ${preset.modifiedAt}",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }

            Row(
                horizontalArrangement = Arrangement.spacedBy(4.dp),
            ) {
                DebugIconButton(buttonId = "presets-item-load-" + preset.name, onClick = onLoad) {
                    Icon(
                        imageVector = Icons.Default.PlayArrow,
                        contentDescription = "Load preset",
                    )
                }
                DebugIconButton(buttonId = "presets-item-delete-" + preset.name, onClick = onDelete) {
                    Icon(
                        imageVector = Icons.Default.Delete,
                        contentDescription = "Delete preset",
                    )
                }
            }
        }
    }
}

@Composable
private fun ExportImportPanel(
    currentParameters: OptimizationParameters,
    currentResult: OptimizationResult?,
    modifier: Modifier = Modifier,
) {
    val scope = rememberCoroutineScope()
    val exporter = remember { ResultExporter() }

    Column(
        modifier = modifier,
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        // Export section
        Card(
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.3f),
            ),
        ) {
            Column(
                modifier = Modifier.padding(12.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Text(
                    text = "Export Results",
                    style = MaterialTheme.typography.titleSmall,
                    color = MaterialTheme.colorScheme.onPrimaryContainer,
                )

                if (currentResult != null) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                    ) {
                        DebugButton(
                            buttonId = "export-json",
                            onClick = {
                                scope.launch {
                                    val outputPath = Paths.get("./exports/results_${System.currentTimeMillis()}.json")
                                    exporter.exportResult(
                                        result = currentResult,
                                        parameters = currentParameters,
                                        outputPath = outputPath,
                                        format = ResultExporter.ExportFormat.JSON,
                                    )
                                }
                            },
                            modifier = Modifier.weight(1f),
                        ) {
                            Text("Export JSON")
                        }

                        DebugButton(
                            buttonId = "export-csv",
                            onClick = {
                                scope.launch {
                                    val outputPath = Paths.get("./exports/results_${System.currentTimeMillis()}.csv")
                                    exporter.exportResult(
                                        result = currentResult,
                                        parameters = currentParameters,
                                        outputPath = outputPath,
                                        format = ResultExporter.ExportFormat.CSV,
                                    )
                                }
                            },
                            modifier = Modifier.weight(1f),
                        ) {
                            Text("Export CSV")
                        }
                    }
                } else {
                    Text(
                        text = "No results available for export",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onPrimaryContainer,
                    )
                }
            }
        }

        // Import section
        Card(
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.secondaryContainer.copy(alpha = 0.3f),
            ),
        ) {
            Column(
                modifier = Modifier.padding(12.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Text(
                    text = "Import Parameters",
                    style = MaterialTheme.typography.titleSmall,
                    color = MaterialTheme.colorScheme.onSecondaryContainer,
                )

                Text(
                    text = "Import functionality will be available in future versions",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSecondaryContainer,
                )
            }
        }
    }
}

@Composable
private fun BatchProcessingPanel(currentParameters: OptimizationParameters, modifier: Modifier = Modifier) {
    val scope = rememberCoroutineScope()
    val batchProcessor = remember { BatchProcessor(com.campro.v5.pipeline.UnifiedOptimizationBridge()) }
    val batchState by batchProcessor.batchState.collectAsState()
    val progress by batchProcessor.progress.collectAsState()
    val results by batchProcessor.results.collectAsState()

    var showBatchDialog by remember { mutableStateOf(false) }

    Column(
        modifier = modifier,
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        // Batch controls
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            DebugButton(
                buttonId = "batch-start",
                onClick = { showBatchDialog = true },
                enabled = batchState !is BatchState.Running,
                modifier = Modifier.weight(1f),
            ) {
                Icon(
                    imageVector = Icons.Default.PlaylistPlay,
                    contentDescription = null,
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text("Start Batch")
            }

            DebugButton(
                buttonId = "batch-cancel",
                onClick = { batchProcessor.cancelBatch() },
                enabled = batchState is BatchState.Running,
                modifier = Modifier.weight(1f),
                colors = ButtonDefaults.buttonColors(
                    containerColor = MaterialTheme.colorScheme.error,
                ),
            ) {
                Icon(
                    imageVector = Icons.Default.Stop,
                    contentDescription = null,
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text("Cancel")
            }
        }

        // Progress display
        if (batchState is BatchState.Running || batchState is BatchState.Completed) {
            BatchProgressCard(
                progress = progress,
                batchState = batchState,
            )
        }

        // Results summary
        if (results.isNotEmpty()) {
            BatchResultsCard(
                results = results,
                onExportResults = {
                    scope.launch {
                        val outputPath = Paths.get("./batch_results_${System.currentTimeMillis()}.json")
                        batchProcessor.exportBatchResults(outputPath)
                    }
                },
            )
        }
    }

    // Batch configuration dialog
    if (showBatchDialog) {
        BatchConfigurationDialog(
            baseParameters = currentParameters,
            onDismiss = { showBatchDialog = false },
            onStartBatch = { parameterSets ->
                scope.launch {
                    val outputDir = com.campro.v5.io.FileIOUtils.createOutputDirectory("./batch_output")
                    batchProcessor.processBatch(parameterSets, outputDir)
                }
                showBatchDialog = false
            },
        )
    }
}

@Composable
private fun BatchProgressCard(progress: BatchProgress, batchState: BatchState) {
    Card(
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.tertiaryContainer.copy(alpha = 0.3f),
        ),
    ) {
        Column(
            modifier = Modifier.padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Text(
                text = "Batch Progress",
                style = MaterialTheme.typography.titleSmall,
                color = MaterialTheme.colorScheme.onTertiaryContainer,
            )

            LinearProgressIndicator(
                progress = if (progress.total > 0) (progress.completed + progress.failed).toFloat() / progress.total else 0f,
                modifier = Modifier.fillMaxWidth(),
            )

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
            ) {
                Text(
                    text = "Completed: ${progress.completed}",
                    style = MaterialTheme.typography.bodySmall,
                    color = Color(0xFF4CAF50),
                )
                Text(
                    text = "Failed: ${progress.failed}",
                    style = MaterialTheme.typography.bodySmall,
                    color = Color(0xFFF44336),
                )
                Text(
                    text = "Total: ${progress.total}",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onTertiaryContainer,
                )
            }

            Text(
                text = "Success Rate: ${String.format("%.1f", progress.successRate)}%",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onTertiaryContainer,
            )
        }
    }
}

@Composable
private fun BatchResultsCard(results: List<BatchResult>, onExportResults: () -> Unit) {
    Card(
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant,
        ),
    ) {
        Column(
            modifier = Modifier.padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text(
                    text = "Batch Results",
                    style = MaterialTheme.typography.titleSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )

                DebugButton(
                    buttonId = "batch-export",
                    onClick = onExportResults,
                    modifier = Modifier.height(32.dp),
                ) {
                    Text("Export")
                }
            }

            val successful = results.count { it.status == BatchResultStatus.Success }
            val failed = results.count { it.status == BatchResultStatus.Failed }
            val totalTime = results.sumOf { it.executionTime }

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
            ) {
                Text(
                    text = "Successful: $successful",
                    style = MaterialTheme.typography.bodySmall,
                    color = Color(0xFF4CAF50),
                )
                Text(
                    text = "Failed: $failed",
                    style = MaterialTheme.typography.bodySmall,
                    color = Color(0xFFF44336),
                )
                Text(
                    text = "Total Time: ${String.format("%.1f", totalTime)}s",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
    }
}

// Dialog components would be implemented here
@Composable
private fun SavePresetDialog(onDismiss: () -> Unit, onSave: (String, String) -> Unit) {
    // Implementation for save preset dialog
}

@Composable
private fun LoadPresetDialog(presets: List<PresetManager.PresetInfo>, onDismiss: () -> Unit, onLoad: (String) -> Unit) {
    // Implementation for load preset dialog
}

@Composable
private fun BatchConfigurationDialog(
    baseParameters: OptimizationParameters,
    onDismiss: () -> Unit,
    onStartBatch: (List<OptimizationParameters>) -> Unit,
) {
    // Implementation for batch configuration dialog
}
