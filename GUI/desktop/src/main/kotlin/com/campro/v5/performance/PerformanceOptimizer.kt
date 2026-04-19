package com.campro.v5.performance

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import org.slf4j.LoggerFactory
import java.util.concurrent.atomic.AtomicLong

/**
 * Performance optimization utilities for Compose UI.
 *
 * Provides performance monitoring, optimization helpers, and memory management
 * for the optimization pipeline UI components.
 */
object PerformanceOptimizer {

    private val logger = LoggerFactory.getLogger(PerformanceOptimizer::class.java)

    /**
     * Performance metrics tracking.
     */
    data class PerformanceMetrics(
        val renderTime: Long = 0,
        val memoryUsage: Long = 0,
        val recompositionCount: Int = 0,
        val lastUpdate: Long = System.currentTimeMillis(),
    )

    private val _performanceMetrics = MutableStateFlow(PerformanceMetrics())
    val performanceMetrics: StateFlow<PerformanceMetrics> = _performanceMetrics.asStateFlow()

    private val recompositionCounter = AtomicLong(0)
    private val renderTimeTracker = mutableMapOf<String, Long>()

    /**
     * Track recomposition for performance monitoring.
     */
    fun trackRecomposition(componentName: String) {
        recompositionCounter.incrementAndGet()
        logger.debug("Recomposition in $componentName: ${recompositionCounter.get()}")
    }

    /**
     * Track render time for a component.
     */
    fun trackRenderTime(componentName: String, renderTime: Long) {
        renderTimeTracker[componentName] = renderTime
        logger.debug("Render time for $componentName: ${renderTime}ms")
    }

    /**
     * Get current memory usage.
     */
    fun getMemoryUsage(): Long {
        val runtime = Runtime.getRuntime()
        return runtime.totalMemory() - runtime.freeMemory()
    }

    /**
     * Update performance metrics.
     */
    fun updateMetrics() {
        val currentMetrics = _performanceMetrics.value
        _performanceMetrics.value = currentMetrics.copy(
            memoryUsage = getMemoryUsage(),
            recompositionCount = recompositionCounter.get().toInt(),
            lastUpdate = System.currentTimeMillis(),
        )
    }

    /**
     * Reset performance counters.
     */
    fun resetCounters() {
        recompositionCounter.set(0)
        renderTimeTracker.clear()
        logger.info("Performance counters reset")
    }

    /**
     * Check if performance is within acceptable limits.
     */
    fun isPerformanceAcceptable(): Boolean {
        val metrics = _performanceMetrics.value
        return metrics.memoryUsage < 200 * 1024 * 1024 &&
            // 200MB limit
            metrics.recompositionCount < 1000 // Reasonable recomposition limit
    }

    /**
     * Get performance recommendations.
     */
    fun getPerformanceRecommendations(): List<String> {
        val recommendations = mutableListOf<String>()
        val metrics = _performanceMetrics.value

        if (metrics.memoryUsage > 150 * 1024 * 1024) {
            recommendations.add("High memory usage detected. Consider optimizing data structures.")
        }

        if (metrics.recompositionCount > 500) {
            recommendations.add("High recomposition count. Consider using remember() for expensive calculations.")
        }

        val slowComponents = renderTimeTracker.filter { it.value > 16 } // > 16ms (60fps)
        if (slowComponents.isNotEmpty()) {
            recommendations.add("Slow rendering detected in: ${slowComponents.keys.joinToString(", ")}")
        }

        return recommendations
    }
}

/**
 * Composable for tracking performance of child components.
 */
@Composable
fun PerformanceTracker(componentName: String, content: @Composable () -> Unit) {
    val startTime = remember { System.currentTimeMillis() }

    DisposableEffect(componentName) {
        onDispose {
            val renderTime = System.currentTimeMillis() - startTime
            PerformanceOptimizer.trackRenderTime(componentName, renderTime)
        }
    }

    content()
}

/**
 * Optimized result viewer with performance optimizations.
 */
@Composable
fun OptimizedResultViewer(result: com.campro.v5.models.OptimizationResult, modifier: Modifier = Modifier) {
    PerformanceTracker("OptimizedResultViewer") {
        // Use remember for expensive calculations
        val processedData = remember(result) {
            processResultData(result)
        }

        // Use derivedStateOf for computed values
        val summaryStats = remember {
            derivedStateOf {
                calculateSummaryStats(processedData)
            }
        }

        Column(
            modifier = modifier,
        ) {
            // Display summary stats
            SummaryStatsCard(stats = summaryStats.value)

            // Use LazyColumn for large datasets
            LazyColumn {
                items(processedData.items.size) { index ->
                    ResultItem(item = processedData.items[index])
                }
            }
        }
    }
}

/**
 * Process result data for display.
 */
private fun processResultData(result: com.campro.v5.models.OptimizationResult): ProcessedResultData = ProcessedResultData(
    items = listOf(
        ResultItem("Status", result.status),
        ResultItem("Execution Time", "${result.executionTime}s"),
        ResultItem("Motion Law Points", result.motionLaw.thetaDeg.size.toString()),
        ResultItem("Optimal Method", result.optimalProfiles.optimalMethod),
        ResultItem("Max Stress", "${result.feaAnalysis.maxStress} MPa"),
    ),
)

/**
 * Calculate summary statistics.
 */
private fun calculateSummaryStats(data: ProcessedResultData): SummaryStats = SummaryStats(
    totalItems = data.items.size,
    hasErrors = data.items.any { it.value.contains("error", ignoreCase = true) },
    averageValue = data.items.mapNotNull { it.value.toDoubleOrNull() }.average(),
)

/**
 * Data classes for processed results.
 */
private data class ProcessedResultData(val items: List<ResultItem>)

private data class ResultItem(val label: String, val value: String)

private data class SummaryStats(val totalItems: Int, val hasErrors: Boolean, val averageValue: Double)

/**
 * Summary stats card component.
 */
@Composable
private fun SummaryStatsCard(stats: SummaryStats, modifier: Modifier = Modifier) {
    Card(
        modifier = modifier,
        colors = CardDefaults.cardColors(
            containerColor = if (stats.hasErrors) {
                MaterialTheme.colorScheme.errorContainer
            } else {
                MaterialTheme.colorScheme.primaryContainer
            },
        ),
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
        ) {
            Text(
                text = "Summary",
                style = MaterialTheme.typography.titleMedium,
            )
            Text(
                text = "Items: ${stats.totalItems}",
                style = MaterialTheme.typography.bodyMedium,
            )
            if (stats.hasErrors) {
                Text(
                    text = "Errors detected",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.error,
                )
            }
        }
    }
}

/**
 * Result item component.
 */
@Composable
private fun ResultItem(item: ResultItem, modifier: Modifier = Modifier) {
    Card(
        modifier = modifier.padding(4.dp),
    ) {
        Row(
            modifier = Modifier.padding(12.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
        ) {
            Text(
                text = item.label,
                style = MaterialTheme.typography.bodyMedium,
            )
            Text(
                text = item.value,
                style = MaterialTheme.typography.bodyMedium,
                fontWeight = FontWeight.Bold,
            )
        }
    }
}

/**
 * Memory management utilities.
 */
object MemoryManager {

    private val logger = LoggerFactory.getLogger(MemoryManager::class.java)

    /**
     * Force garbage collection if memory usage is high.
     */
    fun optimizeMemoryIfNeeded() {
        val memoryUsage = PerformanceOptimizer.getMemoryUsage()
        val memoryUsageMB = memoryUsage / (1024 * 1024)

        if (memoryUsageMB > 150) {
            logger.info("High memory usage detected: ${memoryUsageMB}MB. Running garbage collection.")
            System.gc()

            val newMemoryUsage = PerformanceOptimizer.getMemoryUsage()
            val newMemoryUsageMB = newMemoryUsage / (1024 * 1024)
            logger.info("Memory usage after GC: ${newMemoryUsageMB}MB")
        }
    }

    /**
     * Clear caches and temporary data.
     */
    fun clearCaches() {
        // Clear any application caches
        logger.info("Clearing application caches")
        // Implementation would depend on specific cache implementations
    }

    /**
     * Get memory usage statistics.
     */
    fun getMemoryStats(): MemoryStats {
        val runtime = Runtime.getRuntime()
        return MemoryStats(
            totalMemory = runtime.totalMemory(),
            freeMemory = runtime.freeMemory(),
            usedMemory = runtime.totalMemory() - runtime.freeMemory(),
            maxMemory = runtime.maxMemory(),
        )
    }
}

/**
 * Memory statistics data class.
 */
data class MemoryStats(val totalMemory: Long, val freeMemory: Long, val usedMemory: Long, val maxMemory: Long) {
    val usedMemoryMB: Double
        get() = usedMemory / (1024.0 * 1024.0)

    val totalMemoryMB: Double
        get() = totalMemory / (1024.0 * 1024.0)

    val memoryUsagePercentage: Double
        get() = (usedMemory.toDouble() / totalMemory) * 100.0
}
