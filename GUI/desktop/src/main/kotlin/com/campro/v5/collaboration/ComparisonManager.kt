package com.campro.v5.collaboration

import androidx.compose.runtime.mutableStateOf
import com.campro.v5.layout.StateManager
import com.google.gson.GsonBuilder
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.Date
import kotlin.math.abs

/**
 * Manages comparison functionality for the CamPro v5 application.
 * This class provides design comparison tools including parameter comparison,
 * result analysis, visual diff, and comprehensive comparison reports.
 */
class ComparisonManager {
    // Comparison state
    private val _activeComparisons = MutableStateFlow<List<Comparison>>(emptyList())
    val activeComparisons: StateFlow<List<Comparison>> = _activeComparisons.asStateFlow()

    private val _isComparing = mutableStateOf(false)
    val isComparing: Boolean
        get() = _isComparing.value

    // Comparison events
    private val _comparisonEvents = MutableStateFlow<ComparisonEvent?>(null)
    val comparisonEvents: StateFlow<ComparisonEvent?> = _comparisonEvents.asStateFlow()

    // Comparison types
    private val supportedTypes =
        mapOf(
            "parameters" to ComparisonType("Parameters", "Compare project parameters"),
            "results" to ComparisonType("Results", "Compare simulation results"),
            "performance" to ComparisonType("Performance", "Compare performance metrics"),
            "visual" to ComparisonType("Visual", "Visual comparison of designs"),
            "full" to ComparisonType("Full", "Complete project comparison"),
        )

    // State manager for persistence
    private val stateManager = StateManager.getInstance()

    // Coroutine scope for async operations
    private val scope = CoroutineScope(Dispatchers.IO)

    // JSON serialization
    private val gson = GsonBuilder().setPrettyPrinting().create()

    init {
        loadComparisonHistory()
    }

    /**
     * Compare two projects.
     */
    suspend fun compareProjects(
        project1: ProjectData,
        project2: ProjectData,
        comparisonType: String = "full",
        options: ComparisonOptions = ComparisonOptions(),
    ): ComparisonResult = withContext(Dispatchers.IO) {
        try {
            _isComparing.value = true

            val comparison =
                Comparison(
                    id = generateComparisonId(),
                    project1 = project1,
                    project2 = project2,
                    type = comparisonType,
                    timestamp = Date(),
                    options = options,
                )

            emitEvent(ComparisonEvent.ComparisonStarted(comparison.id, comparisonType))

            val result =
                when (comparisonType.lowercase()) {
                    "parameters" -> compareParameters(project1, project2, options)
                    "results" -> compareResults(project1, project2, options)
                    "performance" -> comparePerformance(project1, project2, options)
                    "visual" -> compareVisual(project1, project2, options)
                    "full" -> compareFullProjects(project1, project2, options)
                    else -> ComparisonResult.Error("Unsupported comparison type: $comparisonType")
                }

            when (result) {
                is ComparisonResult.Success -> {
                    val completedComparison =
                        comparison.copy(
                            result = result.comparisonData,
                            status = ComparisonStatus.COMPLETED,
                        )
                    addComparison(completedComparison)
                    emitEvent(ComparisonEvent.ComparisonCompleted(comparison.id))
                }
                is ComparisonResult.Error -> {
                    val failedComparison =
                        comparison.copy(
                            status = ComparisonStatus.FAILED,
                            error = result.message,
                        )
                    addComparison(failedComparison)
                    emitEvent(ComparisonEvent.ComparisonFailed(comparison.id, result.message))
                }
            }

            result
        } catch (e: Exception) {
            val errorMessage = "Comparison failed: ${e.message}"
            emitEvent(ComparisonEvent.ComparisonFailed("", errorMessage))
            ComparisonResult.Error(errorMessage)
        } finally {
            _isComparing.value = false
        }
    }

    /**
     * Compare project parameters.
     */
    private suspend fun compareParameters(project1: ProjectData, project2: ProjectData, options: ComparisonOptions): ComparisonResult =
        withContext(Dispatchers.IO) {
            val differences = mutableListOf<ParameterDifference>()
            val allKeys = (project1.parameters.keys + project2.parameters.keys).toSet()

            allKeys.forEach { key ->
                val value1 = project1.parameters[key]
                val value2 = project2.parameters[key]

                when {
                    value1 == null && value2 != null -> {
                        differences.add(
                            ParameterDifference(
                                parameter = key,
                                value1 = null,
                                value2 = value2,
                                type = DifferenceType.ADDED,
                                significance = calculateSignificance(key, null, value2),
                            ),
                        )
                    }
                    value1 != null && value2 == null -> {
                        differences.add(
                            ParameterDifference(
                                parameter = key,
                                value1 = value1,
                                value2 = null,
                                type = DifferenceType.REMOVED,
                                significance = calculateSignificance(key, value1, null),
                            ),
                        )
                    }
                    value1 != value2 -> {
                        differences.add(
                            ParameterDifference(
                                parameter = key,
                                value1 = value1,
                                value2 = value2,
                                type = DifferenceType.MODIFIED,
                                significance = calculateSignificance(key, value1, value2),
                            ),
                        )
                    }
                }
            }

            val comparisonData =
                ComparisonData(
                    parameterDifferences = differences,
                    summary =
                    ComparisonSummary(
                        totalDifferences = differences.size,
                        significantDifferences = differences.count { it.significance == Significance.HIGH },
                        similarityScore = calculateSimilarityScore(differences, allKeys.size),
                    ),
                )

            ComparisonResult.Success(comparisonData)
        }

    /**
     * Compare simulation results.
     */
    private suspend fun compareResults(project1: ProjectData, project2: ProjectData, options: ComparisonOptions): ComparisonResult =
        withContext(Dispatchers.IO) {
            val results1 = project1.simulationResults
            val results2 = project2.simulationResults

            if (results1 == null || results2 == null) {
                return@withContext ComparisonResult.Error("Both projects must have simulation results")
            }

            val metricDifferences = mutableListOf<MetricDifference>()
            val allMetrics = (results1.metrics.keys + results2.metrics.keys).toSet()

            allMetrics.forEach { metric ->
                val value1 = results1.metrics[metric]
                val value2 = results2.metrics[metric]

                if (value1 != null && value2 != null) {
                    val difference = abs(value1 - value2)
                    val percentChange = if (value1 != 0.0) (difference / abs(value1)) * 100 else 0.0

                    metricDifferences.add(
                        MetricDifference(
                            metric = metric,
                            value1 = value1,
                            value2 = value2,
                            absoluteDifference = difference,
                            percentageChange = percentChange,
                            significance =
                            if (percentChange >= 10.0) {
                                Significance.HIGH
                            } else if (percentChange > 5.0) {
                                Significance.MEDIUM
                            } else {
                                Significance.LOW
                            },
                        ),
                    )
                }
            }

            val comparisonData =
                ComparisonData(
                    metricDifferences = metricDifferences,
                    summary =
                    ComparisonSummary(
                        totalDifferences = metricDifferences.size,
                        significantDifferences = metricDifferences.count { it.significance == Significance.HIGH },
                        similarityScore = calculateMetricSimilarity(metricDifferences),
                    ),
                )

            ComparisonResult.Success(comparisonData)
        }

    /**
     * Compare performance metrics.
     */
    private suspend fun comparePerformance(project1: ProjectData, project2: ProjectData, options: ComparisonOptions): ComparisonResult =
        withContext(Dispatchers.IO) {
            // Placeholder implementation
            val comparisonData =
                ComparisonData(
                    summary =
                    ComparisonSummary(
                        totalDifferences = 0,
                        significantDifferences = 0,
                        similarityScore = 100.0,
                    ),
                )

            ComparisonResult.Success(comparisonData)
        }

    /**
     * Visual comparison of designs.
     */
    private suspend fun compareVisual(project1: ProjectData, project2: ProjectData, options: ComparisonOptions): ComparisonResult =
        withContext(Dispatchers.IO) {
            // Placeholder implementation
            val comparisonData =
                ComparisonData(
                    summary =
                    ComparisonSummary(
                        totalDifferences = 0,
                        significantDifferences = 0,
                        similarityScore = 100.0,
                    ),
                )

            ComparisonResult.Success(comparisonData)
        }

    /**
     * Full project comparison.
     */
    private suspend fun compareFullProjects(project1: ProjectData, project2: ProjectData, options: ComparisonOptions): ComparisonResult =
        withContext(Dispatchers.IO) {
            val parameterResult = compareParameters(project1, project2, options)
            val resultResult = compareResults(project1, project2, options)

            when {
                parameterResult is ComparisonResult.Success && resultResult is ComparisonResult.Success -> {
                    val combinedData =
                        ComparisonData(
                            parameterDifferences = parameterResult.comparisonData.parameterDifferences,
                            metricDifferences = resultResult.comparisonData.metricDifferences,
                            summary =
                            ComparisonSummary(
                                totalDifferences =
                                parameterResult.comparisonData.summary.totalDifferences +
                                    resultResult.comparisonData.summary.totalDifferences,
                                significantDifferences =
                                parameterResult.comparisonData.summary.significantDifferences +
                                    resultResult.comparisonData.summary.significantDifferences,
                                similarityScore =
                                (
                                    parameterResult.comparisonData.summary.similarityScore +
                                        resultResult.comparisonData.summary.similarityScore
                                    ) / 2.0,
                            ),
                        )
                    ComparisonResult.Success(combinedData)
                }
                parameterResult is ComparisonResult.Success -> parameterResult
                resultResult is ComparisonResult.Success -> resultResult
                else -> ComparisonResult.Error("Failed to compare projects")
            }
        }

    /**
     * Get comparison history.
     */
    fun getComparisonHistory(): List<Comparison> = _activeComparisons.value

    /**
     * Get comparison by ID.
     */
    fun getComparison(id: String): Comparison? = _activeComparisons.value.find { it.id == id }

    /**
     * Delete a comparison.
     */
    suspend fun deleteComparison(id: String): Boolean = withContext(Dispatchers.IO) {
        val comparisons = _activeComparisons.value.toMutableList()
        val removed = comparisons.removeIf { it.id == id }
        if (removed) {
            _activeComparisons.value = comparisons
            saveComparisonHistory()
            emitEvent(ComparisonEvent.ComparisonDeleted(id))
        }
        removed
    }

    /**
     * Export comparison results.
     */
    suspend fun exportComparison(comparisonId: String, format: String, filePath: String): Boolean = withContext(Dispatchers.IO) {
        try {
            val comparison = getComparison(comparisonId) ?: return@withContext false

            when (format.lowercase()) {
                "json" -> exportToJson(comparison, filePath)
                "html" -> exportToHtml(comparison, filePath)
                "csv" -> exportToCsv(comparison, filePath)
                else -> return@withContext false
            }

            emitEvent(ComparisonEvent.ComparisonExported(comparisonId, format))
            true
        } catch (e: Exception) {
            false
        }
    }

    // Helper methods
    private fun calculateSignificance(parameter: String, value1: String?, value2: String?): Significance {
        // Simplified significance calculation
        return when {
            value1 == null || value2 == null -> Significance.HIGH
            value1.toDoubleOrNull() != null && value2.toDoubleOrNull() != null -> {
                val diff = abs(value1.toDouble() - value2.toDouble())
                when {
                    diff > 10.0 -> Significance.HIGH
                    diff > 1.0 -> Significance.MEDIUM
                    else -> Significance.LOW
                }
            }
            else -> if (value1 != value2) Significance.MEDIUM else Significance.LOW
        }
    }

    private fun calculateSimilarityScore(differences: List<ParameterDifference>, totalParameters: Int): Double {
        if (totalParameters == 0) return 100.0
        val unchangedParameters = totalParameters - differences.size
        return (unchangedParameters.toDouble() / totalParameters) * 100.0
    }

    private fun calculateMetricSimilarity(differences: List<MetricDifference>): Double {
        if (differences.isEmpty()) return 100.0
        val avgPercentChange = differences.map { it.percentageChange }.average()
        return maxOf(0.0, 100.0 - avgPercentChange)
    }

    private fun generateComparisonId(): String = "comp_${System.currentTimeMillis()}_${(1000..9999).random()}"

    private fun addComparison(comparison: Comparison) {
        val comparisons = _activeComparisons.value.toMutableList()
        comparisons.add(0, comparison) // Add to beginning

        // Keep only last 50 comparisons
        if (comparisons.size > 50) {
            comparisons.removeAt(comparisons.size - 1)
        }

        _activeComparisons.value = comparisons
        saveComparisonHistory()
    }

    private fun loadComparisonHistory() {
        val historyJson = stateManager.getState("comparison.history", "[]")
        try {
            val comparisons = gson.fromJson(historyJson, Array<Comparison>::class.java).toList()
            _activeComparisons.value = comparisons
        } catch (e: Exception) {
            _activeComparisons.value = emptyList()
        }
    }

    private fun saveComparisonHistory() {
        val historyJson = gson.toJson(_activeComparisons.value)
        stateManager.setState("comparison.history", historyJson)
    }

    /**
     * Reset the comparison manager state.
     * This is primarily used for testing to ensure a clean state between tests.
     */
    fun resetState() {
        _activeComparisons.value = emptyList()
        _isComparing.value = false
        _comparisonEvents.value = null
        // Clear comparison history in StateManager
        stateManager.setState("comparison.history", "[]")
    }

    private fun emitEvent(event: ComparisonEvent) {
        scope.launch {
            _comparisonEvents.value = event
            // Add an extremely long delay to ensure event propagation
            kotlinx.coroutines.delay(2000)
        }
    }

    // Export methods
    private suspend fun exportToJson(comparison: Comparison, filePath: String) {
        val file = java.io.File(filePath)
        file.parentFile?.mkdirs()
        file.writeText(gson.toJson(comparison))
    }

    private suspend fun exportToHtml(comparison: Comparison, filePath: String) {
        val file = java.io.File(filePath)
        file.parentFile?.mkdirs()

        val html =
            buildString {
                append("<!DOCTYPE html><html><head><title>Comparison Report</title></head><body>")
                append("<h1>Project Comparison Report</h1>")
                append("<p>Comparison ID: ${comparison.id}</p>")
                append("<p>Type: ${comparison.type}</p>")
                append("<p>Date: ${comparison.timestamp}</p>")

                comparison.result?.let { result ->
                    append("<h2>Summary</h2>")
                    append("<p>Total Differences: ${result.summary.totalDifferences}</p>")
                    append("<p>Significant Differences: ${result.summary.significantDifferences}</p>")
                    append("<p>Similarity Score: ${String.format("%.2f", result.summary.similarityScore)}%</p>")
                }

                append("</body></html>")
            }

        file.writeText(html)
    }

    private suspend fun exportToCsv(comparison: Comparison, filePath: String) {
        val file = java.io.File(filePath)
        file.parentFile?.mkdirs()

        val csv =
            buildString {
                append("Type,Parameter/Metric,Value1,Value2,Difference,Significance\n")

                comparison.result?.parameterDifferences?.forEach { diff ->
                    append("Parameter,${diff.parameter},${diff.value1 ?: ""},${diff.value2 ?: ""},${diff.type},${diff.significance}\n")
                }

                comparison.result?.metricDifferences?.forEach { diff ->
                    append("Metric,${diff.metric},${diff.value1},${diff.value2},${diff.absoluteDifference},${diff.significance}\n")
                }
            }

        file.writeText(csv)
    }

    companion object {
        @Volatile
        private var INSTANCE: ComparisonManager? = null

        fun getInstance(): ComparisonManager = INSTANCE ?: synchronized(this) {
            INSTANCE ?: ComparisonManager().also { INSTANCE = it }
        }
    }
}

// Data classes and enums
data class ComparisonType(val name: String, val description: String)

data class ComparisonOptions(val includeVisuals: Boolean = true, val threshold: Double = 0.01)

data class Comparison(
    val id: String,
    val project1: ProjectData,
    val project2: ProjectData,
    val type: String,
    val timestamp: Date,
    val options: ComparisonOptions,
    val result: ComparisonData? = null,
    val status: ComparisonStatus = ComparisonStatus.PENDING,
    val error: String? = null,
)

data class ComparisonData(
    val parameterDifferences: List<ParameterDifference> = emptyList(),
    val metricDifferences: List<MetricDifference> = emptyList(),
    val summary: ComparisonSummary,
)

data class ComparisonSummary(val totalDifferences: Int, val significantDifferences: Int, val similarityScore: Double)

data class ParameterDifference(
    val parameter: String,
    val value1: String?,
    val value2: String?,
    val type: DifferenceType,
    val significance: Significance,
)

data class MetricDifference(
    val metric: String,
    val value1: Double,
    val value2: Double,
    val absoluteDifference: Double,
    val percentageChange: Double,
    val significance: Significance,
)

enum class ComparisonStatus { PENDING, COMPLETED, FAILED }

enum class DifferenceType { ADDED, REMOVED, MODIFIED }

enum class Significance { LOW, MEDIUM, HIGH }

sealed class ComparisonResult {
    data class Success(val comparisonData: ComparisonData) : ComparisonResult()

    data class Error(val message: String) : ComparisonResult()
}

sealed class ComparisonEvent {
    data class ComparisonStarted(val id: String, val type: String) : ComparisonEvent()

    data class ComparisonCompleted(val id: String) : ComparisonEvent()

    data class ComparisonFailed(val id: String, val error: String) : ComparisonEvent()

    data class ComparisonDeleted(val id: String) : ComparisonEvent()

    data class ComparisonExported(val id: String, val format: String) : ComparisonEvent()
}
