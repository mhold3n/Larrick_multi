package com.campro.v5.batch

import com.campro.v5.io.FileIOUtils
import com.campro.v5.io.ResultExporter
import com.campro.v5.models.OptimizationParameters
import com.campro.v5.models.OptimizationResult
import com.campro.v5.pipeline.UnifiedOptimizationBridge
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.sync.Semaphore
import org.slf4j.LoggerFactory
import java.nio.file.Path
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter

/**
 * Batch processor for running multiple optimization jobs.
 *
 * Handles parallel processing of multiple parameter sets with progress tracking,
 * error handling, and result aggregation.
 */
class BatchProcessor(private val bridge: UnifiedOptimizationBridge, private val maxConcurrentJobs: Int = 3) {

    private val logger = LoggerFactory.getLogger(BatchProcessor::class.java)

    private val _batchState = MutableStateFlow<BatchState>(BatchState.Idle)
    val batchState: StateFlow<BatchState> = _batchState.asStateFlow()

    private val _progress = MutableStateFlow(BatchProgress(0, 0, 0, 0))
    val progress: StateFlow<BatchProgress> = _progress.asStateFlow()

    private val _results = MutableStateFlow<List<BatchResult>>(emptyList())
    val results: StateFlow<List<BatchResult>> = _results.asStateFlow()

    private var currentBatchJob: Job? = null

    /**
     * Process batch of optimization jobs.
     */
    suspend fun processBatch(parameterSets: List<OptimizationParameters>, outputDir: Path, exportResults: Boolean = true) {
        // Cancel any existing batch
        currentBatchJob?.cancel()

        currentBatchJob = CoroutineScope(Dispatchers.Main).launch {
            try {
                logger.info("Starting batch processing with ${parameterSets.size} parameter sets")
                _batchState.value = BatchState.Running
                _progress.value = BatchProgress(0, parameterSets.size, 0, 0)
                _results.value = emptyList()

                val results = mutableListOf<BatchResult>()
                val semaphore = Semaphore(maxConcurrentJobs)

                // Process jobs in parallel with semaphore
                val jobs = parameterSets.mapIndexed { index, parameters ->
                    async {
                        semaphore.acquire()
                        try {
                            processSingleJob(index, parameters, outputDir, exportResults)
                        } finally {
                            semaphore.release()
                        }
                    }
                }

                // Collect results as they complete
                jobs.forEach { job ->
                    try {
                        val result = job.await()
                        results.add(result)

                        // Update progress
                        val currentProgress = _progress.value
                        val newProgress = when (result.status) {
                            BatchResultStatus.Success -> currentProgress.copy(
                                completed = currentProgress.completed + 1,
                            )
                            BatchResultStatus.Failed -> currentProgress.copy(
                                failed = currentProgress.failed + 1,
                            )
                        }
                        _progress.value = newProgress
                        _results.value = results.toList()
                    } catch (e: Exception) {
                        logger.error("Batch job failed", e)
                        val failedResult = BatchResult(
                            index = -1,
                            parameters = OptimizationParameters.createDefault(),
                            result = null,
                            status = BatchResultStatus.Failed,
                            error = e.message ?: "Unknown error",
                            executionTime = 0.0,
                        )
                        results.add(failedResult)

                        val currentProgress = _progress.value
                        _progress.value = currentProgress.copy(failed = currentProgress.failed + 1)
                        _results.value = results.toList()
                    }
                }

                // Batch completed
                _batchState.value = BatchState.Completed
                logger.info(
                    "Batch processing completed: ${results.count {
                        it.status == BatchResultStatus.Success
                    }} successful, ${results.count { it.status == BatchResultStatus.Failed }} failed",
                )
            } catch (e: CancellationException) {
                logger.info("Batch processing cancelled")
                _batchState.value = BatchState.Cancelled
                throw e
            } catch (e: Exception) {
                logger.error("Batch processing failed", e)
                _batchState.value = BatchState.Failed(e)
            }
        }

        currentBatchJob?.join()
    }

    /**
     * Process single optimization job.
     */
    private suspend fun processSingleJob(
        index: Int,
        parameters: OptimizationParameters,
        outputDir: Path,
        exportResults: Boolean,
    ): BatchResult {
        val startTime = System.currentTimeMillis()

        return try {
            logger.debug("Processing batch job $index")

            // Create job-specific output directory
            val jobOutputDir = outputDir.resolve("job_${index + 1}")
            FileIOUtils.ensureDirectoryExists(jobOutputDir)

            // Run optimization
            val result = bridge.runOptimization(parameters, jobOutputDir).get()

            // Export results if requested
            if (exportResults) {
                val exporter = ResultExporter()
                exporter.exportResult(
                    result = result,
                    parameters = parameters,
                    outputPath = jobOutputDir.resolve("results.json"),
                    format = ResultExporter.ExportFormat.JSON,
                )
            }

            val executionTime = (System.currentTimeMillis() - startTime) / 1000.0

            BatchResult(
                index = index,
                parameters = parameters,
                result = result,
                status = if (result.isSuccess()) BatchResultStatus.Success else BatchResultStatus.Failed,
                error = result.error,
                executionTime = executionTime,
            )
        } catch (e: Exception) {
            val executionTime = (System.currentTimeMillis() - startTime) / 1000.0
            logger.error("Batch job $index failed", e)

            BatchResult(
                index = index,
                parameters = parameters,
                result = null,
                status = BatchResultStatus.Failed,
                error = e.message ?: "Unknown error",
                executionTime = executionTime,
            )
        }
    }

    /**
     * Cancel current batch processing.
     */
    fun cancelBatch() {
        currentBatchJob?.cancel()
        _batchState.value = BatchState.Cancelled
        logger.info("Batch processing cancelled by user")
    }

    /**
     * Reset batch state.
     */
    fun resetBatch() {
        currentBatchJob?.cancel()
        _batchState.value = BatchState.Idle
        _progress.value = BatchProgress(0, 0, 0, 0)
        _results.value = emptyList()
    }

    /**
     * Get batch summary.
     */
    fun getBatchSummary(): BatchSummary {
        val currentResults = _results.value
        val successful = currentResults.count { it.status == BatchResultStatus.Success }
        val failed = currentResults.count { it.status == BatchResultStatus.Failed }
        val totalTime = currentResults.sumOf { it.executionTime }

        return BatchSummary(
            totalJobs = currentResults.size,
            successful = successful,
            failed = failed,
            totalExecutionTime = totalTime,
            averageExecutionTime = if (currentResults.isNotEmpty()) totalTime / currentResults.size else 0.0,
        )
    }

    /**
     * Export batch results.
     */
    suspend fun exportBatchResults(outputPath: Path, format: ResultExporter.ExportFormat = ResultExporter.ExportFormat.JSON): Boolean {
        return try {
            val currentResults = _results.value
            if (currentResults.isEmpty()) {
                logger.warn("No batch results to export")
                return false
            }

            val exporter = ResultExporter()
            val batchSummary = getBatchSummary()

            // Create batch export data
            val batchExportData = mapOf(
                "metadata" to mapOf(
                    "exportedAt" to LocalDateTime.now().format(DateTimeFormatter.ISO_LOCAL_DATE_TIME),
                    "batchSummary" to batchSummary,
                ),
                "results" to currentResults.map { batchResult ->
                    mapOf(
                        "index" to batchResult.index,
                        "status" to batchResult.status.name,
                        "executionTime" to batchResult.executionTime,
                        "error" to batchResult.error,
                        "parameters" to batchResult.parameters,
                        "result" to batchResult.result,
                    )
                },
            )

            // For now, export as JSON (can be extended for other formats)
            val json = com.google.gson.GsonBuilder().setPrettyPrinting().create().toJson(batchExportData)
            java.nio.file.Files.write(outputPath, json.toByteArray())

            logger.info("Exported batch results to: $outputPath")
            true
        } catch (e: Exception) {
            logger.error("Failed to export batch results", e)
            false
        }
    }
}

/**
 * Batch state enumeration.
 */
sealed class BatchState {
    object Idle : BatchState()
    object Running : BatchState()
    object Completed : BatchState()
    object Cancelled : BatchState()
    data class Failed(val error: Throwable) : BatchState()
}

/**
 * Batch progress data.
 */
data class BatchProgress(val completed: Int, val total: Int, val failed: Int, val running: Int) {
    val successRate: Double
        get() = if (total > 0) (completed.toDouble() / total) * 100 else 0.0

    val isComplete: Boolean
        get() = completed + failed >= total
}

/**
 * Batch result for individual job.
 */
data class BatchResult(
    val index: Int,
    val parameters: OptimizationParameters,
    val result: OptimizationResult?,
    val status: BatchResultStatus,
    val error: String?,
    val executionTime: Double,
)

/**
 * Batch result status.
 */
enum class BatchResultStatus {
    Success,
    Failed,
}

/**
 * Batch summary information.
 */
data class BatchSummary(
    val totalJobs: Int,
    val successful: Int,
    val failed: Int,
    val totalExecutionTime: Double,
    val averageExecutionTime: Double,
) {
    val successRate: Double
        get() = if (totalJobs > 0) (successful.toDouble() / totalJobs) * 100 else 0.0
}
