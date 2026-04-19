package com.campro.v5.fea

import com.campro.v5.emitError
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.io.File
import java.util.UUID
import java.util.concurrent.ConcurrentHashMap

/**
 * Manager for handling asynchronous FEA computations.
 * This class provides job scheduling, progress tracking, and cancellation support.
 */
class ComputationManager {
    private val scope = CoroutineScope(Dispatchers.Default + SupervisorJob())
    private val feaEngine: FeaEngine? by lazy {
        try {
            FeaEngine()
        } catch (e: Exception) {
            println("Failed to initialize FEA engine: ${e.message}")
            null
        }
    }
    private val jobs = ConcurrentHashMap<String, Job>()
    private val progressMap = ConcurrentHashMap<String, MutableStateFlow<Float>>()
    private val resultMap = ConcurrentHashMap<String, MutableStateFlow<ComputationResult?>>()

    /**
     * Start a new FEA computation.
     *
     * @param modelFile The model file to analyze
     * @param parameters The parameters for the analysis
     * @param type The type of analysis to run
     * @return The ID of the computation job
     */
    fun startComputation(modelFile: File, parameters: Map<String, String>, type: ComputationType): String {
        val jobId = UUID.randomUUID().toString()
        val progressFlow = MutableStateFlow(0f)
        val resultFlow = MutableStateFlow<ComputationResult?>(null)

        progressMap[jobId] = progressFlow
        resultMap[jobId] = resultFlow

        val job =
            scope.launch {
                try {
                    // Update progress to indicate job has started
                    progressFlow.value = 0.1f

                    // Check if FEA engine is available and functional
                    val engine = feaEngine
                    if (engine == null || !engine.isFunctional()) {
                        throw IllegalStateException("FEA engine not available - native library not loaded or not functional")
                    }

                    // Run the appropriate analysis based on the type
                    val resultsFile =
                        when (type) {
                            ComputationType.GENERAL -> engine.runAnalysis(modelFile, parameters)
                            ComputationType.STRESS -> engine.runStressAnalysis(modelFile, parameters)
                            ComputationType.VIBRATION -> engine.runVibrationAnalysis(modelFile, parameters)
                            ComputationType.MESH_GENERATION -> engine.generateMesh(modelFile, parameters)
                        }

                    // Update progress to indicate job has completed
                    progressFlow.value = 1.0f

                    // Set the result
                    resultFlow.value = ComputationResult.Success(resultsFile)
                } catch (e: CancellationException) {
                    // Job was cancelled
                    progressFlow.value = 0f
                    resultFlow.value = ComputationResult.Cancelled
                    throw e
                } catch (e: Exception) {
                    // Job failed
                    progressFlow.value = 0f
                    resultFlow.value = ComputationResult.Error(e.message ?: "Unknown error")
                    emitError("Computation failed: ${e.message}", "ComputationManager")
                } finally {
                    // Clean up
                    jobs.remove(jobId)
                }
            }

        jobs[jobId] = job
        return jobId
    }

    /**
     * Get the progress of a computation.
     *
     * @param jobId The ID of the computation job
     * @return A flow of progress values from 0.0 to 1.0, or null if the job doesn't exist
     */
    fun getProgress(jobId: String): StateFlow<Float>? = progressMap[jobId]?.asStateFlow()

    /**
     * Get the result of a computation.
     *
     * @param jobId The ID of the computation job
     * @return A flow of the computation result, or null if the job doesn't exist
     */
    fun getResult(jobId: String): StateFlow<ComputationResult?>? = resultMap[jobId]?.asStateFlow()

    /**
     * Cancel a computation.
     *
     * @param jobId The ID of the computation job
     * @return True if the job was cancelled, false if it doesn't exist or is already completed
     */
    fun cancelComputation(jobId: String): Boolean {
        val job = jobs[jobId] ?: return false
        if (job.isCompleted) return false

        job.cancel()
        return true
    }

    /**
     * Cancel all computations.
     */
    fun cancelAll() {
        jobs.forEach { (_, job) -> job.cancel() }
    }

    /**
     * Check if a computation is running.
     *
     * @param jobId The ID of the computation job
     * @return True if the job is running, false otherwise
     */
    fun isRunning(jobId: String): Boolean {
        val job = jobs[jobId] ?: return false
        return job.isActive
    }

    /**
     * Get all running computation job IDs.
     *
     * @return A list of job IDs for all running computations
     */
    fun getRunningJobs(): List<String> = jobs.filter { (_, job) -> job.isActive }.keys.toList()

    /**
     * Clean up resources when the manager is no longer needed.
     */
    fun shutdown() {
        cancelAll()
        scope.cancel()
    }
}

/**
 * Types of FEA computations.
 */
enum class ComputationType {
    GENERAL,
    STRESS,
    VIBRATION,
    MESH_GENERATION,
}

/**
 * Result of a computation.
 */
sealed class ComputationResult {
    /**
     * Computation completed successfully.
     *
     * @param resultsFile The file containing the computation results
     */
    data class Success(val resultsFile: File) : ComputationResult()

    /**
     * Computation failed with an error.
     *
     * @param message The error message
     */
    data class Error(val message: String) : ComputationResult()

    /**
     * Computation was cancelled.
     */
    object Cancelled : ComputationResult()
}

/**
 * Extension function to check if a computation result is successful.
 *
 * @return True if the result is a Success, false otherwise
 */
fun ComputationResult?.isSuccess(): Boolean = this is ComputationResult.Success

/**
 * Extension function to get the results file from a successful computation.
 *
 * @return The results file, or null if the computation was not successful
 */
fun ComputationResult?.getResultsFile(): File? = (this as? ComputationResult.Success)?.resultsFile

/**
 * Extension function to get the error message from a failed computation.
 *
 * @return The error message, or null if the computation did not fail
 */
fun ComputationResult?.getErrorMessage(): String? = (this as? ComputationResult.Error)?.message

/**
 * Extension function to check if a computation was cancelled.
 *
 * @return True if the computation was cancelled, false otherwise
 */
fun ComputationResult?.isCancelled(): Boolean = this is ComputationResult.Cancelled
