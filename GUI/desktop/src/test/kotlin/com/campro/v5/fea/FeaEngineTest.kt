package com.campro.v5.fea

import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Disabled
import org.junit.jupiter.api.Test
import java.io.File
import kotlin.test.assertNotNull

/**
 * Tests for the Rust FEA Engine integration.
 *
 * Note: These tests require the Rust FEA Engine to be available.
 * If the engine is not available, the tests will be skipped.
 */
class FeaEngineTest {
    private lateinit var feaEngine: FeaEngine
    private lateinit var computationManager: ComputationManager
    private lateinit var dataTransfer: DataTransfer
    private lateinit var errorHandler: ErrorHandler
    private lateinit var testModelFile: File

    @BeforeEach
    fun setUp() {
        // Create test model file
        testModelFile = createTestModelFile()

        // Initialize components
        feaEngine = FeaEngine()
        computationManager = ComputationManager()
        dataTransfer = DataTransfer()
        errorHandler = ErrorHandler()

        // Print diagnostic information
        println("[DEBUG_LOG] Test setup complete")
        println("[DEBUG_LOG] Test model file: ${testModelFile.absolutePath}")
    }

    @AfterEach
    fun tearDown() {
        // Clean up test model file
        if (testModelFile.exists()) {
            testModelFile.delete()
        }

        // Shut down computation manager
        computationManager.shutdown()

        println("[DEBUG_LOG] Test teardown complete")
    }

    /**
     * Create a test model file with sample data.
     */
    private fun createTestModelFile(): File {
        val modelFile = File.createTempFile("test_model_", ".json")
        modelFile.deleteOnExit()

        // Create a simple model with a single element
        val modelJson =
            """
            {
                "nodes": [
                    {"id": 1, "x": 0.0, "y": 0.0, "z": 0.0},
                    {"id": 2, "x": 1.0, "y": 0.0, "z": 0.0},
                    {"id": 3, "x": 1.0, "y": 1.0, "z": 0.0},
                    {"id": 4, "x": 0.0, "y": 1.0, "z": 0.0}
                ],
                "elements": [
                    {"id": 1, "type": "quad", "nodeIds": [1, 2, 3, 4], "materialId": 1}
                ],
                "materials": [
                    {"id": 1, "name": "Steel", "youngsModulus": 210000.0, "poissonsRatio": 0.3, "density": 7850.0}
                ],
                "boundaries": [
                    {"id": 1, "type": "fixed", "nodeIds": [1, 4], "values": {"x": 0.0, "y": 0.0, "z": 0.0}},
                    {"id": 2, "type": "force", "nodeIds": [2, 3], "values": {"x": 1000.0, "y": 0.0, "z": 0.0}}
                ]
            }
            """.trimIndent()

        modelFile.writeText(modelJson)
        return modelFile
    }

    /**
     * Test if the FEA engine is available.
     * This test is used to determine if the other tests should be run.
     */
    @Test
    fun testFeaEngineAvailability() {
        val available = feaEngine.isAvailable()
        println("[DEBUG_LOG] FEA engine available: $available")

        // If the engine is not available, print a warning but don't fail the test
        if (!available) {
            println("[DEBUG_LOG] WARNING: FEA engine is not available. Tests will be skipped.")
        }
    }

    /**
     * Test running a simple analysis.
     * This test is disabled by default and will only run if the FEA engine is available.
     */
    @Test
    @Disabled("Requires Rust FEA Engine")
    fun testRunAnalysis() = runBlocking {
        // Skip test if FEA engine is not available
        if (!feaEngine.isAvailable()) {
            println("[DEBUG_LOG] Skipping testRunAnalysis: FEA engine not available")
            return@runBlocking
        }

        // Set up test parameters
        val parameters =
            mapOf(
                "analysis_type" to "static",
                "solver" to "direct",
                "max_iterations" to "100",
                "tolerance" to "1e-6",
            )

        // Run analysis
        println("[DEBUG_LOG] Running analysis...")
        val resultsFile = feaEngine.runAnalysis(testModelFile, parameters)

        // Verify results
        assertTrue(resultsFile.exists(), "Results file should exist")
        assertTrue(resultsFile.length() > 0, "Results file should not be empty")

        println("[DEBUG_LOG] Analysis complete. Results file: ${resultsFile.absolutePath}")
    }

    /**
     * Test the computation manager.
     * This test is disabled by default and will only run if the FEA engine is available.
     */
    @Test
    @Disabled("Requires Rust FEA Engine")
    fun testComputationManager() = runBlocking {
        // Skip test if FEA engine is not available
        if (!feaEngine.isAvailable()) {
            println("[DEBUG_LOG] Skipping testComputationManager: FEA engine not available")
            return@runBlocking
        }

        // Set up test parameters
        val parameters =
            mapOf(
                "analysis_type" to "static",
                "solver" to "direct",
                "max_iterations" to "100",
                "tolerance" to "1e-6",
            )

        // Start computation
        println("[DEBUG_LOG] Starting computation...")
        val jobId =
            computationManager.startComputation(
                testModelFile,
                parameters,
                ComputationType.GENERAL,
            )

        // Wait for computation to complete
        var result: ComputationResult? = null
        val maxWaitTime = 30000L // 30 seconds
        val startTime = System.currentTimeMillis()

        while (System.currentTimeMillis() - startTime < maxWaitTime) {
            val progress = computationManager.getProgress(jobId)?.value ?: 0f
            println("[DEBUG_LOG] Computation progress: $progress")

            result = computationManager.getResult(jobId)?.value
            if (result != null) {
                break
            }

            kotlinx.coroutines.delay(1000) // Wait 1 second
        }

        // Verify result
        assertNotNull(result, "Computation result should not be null")
        assertTrue(result!!.isSuccess(), "Computation should succeed")

        val resultsFile = result!!.getResultsFile()
        assertNotNull(resultsFile, "Results file should not be null")
        assertTrue(resultsFile!!.exists(), "Results file should exist")
        assertTrue(resultsFile.length() > 0, "Results file should not be empty")

        println("[DEBUG_LOG] Computation complete. Results file: ${resultsFile.absolutePath}")
    }

    /**
     * Test data transfer between Kotlin and Rust.
     */
    @Test
    fun testDataTransfer() = runBlocking {
        // Create test data
        val testData =
            mapOf(
                "key1" to "value1",
                "key2" to 123,
                "key3" to listOf(1, 2, 3),
            )

        // Test transferToRust
        println("[DEBUG_LOG] Testing transferToRust...")
        val file1 = dataTransfer.transferToRust(testData)
        assertTrue(file1.exists(), "File should exist")
        assertTrue(file1.length() > 0, "File should not be empty")

        // Test transferLargeDataToRust
        println("[DEBUG_LOG] Testing transferLargeDataToRust...")
        val file2 = dataTransfer.transferLargeDataToRust(testData)
        assertTrue(file2.exists(), "File should exist")
        assertTrue(file2.length() > 0, "File should not be empty")

        // Test caching
        println("[DEBUG_LOG] Testing caching...")
        val cacheFile = dataTransfer.createCacheFile(testData, "test_cache")
        assertTrue(cacheFile.exists(), "Cache file should exist")

        val cachedData = dataTransfer.readFromCache("test_cache", Map::class.java)
        assertNotNull(cachedData, "Cached data should not be null")

        // Clean up
        dataTransfer.clearCache()

        println("[DEBUG_LOG] Data transfer tests complete")
    }

    /**
     * Test error handling.
     */
    @Test
    fun testErrorHandling() = runBlocking {
        // Create test error
        println("[DEBUG_LOG] Creating test error...")
        val error =
            errorHandler.createError(
                ErrorType.COMPUTATION,
                "Test error message",
                Exception("Test exception"),
                mapOf("key" to "value"),
            )

        // Verify error
        assertEquals(ErrorType.COMPUTATION, error.type, "Error type should match")
        assertEquals("Test error message", error.message, "Error message should match")
        assertNotNull(error.cause, "Error cause should not be null")
        assertEquals("Test exception", error.cause?.message, "Error cause message should match")
        assertEquals("value", error.data["key"], "Error data should match")

        // Test error handling
        println("[DEBUG_LOG] Testing error handling...")
        val recoverySuccessful = errorHandler.handleError(error)
        assertFalse(recoverySuccessful, "Recovery should not be successful for test error")

        // Test diagnostics
        println("[DEBUG_LOG] Running diagnostics...")
        val diagnostics = FeaDiagnostics()
        val report = diagnostics.runDiagnostics()

        println("[DEBUG_LOG] Diagnostic report:")
        println(report.getSummary())

        // Verify report
        assertNotNull(report.feaEngineVersion, "FEA engine version should not be null")
        assertTrue(report.maxMemory > 0, "Max memory should be positive")

        println("[DEBUG_LOG] Error handling tests complete")
    }
}
