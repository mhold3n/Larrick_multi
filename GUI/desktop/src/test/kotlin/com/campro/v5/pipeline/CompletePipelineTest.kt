package com.campro.v5.pipeline

import com.campro.v5.models.OptimizationParameters
import kotlinx.coroutines.test.runTest
import org.junit.jupiter.api.*
import org.junit.jupiter.api.Assertions.*
import java.nio.file.Paths
import java.util.concurrent.TimeUnit

/**
 * Complete pipeline tests that validate the entire optimization workflow
 * from parameter input through result generation and export.
 *
 * Tests the complete data flow and integration between all components.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class CompletePipelineTest {

    private lateinit var bridge: UnifiedOptimizationBridge
    private lateinit var testOutputDir: java.nio.file.Path

    @BeforeEach
    fun setup() {
        bridge = UnifiedOptimizationBridge()
        testOutputDir = Paths.get("./test_pipeline_output_${System.currentTimeMillis()}")
        java.nio.file.Files.createDirectories(testOutputDir)
    }

    @AfterEach
    fun tearDown() {
        // Clean up test output directory
        try {
            java.nio.file.Files.walk(testOutputDir)
                .sorted(Comparator.reverseOrder())
                .forEach { path ->
                    java.nio.file.Files.deleteIfExists(path)
                }
        } catch (e: Exception) {
            // Ignore cleanup errors
        }
    }

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    fun `test complete optimization pipeline with default parameters`() = runTest {
        // Given
        val parameters = OptimizationParameters.createDefault()

        // When
        val result = bridge.runOptimization(parameters, testOutputDir).get()

        // Then
        assertNotNull(result)
        assertTrue(result.isSuccess())
        assertTrue(result.executionTime > 0)

        // Verify motion law data
        assertNotNull(result.motionLaw)
        assertTrue(result.motionLaw.thetaDeg.isNotEmpty())
        assertTrue(result.motionLaw.displacement.isNotEmpty())
        assertTrue(result.motionLaw.velocity.isNotEmpty())
        assertTrue(result.motionLaw.acceleration.isNotEmpty())

        // Verify gear profile data
        assertNotNull(result.optimalProfiles)
        assertTrue(result.optimalProfiles.rSun.isNotEmpty())
        assertTrue(result.optimalProfiles.rPlanet.isNotEmpty())
        assertTrue(result.optimalProfiles.rRingInner.isNotEmpty())
        assertTrue(result.optimalProfiles.gearRatio > 0)
        assertTrue(result.optimalProfiles.optimalMethod.isNotEmpty())

        // Verify FEA analysis data
        assertNotNull(result.feaAnalysis)
        assertTrue(result.feaAnalysis.maxStress > 0)
        assertTrue(result.feaAnalysis.naturalFrequencies.isNotEmpty())
        assertTrue(result.feaAnalysis.fatigueLife > 0)
        assertTrue(result.feaAnalysis.recommendations.isNotEmpty())
    }

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    fun `test complete optimization pipeline with quick test parameters`() = runTest {
        // Given
        val parameters = OptimizationParameters.createQuickTest()

        // When
        val result = bridge.runOptimization(parameters, testOutputDir).get()

        // Then
        assertNotNull(result)
        assertTrue(result.isSuccess())
        assertTrue(result.executionTime > 0)

        // Verify quick test produces valid results
        assertNotNull(result.motionLaw)
        assertNotNull(result.optimalProfiles)
        assertNotNull(result.feaAnalysis)

        // Verify results are different from default (if applicable)
        val defaultResult = bridge.runOptimization(
            OptimizationParameters.createDefault(),
            testOutputDir.resolve("default"),
        ).get()

        // Results should be valid even if different
        assertTrue(result.isSuccess())
        assertTrue(defaultResult.isSuccess())
    }

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    fun `test complete optimization pipeline with high performance parameters`() = runTest {
        // Given
        val parameters = OptimizationParameters.createHighPerformance()

        // When
        val result = bridge.runOptimization(parameters, testOutputDir).get()

        // Then
        assertNotNull(result)
        assertTrue(result.isSuccess())
        assertTrue(result.executionTime > 0)

        // Verify high performance produces valid results
        assertNotNull(result.motionLaw)
        assertNotNull(result.optimalProfiles)
        assertNotNull(result.feaAnalysis)

        // Verify results are meaningful
        assertTrue(result.motionLaw.thetaDeg.size > 0)
        assertTrue(result.optimalProfiles.rSun.size > 0)
        assertTrue(result.feaAnalysis.maxStress > 0)
    }

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    fun `test pipeline handles invalid parameters gracefully`() = runTest {
        // Given
        val invalidParameters = OptimizationParameters(
            samplingStepDeg = -1.0, // Invalid negative value
            strokeLengthMm = 0.0, // Invalid zero value
            gearRatio = 0.0, // Invalid zero value
            rpm = -100.0, // Invalid negative value
            planetCount = 0, // Invalid zero value
            rodLength = -50.0, // Invalid negative value
            journalRadius = 0.0, // Invalid zero value
            ringThickness = -10.0, // Invalid negative value
            interferenceBuffer = -5.0, // Invalid negative value
        )

        // When
        val result = bridge.runOptimization(invalidParameters, testOutputDir).get()

        // Then
        assertNotNull(result)
        // Should either succeed with corrected parameters or fail gracefully
        if (result.isSuccess()) {
            // If it succeeds, verify results are valid
            assertNotNull(result.motionLaw)
            assertNotNull(result.optimalProfiles)
            assertNotNull(result.feaAnalysis)
        } else {
            // If it fails, verify error is meaningful
            assertNotNull(result.error)
            assertTrue(result.error!!.isNotEmpty())
        }
    }

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    fun `test pipeline produces consistent results for same parameters`() = runTest {
        // Given
        val parameters = OptimizationParameters.createDefault()

        // When - Run optimization multiple times
        val result1 = bridge.runOptimization(parameters, testOutputDir.resolve("run1")).get()
        val result2 = bridge.runOptimization(parameters, testOutputDir.resolve("run2")).get()
        val result3 = bridge.runOptimization(parameters, testOutputDir.resolve("run3")).get()

        // Then
        assertNotNull(result1)
        assertNotNull(result2)
        assertNotNull(result3)

        // All runs should succeed
        assertTrue(result1.isSuccess())
        assertTrue(result2.isSuccess())
        assertTrue(result3.isSuccess())

        // Results should be consistent (within reasonable tolerance)
        assertEquals(result1.motionLaw.thetaDeg.size, result2.motionLaw.thetaDeg.size)
        assertEquals(result2.motionLaw.thetaDeg.size, result3.motionLaw.thetaDeg.size)

        assertEquals(result1.optimalProfiles.gearRatio, result2.optimalProfiles.gearRatio, 0.001)
        assertEquals(result2.optimalProfiles.gearRatio, result3.optimalProfiles.gearRatio, 0.001)

        assertEquals(result1.feaAnalysis.maxStress, result2.feaAnalysis.maxStress, 0.001)
        assertEquals(result2.feaAnalysis.maxStress, result3.feaAnalysis.maxStress, 0.001)
    }

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    fun `test pipeline handles edge case parameters`() = runTest {
        // Given - Edge case parameters
        val edgeCaseParameters = OptimizationParameters(
            samplingStepDeg = 0.1, // Very small step
            strokeLengthMm = 1000.0, // Very large stroke
            gearRatio = 10.0, // Very large gear ratio
            rpm = 10000.0, // Very high RPM
            planetCount = 8, // Many planets
            rodLength = 500.0, // Very long rod
            journalRadius = 100.0, // Large journal
            ringThickness = 50.0, // Thick ring
            interferenceBuffer = 1.0, // Small buffer
        )

        // When
        val result = bridge.runOptimization(edgeCaseParameters, testOutputDir).get()

        // Then
        assertNotNull(result)

        // Should either succeed or fail gracefully
        if (result.isSuccess()) {
            // If it succeeds, verify results are valid
            assertNotNull(result.motionLaw)
            assertNotNull(result.optimalProfiles)
            assertNotNull(result.feaAnalysis)

            // Verify results are within reasonable bounds
            assertTrue(result.motionLaw.thetaDeg.size > 0)
            assertTrue(result.optimalProfiles.gearRatio > 0)
            assertTrue(result.feaAnalysis.maxStress > 0)
        } else {
            // If it fails, verify error is meaningful
            assertNotNull(result.error)
            assertTrue(result.error!!.isNotEmpty())
        }
    }

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    fun `test pipeline performance with different parameter sets`() = runTest {
        // Given
        val parameterSets = listOf(
            OptimizationParameters.createDefault(),
            OptimizationParameters.createQuickTest(),
            OptimizationParameters.createHighPerformance(),
        )

        val executionTimes = mutableListOf<Double>()

        // When
        parameterSets.forEachIndexed { index, parameters ->
            val startTime = System.currentTimeMillis()
            val result = bridge.runOptimization(parameters, testOutputDir.resolve("perf_$index")).get()
            val endTime = System.currentTimeMillis()

            val executionTime = (endTime - startTime) / 1000.0
            executionTimes.add(executionTime)

            // Then
            assertNotNull(result)
            assertTrue(result.isSuccess())
            assertTrue(executionTime > 0)
        }

        // Verify performance is reasonable
        val averageExecutionTime = executionTimes.average()
        assertTrue(averageExecutionTime < 30.0, "Average execution time $averageExecutionTime seconds exceeds 30 second limit")

        // Verify all parameter sets complete successfully
        assertEquals(3, executionTimes.size)
        assertTrue(executionTimes.all { it > 0 })
    }

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    fun `test pipeline data integrity and validation`() = runTest {
        // Given
        val parameters = OptimizationParameters.createDefault()

        // When
        val result = bridge.runOptimization(parameters, testOutputDir).get()

        // Then
        assertNotNull(result)
        assertTrue(result.isSuccess())

        // Verify motion law data integrity
        val motionLaw = result.motionLaw
        assertEquals(motionLaw.thetaDeg.size, motionLaw.displacement.size)
        assertEquals(motionLaw.thetaDeg.size, motionLaw.velocity.size)
        assertEquals(motionLaw.thetaDeg.size, motionLaw.acceleration.size)

        // Verify gear profile data integrity
        val gearProfiles = result.optimalProfiles
        assertEquals(gearProfiles.rSun.size, gearProfiles.rPlanet.size)
        assertEquals(gearProfiles.rSun.size, gearProfiles.rRingInner.size)

        // Verify FEA analysis data integrity
        val feaAnalysis = result.feaAnalysis
        assertTrue(feaAnalysis.naturalFrequencies.isNotEmpty())
        assertTrue(feaAnalysis.recommendations.isNotEmpty())
        assertTrue(feaAnalysis.maxStress > 0)
        assertTrue(feaAnalysis.fatigueLife > 0)

        // Verify data consistency
        assertTrue(motionLaw.thetaDeg.all { it >= 0 && it <= 360 })
        assertTrue(gearProfiles.rSun.all { it > 0 })
        assertTrue(gearProfiles.rPlanet.all { it > 0 })
        assertTrue(gearProfiles.rRingInner.all { it > 0 })
        assertTrue(feaAnalysis.naturalFrequencies.all { it > 0 })
    }

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    fun `test pipeline error handling and recovery`() = runTest {
        // Given
        val parameters = OptimizationParameters.createDefault()

        // When - Test error handling
        val result = bridge.runOptimization(parameters, testOutputDir).get()

        // Then
        assertNotNull(result)

        // Should either succeed or fail gracefully
        if (result.isSuccess()) {
            // If it succeeds, verify results are valid
            assertNotNull(result.motionLaw)
            assertNotNull(result.optimalProfiles)
            assertNotNull(result.feaAnalysis)
        } else {
            // If it fails, verify error handling
            assertNotNull(result.error)
            assertTrue(result.error!!.isNotEmpty())

            // Verify error is recoverable
            val retryResult = bridge.runOptimization(parameters, testOutputDir.resolve("retry")).get()
            assertNotNull(retryResult)
        }
    }
}
