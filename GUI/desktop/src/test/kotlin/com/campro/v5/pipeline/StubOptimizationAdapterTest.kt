package com.campro.v5.pipeline

import com.campro.v5.models.OptimizationParameters
import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import java.nio.file.Files

/**
 * Contract tests for the GUI-facing stub optimization adapter.
 *
 * Intended behavior:
 * - Keep the UI stable by guaranteeing deterministic result shapes.
 *
 * Current behavior:
 * - Verifies status, array sizes, and fixture snapshot side effects.
 */
class StubOptimizationAdapterTest {
    private val adapter = StubOptimizationAdapter()

    @Test
    fun `stub adapter returns deterministic success payload`() =
        runBlocking {
            val outputDir = Files.createTempDirectory("campro-stub-test")
            val result = adapter.runOptimization(OptimizationParameters.createDefault(), outputDir)

            assertEquals("success", result.status)
            assertTrue(result.motionLaw.thetaDeg.isNotEmpty())
            assertEquals(result.motionLaw.thetaDeg.size, result.motionLaw.displacement.size)
            assertEquals(result.motionLaw.thetaDeg.size, result.optimalProfiles.rSun.size)
            assertTrue(Files.exists(outputDir.resolve("optimization_results.stub.json")))
        }
}
