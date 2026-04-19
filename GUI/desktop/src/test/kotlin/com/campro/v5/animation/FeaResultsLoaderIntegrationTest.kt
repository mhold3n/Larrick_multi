package com.campro.v5.animation

import com.campro.v5.fea.FeaEngine
import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.io.TempDir
import java.io.File
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * Integration test for FEA results export and loading.
 * Tests the round-trip: Rust JNI export -> JSON file -> FeaResultsLoader parsing.
 */
class FeaResultsLoaderIntegrationTest {

    @Test
    fun `test FEA export and loader round-trip`(@TempDir tempDir: File) = runBlocking {
        // Create temporary files for the test
        val modelFile = File(tempDir, "test_model.json")

        // Write a minimal model file (just needs to exist for the JNI call)
        modelFile.writeText("""{"type": "test_model", "nodes": 4, "elements": 3}""")

        // Create FEA engine instance and run analysis via Rust JNI (this will generate realistic JSON)
        val feaEngine = FeaEngine()
        val resultsFile = feaEngine.runAnalysis(
            modelFile,
            emptyMap(), // No parameters needed for this test
        )

        // Verify the results file was created
        assertTrue(resultsFile.exists(), "FEA results file should be created")

        // Print the actual JSON content for debugging
        val jsonContent = resultsFile.readText()
        println("Generated JSON content:")
        println(jsonContent)

        // Load results using FeaResultsLoader
        val analysisData = FeaResultsLoader.loadResults(resultsFile)

        // Validate the loaded data structure
        assertNotNull(analysisData, "Analysis data should not be null")

        // Check displacements (should have 4 nodes based on JNI implementation)
        assertEquals(4, analysisData.displacements.size, "Should have 4 displacement entries")
        analysisData.displacements.forEach { (nodeId, displacement) ->
            assertTrue(nodeId in 1..4, "Node ID should be in range 1-4, got $nodeId")
            assertNotNull(displacement, "Displacement should not be null for node $nodeId")
            // Displacements should be small (microns) but non-zero
            assertTrue(
                displacement.x != 0f || displacement.y != 0f,
                "Displacement should be non-zero for node $nodeId",
            )
        }

        // Check stresses (should have 3 elements based on JNI implementation)
        assertEquals(3, analysisData.stresses.size, "Should have 3 stress entries")
        analysisData.stresses.forEach { (elementId, stress) ->
            assertTrue(elementId in 10..12, "Element ID should be in range 10-12, got $elementId")
            assertTrue(stress > 0f, "Stress should be positive for element $elementId")
            assertTrue(stress >= 100f, "Stress should be >= 100 MPa for element $elementId")
        }

        // Check time steps (should have ~90 steps based on JNI implementation)
        assertTrue(
            analysisData.timeSteps.size >= 80,
            "Should have at least 80 time steps, got ${analysisData.timeSteps.size}",
        )
        assertTrue(
            analysisData.timeSteps.size <= 100,
            "Should have at most 100 time steps, got ${analysisData.timeSteps.size}",
        )

        // Verify time steps are monotonically increasing
        for (i in 1 until analysisData.timeSteps.size) {
            assertTrue(
                analysisData.timeSteps[i] > analysisData.timeSteps[i - 1],
                "Time steps should be monotonically increasing",
            )
        }

        // Verify time step increment is approximately 0.01
        if (analysisData.timeSteps.size > 1) {
            val dt = analysisData.timeSteps[1] - analysisData.timeSteps[0]
            assertTrue(
                dt > 0.009f && dt < 0.011f,
                "Time step increment should be ~0.01, got $dt",
            )
        }
    }

    @Test
    fun `test stress analysis export and loader round-trip`(@TempDir tempDir: File) = runBlocking {
        // Create temporary files for the test
        val modelFile = File(tempDir, "stress_model.json")

        // Write a minimal model file
        modelFile.writeText("""{"type": "stress_model", "nodes": 2, "elements": 4}""")

        // Create FEA engine instance and run stress analysis via Rust JNI
        val feaEngine = FeaEngine()
        val resultsFile = feaEngine.runStressAnalysis(
            modelFile,
            emptyMap(),
        )

        // Verify the results file was created
        assertTrue(resultsFile.exists(), "Stress results file should be created")

        // Load results using FeaResultsLoader
        val analysisData = FeaResultsLoader.loadResults(resultsFile)

        // Validate the loaded data structure
        assertNotNull(analysisData, "Analysis data should not be null")

        // Check displacements (should have 2 nodes based on JNI implementation)
        assertEquals(2, analysisData.displacements.size, "Should have 2 displacement entries")

        // Check stresses (should have 4 elements based on JNI implementation)
        assertEquals(4, analysisData.stresses.size, "Should have 4 stress entries")
        analysisData.stresses.forEach { (elementId, stress) ->
            assertTrue(elementId in 21..24, "Element ID should be in range 21-24, got $elementId")
            assertTrue(stress > 0f, "Stress should be positive for element $elementId")
            assertTrue(stress >= 100f, "Stress should be >= 100 MPa for element $elementId")
        }

        // Check time steps (should have 3 steps based on JNI implementation)
        assertEquals(3, analysisData.timeSteps.size, "Should have 3 time steps")
        assertEquals(0.0f, analysisData.timeSteps[0], "First time step should be 0.0")
        assertEquals(0.01f, analysisData.timeSteps[1], "Second time step should be 0.01")
        assertEquals(0.02f, analysisData.timeSteps[2], "Third time step should be 0.02")
    }
}
