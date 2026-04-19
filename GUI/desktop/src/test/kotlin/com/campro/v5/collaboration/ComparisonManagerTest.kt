package com.campro.v5.collaboration

import com.campro.v5.waitForConditionOrFail
import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertNotNull
import org.junit.jupiter.api.Assertions.assertNull
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.io.TempDir
import java.io.File
import java.nio.file.Path
import java.util.Date

class ComparisonManagerTest {
    private lateinit var comparisonManager: ComparisonManager
    private lateinit var project1: ProjectData
    private lateinit var project2: ProjectData
    private lateinit var project3: ProjectData

    @TempDir
    lateinit var tempDir: Path

    @BeforeEach
    fun setUp() {
        comparisonManager = ComparisonManager.getInstance()
        comparisonManager.resetState()

        // Create test project 1
        project1 =
            ProjectData(
                metadata =
                ProjectMetadata(
                    name = "Project Alpha",
                    description = "First test project",
                    timestamp = Date(),
                    version = "1.0",
                    author = "Test User",
                ),
                parameters =
                mapOf(
                    "Piston Diameter" to "70.0",
                    "Stroke" to "20.0",
                    "Rod Length" to "40.0",
                    "TDC Offset" to "40.0",
                    "Cycle Ratio" to "2.0",
                ),
                simulationResults =
                SimulationResults(
                    parameters =
                    mapOf(
                        "Piston Diameter" to "70.0",
                        "Stroke" to "20.0",
                    ),
                    metrics =
                    mapOf(
                        "Max Force" to 1500.0,
                        "Max Velocity" to 25.0,
                        "Efficiency" to 0.85,
                        "Power Output" to 120.0,
                    ),
                ),
            )

        // Create test project 2 (similar to project1 with some differences)
        project2 =
            ProjectData(
                metadata =
                ProjectMetadata(
                    name = "Project Beta",
                    description = "Second test project",
                    timestamp = Date(),
                    version = "1.1",
                    author = "Test User",
                ),
                parameters =
                mapOf(
                    "Piston Diameter" to "75.0", // Different
                    "Stroke" to "20.0", // Same
                    "Rod Length" to "45.0", // Different
                    "TDC Offset" to "40.0", // Same
                    "Cycle Ratio" to "2.5", // Different
                    "New Parameter" to "10.0", // Added
                ),
                simulationResults =
                SimulationResults(
                    parameters =
                    mapOf(
                        "Piston Diameter" to "75.0",
                        "Stroke" to "20.0",
                    ),
                    metrics =
                    mapOf(
                        "Max Force" to 1650.0, // 10% increase
                        "Max Velocity" to 23.0, // 8% decrease
                        "Efficiency" to 0.88, // 3.5% increase
                        "Power Output" to 135.0, // 12.5% increase
                        "New Metric" to 50.0, // Added
                    ),
                ),
            )

        // Create test project 3 (significantly different)
        project3 =
            ProjectData(
                metadata =
                ProjectMetadata(
                    name = "Project Gamma",
                    description = "Third test project",
                    timestamp = Date(),
                    version = "2.0",
                    author = "Different User",
                ),
                parameters =
                mapOf(
                    "Piston Diameter" to "60.0",
                    "Stroke" to "15.0",
                    "Rod Length" to "35.0",
                    "TDC Offset" to "30.0",
                    "Cycle Ratio" to "1.5",
                ),
                simulationResults =
                SimulationResults(
                    parameters =
                    mapOf(
                        "Piston Diameter" to "60.0",
                        "Stroke" to "15.0",
                    ),
                    metrics =
                    mapOf(
                        "Max Force" to 1200.0,
                        "Max Velocity" to 30.0,
                        "Efficiency" to 0.75,
                        "Power Output" to 90.0,
                    ),
                ),
            )
    }

    @AfterEach
    fun tearDown() {
        // Clean up any test data
    }

    @Test
    fun `test compare project parameters`() = runBlocking {
        val result =
            comparisonManager.compareProjects(
                project1,
                project2,
                comparisonType = "parameters",
            )

        assertTrue(result is ComparisonResult.Success)
        val successResult = result as ComparisonResult.Success
        val comparisonData = successResult.comparisonData

        // Should have differences for changed parameters and added parameter
        assertTrue(comparisonData.parameterDifferences.isNotEmpty())

        // Check specific differences
        val pistonDiameterDiff = comparisonData.parameterDifferences.find { it.parameter == "Piston Diameter" }
        assertNotNull(pistonDiameterDiff)
        assertEquals("70.0", pistonDiameterDiff!!.value1)
        assertEquals("75.0", pistonDiameterDiff.value2)
        assertEquals(DifferenceType.MODIFIED, pistonDiameterDiff.type)

        val newParameterDiff = comparisonData.parameterDifferences.find { it.parameter == "New Parameter" }
        assertNotNull(newParameterDiff)
        assertNull(newParameterDiff!!.value1)
        assertEquals("10.0", newParameterDiff.value2)
        assertEquals(DifferenceType.ADDED, newParameterDiff.type)

        // Check summary
        assertTrue(comparisonData.summary.totalDifferences > 0)
        assertTrue(comparisonData.summary.similarityScore < 100.0)
    }

    @Test
    fun `test compare simulation results`() = runBlocking {
        // Reset state and ensure it's clean before starting
        comparisonManager.resetState()

        val result =
            comparisonManager.compareProjects(
                project1,
                project2,
                comparisonType = "results",
            )

        // Wait for the result to be fully processed
        waitForConditionOrFail(
            maxAttempts = 20,
            delayMs = 100,
            message = "Comparison result was not successful or data was not fully processed",
        ) {
            result is ComparisonResult.Success &&
                (result as ComparisonResult.Success).comparisonData.metricDifferences.isNotEmpty()
        }

        assertTrue(result is ComparisonResult.Success)
        val successResult = result as ComparisonResult.Success
        val comparisonData = successResult.comparisonData

        // Should have metric differences
        assertTrue(comparisonData.metricDifferences.isNotEmpty())

        // Check specific metric differences
        val maxForceDiff = comparisonData.metricDifferences.find { it.metric == "Max Force" }
        assertNotNull(maxForceDiff)
        assertEquals(1500.0, maxForceDiff!!.value1)
        assertEquals(1650.0, maxForceDiff.value2)
        assertEquals(150.0, maxForceDiff.absoluteDifference)
        assertEquals(10.0, maxForceDiff.percentageChange, 0.1)
        assertEquals(Significance.HIGH, maxForceDiff.significance)

        val efficiencyDiff = comparisonData.metricDifferences.find { it.metric == "Efficiency" }
        assertNotNull(efficiencyDiff)
        assertTrue(efficiencyDiff!!.percentageChange < 5.0) // Should be low significance
        assertEquals(Significance.LOW, efficiencyDiff.significance)
    }

    @Test
    fun `test full project comparison`() = runBlocking {
        val result =
            comparisonManager.compareProjects(
                project1,
                project2,
                comparisonType = "full",
            )

        assertTrue(result is ComparisonResult.Success)
        val successResult = result as ComparisonResult.Success
        val comparisonData = successResult.comparisonData

        // Should have both parameter and metric differences
        assertTrue(comparisonData.parameterDifferences.isNotEmpty())
        assertTrue(comparisonData.metricDifferences.isNotEmpty())

        // Summary should combine both types
        val totalDifferences = comparisonData.parameterDifferences.size + comparisonData.metricDifferences.size
        assertEquals(totalDifferences, comparisonData.summary.totalDifferences)
    }

    @Test
    fun `test compare projects with no simulation results`() = runBlocking {
        val projectWithoutResults = project1.copy(simulationResults = null)

        val result =
            comparisonManager.compareProjects(
                projectWithoutResults,
                project2,
                comparisonType = "results",
            )

        assertTrue(result is ComparisonResult.Error)
        val errorResult = result as ComparisonResult.Error
        assertTrue(errorResult.message.contains("Both projects must have simulation results"))
    }

    @Test
    fun `test compare with unsupported type`() = runBlocking {
        val result =
            comparisonManager.compareProjects(
                project1,
                project2,
                comparisonType = "unsupported_type",
            )

        assertTrue(result is ComparisonResult.Error)
        val errorResult = result as ComparisonResult.Error
        assertTrue(errorResult.message.contains("Unsupported comparison type"))
    }

    @Test
    fun `test comparison history tracking`() = runBlocking {
        val initialHistorySize = comparisonManager.getComparisonHistory().size

        val result =
            comparisonManager.compareProjects(
                project1,
                project2,
                comparisonType = "parameters",
            )

        assertTrue(result is ComparisonResult.Success)

        val newHistorySize = comparisonManager.getComparisonHistory().size
        assertEquals(initialHistorySize + 1, newHistorySize)

        val latestComparison = comparisonManager.getComparisonHistory().first()
        assertEquals("parameters", latestComparison.type)
        assertEquals(ComparisonStatus.COMPLETED, latestComparison.status)
        assertNotNull(latestComparison.result)
    }

    @Test
    fun `test get comparison by ID`() = runBlocking {
        val result =
            comparisonManager.compareProjects(
                project1,
                project2,
                comparisonType = "full",
            )

        assertTrue(result is ComparisonResult.Success)

        val comparison = comparisonManager.getComparisonHistory().first()
        val retrievedComparison = comparisonManager.getComparison(comparison.id)

        assertNotNull(retrievedComparison)
        assertEquals(comparison.id, retrievedComparison!!.id)
        assertEquals(comparison.type, retrievedComparison.type)
    }

    @Test
    fun `test delete comparison`() = runBlocking {
        val result =
            comparisonManager.compareProjects(
                project1,
                project2,
                comparisonType = "parameters",
            )

        assertTrue(result is ComparisonResult.Success)

        val comparison = comparisonManager.getComparisonHistory().first()
        val initialSize = comparisonManager.getComparisonHistory().size

        val deleteResult = comparisonManager.deleteComparison(comparison.id)

        assertTrue(deleteResult)
        assertEquals(initialSize - 1, comparisonManager.getComparisonHistory().size)
        assertNull(comparisonManager.getComparison(comparison.id))
    }

    @Test
    fun `test export comparison to JSON`() = runBlocking {
        val result =
            comparisonManager.compareProjects(
                project1,
                project2,
                comparisonType = "full",
            )

        assertTrue(result is ComparisonResult.Success)

        val comparison = comparisonManager.getComparisonHistory().first()
        val exportPath = tempDir.resolve("comparison.json").toString()

        val exportResult =
            comparisonManager.exportComparison(
                comparison.id,
                "json",
                exportPath,
            )

        assertTrue(exportResult)
        assertTrue(File(exportPath).exists())

        val exportedContent = File(exportPath).readText()
        assertTrue(exportedContent.contains("Project Alpha"))
        assertTrue(exportedContent.contains("Project Beta"))
        assertTrue(exportedContent.contains("parameterDifferences"))
    }

    @Test
    fun `test export comparison to HTML`() = runBlocking {
        val result =
            comparisonManager.compareProjects(
                project1,
                project2,
                comparisonType = "parameters",
            )

        assertTrue(result is ComparisonResult.Success)

        val comparison = comparisonManager.getComparisonHistory().first()
        val exportPath = tempDir.resolve("comparison.html").toString()

        val exportResult =
            comparisonManager.exportComparison(
                comparison.id,
                "html",
                exportPath,
            )

        assertTrue(exportResult)
        assertTrue(File(exportPath).exists())

        val exportedContent = File(exportPath).readText()
        assertTrue(exportedContent.contains("<!DOCTYPE html>"))
        assertTrue(exportedContent.contains("Project Comparison Report"))
        assertTrue(exportedContent.contains("parameters"))
    }

    @Test
    fun `test export comparison to CSV`() = runBlocking {
        val result =
            comparisonManager.compareProjects(
                project1,
                project2,
                comparisonType = "full",
            )

        assertTrue(result is ComparisonResult.Success)

        val comparison = comparisonManager.getComparisonHistory().first()
        val exportPath = tempDir.resolve("comparison.csv").toString()

        val exportResult =
            comparisonManager.exportComparison(
                comparison.id,
                "csv",
                exportPath,
            )

        assertTrue(exportResult)
        assertTrue(File(exportPath).exists())

        val exportedContent = File(exportPath).readText()
        assertTrue(exportedContent.contains("Type,Parameter/Metric,Value1,Value2,Difference,Significance"))
        assertTrue(exportedContent.contains("Parameter,"))
        assertTrue(exportedContent.contains("Metric,"))
    }

    @Test
    fun `test comparison with custom options`() = runBlocking {
        val options =
            ComparisonOptions(
                includeVisuals = false,
                threshold = 0.05,
            )

        val result =
            comparisonManager.compareProjects(
                project1,
                project2,
                comparisonType = "parameters",
                options = options,
            )

        assertTrue(result is ComparisonResult.Success)

        val comparison = comparisonManager.getComparisonHistory().first()
        assertEquals(options, comparison.options)
    }

    @Test
    fun `test performance comparison type`() = runBlocking {
        val result =
            comparisonManager.compareProjects(
                project1,
                project2,
                comparisonType = "performance",
            )

        // Performance comparison is a placeholder implementation
        assertTrue(result is ComparisonResult.Success)
        val successResult = result as ComparisonResult.Success
        assertEquals(100.0, successResult.comparisonData.summary.similarityScore)
    }

    @Test
    fun `test visual comparison type`() = runBlocking {
        val result =
            comparisonManager.compareProjects(
                project1,
                project2,
                comparisonType = "visual",
            )

        // Visual comparison is a placeholder implementation
        assertTrue(result is ComparisonResult.Success)
        val successResult = result as ComparisonResult.Success
        assertEquals(100.0, successResult.comparisonData.summary.similarityScore)
    }

    @Test
    fun `test comparison significance levels`() = runBlocking {
        val result =
            comparisonManager.compareProjects(
                project1,
                project3, // More significantly different project
                comparisonType = "results",
            )

        assertTrue(result is ComparisonResult.Success)
        val successResult = result as ComparisonResult.Success
        val comparisonData = successResult.comparisonData

        // Should have high significance differences due to larger changes
        val highSignificanceDiffs =
            comparisonData.metricDifferences.filter {
                it.significance == Significance.HIGH
            }
        assertTrue(highSignificanceDiffs.isNotEmpty())

        // Max Force difference should be high (1500 vs 1200 = 20% change)
        val maxForceDiff = comparisonData.metricDifferences.find { it.metric == "Max Force" }
        assertNotNull(maxForceDiff)
        assertEquals(Significance.HIGH, maxForceDiff!!.significance)
    }

    @Test
    fun `test similarity score calculation`() = runBlocking {
        // Compare identical projects (should have high similarity)
        val identicalResult =
            comparisonManager.compareProjects(
                project1,
                project1,
                comparisonType = "parameters",
            )

        assertTrue(identicalResult is ComparisonResult.Success)
        val identicalData = (identicalResult as ComparisonResult.Success).comparisonData
        assertEquals(100.0, identicalData.summary.similarityScore)
        assertEquals(0, identicalData.summary.totalDifferences)

        // Compare different projects (should have lower similarity)
        val differentResult =
            comparisonManager.compareProjects(
                project1,
                project2,
                comparisonType = "parameters",
            )

        assertTrue(differentResult is ComparisonResult.Success)
        val differentData = (differentResult as ComparisonResult.Success).comparisonData
        assertTrue(differentData.summary.similarityScore < 100.0)
        assertTrue(differentData.summary.totalDifferences > 0)
    }

    @Test
    fun `test comparison events`() = runBlocking {
        // Reset state to ensure clean test
        comparisonManager.resetState()

        // Perform the comparison
        val result = comparisonManager.compareProjects(project1, project2, "parameters")

        // Add a significant delay to allow the operation to complete
        kotlinx.coroutines.delay(3000)

        // Verify the result directly
        assertTrue(result is ComparisonResult.Success, "Comparison should succeed")

        // Verify the comparison history has been updated
        val history = comparisonManager.getComparisonHistory()
        assertTrue(history.isNotEmpty(), "Comparison history should not be empty")
        assertEquals("parameters", history.first().type, "Comparison type should be 'parameters'")
        assertEquals(ComparisonStatus.COMPLETED, history.first().status, "Comparison status should be COMPLETED")
    }

    @Test
    fun `test comparison with missing parameters`() = runBlocking {
        // Reset state to ensure clean test
        comparisonManager.resetState()

        val projectWithMissingParams =
            project1.copy(
                parameters =
                mapOf(
                    "Piston Diameter" to "70.0",
                    "Stroke" to "20.0",
                    // Missing Rod Length, TDC Offset, Cycle Ratio
                ),
            )

        // First compare project1 with projectWithMissingParams to check for removed parameters
        val result1 =
            comparisonManager.compareProjects(
                project1,
                projectWithMissingParams,
                comparisonType = "parameters",
            )

        // Add a significant delay to allow the operation to complete
        kotlinx.coroutines.delay(3000)

        // Verify the result directly
        assertTrue(result1 is ComparisonResult.Success, "Comparison should succeed")
        val successResult1 = result1 as ComparisonResult.Success
        val comparisonData1 = successResult1.comparisonData

        // Verify parameter differences
        assertTrue(comparisonData1.parameterDifferences.isNotEmpty(), "Should have parameter differences")

        // Should detect removed parameters (parameters in project1 but not in projectWithMissingParams)
        val removedParams =
            comparisonData1.parameterDifferences.filter {
                it.type == DifferenceType.REMOVED
            }
        assertTrue(removedParams.isNotEmpty(), "Should detect removed parameters")

        // Now compare projectWithMissingParams with project2 to check for added parameters
        val result2 =
            comparisonManager.compareProjects(
                projectWithMissingParams,
                project2,
                comparisonType = "parameters",
            )

        // Add a significant delay to allow the operation to complete
        kotlinx.coroutines.delay(3000)

        // Verify the result directly
        assertTrue(result2 is ComparisonResult.Success, "Comparison should succeed")
        val successResult2 = result2 as ComparisonResult.Success
        val comparisonData2 = successResult2.comparisonData

        // Verify parameter differences
        assertTrue(comparisonData2.parameterDifferences.isNotEmpty(), "Should have parameter differences")

        // Should detect added parameters (parameters in project2 but not in projectWithMissingParams)
        val addedParams =
            comparisonData2.parameterDifferences.filter {
                it.type == DifferenceType.ADDED
            }
        assertTrue(addedParams.isNotEmpty(), "Should detect added parameters")
    }

    @Test
    fun `test comparison history limit`() = runBlocking {
        // Add many comparisons to test history limit
        repeat(55) {
            // More than the 50 limit
            comparisonManager.compareProjects(
                project1,
                project2,
                comparisonType = "parameters",
            )
        }

        val history = comparisonManager.getComparisonHistory()
        assertTrue(history.size <= 50) // Should not exceed limit
    }

    @Test
    fun `test export with invalid format`() = runBlocking {
        val result =
            comparisonManager.compareProjects(
                project1,
                project2,
                comparisonType = "parameters",
            )

        assertTrue(result is ComparisonResult.Success)

        val comparison = comparisonManager.getComparisonHistory().first()
        val exportPath = tempDir.resolve("comparison.invalid").toString()

        val exportResult =
            comparisonManager.exportComparison(
                comparison.id,
                "invalid_format",
                exportPath,
            )

        assertFalse(exportResult)
    }

    @Test
    fun `test export non-existent comparison`() = runBlocking {
        val exportPath = tempDir.resolve("comparison.json").toString()

        val exportResult =
            comparisonManager.exportComparison(
                "non_existent_id",
                "json",
                exportPath,
            )

        assertFalse(exportResult)
    }
}
