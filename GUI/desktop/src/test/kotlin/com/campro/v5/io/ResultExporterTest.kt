package com.campro.v5.io

import com.campro.v5.models.OptimizationParameters
import com.campro.v5.models.OptimizationResult
import com.campro.v5.models.MotionLawData
import com.campro.v5.models.GearProfileData
import com.campro.v5.models.ToothProfileData
import com.campro.v5.models.FEAAnalysisData
import org.junit.jupiter.api.*
import org.junit.jupiter.api.Assertions.*
import java.nio.file.Files
import java.nio.file.Path
import java.io.IOException

/**
 * Tests for ResultExporter.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class ResultExporterTest {

    private lateinit var tempDir: Path
    private lateinit var exporter: ResultExporter
    private lateinit var testParameters: OptimizationParameters
    private lateinit var testResult: OptimizationResult

    @BeforeEach
    fun setup() {
        tempDir = Files.createTempDirectory("result_exporter_test")
        exporter = ResultExporter()
        testParameters = OptimizationParameters.createDefault()
        testResult = createTestOptimizationResult()
    }

    @AfterEach
    fun tearDown() {
        // Clean up temp directory
        Files.walk(tempDir)
            .sorted(Comparator.reverseOrder())
            .forEach { path ->
                try {
                    Files.deleteIfExists(path)
                } catch (e: IOException) {
                    // Ignore cleanup errors
                }
            }
    }

    @Test
    fun `test export to JSON`() {
        // Given
        val outputPath = tempDir.resolve("test_result.json")

        // When
        val exportedPath = exporter.exportResult(
            result = testResult,
            parameters = testParameters,
            outputPath = outputPath,
            format = ResultExporter.ExportFormat.JSON,
        )

        // Then
        assertTrue(Files.exists(exportedPath))
        assertTrue(Files.size(exportedPath) > 0)

        val content = Files.readString(exportedPath)
        assertTrue(content.contains("\"status\""))
        assertTrue(content.contains("\"motionLaw\""))
        assertTrue(content.contains("\"optimalProfiles\""))
        assertTrue(content.contains("\"parameters\""))
    }

    @Test
    fun `test export to CSV`() {
        // Given
        val outputPath = tempDir.resolve("test_result.csv")

        // When
        val exportedPath = exporter.exportResult(
            result = testResult,
            parameters = testParameters,
            outputPath = outputPath,
            format = ResultExporter.ExportFormat.CSV,
        )

        // Then
        assertTrue(Files.exists(exportedPath))
        assertTrue(Files.size(exportedPath) > 0)

        val content = Files.readString(exportedPath)
        assertTrue(content.contains("Optimization Results Export"))
        assertTrue(content.contains("Parameters"))
        assertTrue(content.contains("Motion Law Data"))
        assertTrue(content.contains("Gear Profile Data"))
        assertTrue(content.contains("FEA Analysis"))
    }

    @Test
    fun `test export to PDF`() {
        // Given
        val outputPath = tempDir.resolve("test_result.pdf")

        // When
        val exportedPath = exporter.exportResult(
            result = testResult,
            parameters = testParameters,
            outputPath = outputPath,
            format = ResultExporter.ExportFormat.PDF,
        )

        // Then
        assertTrue(Files.exists(exportedPath))
        assertTrue(Files.size(exportedPath) > 0)

        val content = Files.readString(exportedPath)
        assertTrue(content.contains("OPTIMIZATION RESULTS REPORT"))
        assertTrue(content.contains("PARAMETERS"))
        assertTrue(content.contains("MOTION LAW ANALYSIS"))
        assertTrue(content.contains("GEAR PROFILE ANALYSIS"))
        assertTrue(content.contains("FEA ANALYSIS"))
    }

    @Test
    fun `test export to Excel`() {
        // Given
        val outputPath = tempDir.resolve("test_result.xlsx")

        // When
        val exportedPath = exporter.exportResult(
            result = testResult,
            parameters = testParameters,
            outputPath = outputPath,
            format = ResultExporter.ExportFormat.EXCEL,
        )

        // Then
        assertTrue(Files.exists(exportedPath))
        assertTrue(Files.size(exportedPath) > 0)
    }

    @Test
    fun `test file extension handling`() {
        // Given
        val outputPathWithoutExt = tempDir.resolve("test_result")

        // When
        val jsonPath = exporter.exportResult(
            result = testResult,
            parameters = testParameters,
            outputPath = outputPathWithoutExt,
            format = ResultExporter.ExportFormat.JSON,
        )

        // Then
        assertTrue(jsonPath.fileName.toString().endsWith(".json"))
        assertTrue(Files.exists(jsonPath))
    }

    @Test
    fun `test export with failed result`() {
        // Given
        val failedResult = testResult.copy(
            status = "failed",
            error = "Test error",
        )
        val outputPath = tempDir.resolve("failed_result.json")

        // When
        val exportedPath = exporter.exportResult(
            result = failedResult,
            parameters = testParameters,
            outputPath = outputPath,
            format = ResultExporter.ExportFormat.JSON,
        )

        // Then
        assertTrue(Files.exists(exportedPath))

        val content = Files.readString(exportedPath)
        assertTrue(content.contains("\"failed\""))
        assertTrue(content.contains("Test error"))
    }

    private fun createTestOptimizationResult(): OptimizationResult = OptimizationResult(
        status = "success",
        motionLaw = MotionLawData(
            thetaDeg = doubleArrayOf(0.0, 90.0, 180.0),
            displacement = doubleArrayOf(0.0, 50.0, 100.0),
            velocity = doubleArrayOf(100.0, 0.0, -100.0),
            acceleration = doubleArrayOf(0.0, -1000.0, 0.0),
        ),
        optimalProfiles = GearProfileData(
            rSun = doubleArrayOf(110.0, 115.0, 120.0),
            rPlanet = doubleArrayOf(175.0, 180.0, 185.0),
            rRingInner = doubleArrayOf(460.0, 470.0, 480.0),
            gearRatio = 2.0,
            optimalMethod = "litvin",
            efficiencyAnalysis = null,
        ),
        toothProfiles = ToothProfileData(
            sunTeeth = null,
            planetTeeth = null,
            ringTeeth = null,
        ),
        feaAnalysis = FEAAnalysisData(
            maxStress = 150.0,
            naturalFrequencies = doubleArrayOf(100.0, 200.0, 300.0),
            fatigueLife = 1000000.0,
            modeShapes = arrayOf("Mode 1", "Mode 2", "Mode 3"),
            recommendations = arrayOf("Recommendation 1", "Recommendation 2"),
        ),
        executionTime = 1.5,
    )
}
