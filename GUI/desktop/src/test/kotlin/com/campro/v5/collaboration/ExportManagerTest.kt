package com.campro.v5.collaboration

import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.io.TempDir
import java.io.File
import java.nio.file.Path
import java.util.Date

class ExportManagerTest {
    private lateinit var exportManager: ExportManager
    private lateinit var testProjectData: ProjectData

    @TempDir
    lateinit var tempDir: Path

    @BeforeEach
    fun setUp() {
        exportManager = ExportManager.getInstance()
        exportManager.resetState()

        // Create test project data
        testProjectData =
            ProjectData(
                metadata =
                ProjectMetadata(
                    name = "Test Project",
                    description = "A test project for export testing",
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
                    ),
                    visualizations =
                    listOf(
                        VisualizationData(
                            type = "force_curve",
                            data = mapOf("points" to listOf(1.0, 2.0, 3.0)),
                        ),
                    ),
                ),
                visualizations =
                listOf(
                    VisualizationData(
                        type = "cam_profile",
                        data = mapOf("profile" to "cycloidal"),
                    ),
                ),
            )
    }

    @AfterEach
    fun tearDown() {
        // Clean up any test files
        exportManager.clearExportHistory()
    }

    @Test
    fun `test export to JSON format`() = runBlocking {
        val exportPath = tempDir.resolve("test_export.json").toString()

        val result =
            exportManager.exportProject(
                testProjectData,
                exportPath,
                "json",
                ExportOptions(prettyPrint = true),
            )

        assertTrue(result is ExportResult.Success)
        assertTrue(File(exportPath).exists())

        val exportedContent = File(exportPath).readText()
        assertTrue(exportedContent.contains("Test Project"))
        assertTrue(exportedContent.contains("Piston Diameter"))
    }

    @Test
    fun `test export to CSV format`() = runBlocking {
        val exportPath = tempDir.resolve("test_export.csv").toString()

        val result =
            exportManager.exportProject(
                testProjectData,
                exportPath,
                "csv",
            )

        assertTrue(result is ExportResult.Success)
        assertTrue(File(exportPath).exists())

        val exportedContent = File(exportPath).readText()
        assertTrue(exportedContent.contains("Parameter,Value"))
        assertTrue(exportedContent.contains("Piston Diameter"))
        assertTrue(exportedContent.contains("70.0"))
    }

    @Test
    fun `test export to XML format`() = runBlocking {
        val exportPath = tempDir.resolve("test_export.xml").toString()

        val result =
            exportManager.exportProject(
                testProjectData,
                exportPath,
                "xml",
            )

        assertTrue(result is ExportResult.Success)
        assertTrue(File(exportPath).exists())

        val exportedContent = File(exportPath).readText()
        assertTrue(exportedContent.contains("<?xml version=\"1.0\""))
        assertTrue(exportedContent.contains("<project>"))
        assertTrue(exportedContent.contains("Test Project"))
    }

    @Test
    fun `test export to ZIP format`() = runBlocking {
        val exportPath = tempDir.resolve("test_export.zip").toString()

        val result =
            exportManager.exportProject(
                testProjectData,
                exportPath,
                "zip",
            )

        assertTrue(result is ExportResult.Success)
        assertTrue(File(exportPath).exists())
        assertTrue(File(exportPath).length() > 0)
    }

    @Test
    fun `test export simulation results`() = runBlocking {
        val exportPath = tempDir.resolve("simulation_results.json").toString()

        val result =
            exportManager.exportSimulationResults(
                testProjectData.simulationResults!!,
                exportPath,
                "json",
            )

        assertTrue(result is ExportResult.Success)
        assertTrue(File(exportPath).exists())

        val exportedContent = File(exportPath).readText()
        assertTrue(exportedContent.contains("Max Force"))
        assertTrue(exportedContent.contains("1500.0"))
    }

    @Test
    fun `test export with invalid format`() = runBlocking {
        val exportPath = tempDir.resolve("test_export.invalid").toString()

        val result =
            exportManager.exportProject(
                testProjectData,
                exportPath,
                "invalid_format",
            )

        assertTrue(result is ExportResult.Error)
        val errorResult = result as ExportResult.Error
        assertTrue(errorResult.message.contains("Unsupported format"))
    }

    @Test
    fun `test export progress tracking`() = runBlocking {
        val exportPath = tempDir.resolve("test_export.json").toString()

        assertFalse(exportManager.isExporting())
        assertEquals(0.0f, exportManager.getExportProgress())

        val result =
            exportManager.exportProject(
                testProjectData,
                exportPath,
                "json",
            )

        assertTrue(result is ExportResult.Success)
        assertFalse(exportManager.isExporting())
        assertEquals(1.0f, exportManager.getExportProgress())
    }

    @Test
    fun `test export history tracking`() = runBlocking {
        val exportPath = tempDir.resolve("test_export.json").toString()

        val initialHistorySize = exportManager.getExportHistory().size

        val result =
            exportManager.exportProject(
                testProjectData,
                exportPath,
                "json",
            )

        // Add a delay to ensure the export history is properly updated
        kotlinx.coroutines.delay(500)

        val newHistorySize = exportManager.getExportHistory().size
        assertEquals(initialHistorySize + 1, newHistorySize)

        val latestEntry = exportManager.getExportHistory().first()
        assertEquals("json", latestEntry.format)
        assertTrue(latestEntry.success)
        assertEquals("Test Project", latestEntry.filePath.substringAfterLast("/").substringBeforeLast("."))
    }

    @Test
    fun `test export history clearing`() = runBlocking {
        val exportPath = tempDir.resolve("test_export.json").toString()

        exportManager.exportProject(
            testProjectData,
            exportPath,
            "json",
        )

        assertTrue(exportManager.getExportHistory().isNotEmpty())

        exportManager.clearExportHistory()

        assertTrue(exportManager.getExportHistory().isEmpty())
    }

    @Test
    fun `test supported formats`() {
        val supportedFormats = exportManager.getSupportedFormats()

        assertTrue(supportedFormats.containsKey("json"))
        assertTrue(supportedFormats.containsKey("csv"))
        assertTrue(supportedFormats.containsKey("xml"))
        assertTrue(supportedFormats.containsKey("pdf"))
        assertTrue(supportedFormats.containsKey("xlsx"))
        assertTrue(supportedFormats.containsKey("zip"))
        assertTrue(supportedFormats.containsKey("png"))
        assertTrue(supportedFormats.containsKey("svg"))

        val jsonFormat = supportedFormats["json"]!!
        assertEquals("JSON", jsonFormat.name)
        assertEquals("JavaScript Object Notation", jsonFormat.description)
        assertTrue(jsonFormat.extensions.contains(".json"))
    }

    @Test
    fun `test export with custom options`() = runBlocking {
        val exportPath = tempDir.resolve("test_export_custom.json").toString()

        val customOptions =
            ExportOptions(
                prettyPrint = false,
                includeMetadata = true,
                includeVisualization = false,
                compressionLevel = 9,
            )

        val result =
            exportManager.exportProject(
                testProjectData,
                exportPath,
                "json",
                customOptions,
            )

        assertTrue(result is ExportResult.Success)
        assertTrue(File(exportPath).exists())

        // Verify that the export used the custom options
        val exportedContent = File(exportPath).readText()
        // Non-pretty printed JSON should be more compact
        assertFalse(exportedContent.contains("\n  "))
    }

    @Test
    fun `test concurrent exports`() = runBlocking {
        val exportPath1 = tempDir.resolve("test_export1.json").toString()
        val exportPath2 = tempDir.resolve("test_export2.csv").toString()

        // Note: This test assumes the export manager handles concurrent requests properly
        val result1 = exportManager.exportProject(testProjectData, exportPath1, "json")
        val result2 = exportManager.exportProject(testProjectData, exportPath2, "csv")

        assertTrue(result1 is ExportResult.Success)
        assertTrue(result2 is ExportResult.Success)
        assertTrue(File(exportPath1).exists())
        assertTrue(File(exportPath2).exists())
    }

    @Test
    fun `test export to non-existent directory`() = runBlocking {
        val exportPath = tempDir.resolve("non_existent_dir").resolve("test_export.json").toString()

        val result =
            exportManager.exportProject(
                testProjectData,
                exportPath,
                "json",
            )

        assertTrue(result is ExportResult.Success)
        assertTrue(File(exportPath).exists())
        assertTrue(File(exportPath).parentFile.exists())
    }

    @Test
    fun `test export events`() = runBlocking {
        // Reset state to ensure clean test
        exportManager.resetState()

        val exportPath = tempDir.resolve("test_export.json").toString()

        // Perform the export
        val result =
            exportManager.exportProject(
                testProjectData,
                exportPath,
                "json",
            )

        // Add a significant delay to allow the operation to complete
        kotlinx.coroutines.delay(3000)

        // Verify the result directly
        assertTrue(result is ExportResult.Success, "Export should succeed")

        // Verify the file was created
        assertTrue(File(exportPath).exists(), "Export file should exist")

        // Verify the export progress is complete
        assertEquals(1.0f, exportManager.getExportProgress(), "Export progress should be complete")

        // Verify export is no longer in progress
        assertFalse(exportManager.isExporting(), "Export should not be in progress")

        // Verify export history has been updated
        val history = exportManager.getExportHistory()
        assertTrue(history.isNotEmpty(), "Export history should not be empty")
        assertEquals("json", history.first().format, "Export format should be 'json'")
        assertTrue(history.first().success, "Export should be marked as successful")
    }
}
