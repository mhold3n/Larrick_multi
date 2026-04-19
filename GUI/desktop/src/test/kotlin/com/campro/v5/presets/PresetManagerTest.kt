package com.campro.v5.presets

import com.campro.v5.models.OptimizationParameters
import org.junit.jupiter.api.*
import org.junit.jupiter.api.Assertions.*
import java.nio.file.Files
import java.nio.file.Path

/**
 * Tests for PresetManager.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class PresetManagerTest {

    private lateinit var tempDir: Path
    private lateinit var presetManager: PresetManager
    private lateinit var testParameters: OptimizationParameters

    @BeforeEach
    fun setup() {
        tempDir = Files.createTempDirectory("preset_manager_test")
        presetManager = PresetManager(presetsDirectory = tempDir)
        testParameters = OptimizationParameters.createDefault()
    }

    @AfterEach
    fun tearDown() {
        // Clean up temp directory
        Files.walk(tempDir)
            .sorted(Comparator.reverseOrder())
            .forEach { path ->
                try {
                    Files.deleteIfExists(path)
                } catch (e: Exception) {
                    // Ignore cleanup errors
                }
            }
    }

    @Test
    fun `test save and load preset`() {
        // Given
        val preset = presetManager.createPreset(
            name = "Test Preset",
            description = "Test description",
            parameters = testParameters,
            tags = listOf("test", "default"),
        )

        // When
        val saveResult = presetManager.savePreset(preset)
        val loadedPreset = presetManager.loadPreset("Test Preset")

        // Then
        assertTrue(saveResult)
        assertNotNull(loadedPreset)
        assertEquals("Test Preset", loadedPreset?.name)
        assertEquals("Test description", loadedPreset?.description)
        assertEquals(testParameters.samplingStepDeg, loadedPreset?.parameters?.samplingStepDeg)
        assertEquals(testParameters.strokeLengthMm, loadedPreset?.parameters?.strokeLengthMm)
        assertEquals(listOf("test", "default"), loadedPreset?.tags)
    }

    @Test
    fun `test preset exists check`() {
        // Given
        val preset = presetManager.createPreset(
            name = "Existing Preset",
            parameters = testParameters,
        )
        presetManager.savePreset(preset)

        // When & Then
        assertTrue(presetManager.presetExists("Existing Preset"))
        assertFalse(presetManager.presetExists("Non-existing Preset"))
    }

    @Test
    fun `test get available presets`() {
        // Given
        val preset1 = presetManager.createPreset(
            name = "Preset 1",
            description = "First preset",
            parameters = testParameters,
        )
        val preset2 = presetManager.createPreset(
            name = "Preset 2",
            description = "Second preset",
            parameters = OptimizationParameters.createQuickTest(),
        )

        presetManager.savePreset(preset1)
        presetManager.savePreset(preset2)

        // When
        val availablePresets = presetManager.getAvailablePresets()

        // Then
        assertEquals(2, availablePresets.size)
        assertTrue(availablePresets.any { it.name == "Preset 1" })
        assertTrue(availablePresets.any { it.name == "Preset 2" })
        assertTrue(availablePresets.any { it.description == "First preset" })
        assertTrue(availablePresets.any { it.description == "Second preset" })
    }

    @Test
    fun `test delete preset`() {
        // Given
        val preset = presetManager.createPreset(
            name = "To Delete",
            parameters = testParameters,
        )
        presetManager.savePreset(preset)
        assertTrue(presetManager.presetExists("To Delete"))

        // When
        val deleteResult = presetManager.deletePreset("To Delete")

        // Then
        assertTrue(deleteResult)
        assertFalse(presetManager.presetExists("To Delete"))
    }

    @Test
    fun `test update preset`() {
        // Given
        val originalPreset = presetManager.createPreset(
            name = "Update Test",
            description = "Original description",
            parameters = testParameters,
            tags = listOf("original"),
        )
        presetManager.savePreset(originalPreset)

        val newParameters = OptimizationParameters.createQuickTest()

        // When
        val updateResult = presetManager.updatePreset(
            name = "Update Test",
            description = "Updated description",
            parameters = newParameters,
            tags = listOf("updated", "test"),
        )

        // Then
        assertTrue(updateResult)

        val updatedPreset = presetManager.loadPreset("Update Test")
        assertNotNull(updatedPreset)
        assertEquals("Updated description", updatedPreset?.description)
        assertEquals(newParameters.samplingStepDeg, updatedPreset?.parameters?.samplingStepDeg)
        assertEquals(listOf("updated", "test"), updatedPreset?.tags)
        assertNotEquals(originalPreset.modifiedAt, updatedPreset?.modifiedAt)
    }

    @Test
    fun `test duplicate preset`() {
        // Given
        val originalPreset = presetManager.createPreset(
            name = "Original",
            description = "Original preset",
            parameters = testParameters,
        )
        presetManager.savePreset(originalPreset)

        // When
        val duplicateResult = presetManager.duplicatePreset("Original", "Duplicate")

        // Then
        assertTrue(duplicateResult)

        val original = presetManager.loadPreset("Original")
        val duplicate = presetManager.loadPreset("Duplicate")

        assertNotNull(original)
        assertNotNull(duplicate)
        assertEquals("Original", original?.name)
        assertEquals("Duplicate", duplicate?.name)
        assertEquals(original?.parameters?.samplingStepDeg, duplicate?.parameters?.samplingStepDeg)
        assertNotEquals(original?.createdAt, duplicate?.createdAt)
    }

    @Test
    fun `test export and import preset`() {
        // Given
        val preset = presetManager.createPreset(
            name = "Export Test",
            description = "Preset for export",
            parameters = testParameters,
            tags = listOf("export", "test"),
        )
        presetManager.savePreset(preset)

        val exportPath = tempDir.resolve("exported_preset.json")

        // When
        val exportResult = presetManager.exportPreset("Export Test", exportPath)
        val importResult = presetManager.importPreset(exportPath, "Imported Preset")

        // Then
        assertTrue(exportResult)
        assertTrue(importResult)

        val importedPreset = presetManager.loadPreset("Imported Preset")
        assertNotNull(importedPreset)
        assertEquals("Imported Preset", importedPreset?.name)
        assertEquals("Preset for export", importedPreset?.description)
        assertEquals(testParameters.samplingStepDeg, importedPreset?.parameters?.samplingStepDeg)
        assertEquals(listOf("export", "test"), importedPreset?.tags)
    }

    @Test
    fun `test search presets`() {
        // Given
        val preset1 = presetManager.createPreset(
            name = "High Performance",
            description = "Optimized for speed",
            parameters = OptimizationParameters.createHighPerformance(),
            tags = listOf("performance", "fast"),
        )
        val preset2 = presetManager.createPreset(
            name = "Quick Test",
            description = "Fast testing preset",
            parameters = OptimizationParameters.createQuickTest(),
            tags = listOf("test", "development"),
        )
        val preset3 = presetManager.createPreset(
            name = "Default Settings",
            description = "Standard configuration",
            parameters = OptimizationParameters.createDefault(),
            tags = listOf("default", "standard"),
        )

        presetManager.savePreset(preset1)
        presetManager.savePreset(preset2)
        presetManager.savePreset(preset3)

        // When & Then
        val performanceResults = presetManager.searchPresets("performance")
        assertEquals(1, performanceResults.size)
        assertEquals("High Performance", performanceResults[0].name)

        val testResults = presetManager.searchPresets("test")
        assertEquals(2, testResults.size)
        assertTrue(testResults.any { it.name == "Quick Test" })
        assertTrue(testResults.any { it.name == "Default Settings" })

        val fastResults = presetManager.searchPresets("fast")
        assertEquals(2, fastResults.size)
        assertTrue(fastResults.any { it.name == "High Performance" })
        assertTrue(fastResults.any { it.name == "Quick Test" })
    }

    @Test
    fun `test load non-existent preset`() {
        // When
        val result = presetManager.loadPreset("Non-existent")

        // Then
        assertNull(result)
    }

    @Test
    fun `test delete non-existent preset`() {
        // When
        val result = presetManager.deletePreset("Non-existent")

        // Then
        assertFalse(result)
    }
}
