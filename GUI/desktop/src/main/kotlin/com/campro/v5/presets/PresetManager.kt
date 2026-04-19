package com.campro.v5.presets

import com.campro.v5.models.OptimizationParameters
import com.google.gson.Gson
import com.google.gson.GsonBuilder
import org.slf4j.LoggerFactory
import java.io.IOException
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter

/**
 * Manager for optimization parameter presets.
 *
 * Handles saving, loading, and managing parameter presets with metadata,
 * validation, and file system operations.
 */
class PresetManager(private val presetsDirectory: Path = Paths.get("./presets")) {

    private val logger = LoggerFactory.getLogger(PresetManager::class.java)
    private val gson: Gson = GsonBuilder()
        .setPrettyPrinting()
        .setDateFormat("yyyy-MM-dd HH:mm:ss")
        .create()

    init {
        ensurePresetsDirectoryExists()
    }

    /**
     * Preset data class with metadata.
     */
    data class Preset(
        val name: String,
        val description: String = "",
        val parameters: OptimizationParameters,
        val createdAt: String = LocalDateTime.now().format(DateTimeFormatter.ISO_LOCAL_DATE_TIME),
        val modifiedAt: String = LocalDateTime.now().format(DateTimeFormatter.ISO_LOCAL_DATE_TIME),
        val tags: List<String> = emptyList(),
        val version: String = "1.0",
    )

    /**
     * Save preset to file system.
     */
    fun savePreset(preset: Preset): Boolean = try {
        val presetFile = getPresetFile(preset.name)
        val json = gson.toJson(preset)

        Files.write(presetFile, json.toByteArray())
        logger.info("Saved preset: ${preset.name}")
        true
    } catch (e: IOException) {
        logger.error("Failed to save preset: ${preset.name}", e)
        false
    }

    /**
     * Load preset from file system.
     */
    fun loadPreset(name: String): Preset? {
        return try {
            val presetFile = getPresetFile(name)
            if (!Files.exists(presetFile)) {
                logger.warn("Preset file not found: $name")
                return null
            }

            val json = Files.readString(presetFile)
            val preset = gson.fromJson(json, Preset::class.java)
            logger.info("Loaded preset: $name")
            preset
        } catch (e: Exception) {
            logger.error("Failed to load preset: $name", e)
            null
        }
    }

    /**
     * Delete preset from file system.
     */
    fun deletePreset(name: String): Boolean = try {
        val presetFile = getPresetFile(name)
        if (Files.exists(presetFile)) {
            Files.delete(presetFile)
            logger.info("Deleted preset: $name")
            true
        } else {
            logger.warn("Preset file not found for deletion: $name")
            false
        }
    } catch (e: IOException) {
        logger.error("Failed to delete preset: $name", e)
        false
    }

    /**
     * Get list of available presets.
     */
    fun getAvailablePresets(): List<PresetInfo> {
        return try {
            if (!Files.exists(presetsDirectory)) {
                return emptyList()
            }

            Files.list(presetsDirectory)
                .filter { path -> path.fileName.toString().endsWith(".json") }
                .map { path ->
                    try {
                        val json = Files.readString(path)
                        val preset = gson.fromJson(json, Preset::class.java)
                        PresetInfo(
                            name = preset.name,
                            description = preset.description,
                            createdAt = preset.createdAt,
                            modifiedAt = preset.modifiedAt,
                            tags = preset.tags,
                        )
                    } catch (e: Exception) {
                        logger.warn("Failed to read preset info from: $path", e)
                        null
                    }
                }
                .toList()
                .filterNotNull()
                .sortedBy { it.name }
        } catch (e: IOException) {
            logger.error("Failed to list presets", e)
            emptyList()
        }
    }

    /**
     * Check if preset exists.
     */
    fun presetExists(name: String): Boolean = Files.exists(getPresetFile(name))

    /**
     * Create preset from parameters.
     */
    fun createPreset(name: String, description: String = "", parameters: OptimizationParameters, tags: List<String> = emptyList()): Preset =
        Preset(
            name = name,
            description = description,
            parameters = parameters,
            tags = tags,
        )

    /**
     * Update existing preset.
     */
    fun updatePreset(
        name: String,
        description: String? = null,
        parameters: OptimizationParameters? = null,
        tags: List<String>? = null,
    ): Boolean {
        return try {
            val existingPreset = loadPreset(name) ?: return false

            val updatedPreset = existingPreset.copy(
                description = description ?: existingPreset.description,
                parameters = parameters ?: existingPreset.parameters,
                tags = tags ?: existingPreset.tags,
                modifiedAt = LocalDateTime.now().format(DateTimeFormatter.ISO_LOCAL_DATE_TIME),
            )

            savePreset(updatedPreset)
        } catch (e: Exception) {
            logger.error("Failed to update preset: $name", e)
            false
        }
    }

    /**
     * Duplicate existing preset with new name.
     */
    fun duplicatePreset(originalName: String, newName: String): Boolean {
        return try {
            val originalPreset = loadPreset(originalName) ?: return false

            val duplicatedPreset = originalPreset.copy(
                name = newName,
                createdAt = LocalDateTime.now().format(DateTimeFormatter.ISO_LOCAL_DATE_TIME),
                modifiedAt = LocalDateTime.now().format(DateTimeFormatter.ISO_LOCAL_DATE_TIME),
            )

            savePreset(duplicatedPreset)
        } catch (e: Exception) {
            logger.error("Failed to duplicate preset: $originalName to $newName", e)
            false
        }
    }

    /**
     * Export preset to external file.
     */
    fun exportPreset(name: String, exportPath: Path): Boolean {
        return try {
            val preset = loadPreset(name) ?: return false
            val json = gson.toJson(preset)
            Files.write(exportPath, json.toByteArray())
            logger.info("Exported preset: $name to $exportPath")
            true
        } catch (e: IOException) {
            logger.error("Failed to export preset: $name", e)
            false
        }
    }

    /**
     * Import preset from external file.
     */
    fun importPreset(importPath: Path, newName: String? = null): Boolean = try {
        val json = Files.readString(importPath)
        val preset = gson.fromJson(json, Preset::class.java)

        val finalName = newName ?: preset.name
        val importedPreset = preset.copy(
            name = finalName,
            createdAt = LocalDateTime.now().format(DateTimeFormatter.ISO_LOCAL_DATE_TIME),
            modifiedAt = LocalDateTime.now().format(DateTimeFormatter.ISO_LOCAL_DATE_TIME),
        )

        savePreset(importedPreset)
    } catch (e: Exception) {
        logger.error("Failed to import preset from: $importPath", e)
        false
    }

    /**
     * Search presets by name or tags.
     */
    fun searchPresets(query: String): List<PresetInfo> {
        val allPresets = getAvailablePresets()
        val lowercaseQuery = query.lowercase()

        return allPresets.filter { preset ->
            preset.name.lowercase().contains(lowercaseQuery) ||
                preset.description.lowercase().contains(lowercaseQuery) ||
                preset.tags.any { tag -> tag.lowercase().contains(lowercaseQuery) }
        }
    }

    /**
     * Get preset file path.
     */
    private fun getPresetFile(name: String): Path {
        val safeName = name.replace(Regex("[^a-zA-Z0-9_-]"), "_")
        return presetsDirectory.resolve("$safeName.json")
    }

    /**
     * Ensure presets directory exists.
     */
    private fun ensurePresetsDirectoryExists() {
        try {
            Files.createDirectories(presetsDirectory)
            logger.info("Presets directory ready: $presetsDirectory")
        } catch (e: IOException) {
            logger.error("Failed to create presets directory: $presetsDirectory", e)
            throw e
        }
    }

    /**
     * Preset information for listing.
     */
    data class PresetInfo(val name: String, val description: String, val createdAt: String, val modifiedAt: String, val tags: List<String>)

    companion object {
        /**
         * Create default presets.
         */
        fun createDefaultPresets(presetManager: PresetManager) {
            // Default preset
            val defaultPreset = presetManager.createPreset(
                name = "Default",
                description = "Default optimization parameters",
                parameters = OptimizationParameters.createDefault(),
                tags = listOf("default", "basic"),
            )
            presetManager.savePreset(defaultPreset)

            // Quick test preset
            val quickTestPreset = presetManager.createPreset(
                name = "Quick Test",
                description = "Fast optimization for testing",
                parameters = OptimizationParameters.createQuickTest(),
                tags = listOf("test", "fast", "development"),
            )
            presetManager.savePreset(quickTestPreset)

            // High performance preset
            val highPerfPreset = presetManager.createPreset(
                name = "High Performance",
                description = "High-performance optimization parameters",
                parameters = OptimizationParameters.createHighPerformance(),
                tags = listOf("performance", "production", "optimized"),
            )
            presetManager.savePreset(highPerfPreset)
        }
    }
}
