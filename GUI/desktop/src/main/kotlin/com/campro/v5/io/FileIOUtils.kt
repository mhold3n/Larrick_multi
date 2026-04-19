package com.campro.v5.io

import org.slf4j.LoggerFactory
import java.io.IOException
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter

/**
 * Utility class for file I/O operations.
 *
 * Provides robust file handling with error management, directory creation,
 * and file validation for the optimization pipeline.
 */
object FileIOUtils {

    private val logger = LoggerFactory.getLogger(FileIOUtils::class.java)

    /**
     * Create output directory with timestamp.
     */
    fun createOutputDirectory(baseDir: String = "./output"): Path {
        val timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"))
        val outputDir = Paths.get(baseDir, "optimization_$timestamp")

        return try {
            Files.createDirectories(outputDir)
            logger.info("Created output directory: $outputDir")
            outputDir
        } catch (e: IOException) {
            logger.error("Failed to create output directory: $outputDir", e)
            throw e
        }
    }

    /**
     * Ensure directory exists, create if necessary.
     */
    fun ensureDirectoryExists(path: Path): Path = try {
        if (!Files.exists(path)) {
            Files.createDirectories(path)
            logger.info("Created directory: $path")
        }
        path
    } catch (e: IOException) {
        logger.error("Failed to create directory: $path", e)
        throw e
    }

    /**
     * Validate file path for writing.
     */
    fun validateWritePath(path: Path): Boolean = try {
        val parent = path.parent
        if (parent != null && !Files.exists(parent)) {
            Files.createDirectories(parent)
        }

        // Check if we can write to the directory
        val testFile = parent?.resolve(".write_test")
        if (testFile != null) {
            Files.write(testFile, "test".toByteArray())
            Files.deleteIfExists(testFile)
        }

        true
    } catch (e: Exception) {
        logger.error("Invalid write path: $path", e)
        false
    }

    /**
     * Get file extension from path.
     */
    fun getFileExtension(path: Path): String {
        val fileName = path.fileName.toString()
        val lastDot = fileName.lastIndexOf('.')
        return if (lastDot > 0) fileName.substring(lastDot + 1).lowercase() else ""
    }

    /**
     * Generate unique filename to avoid conflicts.
     */
    fun generateUniqueFilename(basePath: Path, extension: String): Path {
        var counter = 1
        var newPath = basePath

        while (Files.exists(newPath)) {
            val baseName = basePath.fileName.toString()
            val nameWithoutExt = if (baseName.contains('.')) {
                baseName.substring(0, baseName.lastIndexOf('.'))
            } else {
                baseName
            }

            val newName = "${nameWithoutExt}_$counter.$extension"
            newPath = basePath.parent?.resolve(newName) ?: Paths.get(newName)
            counter++
        }

        return newPath
    }

    /**
     * Get file size in human-readable format.
     */
    fun getFileSizeString(path: Path): String = try {
        val size = Files.size(path)
        when {
            size < 1024 -> "$size B"
            size < 1024 * 1024 -> "${size / 1024} KB"
            size < 1024 * 1024 * 1024 -> "${size / (1024 * 1024)} MB"
            else -> "${size / (1024 * 1024 * 1024)} GB"
        }
    } catch (e: IOException) {
        logger.error("Failed to get file size: $path", e)
        "Unknown"
    }

    /**
     * Clean up temporary files.
     */
    fun cleanupTempFiles(directory: Path, pattern: String = "temp_*") {
        try {
            Files.list(directory)
                .filter { path -> path.fileName.toString().matches(pattern.toRegex()) }
                .forEach { path ->
                    try {
                        Files.deleteIfExists(path)
                        logger.debug("Deleted temp file: $path")
                    } catch (e: IOException) {
                        logger.warn("Failed to delete temp file: $path", e)
                    }
                }
        } catch (e: IOException) {
            logger.error("Failed to cleanup temp files in: $directory", e)
        }
    }

    /**
     * Create backup of existing file.
     */
    fun createBackup(path: Path): Path? = try {
        if (Files.exists(path)) {
            val timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"))
            val backupPath = path.parent?.resolve("${path.fileName}.backup_$timestamp")

            if (backupPath != null) {
                Files.copy(path, backupPath)
                logger.info("Created backup: $backupPath")
                backupPath
            } else {
                null
            }
        } else {
            null
        }
    } catch (e: IOException) {
        logger.error("Failed to create backup for: $path", e)
        null
    }

    /**
     * Validate file exists and is readable.
     */
    fun validateReadPath(path: Path): Boolean = try {
        Files.exists(path) && Files.isReadable(path)
    } catch (e: Exception) {
        logger.error("Invalid read path: $path", e)
        false
    }

    /**
     * Get available disk space for directory.
     */
    fun getAvailableDiskSpace(path: Path): Long = try {
        val file = path.toFile()
        file.usableSpace
    } catch (e: Exception) {
        logger.error("Failed to get disk space for: $path", e)
        0L
    }

    /**
     * Check if file is locked by another process.
     */
    fun isFileLocked(path: Path): Boolean = try {
        val file = path.toFile()
        file.canWrite() && file.canRead()
    } catch (e: Exception) {
        logger.error("Failed to check file lock status: $path", e)
        true // Assume locked if we can't check
    }
}
