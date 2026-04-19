package com.campro.v5.utils

import org.slf4j.LoggerFactory
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.StandardOpenOption
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter

/**
 * Utility class for file operations.
 */
object FileUtils {

    private val logger = LoggerFactory.getLogger(FileUtils::class.java)

    /**
     * Create directory if it doesn't exist.
     */
    fun createDirectoryIfNotExists(directory: Path) {
        if (!Files.exists(directory)) {
            try {
                Files.createDirectories(directory)
                logger.debug("Created directory: $directory")
            } catch (e: Exception) {
                logger.error("Failed to create directory: $directory", e)
                throw RuntimeException("Failed to create directory", e)
            }
        }
    }

    /**
     * Create temporary directory with timestamp.
     */
    fun createTempDirectory(prefix: String = "campro"): Path {
        val timestamp = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss"))
        val tempDirName = "${prefix}_$timestamp"
        val tempDir = Files.createTempDirectory(tempDirName)
        logger.debug("Created temporary directory: $tempDir")
        return tempDir
    }

    /**
     * Write string to file.
     */
    fun writeStringToFile(content: String, filePath: Path) {
        try {
            Files.write(filePath, content.toByteArray(), StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)
            logger.debug("Written string to file: $filePath")
        } catch (e: Exception) {
            logger.error("Failed to write string to file: $filePath", e)
            throw RuntimeException("Failed to write string to file", e)
        }
    }

    /**
     * Read string from file.
     */
    fun readStringFromFile(filePath: Path): String = try {
        val content = String(Files.readAllBytes(filePath))
        logger.debug("Read string from file: $filePath")
        content
    } catch (e: Exception) {
        logger.error("Failed to read string from file: $filePath", e)
        throw RuntimeException("Failed to read string from file", e)
    }

    /**
     * Check if file exists.
     */
    fun fileExists(filePath: Path): Boolean = Files.exists(filePath)

    /**
     * Get file size in bytes.
     */
    fun getFileSize(filePath: Path): Long = try {
        Files.size(filePath)
    } catch (e: Exception) {
        logger.error("Failed to get file size: $filePath", e)
        0L
    }

    /**
     * Delete file if it exists.
     */
    fun deleteFileIfExists(filePath: Path) {
        if (Files.exists(filePath)) {
            try {
                Files.delete(filePath)
                logger.debug("Deleted file: $filePath")
            } catch (e: Exception) {
                logger.error("Failed to delete file: $filePath", e)
                throw RuntimeException("Failed to delete file", e)
            }
        }
    }

    /**
     * Clean up temporary directory and all its contents.
     */
    fun cleanupTempDirectory(directory: Path) {
        if (Files.exists(directory)) {
            try {
                Files.walk(directory)
                    .sorted(Comparator.reverseOrder())
                    .forEach { path ->
                        Files.deleteIfExists(path)
                    }
                logger.debug("Cleaned up temporary directory: $directory")
            } catch (e: Exception) {
                logger.error("Failed to cleanup temporary directory: $directory", e)
                throw RuntimeException("Failed to cleanup temporary directory", e)
            }
        }
    }

    /**
     * Get file extension.
     */
    fun getFileExtension(filePath: Path): String {
        val fileName = filePath.fileName.toString()
        val lastDotIndex = fileName.lastIndexOf('.')
        return if (lastDotIndex > 0) {
            fileName.substring(lastDotIndex + 1)
        } else {
            ""
        }
    }

    /**
     * Get file name without extension.
     */
    fun getFileNameWithoutExtension(filePath: Path): String {
        val fileName = filePath.fileName.toString()
        val lastDotIndex = fileName.lastIndexOf('.')
        return if (lastDotIndex > 0) {
            fileName.substring(0, lastDotIndex)
        } else {
            fileName
        }
    }
}
