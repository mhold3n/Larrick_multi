package com.campro.v5.fea

import com.campro.v5.emitError
import com.google.gson.Gson
import com.google.gson.GsonBuilder
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.*
import java.nio.channels.FileChannel
import java.nio.file.StandardOpenOption
import java.util.zip.GZIPInputStream
import java.util.zip.GZIPOutputStream

/**
 * Handles data transfer between Kotlin and Rust.
 * This class provides efficient mechanisms for transferring data, including memory-mapped files
 * for large datasets and data compression.
 */
class DataTransfer {
    private val gson: Gson = GsonBuilder().setPrettyPrinting().create()

    /**
     * Transfer data from Kotlin to Rust.
     *
     * @param data The data to transfer
     * @param compress Whether to compress the data
     * @return A file containing the data
     */
    suspend fun transferToRust(data: Any, compress: Boolean = false): File = withContext(Dispatchers.IO) {
        try {
            // Convert data to JSON
            val json = gson.toJson(data)

            // Create a temporary file
            val tempFile = File.createTempFile("kotlin_to_rust_", if (compress) ".json.gz" else ".json")
            tempFile.deleteOnExit()

            // Write data to file
            if (compress) {
                GZIPOutputStream(FileOutputStream(tempFile)).use { gzipOut ->
                    gzipOut.write(json.toByteArray())
                }
            } else {
                tempFile.writeText(json)
            }

            return@withContext tempFile
        } catch (e: Exception) {
            emitError("Failed to transfer data to Rust: ${e.message}", "DataTransfer")
            throw e
        }
    }

    /**
     * Transfer large data from Kotlin to Rust using memory-mapped files.
     *
     * @param data The data to transfer
     * @param compress Whether to compress the data
     * @return A file containing the data
     */
    suspend fun transferLargeDataToRust(data: Any, compress: Boolean = false): File = withContext(Dispatchers.IO) {
        try {
            // Convert data to JSON
            val json = gson.toJson(data)
            val jsonBytes = json.toByteArray()

            // Create a temporary file
            val tempFile = File.createTempFile("kotlin_to_rust_large_", if (compress) ".json.gz" else ".json")
            tempFile.deleteOnExit()

            // Use memory-mapped file for efficient writing
            val path = tempFile.toPath()

            if (compress) {
                // For compressed data, we need to use regular streams
                GZIPOutputStream(FileOutputStream(tempFile)).use { gzipOut ->
                    gzipOut.write(jsonBytes)
                }
            } else {
                // For uncompressed data, we can use memory-mapped files
                val channel =
                    FileChannel.open(
                        path,
                        StandardOpenOption.READ,
                        StandardOpenOption.WRITE,
                        StandardOpenOption.CREATE,
                    )

                channel.use { fileChannel ->
                    val buffer =
                        fileChannel.map(
                            FileChannel.MapMode.READ_WRITE,
                            0,
                            jsonBytes.size.toLong(),
                        )

                    buffer.put(jsonBytes)
                    buffer.force()
                }
            }

            return@withContext tempFile
        } catch (e: Exception) {
            emitError("Failed to transfer large data to Rust: ${e.message}", "DataTransfer")
            throw e
        }
    }

    /**
     * Transfer data from Rust to Kotlin.
     *
     * @param file The file containing the data
     * @param type The class of the data
     * @return The data
     */
    suspend fun <T> transferFromRust(file: File, type: Class<T>): T = withContext(Dispatchers.IO) {
        try {
            val json =
                if (file.name.endsWith(".gz")) {
                    // Read compressed data
                    GZIPInputStream(FileInputStream(file)).use { gzipIn ->
                        gzipIn.bufferedReader().use { reader ->
                            reader.readText()
                        }
                    }
                } else {
                    // Read uncompressed data
                    file.readText()
                }

            return@withContext gson.fromJson(json, type)
        } catch (e: Exception) {
            emitError("Failed to transfer data from Rust: ${e.message}", "DataTransfer")
            throw e
        }
    }

    /**
     * Transfer large data from Rust to Kotlin using memory-mapped files.
     *
     * @param file The file containing the data
     * @param type The class of the data
     * @return The data
     */
    suspend fun <T> transferLargeDataFromRust(file: File, type: Class<T>): T = withContext(Dispatchers.IO) {
        try {
            val json =
                if (file.name.endsWith(".gz")) {
                    // Read compressed data
                    GZIPInputStream(FileInputStream(file)).use { gzipIn ->
                        gzipIn.bufferedReader().use { reader ->
                            reader.readText()
                        }
                    }
                } else {
                    // Read uncompressed data using memory-mapped files
                    val path = file.toPath()
                    val channel = FileChannel.open(path, StandardOpenOption.READ)

                    channel.use { fileChannel ->
                        val buffer =
                            fileChannel.map(
                                FileChannel.MapMode.READ_ONLY,
                                0,
                                fileChannel.size(),
                            )

                        val bytes = ByteArray(buffer.remaining())
                        buffer.get(bytes)
                        String(bytes)
                    }
                }

            return@withContext gson.fromJson(json, type)
        } catch (e: Exception) {
            emitError("Failed to transfer large data from Rust: ${e.message}", "DataTransfer")
            throw e
        }
    }

    /**
     * Transfer a mesh from Rust to Kotlin.
     *
     * @param file The file containing the mesh data
     * @return The mesh
     */
    suspend fun transferMeshFromRust(file: File): Mesh = withContext(Dispatchers.IO) {
        try {
            return@withContext transferFromRust(file, Mesh::class.java)
        } catch (e: Exception) {
            emitError("Failed to transfer mesh from Rust: ${e.message}", "DataTransfer")
            throw e
        }
    }

    /**
     * Transfer analysis results from Rust to Kotlin.
     *
     * @param file The file containing the analysis results
     * @return The analysis results
     */
    suspend fun transferAnalysisResultsFromRust(file: File): AnalysisResults = withContext(Dispatchers.IO) {
        try {
            return@withContext transferFromRust(file, AnalysisResults::class.java)
        } catch (e: Exception) {
            emitError("Failed to transfer analysis results from Rust: ${e.message}", "DataTransfer")
            throw e
        }
    }

    /**
     * Create a cache file for data.
     *
     * @param data The data to cache
     * @param cacheKey A unique key for the cache
     * @param compress Whether to compress the data
     * @return The cache file
     */
    suspend fun createCacheFile(data: Any, cacheKey: String, compress: Boolean = true): File = withContext(Dispatchers.IO) {
        try {
            // Convert data to JSON
            val json = gson.toJson(data)

            // Create cache directory if it doesn't exist
            val cacheDir = File(System.getProperty("java.io.tmpdir"), "campro_cache")
            if (!cacheDir.exists()) {
                cacheDir.mkdirs()
            }

            // Create cache file
            val cacheFile = File(cacheDir, "$cacheKey${if (compress) ".json.gz" else ".json"}")

            // Write data to file
            if (compress) {
                GZIPOutputStream(FileOutputStream(cacheFile)).use { gzipOut ->
                    gzipOut.write(json.toByteArray())
                }
            } else {
                cacheFile.writeText(json)
            }

            return@withContext cacheFile
        } catch (e: Exception) {
            emitError("Failed to create cache file: ${e.message}", "DataTransfer")
            throw e
        }
    }

    /**
     * Read data from a cache file.
     *
     * @param cacheKey A unique key for the cache
     * @param type The class of the data
     * @return The data, or null if the cache file doesn't exist
     */
    suspend fun <T> readFromCache(cacheKey: String, type: Class<T>): T? = withContext(Dispatchers.IO) {
        try {
            // Check if cache file exists
            val cacheDir = File(System.getProperty("java.io.tmpdir"), "campro_cache")
            val cacheFileGz = File(cacheDir, "$cacheKey.json.gz")
            val cacheFileJson = File(cacheDir, "$cacheKey.json")

            val cacheFile =
                when {
                    cacheFileGz.exists() -> cacheFileGz
                    cacheFileJson.exists() -> cacheFileJson
                    else -> return@withContext null
                }

            // Read data from file
            val json =
                if (cacheFile.name.endsWith(".gz")) {
                    // Read compressed data
                    GZIPInputStream(FileInputStream(cacheFile)).use { gzipIn ->
                        gzipIn.bufferedReader().use { reader ->
                            reader.readText()
                        }
                    }
                } else {
                    // Read uncompressed data
                    cacheFile.readText()
                }

            return@withContext gson.fromJson(json, type)
        } catch (e: Exception) {
            emitError("Failed to read from cache: ${e.message}", "DataTransfer")
            return@withContext null
        }
    }

    /**
     * Clear the cache.
     */
    suspend fun clearCache() = withContext(Dispatchers.IO) {
        try {
            val cacheDir = File(System.getProperty("java.io.tmpdir"), "campro_cache")
            if (cacheDir.exists()) {
                cacheDir.listFiles()?.forEach { it.delete() }
            }
        } catch (e: Exception) {
            emitError("Failed to clear cache: ${e.message}", "DataTransfer")
        }
    }
}

/**
 * Represents a mesh for finite element analysis.
 */
data class Mesh(val nodes: List<Node>, val elements: List<Element>, val boundaries: List<Boundary>)

/**
 * Represents a node in a mesh.
 */
data class Node(val id: Int, val x: Double, val y: Double, val z: Double)

/**
 * Represents an element in a mesh.
 */
data class Element(val id: Int, val type: String, val nodeIds: List<Int>, val materialId: Int)

/**
 * Represents a boundary condition in a mesh.
 */
data class Boundary(val id: Int, val type: String, val nodeIds: List<Int>, val values: Map<String, Double>)

/**
 * Represents the results of a finite element analysis.
 */
data class AnalysisResults(
    val displacements: Map<Int, Vector3D>,
    val stresses: Map<Int, Stress>,
    val strains: Map<Int, Strain>,
    val forces: Map<Int, Vector3D>,
    val eigenvalues: List<Double>? = null,
    val eigenvectors: Map<Int, List<Vector3D>>? = null,
    val timeSteps: List<Double>? = null,
)

/**
 * Represents a 3D vector.
 */
data class Vector3D(val x: Double, val y: Double, val z: Double)

/**
 * Represents stress at a point.
 */
data class Stress(val xx: Double, val yy: Double, val zz: Double, val xy: Double, val yz: Double, val zx: Double, val vonMises: Double)

/**
 * Represents strain at a point.
 */
data class Strain(val xx: Double, val yy: Double, val zz: Double, val xy: Double, val yz: Double, val zx: Double, val equivalent: Double)
