package com.campro.v5.fea

import com.campro.v5.emitError
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.IOException
import java.nio.file.Files
import java.nio.file.Path

/**
 * Wrapper class for the Rust FEA (Finite Element Analysis) engine.
 * This class provides a simplified interface to the Rust FEA engine through JNI.
 */
class FeaEngine {
    companion object {
        // Flag to track whether the native library is available
        private var nativeLibraryAvailable = false

        // Load the native library
        init {
            try {
                // Extract the native library to a temporary file
                val libraryPath = extractNativeLibrary()
                System.load(libraryPath.toString())
                println("Loaded Rust FEA engine native library from $libraryPath")

                // Verify the library is working correctly
                if (verifyNativeLibrary()) {
                    println("Native library verification successful")
                } else {
                    println("Native library verification failed")
                }
            } catch (e: Throwable) {
                val errorMessage =
                    when (e) {
                        is UnsatisfiedLinkError -> "Native library loading error: ${e.message}"
                        is IllegalStateException -> "Resource not found: ${e.message}"
                        is IOException -> "I/O error while extracting library: ${e.message}"
                        else -> "Unexpected error: ${e.message}"
                    }
                println("Failed to load Rust FEA engine native library: $errorMessage")
                e.printStackTrace()
                // Don't re-throw the exception - allow the application to continue
                // without FEA functionality. Set flag to indicate library is not available
                nativeLibraryAvailable = false
            }
        }

        /**
         * Extract the native library from the resources to a temporary file.
         *
         * @return The path to the extracted native library
         */
        private fun extractNativeLibrary(): Path {
            // Normalize OS name to remove spaces and version numbers
            val osName =
                when {
                    System.getProperty("os.name").toLowerCase().contains("win") -> "windows"
                    System.getProperty("os.name").toLowerCase().contains("mac") -> "mac"
                    System.getProperty("os.name").toLowerCase().contains("nix") ||
                        System.getProperty("os.name").toLowerCase().contains("nux") -> "linux"
                    else -> throw UnsupportedOperationException("Unsupported OS: ${System.getProperty("os.name")}")
                }

            // Normalize architecture
            val osArch =
                when {
                    System.getProperty("os.arch").toLowerCase().contains("amd64") -> "x86_64"
                    System.getProperty("os.arch").toLowerCase().contains("x86_64") -> "x86_64"
                    // Add ARM support if needed
                    else -> System.getProperty("os.arch").toLowerCase()
                }

            val libraryName =
                when {
                    osName == "windows" -> "campro_fea.dll"
                    osName == "mac" -> "libcampro_fea.dylib"
                    osName == "linux" -> "libcampro_fea.so"
                    else -> throw UnsupportedOperationException("Unsupported operating system: $osName")
                }

            val resourcePath = "/native/$osName/$osArch/$libraryName"
            println("Attempting to load native library from resource path: $resourcePath")

            val inputStream =
                FeaEngine::class.java.getResourceAsStream(resourcePath)
                    ?: throw IllegalStateException("Native library not found at $resourcePath")

            val tempDir = Files.createTempDirectory("campro_fea")
            val tempFile = tempDir.resolve(libraryName)

            inputStream.use { input ->
                Files.newOutputStream(tempFile).use { output ->
                    input.copyTo(output)
                }
            }

            println("Extracted native library to: $tempFile")

            // Ensure the library is deleted when the JVM exits
            tempFile.toFile().deleteOnExit()
            tempDir.toFile().deleteOnExit()

            return tempFile
        }

        /**
         * Verify that the native library is working correctly.
         *
         * @return true if the library is working correctly, false otherwise
         */
        private fun verifyNativeLibrary(): Boolean {
            try {
                // Try to call a simple native method to verify the library is working
                val testValue = testNativeLibraryNative()
                return testValue == 42 // Expected return value
            } catch (e: UnsatisfiedLinkError) {
                println("Native library verification failed: ${e.message}")
                return false
            }
        }

        /**
         * Test native method to verify that the library is working correctly.
         * The Rust implementation should return 42.
         */
        private external fun testNativeLibraryNative(): Int

        @JvmStatic fun isNativeAvailable(): Boolean = nativeLibraryAvailable
    }

    /**
     * Run a finite element analysis on the given model.
     *
     * @param modelFile The path to the model file
     * @param parameters The parameters for the analysis
     * @return The path to the results file
     */
    suspend fun runAnalysis(modelFile: File, parameters: Map<String, String>): File = withContext(Dispatchers.IO) {
        try {
            // Create a temporary file for the results
            val resultsFile = File.createTempFile("fea_results_", ".json")
            resultsFile.deleteOnExit()

            // Convert parameters to a format that can be passed to the native method
            val parameterArray = parameters.entries.flatMap { listOf(it.key, it.value) }.toTypedArray()

            // Call the native method
            runAnalysisNative(
                modelFile.absolutePath,
                resultsFile.absolutePath,
                parameterArray,
            )

            return@withContext resultsFile
        } catch (e: Exception) {
            emitError("Failed to run FEA analysis: ${e.message}")
            throw e
        }
    }

    /**
     * Run a stress analysis on the given model.
     *
     * @param modelFile The path to the model file
     * @param parameters The parameters for the analysis
     * @return The path to the results file
     */
    suspend fun runStressAnalysis(modelFile: File, parameters: Map<String, String>): File = withContext(Dispatchers.IO) {
        try {
            // Create a temporary file for the results
            val resultsFile = File.createTempFile("stress_results_", ".json")
            resultsFile.deleteOnExit()

            // Convert parameters to a format that can be passed to the native method
            val parameterArray = parameters.entries.flatMap { listOf(it.key, it.value) }.toTypedArray()

            // Call the native method
            runStressAnalysisNative(
                modelFile.absolutePath,
                resultsFile.absolutePath,
                parameterArray,
            )

            return@withContext resultsFile
        } catch (e: Exception) {
            emitError("Failed to run stress analysis: ${e.message}")
            throw e
        }
    }

    /**
     * Run a vibration analysis on the given model.
     *
     * @param modelFile The path to the model file
     * @param parameters The parameters for the analysis
     * @return The path to the results file
     */
    suspend fun runVibrationAnalysis(modelFile: File, parameters: Map<String, String>): File = withContext(Dispatchers.IO) {
        try {
            // Create a temporary file for the results
            val resultsFile = File.createTempFile("vibration_results_", ".json")
            resultsFile.deleteOnExit()

            // Convert parameters to a format that can be passed to the native method
            val parameterArray = parameters.entries.flatMap { listOf(it.key, it.value) }.toTypedArray()

            // Call the native method
            runVibrationAnalysisNative(
                modelFile.absolutePath,
                resultsFile.absolutePath,
                parameterArray,
            )

            return@withContext resultsFile
        } catch (e: Exception) {
            emitError("Failed to run vibration analysis: ${e.message}")
            throw e
        }
    }

    /**
     * Generate a mesh for the given model.
     *
     * @param modelFile The path to the model file
     * @param parameters The parameters for mesh generation
     * @return The path to the mesh file
     */
    suspend fun generateMesh(modelFile: File, parameters: Map<String, String>): File = withContext(Dispatchers.IO) {
        try {
            // Create a temporary file for the mesh
            val meshFile = File.createTempFile("mesh_", ".json")
            meshFile.deleteOnExit()

            // Convert parameters to a format that can be passed to the native method
            val parameterArray = parameters.entries.flatMap { listOf(it.key, it.value) }.toTypedArray()

            // Call the native method
            generateMeshNative(
                modelFile.absolutePath,
                meshFile.absolutePath,
                parameterArray,
            )

            return@withContext meshFile
        } catch (e: Exception) {
            emitError("Failed to generate mesh: ${e.message}")
            throw e
        }
    }

    /**
     * Check if the Rust FEA engine is available.
     *
     * @return True if the Rust FEA engine is available, false otherwise
     */
    fun isAvailable(): Boolean = try {
        // Use the companion object's testNativeLibraryNative() function
        // instead of the non-existent checkAvailabilityNative() function
        val testValue = Companion.testNativeLibraryNative()
        testValue == 42 // Expected return value
    } catch (e: Throwable) {
        // Catch UnsatisfiedLinkError and any other linkage/runtime issues
        false
    }

    /**
     * Get the version of the Rust FEA engine.
     *
     * @return The version of the Rust FEA engine
     */
    fun getVersion(): String = try {
        // Return a hardcoded version string instead of calling the non-existent native method
        // This is a temporary fix until the native method is implemented
        "1.0.0"
    } catch (e: Exception) {
        "Unknown"
    }

    // Native methods

    /**
     * Native method to run a finite element analysis.
     *
     * @param modelFilePath The path to the model file
     * @param resultsFilePath The path to the results file
     * @param parameters The parameters for the analysis
     */
    private external fun runAnalysisNative(modelFilePath: String, resultsFilePath: String, parameters: Array<String>)

    /**
     * Native method to run a stress analysis.
     *
     * @param modelFilePath The path to the model file
     * @param resultsFilePath The path to the results file
     * @param parameters The parameters for the analysis
     */
    private external fun runStressAnalysisNative(modelFilePath: String, resultsFilePath: String, parameters: Array<String>)

    /**
     * Native method to run a vibration analysis.
     *
     * @param modelFilePath The path to the model file
     * @param resultsFilePath The path to the results file
     * @param parameters The parameters for the analysis
     */
    private external fun runVibrationAnalysisNative(modelFilePath: String, resultsFilePath: String, parameters: Array<String>)

    /**
     * Native method to generate a mesh.
     *
     * @param modelFilePath The path to the model file
     * @param meshFilePath The path to the mesh file
     * @param parameters The parameters for mesh generation
     */
    private external fun generateMeshNative(modelFilePath: String, meshFilePath: String, parameters: Array<String>)

    /**
     * Native method to check if the Rust FEA engine is available.
     */
    private external fun checkAvailabilityNative()

    /**
     * Native method to get the version of the Rust FEA engine.
     *
     * @return The version of the Rust FEA engine
     */
    private external fun getVersionNative(): String

    /**
     * Test native method to verify that the library is working correctly.
     * The Rust implementation should return 42.
     */
    private external fun testNativeLibraryNative(): Int

    /**
     * Check if this FEA engine instance is functional.
     * This verifies that the native methods are available and working.
     */
    fun isFunctional(): Boolean = try {
        testNativeLibraryNative() == 42
    } catch (_: Throwable) {
        false
    }
}
