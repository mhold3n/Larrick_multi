package com.campro.v5.file

import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import com.campro.v5.layout.StateManager
import com.google.gson.Gson
import com.google.gson.GsonBuilder
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.withContext
import java.io.File
import java.util.concurrent.ConcurrentHashMap

/**
 * Handles file format conversion for the CamPro v5 application.
 * This class provides import from legacy formats, export to standard formats,
 * and batch conversion capabilities.
 */
class FormatConverter {
    // Supported import formats
    private val importFormats = ConcurrentHashMap<String, FormatImporter>()

    // Supported export formats
    private val exportFormats = ConcurrentHashMap<String, FormatExporter>()

    // Conversion events
    private val _conversionEvents = MutableStateFlow<ConversionEvent?>(null)
    val conversionEvents: StateFlow<ConversionEvent?> = _conversionEvents.asStateFlow()

    // State manager for persistence
    private val stateManager = StateManager.getInstance()

    init {
        // Register default importers
        registerDefaultImporters()

        // Register default exporters
        registerDefaultExporters()
    }

    /**
     * Register default importers.
     */
    private fun registerDefaultImporters() {
        // JSON importer
        registerImporter(
            FormatImporter(
                format = "json",
                name = "JSON",
                description = "JavaScript Object Notation",
                extensions = listOf("json"),
                importFn = { file ->
                    try {
                        val json = file.readText()
                        val gson = Gson()
                        val project = gson.fromJson(json, Project::class.java)
                        ConversionResult.Success(project)
                    } catch (e: Exception) {
                        ConversionResult.Error("Failed to import JSON: ${e.message}")
                    }
                },
            ),
        )

        // CSV importer
        registerImporter(
            FormatImporter(
                format = "csv",
                name = "CSV",
                description = "Comma-Separated Values",
                extensions = listOf("csv"),
                importFn = { file ->
                    try {
                        val lines = file.readLines()
                        if (lines.isEmpty()) {
                            return@FormatImporter ConversionResult.Error("Empty CSV file")
                        }

                        // Parse header
                        val header = lines[0].split(",")
                        if (header.size < 2 || header[0].trim() != "Parameter" || header[1].trim() != "Value") {
                            return@FormatImporter ConversionResult.Error("Invalid CSV format: Expected 'Parameter,Value' header")
                        }

                        // Parse parameters
                        val parameters = mutableMapOf<String, String>()
                        for (i in 1 until lines.size) {
                            val parts = lines[i].split(",")
                            if (parts.size >= 2) {
                                parameters[parts[0].trim()] = parts[1].trim()
                            }
                        }

                        // Create project
                        val project =
                            Project(
                                name = file.nameWithoutExtension,
                                parameters = parameters,
                                metadata =
                                ProjectMetadata(
                                    createdAt = System.currentTimeMillis(),
                                    modifiedAt = System.currentTimeMillis(),
                                    description = "Imported from CSV",
                                ),
                            )

                        ConversionResult.Success(project)
                    } catch (e: Exception) {
                        ConversionResult.Error("Failed to import CSV: ${e.message}")
                    }
                },
            ),
        )

        // XML importer
        registerImporter(
            FormatImporter(
                format = "xml",
                name = "XML",
                description = "Extensible Markup Language",
                extensions = listOf("xml"),
                importFn = { file ->
                    try {
                        // This is a placeholder. In a real implementation, this would use an XML parser.
                        ConversionResult.Error("XML import not implemented yet")
                    } catch (e: Exception) {
                        ConversionResult.Error("Failed to import XML: ${e.message}")
                    }
                },
            ),
        )

        // Legacy CamPro v3 importer
        registerImporter(
            FormatImporter(
                format = "campro3",
                name = "CamPro v3",
                description = "CamPro v3 Project File",
                extensions = listOf("cp3", "campro3"),
                importFn = { file ->
                    try {
                        // This is a placeholder. In a real implementation, this would parse the CamPro v3 format.
                        ConversionResult.Error("CamPro v3 import not implemented yet")
                    } catch (e: Exception) {
                        ConversionResult.Error("Failed to import CamPro v3 file: ${e.message}")
                    }
                },
            ),
        )

        // Legacy CamPro v4 importer
        registerImporter(
            FormatImporter(
                format = "campro4",
                name = "CamPro v4",
                description = "CamPro v4 Project File",
                extensions = listOf("cp4", "campro4"),
                importFn = { file ->
                    try {
                        // This is a placeholder. In a real implementation, this would parse the CamPro v4 format.
                        ConversionResult.Error("CamPro v4 import not implemented yet")
                    } catch (e: Exception) {
                        ConversionResult.Error("Failed to import CamPro v4 file: ${e.message}")
                    }
                },
            ),
        )
    }

    /**
     * Register default exporters.
     */
    private fun registerDefaultExporters() {
        // JSON exporter
        registerExporter(
            FormatExporter(
                format = "json",
                name = "JSON",
                description = "JavaScript Object Notation",
                extensions = listOf("json"),
                exportFn = { project, file ->
                    try {
                        val gson = GsonBuilder().setPrettyPrinting().create()
                        val json = gson.toJson(project)
                        file.writeText(json)
                        ConversionResult.Success(file)
                    } catch (e: Exception) {
                        ConversionResult.Error("Failed to export JSON: ${e.message}")
                    }
                },
            ),
        )

        // CSV exporter
        registerExporter(
            FormatExporter(
                format = "csv",
                name = "CSV",
                description = "Comma-Separated Values",
                extensions = listOf("csv"),
                exportFn = { project, file ->
                    try {
                        val csv = StringBuilder()
                        csv.appendLine("Parameter,Value")
                        project.parameters.forEach { (key, value) ->
                            csv.appendLine("$key,$value")
                        }
                        file.writeText(csv.toString())
                        ConversionResult.Success(file)
                    } catch (e: Exception) {
                        ConversionResult.Error("Failed to export CSV: ${e.message}")
                    }
                },
            ),
        )

        // XML exporter
        registerExporter(
            FormatExporter(
                format = "xml",
                name = "XML",
                description = "Extensible Markup Language",
                extensions = listOf("xml"),
                exportFn = { project, file ->
                    try {
                        // This is a placeholder. In a real implementation, this would use an XML writer.
                        ConversionResult.Error("XML export not implemented yet")
                    } catch (e: Exception) {
                        ConversionResult.Error("Failed to export XML: ${e.message}")
                    }
                },
            ),
        )

        // PDF report exporter
        registerExporter(
            FormatExporter(
                format = "pdf",
                name = "PDF Report",
                description = "Portable Document Format Report",
                extensions = listOf("pdf"),
                exportFn = { project, file ->
                    try {
                        // This is a placeholder. In a real implementation, this would generate a PDF report.
                        ConversionResult.Error("PDF export not implemented yet")
                    } catch (e: Exception) {
                        ConversionResult.Error("Failed to export PDF: ${e.message}")
                    }
                },
            ),
        )

        // Excel exporter
        registerExporter(
            FormatExporter(
                format = "xlsx",
                name = "Excel Workbook",
                description = "Microsoft Excel Workbook",
                extensions = listOf("xlsx"),
                exportFn = { project, file ->
                    try {
                        // This is a placeholder. In a real implementation, this would generate an Excel file.
                        ConversionResult.Error("Excel export not implemented yet")
                    } catch (e: Exception) {
                        ConversionResult.Error("Failed to export Excel: ${e.message}")
                    }
                },
            ),
        )
    }

    /**
     * Register a format importer.
     *
     * @param importer The importer to register
     */
    fun registerImporter(importer: FormatImporter) {
        importFormats[importer.format] = importer
    }

    /**
     * Register a format exporter.
     *
     * @param exporter The exporter to register
     */
    fun registerExporter(exporter: FormatExporter) {
        exportFormats[exporter.format] = exporter
    }

    /**
     * Get a format importer by format.
     *
     * @param format The format
     * @return The importer, or null if it wasn't found
     */
    fun getImporter(format: String): FormatImporter? = importFormats[format]

    /**
     * Get a format exporter by format.
     *
     * @param format The format
     * @return The exporter, or null if it wasn't found
     */
    fun getExporter(format: String): FormatExporter? = exportFormats[format]

    /**
     * Get all format importers.
     *
     * @return A list of all importers
     */
    fun getAllImporters(): List<FormatImporter> = importFormats.values.toList()

    /**
     * Get all format exporters.
     *
     * @return A list of all exporters
     */
    fun getAllExporters(): List<FormatExporter> = exportFormats.values.toList()

    /**
     * Import a project from a file.
     *
     * @param file The file to import from
     * @param format The format to import from, or null to auto-detect
     * @return The result of the import operation
     */
    suspend fun importProject(file: File, format: String? = null): ConversionResult<Project> = withContext(Dispatchers.IO) {
        try {
            // Check if the file exists
            if (!file.exists() || !file.isFile) {
                return@withContext ConversionResult.Error("File not found: ${file.absolutePath}")
            }

            // Determine format
            val importFormat =
                if (format != null) {
                    // Use specified format
                    importFormats[format] ?: return@withContext ConversionResult.Error("Unsupported import format: $format")
                } else {
                    // Auto-detect format from file extension
                    val extension = file.extension.lowercase()
                    importFormats.values.find { it.extensions.contains(extension) }
                        ?: return@withContext ConversionResult.Error("Unsupported file extension: $extension")
                }

            // Import project
            val result = importFormat.importFn(file)

            // Emit event
            if (result is ConversionResult.Success) {
                _conversionEvents.value = ConversionEvent.ProjectImported(file.absolutePath, importFormat.format)
            } else if (result is ConversionResult.Error) {
                _conversionEvents.value = ConversionEvent.ImportFailed(file.absolutePath, importFormat.format, result.message)
            }

            return@withContext result
        } catch (e: Exception) {
            val errorMessage = "Failed to import project: ${e.message}"
            _conversionEvents.value = ConversionEvent.ImportFailed(file.absolutePath, format ?: "unknown", errorMessage)
            return@withContext ConversionResult.Error(errorMessage)
        }
    }

    /**
     * Export a project to a file.
     *
     * @param project The project to export
     * @param file The file to export to
     * @param format The format to export to, or null to auto-detect
     * @return The result of the export operation
     */
    suspend fun exportProject(project: Project, file: File, format: String? = null): ConversionResult<File> = withContext(Dispatchers.IO) {
        try {
            // Determine format
            val exportFormat =
                if (format != null) {
                    // Use specified format
                    exportFormats[format] ?: return@withContext ConversionResult.Error("Unsupported export format: $format")
                } else {
                    // Auto-detect format from file extension
                    val extension = file.extension.lowercase()
                    exportFormats.values.find { it.extensions.contains(extension) }
                        ?: return@withContext ConversionResult.Error("Unsupported file extension: $extension")
                }

            // Export project
            val result = exportFormat.exportFn(project, file)

            // Emit event
            if (result is ConversionResult.Success) {
                _conversionEvents.value = ConversionEvent.ProjectExported(file.absolutePath, exportFormat.format)
            } else if (result is ConversionResult.Error) {
                _conversionEvents.value = ConversionEvent.ExportFailed(file.absolutePath, exportFormat.format, result.message)
            }

            return@withContext result
        } catch (e: Exception) {
            val errorMessage = "Failed to export project: ${e.message}"
            _conversionEvents.value = ConversionEvent.ExportFailed(file.absolutePath, format ?: "unknown", errorMessage)
            return@withContext ConversionResult.Error(errorMessage)
        }
    }

    /**
     * Perform batch conversion of multiple files.
     *
     * @param inputFiles The input files to convert
     * @param outputDir The output directory
     * @param inputFormat The input format, or null to auto-detect
     * @param outputFormat The output format
     * @return A list of conversion results
     */
    suspend fun batchConvert(
        inputFiles: List<File>,
        outputDir: File,
        inputFormat: String? = null,
        outputFormat: String,
    ): List<BatchConversionResult> = withContext(Dispatchers.IO) {
        // Check if output format is supported
        val exporter =
            exportFormats[outputFormat]
                ?: return@withContext listOf(
                    BatchConversionResult(null, null, ConversionResult.Error("Unsupported output format: $outputFormat")),
                )

        // Create output directory if it doesn't exist
        outputDir.mkdirs()

        // Process each input file
        val results = mutableListOf<BatchConversionResult>()

        for (inputFile in inputFiles) {
            try {
                // Import project
                val importResult = importProject(inputFile, inputFormat)

                if (importResult is ConversionResult.Success) {
                    // Create output file
                    val outputFile = File(outputDir, "${inputFile.nameWithoutExtension}.${exporter.extensions.first()}")

                    // Export project
                    val exportResult = exportProject(importResult.data, outputFile, outputFormat)

                    results.add(BatchConversionResult(inputFile, outputFile, exportResult))
                } else {
                    results.add(BatchConversionResult(inputFile, null, importResult))
                }
            } catch (e: Exception) {
                results.add(BatchConversionResult(inputFile, null, ConversionResult.Error("Failed to convert file: ${e.message}")))
            }
        }

        // Emit event
        _conversionEvents.value =
            ConversionEvent.BatchConversionCompleted(results.size, results.count { it.result is ConversionResult.Success })

        return@withContext results
    }

    companion object {
        // Singleton instance
        private var instance: FormatConverter? = null

        /**
         * Get the singleton instance of the FormatConverter.
         *
         * @return The FormatConverter instance
         */
        fun getInstance(): FormatConverter {
            if (instance == null) {
                instance = FormatConverter()
            }
            return instance!!
        }
    }
}

/**
 * A format importer.
 *
 * @param format The format identifier
 * @param name The display name of the format
 * @param description The description of the format
 * @param extensions The file extensions associated with the format
 * @param importFn The function to import a file in this format
 */
data class FormatImporter(
    val format: String,
    val name: String,
    val description: String,
    val extensions: List<String>,
    val importFn: suspend (File) -> ConversionResult<Project>,
)

/**
 * A format exporter.
 *
 * @param format The format identifier
 * @param name The display name of the format
 * @param description The description of the format
 * @param extensions The file extensions associated with the format
 * @param exportFn The function to export a project in this format
 */
data class FormatExporter(
    val format: String,
    val name: String,
    val description: String,
    val extensions: List<String>,
    val exportFn: suspend (Project, File) -> ConversionResult<File>,
)

/**
 * The result of a conversion operation.
 */
sealed class ConversionResult<out T> {
    /**
     * A successful conversion.
     *
     * @param data The converted data
     */
    data class Success<T>(val data: T) : ConversionResult<T>()

    /**
     * A failed conversion.
     *
     * @param message The error message
     */
    data class Error(val message: String) : ConversionResult<Nothing>()
}

/**
 * The result of a batch conversion operation.
 *
 * @param inputFile The input file
 * @param outputFile The output file, or null if conversion failed
 * @param result The result of the conversion
 */
data class BatchConversionResult(val inputFile: File?, val outputFile: File?, val result: ConversionResult<*>)

/**
 * Conversion events emitted by the FormatConverter.
 */
sealed class ConversionEvent {
    /**
     * Event emitted when a project is imported.
     *
     * @param filePath The path to the imported file
     * @param format The format of the imported file
     */
    data class ProjectImported(val filePath: String, val format: String) : ConversionEvent()

    /**
     * Event emitted when a project is exported.
     *
     * @param filePath The path to the exported file
     * @param format The format of the exported file
     */
    data class ProjectExported(val filePath: String, val format: String) : ConversionEvent()

    /**
     * Event emitted when a project import fails.
     *
     * @param filePath The path to the file
     * @param format The format of the file
     * @param message The error message
     */
    data class ImportFailed(val filePath: String, val format: String, val message: String) : ConversionEvent()

    /**
     * Event emitted when a project export fails.
     *
     * @param filePath The path to the file
     * @param format The format of the file
     * @param message The error message
     */
    data class ExportFailed(val filePath: String, val format: String, val message: String) : ConversionEvent()

    /**
     * Event emitted when a batch conversion is completed.
     *
     * @param totalFiles The total number of files processed
     * @param successfulFiles The number of files successfully converted
     */
    data class BatchConversionCompleted(val totalFiles: Int, val successfulFiles: Int) : ConversionEvent()
}

/**
 * Composable function to remember a FormatConverter instance.
 *
 * @return The remembered FormatConverter instance
 */
@Composable
fun rememberFormatConverter(): FormatConverter = remember { FormatConverter.getInstance() }
