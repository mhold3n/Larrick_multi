package com.campro.v5.io

import com.campro.v5.models.OptimizationParameters
import com.campro.v5.models.OptimizationResult
import com.google.gson.Gson
import com.google.gson.GsonBuilder
import org.slf4j.LoggerFactory
import java.io.FileWriter
import java.io.IOException
import java.nio.file.Path
import java.nio.file.Paths
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter

/**
 * Result exporter for optimization results.
 *
 * Supports multiple export formats including JSON, CSV, and PDF.
 * Provides comprehensive result export with metadata and formatting.
 */
class ResultExporter {

    private val logger = LoggerFactory.getLogger(ResultExporter::class.java)
    private val gson: Gson = GsonBuilder()
        .setPrettyPrinting()
        .setDateFormat("yyyy-MM-dd HH:mm:ss")
        .create()

    /**
     * Export format enumeration.
     */
    enum class ExportFormat {
        JSON,
        CSV,
        PDF,
        EXCEL,
    }

    /**
     * Export optimization result to specified format.
     */
    fun exportResult(result: OptimizationResult, parameters: OptimizationParameters, outputPath: Path, format: ExportFormat): Path =
        when (format) {
            ExportFormat.JSON -> exportToJson(result, parameters, outputPath)
            ExportFormat.CSV -> exportToCsv(result, parameters, outputPath)
            ExportFormat.PDF -> exportToPdf(result, parameters, outputPath)
            ExportFormat.EXCEL -> exportToExcel(result, parameters, outputPath)
        }

    /**
     * Export to JSON format.
     */
    private fun exportToJson(result: OptimizationResult, parameters: OptimizationParameters, outputPath: Path): Path {
        val jsonPath = ensureJsonExtension(outputPath)

        return try {
            val exportData = createExportData(result, parameters)
            val json = gson.toJson(exportData)

            FileWriter(jsonPath.toFile()).use { writer ->
                writer.write(json)
            }

            logger.info("Exported results to JSON: $jsonPath")
            jsonPath
        } catch (e: IOException) {
            logger.error("Failed to export JSON: $jsonPath", e)
            throw e
        }
    }

    /**
     * Export to CSV format.
     */
    private fun exportToCsv(result: OptimizationResult, parameters: OptimizationParameters, outputPath: Path): Path {
        val csvPath = ensureCsvExtension(outputPath)

        return try {
            FileWriter(csvPath.toFile()).use { writer ->
                // Write metadata
                writer.write("Optimization Results Export\n")
                writer.write("Generated: ${LocalDateTime.now().format(DateTimeFormatter.ISO_LOCAL_DATE_TIME)}\n")
                writer.write("Status: ${result.status}\n")
                writer.write("Execution Time: ${result.executionTime}s\n\n")

                // Write parameters
                writer.write("Parameters\n")
                writer.write("Parameter,Value,Unit\n")
                writeParametersToCsv(writer, parameters)
                writer.write("\n")

                // Write motion law data
                writer.write("Motion Law Data\n")
                writer.write("Theta (deg),Displacement (mm),Velocity (mm/ω),Acceleration (mm/ω²)\n")
                writeMotionLawToCsv(writer, result.motionLaw)
                writer.write("\n")

                // Write gear profile data
                writer.write("Gear Profile Data\n")
                writer.write("Index,Sun Radius (mm),Planet Radius (mm),Ring Radius (mm)\n")
                writeGearProfilesToCsv(writer, result.optimalProfiles)
                writer.write("\n")

                // Write FEA analysis
                writer.write("FEA Analysis\n")
                writer.write("Metric,Value,Unit\n")
                writer.write("Max Stress,${result.feaAnalysis.maxStress},MPa\n")
                writer.write("Fatigue Life,${result.feaAnalysis.fatigueLife},cycles\n")
                writer.write("Natural Frequencies,${result.feaAnalysis.naturalFrequencies.joinToString(";")},Hz\n")
            }

            logger.info("Exported results to CSV: $csvPath")
            csvPath
        } catch (e: IOException) {
            logger.error("Failed to export CSV: $csvPath", e)
            throw e
        }
    }

    /**
     * Export to PDF format (simplified implementation).
     */
    private fun exportToPdf(result: OptimizationResult, parameters: OptimizationParameters, outputPath: Path): Path {
        val pdfPath = ensurePdfExtension(outputPath)

        return try {
            // For now, create a text-based PDF-like format
            // In a full implementation, you would use a PDF library like iText
            FileWriter(pdfPath.toFile()).use { writer ->
                writer.write("OPTIMIZATION RESULTS REPORT\n")
                writer.write("==========================\n\n")
                writer.write("Generated: ${LocalDateTime.now().format(DateTimeFormatter.ISO_LOCAL_DATE_TIME)}\n")
                writer.write("Status: ${result.status}\n")
                writer.write("Execution Time: ${result.executionTime}s\n\n")

                writer.write("PARAMETERS\n")
                writer.write("----------\n")
                writeParametersToText(writer, parameters)
                writer.write("\n")

                writer.write("MOTION LAW ANALYSIS\n")
                writer.write("------------------\n")
                writer.write("Data Points: ${result.motionLaw.thetaDeg.size}\n")
                writer.write("Max Displacement: ${result.motionLaw.displacement.maxOrNull()} mm\n")
                writer.write("Max Velocity: ${result.motionLaw.velocity.maxOrNull()} mm/ω\n")
                writer.write("Max Acceleration: ${result.motionLaw.acceleration.maxOrNull()} mm/ω²\n\n")

                writer.write("GEAR PROFILE ANALYSIS\n")
                writer.write("--------------------\n")
                writer.write("Optimal Method: ${result.optimalProfiles.optimalMethod}\n")
                writer.write("Gear Ratio: ${result.optimalProfiles.gearRatio}\n")
                writer.write("Average Sun Radius: ${result.optimalProfiles.rSun.average()} mm\n")
                writer.write("Average Planet Radius: ${result.optimalProfiles.rPlanet.average()} mm\n")
                writer.write("Average Ring Radius: ${result.optimalProfiles.rRingInner.average()} mm\n\n")

                writer.write("FEA ANALYSIS\n")
                writer.write("------------\n")
                writer.write("Max Stress: ${result.feaAnalysis.maxStress} MPa\n")
                writer.write("Fatigue Life: ${result.feaAnalysis.fatigueLife} cycles\n")
                writer.write("Natural Frequencies: ${result.feaAnalysis.naturalFrequencies.joinToString(", ")} Hz\n")
                writer.write("Recommendations:\n")
                result.feaAnalysis.recommendations.forEach { rec ->
                    writer.write("  • $rec\n")
                }
            }

            logger.info("Exported results to PDF: $pdfPath")
            pdfPath
        } catch (e: IOException) {
            logger.error("Failed to export PDF: $pdfPath", e)
            throw e
        }
    }

    /**
     * Export to Excel format (simplified implementation).
     */
    private fun exportToExcel(result: OptimizationResult, parameters: OptimizationParameters, outputPath: Path): Path {
        val excelPath = ensureExcelExtension(outputPath)

        return try {
            // For now, create a CSV file with .xlsx extension
            // In a full implementation, you would use Apache POI
            exportToCsv(result, parameters, excelPath)

            logger.info("Exported results to Excel: $excelPath")
            excelPath
        } catch (e: IOException) {
            logger.error("Failed to export Excel: $excelPath", e)
            throw e
        }
    }

    /**
     * Create comprehensive export data structure.
     */
    private fun createExportData(result: OptimizationResult, parameters: OptimizationParameters): Map<String, Any> = mapOf(
        "metadata" to mapOf(
            "exportedAt" to LocalDateTime.now().format(DateTimeFormatter.ISO_LOCAL_DATE_TIME),
            "version" to "1.0",
            "format" to "optimization_result",
        ),
        "summary" to mapOf(
            "status" to result.status,
            "executionTime" to result.executionTime,
            "error" to result.error,
        ),
        "parameters" to parameters,
        "results" to mapOf(
            "motionLaw" to result.motionLaw,
            "optimalProfiles" to result.optimalProfiles,
            "toothProfiles" to result.toothProfiles,
            "feaAnalysis" to result.feaAnalysis,
        ),
    )

    /**
     * Write parameters to CSV format.
     */
    private fun writeParametersToCsv(writer: FileWriter, parameters: OptimizationParameters) {
        val parameterMap = mapOf(
            "Sampling Step" to "${parameters.samplingStepDeg}°",
            "Stroke Length" to "${parameters.strokeLengthMm}mm",
            "Gear Ratio" to parameters.gearRatio.toString(),
            "RPM" to parameters.rpm.toString(),
            "Planet Count" to parameters.planetCount.toString(),
            "Rod Length" to "${parameters.rodLength}mm",
            "Journal Radius" to "${parameters.journalRadius}mm",
            "Ring Thickness" to "${parameters.ringThickness}mm",
            "Interference Buffer" to "${parameters.interferenceBuffer}mm",
        )

        parameterMap.forEach { (name, value) ->
            writer.write("$name,$value,\n")
        }
    }

    /**
     * Write motion law data to CSV format.
     */
    private fun writeMotionLawToCsv(writer: FileWriter, motionLaw: com.campro.v5.models.MotionLawData) {
        for (i in motionLaw.thetaDeg.indices) {
            writer.write("${motionLaw.thetaDeg[i]},${motionLaw.displacement[i]},${motionLaw.velocity[i]},${motionLaw.acceleration[i]}\n")
        }
    }

    /**
     * Write gear profile data to CSV format.
     */
    private fun writeGearProfilesToCsv(writer: FileWriter, gearProfiles: com.campro.v5.models.GearProfileData) {
        val maxLength = maxOf(gearProfiles.rSun.size, gearProfiles.rPlanet.size, gearProfiles.rRingInner.size)

        for (i in 0 until maxLength) {
            val sunRadius = if (i < gearProfiles.rSun.size) gearProfiles.rSun[i] else ""
            val planetRadius = if (i < gearProfiles.rPlanet.size) gearProfiles.rPlanet[i] else ""
            val ringRadius = if (i < gearProfiles.rRingInner.size) gearProfiles.rRingInner[i] else ""

            writer.write("$i,$sunRadius,$planetRadius,$ringRadius\n")
        }
    }

    /**
     * Write parameters to text format.
     */
    private fun writeParametersToText(writer: FileWriter, parameters: OptimizationParameters) {
        writer.write("Sampling Step: ${parameters.samplingStepDeg}°\n")
        writer.write("Stroke Length: ${parameters.strokeLengthMm}mm\n")
        writer.write("Gear Ratio: ${parameters.gearRatio}\n")
        writer.write("RPM: ${parameters.rpm}\n")
        writer.write("Planet Count: ${parameters.planetCount}\n")
        writer.write("Rod Length: ${parameters.rodLength}mm\n")
        writer.write("Journal Radius: ${parameters.journalRadius}mm\n")
        writer.write("Ring Thickness: ${parameters.ringThickness}mm\n")
        writer.write("Interference Buffer: ${parameters.interferenceBuffer}mm\n")
    }

    /**
     * Ensure file has JSON extension.
     */
    private fun ensureJsonExtension(path: Path): Path = if (FileIOUtils.getFileExtension(path) == "json") {
        path
    } else {
        path.parent?.resolve("${path.fileName}.json") ?: Paths.get("${path.fileName}.json")
    }

    /**
     * Ensure file has CSV extension.
     */
    private fun ensureCsvExtension(path: Path): Path = if (FileIOUtils.getFileExtension(path) == "csv") {
        path
    } else {
        path.parent?.resolve("${path.fileName}.csv") ?: Paths.get("${path.fileName}.csv")
    }

    /**
     * Ensure file has PDF extension.
     */
    private fun ensurePdfExtension(path: Path): Path = if (FileIOUtils.getFileExtension(path) == "pdf") {
        path
    } else {
        path.parent?.resolve("${path.fileName}.pdf") ?: Paths.get("${path.fileName}.pdf")
    }

    /**
     * Ensure file has Excel extension.
     */
    private fun ensureExcelExtension(path: Path): Path = if (FileIOUtils.getFileExtension(path) in listOf("xlsx", "xls")) {
        path
    } else {
        path.parent?.resolve("${path.fileName}.xlsx") ?: Paths.get("${path.fileName}.xlsx")
    }
}
