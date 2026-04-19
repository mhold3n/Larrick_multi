package com.campro.v5.pipeline

import com.campro.v5.models.FEAAnalysisData
import com.campro.v5.models.GearProfileData
import com.campro.v5.models.MotionLawData
import com.campro.v5.models.OptimizationParameters
import com.campro.v5.models.ConvergenceStatus
import com.campro.v5.models.OptimizationResult
import com.campro.v5.models.ToothProfileData
import com.campro.v5.utils.SimpleJsonUtils
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.slf4j.LoggerFactory
import java.io.File
import java.nio.file.Path
import java.util.concurrent.CompletableFuture
import java.util.concurrent.TimeUnit

/**
 * Bridge between Kotlin UI and Python unified optimization pipeline.
 *
 * This class handles parameter validation, conversion, result parsing,
 * and error handling for the unified optimization pipeline.
 */
class UnifiedOptimizationBridge(
    private val mode: String = "optimize",
    private val scriptOverride: String? = null,
    private val legacyCli: Boolean = false,
    private val allowReal: Boolean = false,
) {

    private val logger = LoggerFactory.getLogger(UnifiedOptimizationBridge::class.java)

    companion object {
        private const val DEFAULT_TIMEOUT_SECONDS = 30L
        private const val MAX_RETRIES = 3
        private const val PYTHON_EXE_PROP = "larrick.gui.pythonExe"
        private const val PYTHON_EXE_ENV = "LARRICK_GUI_PYTHON_EXE"
        private const val BRIDGE_SCRIPT_PROP = "larrick.gui.bridgeScript"
        private const val BRIDGE_SCRIPT_ENV = "LARRICK_GUI_BRIDGE_SCRIPT"
        private const val LARRICK_ROOT_ENV = "LARRICK_MULTI_ROOT"
        private const val LEGACY_SCRIPT_PATH = "../scripts/kotlin_bridge_cli.py"
    }

    /**
     * Run unified optimization pipeline with Kotlin parameters.
     *
     * @param parameters Optimization parameters from UI
     * @param outputDir Output directory for results
     * @return CompletableFuture with optimization result
     */
    suspend fun runOptimization(parameters: OptimizationParameters, outputDir: Path): CompletableFuture<OptimizationResult> =
        withContext(Dispatchers.IO) {
            return@withContext CompletableFuture.supplyAsync {
                try {
                    logger.info("Starting unified optimization pipeline")

                    // Validate parameters
                    validateParameters(parameters)

                    // Convert Kotlin parameters to Python format
                    val pythonParams = convertParametersToPython(parameters)

                    // Create temporary files for communication
                    val inputFile = createInputFile(pythonParams, outputDir)
                    val outputFile = createOutputFile(outputDir)

                    // Run Python pipeline
                    val success = runPythonPipeline(inputFile, outputFile, outputDir)

                    if (success) {
                        // Parse results
                        val result = parseResults(outputFile)
                        logger.info("Optimization completed successfully")
                        result
                    } else {
                        logger.error("Python pipeline execution failed")
                        createErrorResult("Pipeline execution failed")
                    }
                } catch (e: Exception) {
                    logger.error("Optimization failed: ${e.message}", e)
                    createErrorResult(e.message ?: "Unknown error")
                }
            }
        }

    /**
     * Validate optimization parameters.
     */
    private fun validateParameters(parameters: OptimizationParameters) {
        require(parameters.samplingStepDeg > 0) { "Sampling step must be positive" }
        require(parameters.strokeLengthMm > 0) { "Stroke length must be positive" }
        require(parameters.pistonDiameterMm > 0) { "Piston diameter must be positive" }
        require(parameters.gearRatio > 0) { "Gear ratio must be positive" }
        require(parameters.rpm > 0) { "RPM must be positive" }
        require(parameters.planetCount > 0) { "Planet count must be positive" }
        require(parameters.rodLength > 0) { "Rod length must be positive" }
        require(parameters.journalRadius > 0) { "Journal radius must be positive" }
        require(parameters.ringThickness > 0) { "Ring thickness must be positive" }
        require(parameters.interferenceBuffer >= 0) { "Interference buffer must be non-negative" }
    }

    /**
     * Convert Kotlin parameters to Python format.
     */
    private fun convertParametersToPython(parameters: OptimizationParameters): Map<String, Any> = mapOf(
        "samplingStepDeg" to parameters.samplingStepDeg,
        "ringRotationDeg" to parameters.ringRotationDeg,
        "gearRatio" to parameters.gearRatio,
        "strokeLengthMm" to parameters.strokeLengthMm,
        "pistonDiameterMm" to parameters.pistonDiameterMm,
        "pistonAreaMm2" to parameters.pistonAreaMm2(),
        "rodLength" to parameters.rodLength,
        "journalRadius" to parameters.journalRadius,
        "interferenceBuffer" to parameters.interferenceBuffer,
        "ringThickness" to parameters.ringThickness,
        "rpm" to parameters.rpm,
        "planetCount" to parameters.planetCount,
        "carrierOffsetDeg" to parameters.carrierOffsetDeg,
        "rampBeforeTdcDeg" to parameters.rampBeforeTdcDeg,
        "rampAfterTdcDeg" to parameters.rampAfterTdcDeg,
        "dwellTdcDeg" to parameters.dwellTdcDeg,
        "rampBeforeBdcDeg" to parameters.rampBeforeBdcDeg,
        "rampAfterBdcDeg" to parameters.rampAfterBdcDeg,
        "dwellBdcDeg" to parameters.dwellBdcDeg,
        "constantVelocityTdcDeg" to parameters.constantVelocityTdcDeg,
        "constantVelocityBdcDeg" to parameters.constantVelocityBdcDeg,
        // Add additional parameters that might be needed
        "planetRadiusBaseFactor" to 0.2,
        "planetRadiusVariationFactor" to 0.1,
        "sunRadiusBaseFactor" to 0.15,
        "sunRadiusVariationFactor" to 0.05,
        "strokeBasedVariationFactor" to 0.3,  // 30% of stroke length for significant asymmetry
        "strokeAchievableFactor" to 0.9,
        "clearanceSafetyMargin" to 0.2,
        "adjustmentSplitFactor" to 0.6,
        // New instantaneous ratio hyperparameters (calculated from user inputs)
        "rMin" to parameters.calculateGearRatioBounds().first,
        "rMax" to parameters.calculateGearRatioBounds().second,
        "rSmoothnessWeight" to parameters.rSmoothnessWeight,
        "motionVariationWeight" to parameters.motionVariationWeight,
        // Journal offset optimization parameters (calculated from user inputs)
        "journalOffsetMin" to parameters.journalOffsetMin,
        "journalOffsetMax" to parameters.journalOffsetMax,
        // User-driven bounds
        "maxGearRatioVariation" to parameters.maxGearRatioVariation,
        "maxJournalOffsetPercent" to parameters.maxJournalOffsetPercent,
        // Optional symmetry prior controls
        "enableSymmetryPrior" to parameters.enableSymmetryPrior,
        "symmetryWeight" to parameters.symmetryWeight,
    )

    /**
     * Create input parameter file for Python pipeline.
     */
    private fun createInputFile(parameters: Map<String, Any>, outputDir: Path): Path {
        val inputFile = outputDir.resolve("input_parameters.json")
        SimpleJsonUtils.writeJsonFile(parameters, inputFile)
        return inputFile
    }

    /**
     * Create output file path for Python pipeline results.
     */
    private fun createOutputFile(outputDir: Path): Path = outputDir.resolve("optimization_results.json")

    /**
     * Run Python pipeline with retry logic.
     */
    private fun runPythonPipeline(inputFile: Path, outputFile: Path, outputDir: Path): Boolean {
        var retryCount = 0
        var lastException: Exception? = null

        while (retryCount < MAX_RETRIES) {
            try {
                val command = buildPythonCommand(inputFile, outputFile, outputDir)
                logger.debug("Running Python command: ${command.joinToString(" ")}")

                val process = ProcessBuilder(command)
                    .directory(File(System.getProperty("user.dir")))
                    .start()

                val completed = process.waitFor(DEFAULT_TIMEOUT_SECONDS, TimeUnit.SECONDS)
                val exitCode = if (completed) process.exitValue() else -1
                val stdout = process.inputStream.bufferedReader().use { it.readText() }
                val stderr = process.errorStream.bufferedReader().use { it.readText() }
                val outputExists = outputFile.toFile().exists()

                if (completed && exitCode == 0) {
                    logger.info("Python pipeline completed successfully")
                    return true
                }

                if (!completed) {
                    process.destroyForcibly()
                    logger.warn("Python pipeline timed out (attempt ${retryCount + 1})")
                    lastException = RuntimeException("Pipeline timed out")
                } else {
                    logger.warn("Python pipeline exited with code $exitCode (attempt ${retryCount + 1})")
                    if (stdout.isNotBlank()) {
                        logger.debug("Python stdout: $stdout")
                    }
                    if (stderr.isNotBlank()) {
                        logger.debug("Python stderr: $stderr")
                    }
                    lastException = RuntimeException("Pipeline exited with code $exitCode")
                }

                if (outputExists) {
                    logger.info("Python pipeline produced output file despite non-zero exit code; attempting to parse results")
                    return true
                }
            } catch (e: Exception) {
                logger.warn("Python pipeline exception (attempt ${retryCount + 1}): ${e.message}")
                lastException = e
            }

            retryCount++
            if (retryCount < MAX_RETRIES) {
                Thread.sleep(1000) // Wait 1 second before retry
            }
        }

        logger.error("Python pipeline failed after $MAX_RETRIES attempts", lastException)
        return false
    }

    /**
     * Build Python command for pipeline execution.
     */
    private fun buildPythonCommand(inputFile: Path, outputFile: Path, outputDir: Path): List<String> {
        val pythonExe = resolvePythonExecutable()
        val scriptPath = resolveBridgeScriptPath()
        val command = mutableListOf(
            pythonExe,
            scriptPath,
            "--input",
            inputFile.toString(),
            "--output",
            outputFile.toString(),
            "--output-dir",
            outputDir.toString(),
        )
        if (!legacyCli) {
            command.add("--mode")
            command.add(mode)
            if (allowReal) {
                command.add("--real")
            }
        }
        return command
    }

    private fun resolvePythonExecutable(): String {
        val property = System.getProperty(PYTHON_EXE_PROP)?.trim().orEmpty()
        if (property.isNotEmpty()) {
            return property
        }
        val env = System.getenv(PYTHON_EXE_ENV)?.trim().orEmpty()
        if (env.isNotEmpty()) {
            return env
        }
        return "python3"
    }

    private fun resolveBridgeScriptPath(): String {
        if (!scriptOverride.isNullOrBlank()) {
            return scriptOverride
        }
        if (legacyCli) {
            return LEGACY_SCRIPT_PATH
        }
        val property = System.getProperty(BRIDGE_SCRIPT_PROP)?.trim().orEmpty()
        if (property.isNotEmpty()) {
            return property
        }
        val env = System.getenv(BRIDGE_SCRIPT_ENV)?.trim().orEmpty()
        if (env.isNotEmpty()) {
            return env
        }

        val userDir = Path.of(System.getProperty("user.dir")).normalize()
        val candidates = mutableListOf<Path>()
        System.getenv(LARRICK_ROOT_ENV)?.let { root ->
            candidates.add(Path.of(root).resolve("scripts/larrick_gui_bridge.py"))
        }
        candidates.add(userDir.resolve("../../scripts/larrick_gui_bridge.py").normalize())
        candidates.add(userDir.resolve("../scripts/larrick_gui_bridge.py").normalize())
        candidates.add(userDir.resolve("scripts/larrick_gui_bridge.py").normalize())

        val existing = candidates.firstOrNull { it.toFile().exists() }
        return (existing ?: candidates.first()).toString()
    }

    /**
     * Parse optimization results from Python output.
     */
    private fun parseResults(outputFile: Path): OptimizationResult {
        try {
            val resultData = SimpleJsonUtils.readJsonFile(outputFile)

            val motionLawMap = resultData["motion_law"] as? Map<String, Any> ?: emptyMap()
            val profilesMap = resultData["optimal_profiles"] as? Map<String, Any> ?: emptyMap()
            val toothMap = resultData["tooth_profiles"] as? Map<String, Any> ?: emptyMap()
            val feaMap = resultData["fea"] as? Map<String, Any> ?: emptyMap()
            val convergenceMap = resultData["convergence_status"] as? Map<String, Any>

            return OptimizationResult(
                status = (resultData["status"] as? String) ?: "failed",
                motionLaw = parseMotionLaw(motionLawMap),
                optimalProfiles = parseGearProfiles(profilesMap),
                toothProfiles = parseToothProfiles(toothMap),
                feaAnalysis = parseFEAAnalysis(feaMap),
                executionTime = (resultData["execution_time"] as? Number)?.toDouble() ?: 0.0,
                error = resultData["error"] as? String,
                convergence = convergenceMap?.let { parseConvergenceStatus(it) },
            )
        } catch (e: Exception) {
            logger.error("Failed to parse results: ${e.message}", e)
            return createErrorResult("Failed to parse results: ${e.message}")
        }
    }

    /**
     * Parse motion law data from Python result.
     */
    private fun parseMotionLaw(motionLawData: Map<String, Any>): MotionLawData {
        fun toDoubleArray(key: String): DoubleArray {
            val list = motionLawData[key] as? List<*> ?: return doubleArrayOf()
            return list.mapNotNull { (it as? Number)?.toDouble() }.toDoubleArray()
        }

        fun DoubleArray.applyScale(scale: Double): DoubleArray {
            if (isEmpty() || scale == 1.0) return this
            return DoubleArray(size) { this[it] * scale }
        }

        val lengthScale = 1_000.0 // Python pipeline provides meters; UI expects millimeters

        val displacement = toDoubleArray("displacement").applyScale(lengthScale)
        val velocity = toDoubleArray("velocity").applyScale(lengthScale)
        val acceleration = toDoubleArray("acceleration").applyScale(lengthScale)

        return MotionLawData(
            thetaDeg = toDoubleArray("theta_deg"),
            displacement = displacement,
            velocity = velocity,
            acceleration = acceleration,
        )
    }

    /**
     * Parse gear profile data from Python result.
     */
    private fun parseGearProfiles(profilesData: Map<String, Any>): GearProfileData {
        fun toDoubleArray(key: String): DoubleArray {
            val list = profilesData[key] as? List<*> ?: return doubleArrayOf()
            return list.mapNotNull { (it as? Number)?.toDouble() }.toDoubleArray()
        }

        val rSun = toDoubleArray("r_sun")
        val rPlanet = toDoubleArray("r_planet")
        val rRingInner = toDoubleArray("r_ring_inner")

        val gearRatio = (profilesData["gear_ratio"] as? Number)?.toDouble() ?: 0.0
        // Use solver_status as a proxy for method if available
        val optimalMethod = (profilesData["optimal_solution"] as? String)
            ?: (profilesData["solver_status"] as? String)
            ?: "unknown"

        val efficiencyAnalysis = profilesData["efficiency_analysis"] as? Map<String, Any>
        
        // New fields for variable instantaneous ratio
        val instantaneousRatio = toDoubleArray("instantaneous_ratio")
        val journalOffset = toDoubleArray("journal_offset")
        val accumulatedPlanetAngleDeg = (profilesData["accumulated_planet_angle_deg"] as? Number)?.toDouble() ?: 0.0
        // Discrete efficiency values for each angle
        val forceTransferEfficiency = toDoubleArray("force_transfer_efficiency")
        val powerTransferEfficiency = toDoubleArray("power_transfer_efficiency")
        val thermalEfficiency = toDoubleArray("thermal_efficiency_curve")

        val powerEfficiencySamples = if (powerTransferEfficiency.isNotEmpty()) {
            powerTransferEfficiency
        } else {
            forceTransferEfficiency
        }

        val forceEfficiencySamples = if (forceTransferEfficiency.isNotEmpty()) {
            forceTransferEfficiency
        } else {
            powerEfficiencySamples
        }

        return GearProfileData(
            rSun = rSun,
            rPlanet = rPlanet,
            rRingInner = rRingInner,
            gearRatio = gearRatio,
            optimalMethod = optimalMethod,
            efficiencyAnalysis = efficiencyAnalysis,
            instantaneousRatio = instantaneousRatio,
            journalOffset = journalOffset,
            accumulatedPlanetAngleDeg = accumulatedPlanetAngleDeg,
            forceTransferEfficiency = forceEfficiencySamples,
            powerTransferEfficiency = powerEfficiencySamples,
            thermalEfficiency = thermalEfficiency,
        )
    }

    /**
     * Parse tooth profile data from Python result.
     */
    private fun parseToothProfiles(toothData: Map<String, Any>): ToothProfileData = ToothProfileData(
        sunTeeth = parseToothArray(toothData["sun_teeth"]),
        planetTeeth = parseToothArray(toothData["planet_teeth"]),
        ringTeeth = parseToothArray(toothData["ring_teeth"]),
    )

    /**
     * Parse tooth array data.
     */
    private fun parseToothArray(toothData: Any?): Array<DoubleArray>? = if (toothData is List<*>) {
        toothData.map { tooth ->
            if (tooth is List<*>) {
                tooth.map { it as Double }.toDoubleArray()
            } else {
                doubleArrayOf()
            }
        }.toTypedArray()
    } else {
        null
    }

    /**
     * Parse FEA analysis data from Python result.
     */
    private fun parseFEAAnalysis(feaData: Map<String, Any>): FEAAnalysisData {
        val analysisSummary = feaData["analysis_summary"] as? Map<String, Any> ?: emptyMap()
        val stressAnalysis = feaData["stress_analysis"] as? Map<String, Any> ?: emptyMap()
        val fatigueAnalysis = feaData["fatigue_analysis"] as? Map<String, Any> ?: emptyMap()
        val vibrationAnalysis = feaData["vibration_analysis"] as? Map<String, Any> ?: emptyMap()

        // Extract stress data
        val maxStress = (stressAnalysis["max_stress"] as? Number)?.toDouble() 
            ?: (analysisSummary["max_stress"] as? Number)?.toDouble() ?: 0.0

        // Extract natural frequencies
        val naturalFrequencies = (vibrationAnalysis["natural_frequencies"] as? List<*>)?.mapNotNull { (it as? Number)?.toDouble() }?.toDoubleArray()
            ?: (analysisSummary["natural_frequencies"] as? List<*>)?.mapNotNull { (it as? Number)?.toDouble() }?.toDoubleArray()
            ?: doubleArrayOf()

        // Extract fatigue life
        val fatigueLife = (fatigueAnalysis["fatigue_life"] as? Number)?.toDouble()
            ?: (analysisSummary["fatigue_life"] as? Number)?.toDouble() ?: 0.0

        // Generate recommendations based on analysis results
        val recommendations = generateRecommendations(maxStress, fatigueLife, naturalFrequencies)

        return FEAAnalysisData(
            maxStress = maxStress,
            naturalFrequencies = naturalFrequencies,
            fatigueLife = fatigueLife,
            modeShapes = (vibrationAnalysis["mode_shapes"] as? List<String>)?.toTypedArray() ?: emptyArray(),
            recommendations = recommendations,
        )
    }

    private fun parseConvergenceStatus(data: Map<String, Any>): ConvergenceStatus {
        val constraintViolations = data["constraint_violations"] as? Map<String, Any>

        fun Number?.toDoubleOrNull(): Double? = this?.toDouble()
        fun Number?.toIntOrNull(): Int? = this?.toInt()

        return ConvergenceStatus(
            converged = data["converged"] as? Boolean ?: false,
            kktError = (data["kkt_error"] as? Number).toDoubleOrNull(),
            constraintTotalViolation = (constraintViolations?.get("total_violation") as? Number).toDoubleOrNull(),
            iterations = (data["iterations"] as? Number).toIntOrNull(),
            objectiveValue = (data["objective_value"] as? Number).toDoubleOrNull(),
            solverSuccess = data["solver_success"] as? Boolean,
        )
    }

    /**
     * Generate engineering recommendations based on FEA analysis results.
     */
    private fun generateRecommendations(maxStress: Double, fatigueLife: Double, naturalFrequencies: DoubleArray): Array<String> {
        val recommendations = mutableListOf<String>()

        // Stress-based recommendations
        when {
            maxStress > 500.0 -> {
                recommendations.add("⚠️ High stress detected (>500 MPa). Consider material upgrade or design modification.")
                recommendations.add("🔧 Increase gear thickness or use higher strength material.")
            }
            maxStress > 200.0 -> {
                recommendations.add("✅ Stress levels are acceptable but monitor for fatigue.")
                recommendations.add("📊 Consider stress concentration reduction at critical points.")
            }
            else -> {
                recommendations.add("✅ Stress levels are well within safe limits.")
            }
        }

        // Fatigue-based recommendations
        when {
            fatigueLife < 10000 -> {
                recommendations.add("⚠️ Low fatigue life (<10K cycles). Design needs improvement.")
                recommendations.add("🔧 Consider surface treatment or material change.")
            }
            fatigueLife < 100000 -> {
                recommendations.add("⚠️ Moderate fatigue life. Monitor operating conditions.")
                recommendations.add("📊 Consider reducing operating speed or load.")
            }
            else -> {
                recommendations.add("✅ Fatigue life is adequate for most applications.")
            }
        }

        // Natural frequency recommendations
        if (naturalFrequencies.isNotEmpty()) {
            val firstNaturalFreq = naturalFrequencies[0]
            when {
                firstNaturalFreq < 50.0 -> {
                    recommendations.add("⚠️ Low natural frequency. Risk of resonance.")
                    recommendations.add("🔧 Increase stiffness or reduce mass.")
                }
                firstNaturalFreq > 200.0 -> {
                    recommendations.add("✅ Natural frequencies are well above operating range.")
                }
                else -> {
                    recommendations.add("✅ Natural frequencies are acceptable.")
                }
            }
        }

        return recommendations.toTypedArray()
    }

    /**
     * Create error result for failed optimization.
     */
    private fun createErrorResult(errorMessage: String): OptimizationResult = OptimizationResult(
        status = "failed",
        motionLaw = MotionLawData(doubleArrayOf(), doubleArrayOf(), doubleArrayOf(), doubleArrayOf()),
        optimalProfiles = GearProfileData(doubleArrayOf(), doubleArrayOf(), doubleArrayOf(), 0.0, "none", null),
        toothProfiles = ToothProfileData(null, null, null),
        feaAnalysis = FEAAnalysisData(0.0, doubleArrayOf(), 0.0, emptyArray(), emptyArray()),
        executionTime = 0.0,
        error = errorMessage,
    )

    /**
     * Check if Python pipeline is available.
     */
    fun isPipelineAvailable(): Boolean = try {
        val command = listOf("python", "-c", "import campro.pipeline.unified_optimizer; print('OK')")
        val process = ProcessBuilder(command).start()
        val success = process.waitFor(5, TimeUnit.SECONDS)
        success && process.exitValue() == 0
    } catch (e: Exception) {
        logger.warn("Python pipeline not available: ${e.message}")
        false
    }

    /**
     * Get pipeline version information.
     */
    fun getPipelineVersion(): String = try {
        val command = listOf("python", "-c", "import campro; print(campro.__version__)")
        val process = ProcessBuilder(command).start()
        val success = process.waitFor(5, TimeUnit.SECONDS)

        if (success && process.exitValue() == 0) {
            process.inputStream.bufferedReader().readText().trim()
        } else {
            "unknown"
        }
    } catch (e: Exception) {
        logger.warn("Could not get pipeline version: ${e.message}")
        "unknown"
    }
}
