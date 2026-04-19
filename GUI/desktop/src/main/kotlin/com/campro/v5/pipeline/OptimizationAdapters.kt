package com.campro.v5.pipeline

import com.campro.v5.models.FEAAnalysisData
import com.campro.v5.models.GearProfileData
import com.campro.v5.models.MotionLawData
import com.campro.v5.models.OptimizationParameters
import com.campro.v5.models.OptimizationResult
import com.campro.v5.models.ToothProfileData
import com.campro.v5.utils.SimpleJsonUtils
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.slf4j.LoggerFactory
import java.nio.file.Files
import java.nio.file.Path
import kotlin.math.PI
import kotlin.math.sin

/**
 * Adapter backed by the legacy Python bridge.
 *
 * Intended behavior:
 * - Preserve compatibility while external repositories are integrated.
 *
 * Current behavior:
 * - Delegates to `UnifiedOptimizationBridge` and blocks until completion.
 */
class LegacyPythonOptimizationAdapter(
    private val bridge: UnifiedOptimizationBridge = UnifiedOptimizationBridge(
        mode = "optimize",
        scriptOverride = "../scripts/kotlin_bridge_cli.py",
        legacyCli = true,
    ),
) : OptimizationPort {
    override suspend fun runOptimization(parameters: OptimizationParameters, outputDir: Path): OptimizationResult =
        withContext(Dispatchers.IO) {
            bridge.runOptimization(parameters, outputDir).get()
        }

    override fun backendName(): String = "legacy-python"
}

/**
 * First-party Larrick bridge adapter for optimization payloads.
 */
class LarrickOptimizationAdapter(
    private val bridge: UnifiedOptimizationBridge = UnifiedOptimizationBridge(
        mode = "optimize",
        legacyCli = false,
        allowReal = false,
    ),
    private val modeName: String = "larrick-stub",
) : OptimizationPort {
    override suspend fun runOptimization(parameters: OptimizationParameters, outputDir: Path): OptimizationResult =
        withContext(Dispatchers.IO) {
            bridge.runOptimization(parameters, outputDir).get()
        }

    override fun backendName(): String = modeName
}

/**
 * Deterministic GUI-first stub adapter.
 *
 * Intended behavior:
 * - Produce stable, realistic-shaped payloads while no external engine is connected.
 *
 * Current behavior:
 * - Generates deterministic series from user parameters and writes a fixture JSON
 *   to the output directory for traceability.
 */
class StubOptimizationAdapter : OptimizationPort {
    private val logger = LoggerFactory.getLogger(StubOptimizationAdapter::class.java)

    override suspend fun runOptimization(parameters: OptimizationParameters, outputDir: Path): OptimizationResult =
        withContext(Dispatchers.Default) {
            val result = generateDeterministicResult(parameters)
            writeFixtureSnapshot(result, outputDir)
            result
        }

    override fun backendName(): String = "stub"

    private fun generateDeterministicResult(parameters: OptimizationParameters): OptimizationResult {
        val pointCount = 73
        val theta = DoubleArray(pointCount) { idx -> idx * (360.0 / (pointCount - 1)) }
        val stroke = parameters.strokeLengthMm
        val rpmScale = (parameters.rpm / 3000.0).coerceIn(0.5, 3.0)

        val displacement = DoubleArray(pointCount) { idx ->
            val radians = theta[idx] * PI / 180.0
            (stroke / 2.0) * (1.0 - sin(radians))
        }
        val velocity = DoubleArray(pointCount) { idx ->
            val radians = theta[idx] * PI / 180.0
            (stroke * rpmScale) * 0.35 * kotlin.math.cos(radians)
        }
        val acceleration = DoubleArray(pointCount) { idx ->
            val radians = theta[idx] * PI / 180.0
            -(stroke * rpmScale * rpmScale) * 0.8 * sin(radians)
        }

        val rSun = DoubleArray(pointCount) { idx ->
            95.0 + parameters.gearRatio * 5.0 + idx * 0.2
        }
        val rPlanet = DoubleArray(pointCount) { idx ->
            140.0 + parameters.journalRadius * 1.5 + idx * 0.15
        }
        val rRingInner = DoubleArray(pointCount) { idx ->
            360.0 + parameters.ringThickness * 2.0 + idx * 0.35
        }
        val ratioWave = DoubleArray(pointCount) { idx ->
            val radians = theta[idx] * PI / 180.0
            parameters.gearRatio + 0.08 * sin(radians * 2.0)
        }
        val journalOffset = DoubleArray(pointCount) { idx ->
            val radians = theta[idx] * PI / 180.0
            0.4 * parameters.journalRadius * sin(radians)
        }

        return OptimizationResult(
            status = "success",
            motionLaw = MotionLawData(theta, displacement, velocity, acceleration),
            optimalProfiles = GearProfileData(
                rSun = rSun,
                rPlanet = rPlanet,
                rRingInner = rRingInner,
                gearRatio = parameters.gearRatio,
                optimalMethod = "stub-larrak-ready",
                efficiencyAnalysis = mapOf("mode" to "stub", "source" to "local-fixture"),
                instantaneousRatio = ratioWave,
                journalOffset = journalOffset,
                accumulatedPlanetAngleDeg = theta.last(),
                forceTransferEfficiency = DoubleArray(pointCount) { 0.86 },
                powerTransferEfficiency = DoubleArray(pointCount) { 0.83 },
                thermalEfficiency = DoubleArray(pointCount) { 0.79 },
            ),
            toothProfiles = ToothProfileData(
                sunTeeth = null,
                planetTeeth = null,
                ringTeeth = null,
            ),
            feaAnalysis = FEAAnalysisData(
                maxStress = 172.0,
                naturalFrequencies = doubleArrayOf(118.0, 242.0, 389.0),
                fatigueLife = 850_000.0,
                modeShapes = arrayOf("ModeA", "ModeB", "ModeC"),
                recommendations = arrayOf(
                    "Stub mode active: connect larrak-analysis for real FEA.",
                    "Stub mode active: connect larrak-engines for native dynamics.",
                ),
            ),
            executionTime = 0.08,
            error = null,
        )
    }

    private fun writeFixtureSnapshot(result: OptimizationResult, outputDir: Path) {
        runCatching {
            Files.createDirectories(outputDir)
            val payload = mapOf(
                "status" to result.status,
                "backend" to backendName(),
                "execution_time" to result.executionTime,
                "motion_law" to mapOf(
                    "theta_deg" to result.motionLaw.thetaDeg.toList(),
                    "displacement" to result.motionLaw.displacement.toList(),
                    "velocity" to result.motionLaw.velocity.toList(),
                    "acceleration" to result.motionLaw.acceleration.toList(),
                ),
                "optimal_profiles" to mapOf(
                    "gear_ratio" to result.optimalProfiles.gearRatio,
                    "r_sun" to result.optimalProfiles.rSun.toList(),
                    "r_planet" to result.optimalProfiles.rPlanet.toList(),
                    "r_ring_inner" to result.optimalProfiles.rRingInner.toList(),
                    "instantaneous_ratio" to result.optimalProfiles.instantaneousRatio.toList(),
                    "journal_offset" to result.optimalProfiles.journalOffset.toList(),
                ),
                "fea" to mapOf(
                    "analysis_summary" to mapOf(
                        "max_stress" to result.feaAnalysis.maxStress,
                        "fatigue_life" to result.feaAnalysis.fatigueLife,
                        "natural_frequencies" to result.feaAnalysis.naturalFrequencies.toList(),
                    ),
                ),
            )
            SimpleJsonUtils.writeJsonFile(payload, outputDir.resolve("optimization_results.stub.json"))
        }.onFailure { logger.warn("Failed to write stub fixture snapshot: ${it.message}") }
    }
}

/**
 * Runtime selector for optimization backend adapters.
 *
 * Intended behavior:
 * - Keep GUI default frontend-only while permitting explicit compatibility mode.
 *
 * Current behavior:
 * - Reads `campro.backend.mode` and selects `stub` unless explicitly set to `legacy`.
 */
object OptimizationBackendProvider {
    private val logger = LoggerFactory.getLogger(OptimizationBackendProvider::class.java)
    private const val MODE_PROPERTY = "campro.backend.mode"

    fun createOptimizationPort(): OptimizationPort {
        val mode = System.getProperty(MODE_PROPERTY, "stub").trim().lowercase()
        return when (mode) {
            "legacy", "python", "legacy-campro" -> {
                logger.info("Using legacy optimization backend mode")
                LegacyPythonOptimizationAdapter()
            }
            "larrick-stub" -> {
                logger.info("Using larrick-stub optimization backend mode")
                LarrickOptimizationAdapter(
                    bridge = UnifiedOptimizationBridge(mode = "optimize", allowReal = false),
                    modeName = "larrick-stub",
                )
            }
            "larrick-real" -> {
                logger.info("Using larrick-real optimization backend mode")
                LarrickOptimizationAdapter(
                    bridge = UnifiedOptimizationBridge(mode = "optimize", allowReal = true),
                    modeName = "larrick-real",
                )
            }
            else -> {
                logger.info("Using stub optimization backend mode")
                StubOptimizationAdapter()
            }
        }
    }
}
