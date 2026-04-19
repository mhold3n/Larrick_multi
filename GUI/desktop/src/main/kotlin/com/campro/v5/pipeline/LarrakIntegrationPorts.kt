package com.campro.v5.pipeline

import com.campro.v5.models.OptimizationParameters
import com.campro.v5.models.OptimizationResult
import java.nio.file.Path

/**
 * Frontend contract layer for big-bang Larrak integration.
 *
 * Intended behavior:
 * - GUI modules depend on these ports only.
 * - Concrete adapters can be swapped from stubs to real Larrak repos.
 *
 * Current behavior:
 * - Optimization supports `stub`, `larrick-stub`, `larrick-real`, and
 *   `legacy-campro` adapter modes.
 * - Orchestration/simulation/analysis/engine now have bridge-backed adapters.
 */

/**
 * Primary optimization boundary that the GUI consumes.
 */
interface OptimizationPort {
    suspend fun runOptimization(parameters: OptimizationParameters, outputDir: Path): OptimizationResult

    fun backendName(): String
}

/**
 * Placeholder analysis boundary for future `larrak-analysis`.
 */
interface AnalysisPort {
    suspend fun summarize(inputs: AnalysisRequest): AnalysisResponse
}

/**
 * Placeholder engine boundary for future `larrak-engines`.
 */
interface EnginePort {
    suspend fun evaluate(request: EngineRequest): EngineResponse
}

/**
 * Placeholder orchestration boundary for future `larrak-orchestration`.
 */
interface OrchestrationPort {
    suspend fun plan(request: OrchestrationRequest): OrchestrationResponse
}

/**
 * Placeholder simulation boundary for future `larrak-simulation`.
 */
interface SimulationPort {
    suspend fun simulate(request: SimulationRequest): SimulationResponse
}

/**
 * Placeholder core boundary for future shared `larrak-core`.
 */
interface CorePort {
    fun versionTag(): String
}

data class AnalysisRequest(val payload: Map<String, Any>)

data class AnalysisResponse(val status: String, val payload: Map<String, Any>)

data class EngineRequest(val payload: Map<String, Any>)

data class EngineResponse(val status: String, val payload: Map<String, Any>)

data class OrchestrationRequest(val payload: Map<String, Any>)

data class OrchestrationResponse(val status: String, val payload: Map<String, Any>)

data class SimulationRequest(val payload: Map<String, Any>)

data class SimulationResponse(val status: String, val payload: Map<String, Any>)
