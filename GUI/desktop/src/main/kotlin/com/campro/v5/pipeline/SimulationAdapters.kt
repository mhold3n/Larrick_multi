package com.campro.v5.pipeline

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.slf4j.LoggerFactory

/**
 * Simulation adapter that routes GUI requests to the Python bridge.
 *
 * Intended behavior:
 * - Give the UI a stable simulation boundary without coupling to extracted
 *   package internals.
 */
class LarrickSimulationAdapter(
    private val bridgeClient: LarrickBridgeClient = LarrickBridgeClient(),
    private val allowReal: Boolean = false,
) : SimulationPort {
    private val logger = LoggerFactory.getLogger(LarrickSimulationAdapter::class.java)

    override suspend fun simulate(request: SimulationRequest): SimulationResponse =
        withContext(Dispatchers.IO) {
            logger.info("Running simulation bridge request (allowReal=$allowReal)")
            val raw = bridgeClient.runMode(
                mode = "simulate",
                payload = request.payload,
                outputDir = BridgeOutputPaths.defaultOutputDir(),
                allowReal = allowReal,
            )
            SimulationResponse(
                status = (raw["status"] as? String) ?: "failed",
                payload = raw,
            )
        }
}

/**
 * Provider for simulation adapter based on backend mode.
 */
object SimulationBackendProvider {
    private const val MODE_PROPERTY = "campro.backend.mode"

    fun createSimulationPort(): SimulationPort {
        val mode = System.getProperty(MODE_PROPERTY, "stub").trim().lowercase()
        return LarrickSimulationAdapter(allowReal = mode == "larrick-real")
    }
}
