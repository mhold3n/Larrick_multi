package com.campro.v5.pipeline

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.slf4j.LoggerFactory

/**
 * Orchestration adapter that routes GUI requests to the Python bridge.
 *
 * Intended behavior:
 * - Keep Kotlin orchestration callers isolated from Python implementation
 *   details.
 * - Support deterministic stub mode and opt-in real mode.
 */
class LarrickOrchestrationAdapter(
    private val bridgeClient: LarrickBridgeClient = LarrickBridgeClient(),
    private val allowReal: Boolean = false,
) : OrchestrationPort {
    private val logger = LoggerFactory.getLogger(LarrickOrchestrationAdapter::class.java)

    override suspend fun plan(request: OrchestrationRequest): OrchestrationResponse =
        withContext(Dispatchers.IO) {
            logger.info("Running orchestration bridge request (allowReal=$allowReal)")
            val raw = bridgeClient.runMode(
                mode = "orchestrate",
                payload = request.payload,
                outputDir = BridgeOutputPaths.defaultOutputDir(),
                allowReal = allowReal,
            )
            OrchestrationResponse(
                status = (raw["status"] as? String) ?: "failed",
                payload = raw,
            )
        }
}

/**
 * Provider for orchestration adapter based on backend mode.
 */
object OrchestrationBackendProvider {
    private const val MODE_PROPERTY = "campro.backend.mode"

    fun createOrchestrationPort(): OrchestrationPort {
        val mode = System.getProperty(MODE_PROPERTY, "stub").trim().lowercase()
        return LarrickOrchestrationAdapter(allowReal = mode == "larrick-real")
    }
}
