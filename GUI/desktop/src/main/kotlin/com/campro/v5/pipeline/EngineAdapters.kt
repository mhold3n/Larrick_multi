package com.campro.v5.pipeline

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Engine adapter that routes GUI requests to the Python bridge.
 */
class LarrickEngineAdapter(
    private val bridgeClient: LarrickBridgeClient = LarrickBridgeClient(),
) : EnginePort {
    override suspend fun evaluate(request: EngineRequest): EngineResponse =
        withContext(Dispatchers.IO) {
            val raw = bridgeClient.runMode(
                mode = "engine_eval",
                payload = request.payload,
                outputDir = BridgeOutputPaths.defaultOutputDir(),
                allowReal = false,
            )
            EngineResponse(
                status = (raw["status"] as? String) ?: "failed",
                payload = raw,
            )
        }
}

object EngineBackendProvider {
    fun createEnginePort(): EnginePort = LarrickEngineAdapter()
}
