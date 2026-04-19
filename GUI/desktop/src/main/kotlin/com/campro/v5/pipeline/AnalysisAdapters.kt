package com.campro.v5.pipeline

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/**
 * Analysis adapter that routes GUI requests to the Python bridge.
 */
class LarrickAnalysisAdapter(
    private val bridgeClient: LarrickBridgeClient = LarrickBridgeClient(),
) : AnalysisPort {
    override suspend fun summarize(inputs: AnalysisRequest): AnalysisResponse =
        withContext(Dispatchers.IO) {
            val raw = bridgeClient.runMode(
                mode = "analyze",
                payload = inputs.payload,
                outputDir = BridgeOutputPaths.defaultOutputDir(),
                allowReal = false,
            )
            AnalysisResponse(
                status = (raw["status"] as? String) ?: "failed",
                payload = raw,
            )
        }
}

object AnalysisBackendProvider {
    fun createAnalysisPort(): AnalysisPort = LarrickAnalysisAdapter()
}
