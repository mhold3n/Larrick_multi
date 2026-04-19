package com.campro.v5.pipeline

import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test

/**
 * Contract tests for bridge-backed Larrick adapters.
 */
class LarrickBridgeAdaptersTest {
    @Test
    fun `bridge client returns stub simulation payload`() {
        val client = LarrickBridgeClient()
        val result = client.runMode(
            mode = "simulate",
            payload = mapOf("rpm" to 3000.0, "fidelity" to 1),
            outputDir = BridgeOutputPaths.defaultOutputDir(),
            allowReal = false,
        )

        assertEquals("success", result["status"])
        assertEquals("simulate", result["mode"])
        assertTrue(result.containsKey("payload"))
    }

    @Test
    fun `orchestration adapter maps bridge status`() =
        runBlocking {
            val adapter = LarrickOrchestrationAdapter(allowReal = false)
            val response = adapter.plan(
                OrchestrationRequest(
                    payload = mapOf("rpm" to 3000.0, "sim_budget" to 2, "max_iterations" to 1),
                ),
            )

            assertEquals("success", response.status)
            assertTrue(response.payload.containsKey("payload"))
        }
}
