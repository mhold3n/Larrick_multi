package com.campro.v5.animation

import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows
import java.io.File

class FeaResultsLoaderTest {

    @Test
    fun `parses displacement and stress from array schema`() {
        val tmp = File.createTempFile("fea_results_array_schema", ".json")
        tmp.writeText(
            """
            {
              "displacements": [
                {"node": 1, "x": 0.1, "y": -0.02},
                {"node": 2, "x": 0.0, "y": 0.00}
              ],
              "stresses": [
                {"element": 5, "vonMises": 120.5},
                {"element": 6, "value": 80.0}
              ],
              "timeSteps": [0.0, 0.01, 0.02]
            }
            """.trimIndent(),
        )

        val data = FeaResultsLoader.loadResults(tmp)
        assertEquals(2, data.displacements.size)
        assertTrue(data.displacements.containsKey(1))
        assertEquals(0.1f, data.displacements[1]!!.x)
        assertEquals(-0.02f, data.displacements[1]!!.y)

        assertEquals(2, data.stresses.size)
        assertEquals(120.5f, data.stresses[5])
        assertEquals(80.0f, data.stresses[6])

        assertEquals(listOf(0.0f, 0.01f, 0.02f), data.timeSteps)

        tmp.delete()
    }

    @Test
    fun `parses displacement and stress from object schema`() {
        val tmp = File.createTempFile("fea_results_obj_schema", ".json")
        tmp.writeText(
            """
            {
              "displacements": {
                "1": {"x": 1.5, "y": 2.5},
                "3": {"x": 0.0, "y": 0.0}
              },
              "stresses": {
                "10": 200.0,
                "11": 150.0
              }
            }
            """.trimIndent(),
        )

        val data = FeaResultsLoader.loadResults(tmp)
        assertEquals(2, data.displacements.size)
        assertEquals(1.5f, data.displacements[1]!!.x)
        assertEquals(2.5f, data.displacements[1]!!.y)

        assertEquals(2, data.stresses.size)
        assertEquals(200.0f, data.stresses[10])
        assertEquals(150.0f, data.stresses[11])

        // timeSteps optional
        assertNotNull(data.timeSteps)

        tmp.delete()
    }

    @Test
    fun `handles missing timeSteps field gracefully`() {
        val tmp = File.createTempFile("fea_results_no_timesteps", ".json")
        tmp.writeText(
            """
            {
              "displacements": [
                {"node": 1, "x": 0.1, "y": -0.02}
              ],
              "stresses": [
                {"element": 5, "vonMises": 120.5}
              ]
            }
            """.trimIndent(),
        )

        val data = FeaResultsLoader.loadResults(tmp)
        assertNotNull(data.timeSteps)
        assertTrue(data.timeSteps.isEmpty(), "timeSteps should be empty when missing from JSON")

        assertEquals(1, data.displacements.size)
        assertEquals(1, data.stresses.size)

        tmp.delete()
    }

    @Test
    fun `handles empty displacements array`() {
        val tmp = File.createTempFile("fea_results_empty_displacements", ".json")
        tmp.writeText(
            """
            {
              "displacements": [],
              "stresses": [
                {"element": 5, "vonMises": 120.5}
              ],
              "timeSteps": [0.0, 0.01]
            }
            """.trimIndent(),
        )

        val data = FeaResultsLoader.loadResults(tmp)
        assertTrue(data.displacements.isEmpty(), "displacements should be empty")
        assertEquals(1, data.stresses.size)
        assertEquals(2, data.timeSteps.size)

        tmp.delete()
    }

    @Test
    fun `handles malformed stress data`() {
        val tmp = File.createTempFile("fea_results_malformed_stress", ".json")
        tmp.writeText(
            """
            {
              "displacements": [
                {"node": 1, "x": 0.1, "y": -0.02}
              ],
              "stresses": [
                {"element": 5, "vonMises": "invalid_number"},
                {"element": 6, "value": 80.0}
              ],
              "timeSteps": [0.0, 0.01]
            }
            """.trimIndent(),
        )

        // Should handle malformed data gracefully
        val data = FeaResultsLoader.loadResults(tmp)
        assertEquals(1, data.displacements.size)
        // Malformed stress entry should be skipped
        assertEquals(1, data.stresses.size)
        assertEquals(80.0f, data.stresses[6])

        tmp.delete()
    }

    @Test
    fun `handles very short time series`() {
        val tmp = File.createTempFile("fea_results_short_timeseries", ".json")
        tmp.writeText(
            """
            {
              "displacements": [
                {"node": 1, "x": 0.1, "y": -0.02}
              ],
              "stresses": [
                {"element": 5, "vonMises": 120.5}
              ],
              "timeSteps": [0.0]
            }
            """.trimIndent(),
        )

        val data = FeaResultsLoader.loadResults(tmp)
        assertEquals(1, data.displacements.size)
        assertEquals(1, data.stresses.size)
        assertEquals(1, data.timeSteps.size)
        assertEquals(0.0f, data.timeSteps[0])

        tmp.delete()
    }

    @Test
    fun `handles single point time series`() {
        val tmp = File.createTempFile("fea_results_single_point", ".json")
        tmp.writeText(
            """
            {
              "displacements": [
                {"node": 1, "x": 0.1, "y": -0.02}
              ],
              "stresses": [
                {"element": 5, "vonMises": 120.5}
              ],
              "timeSteps": [0.0]
            }
            """.trimIndent(),
        )

        val data = FeaResultsLoader.loadResults(tmp)
        assertNotNull(data)
        assertEquals(1, data.timeSteps.size)
        assertEquals(0.0f, data.timeSteps[0])

        tmp.delete()
    }

    @Test
    fun `handles missing displacements field`() {
        val tmp = File.createTempFile("fea_results_no_displacements", ".json")
        tmp.writeText(
            """
            {
              "stresses": [
                {"element": 5, "vonMises": 120.5}
              ],
              "timeSteps": [0.0, 0.01]
            }
            """.trimIndent(),
        )

        val data = FeaResultsLoader.loadResults(tmp)
        assertTrue(data.displacements.isEmpty(), "displacements should be empty when missing")
        assertEquals(1, data.stresses.size)
        assertEquals(2, data.timeSteps.size)

        tmp.delete()
    }

    @Test
    fun `handles missing stresses field`() {
        val tmp = File.createTempFile("fea_results_no_stresses", ".json")
        tmp.writeText(
            """
            {
              "displacements": [
                {"node": 1, "x": 0.1, "y": -0.02}
              ],
              "timeSteps": [0.0, 0.01]
            }
            """.trimIndent(),
        )

        val data = FeaResultsLoader.loadResults(tmp)
        assertEquals(1, data.displacements.size)
        assertTrue(data.stresses.isEmpty(), "stresses should be empty when missing")
        assertEquals(2, data.timeSteps.size)

        tmp.delete()
    }

    @Test
    fun `handles completely empty JSON`() {
        val tmp = File.createTempFile("fea_results_empty", ".json")
        tmp.writeText("{}")

        val data = FeaResultsLoader.loadResults(tmp)
        assertTrue(data.displacements.isEmpty(), "displacements should be empty")
        assertTrue(data.stresses.isEmpty(), "stresses should be empty")
        assertNotNull(data.timeSteps, "timeSteps should not be null")
        assertTrue(data.timeSteps.isEmpty(), "timeSteps should be empty")

        tmp.delete()
    }

    @Test
    fun `handles invalid JSON gracefully`() {
        val tmp = File.createTempFile("fea_results_invalid", ".json")
        tmp.writeText("{ invalid json }")

        assertThrows<Exception> {
            FeaResultsLoader.loadResults(tmp)
        }

        tmp.delete()
    }

    @Test
    fun `handles large datasets efficiently`() {
        val tmp = File.createTempFile("fea_results_large", ".json")

        // Generate large dataset
        val displacements = (1..1000).joinToString(",") {
            """{"node": $it, "x": ${it * 0.001}, "y": ${it * -0.0005}}"""
        }
        val stresses = (1..1000).joinToString(",") {
            """{"element": $it, "vonMises": ${it * 0.1}}"""
        }
        val timeSteps = (0..100).joinToString(",") { "${it * 0.01}" }

        tmp.writeText(
            """
            {
              "displacements": [$displacements],
              "stresses": [$stresses],
              "timeSteps": [$timeSteps]
            }
            """.trimIndent(),
        )

        val data = FeaResultsLoader.loadResults(tmp)
        assertEquals(1000, data.displacements.size)
        assertEquals(1000, data.stresses.size)
        assertEquals(101, data.timeSteps.size)

        // Verify some sample data
        assertEquals(0.001f, data.displacements[1]!!.x) // 1 * 0.001
        assertEquals(-0.0005f, data.displacements[1]!!.y) // 1 * -0.0005
        assertEquals(0.1f, data.stresses[1]) // 1 * 0.1

        tmp.delete()
    }
}
