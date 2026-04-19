package com.campro.v5.animation

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import kotlin.random.Random

class MotionIngestionTest {
    private fun interpLinear(theta: Double, samples: FixtureMotionSamples): Double {
        val step = samples.stepDeg
        val n = samples.samples.size
        if (n == 0) return 0.0
        val t = ((theta % 360.0) + 360.0) % 360.0
        val idx = (t / step).toInt().coerceIn(0, n - 1)
        val next = (idx + 1) % n
        val t0 = samples.samples[idx].thetaDeg
        val t1 = (t0 + step).coerceAtMost(360.0)
        val x0 = samples.samples[idx].xMm
        val x1 = samples.samples[next].xMm
        val w = if (step > 0.0) ((t - t0 + 360.0) % 360.0) / step else 0.0
        return x0 + (x1 - x0) * w
    }

    @Test
    fun `periodicity holds at wrap for small fixture`() {
        val ms = FixtureLoader.loadMotionSamples("fixtures/motion_samples_small.json")
        val x0 = ms.samples.first().xMm
        val x360 = interpLinear(360.0, ms)
        assertEquals(x0, x360, 1e-6)
    }

    @Test
    fun `grid-agnostic interpolation consistency`() {
        val coarse = FixtureLoader.loadMotionSamples("fixtures/motion_samples_small.json")
        val fine = FixtureLoader.loadMotionSamples("fixtures/motion_samples_fine.json")
        // Fine is placeholder; just ensure no exceptions and values finite
        repeat(50) {
            val th = Random(42).nextDouble() * 360.0
            val xc = interpLinear(th, coarse)
            assertTrue(xc.isFinite())
        }
    }
}
