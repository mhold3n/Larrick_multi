package com.campro.v5.animation

import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import kotlin.random.Random
import kotlin.system.measureTimeMillis

class InterpolationPerformanceTest {
    @Test
    fun `angle interpolator handles 10k queries quickly and robustly`() {
        val values = (0 until 360).map { it.toDouble() }
        val step = 1.0
        val rnd = Random(1234)
        var sum = 0.0
        val elapsed =
            measureTimeMillis {
                repeat(10_000) {
                    val th = rnd.nextDouble() * 360.0
                    val v = AngleInterpolator.linear(th, step, values)
                    // accumulate to prevent JIT removing work
                    sum += v
                }
            }
        // Sanity: sum must be finite and time reasonable (< 200 ms on typical CI)
        assertTrue(sum.isFinite())
        assertTrue(elapsed < 200, "Interpolation took ${elapsed}ms, expected <200ms")
    }
}
