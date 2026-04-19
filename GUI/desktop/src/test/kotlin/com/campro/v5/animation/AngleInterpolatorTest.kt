package com.campro.v5.animation

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test

class AngleInterpolatorTest {
    private fun lerp(a: Double, b: Double, t: Double): Double = a + (b - a) * t

    @Test
    fun `wrap-around interpolation basic`() {
        val a = 350.0
        val b = 10.0
        val mid = (a + ((b + 360.0) - a) * 0.5) % 360.0
        assertEquals(0.0, mid, 1e-9)
    }

    @Test
    fun `linear interpolator matches endpoints and mid`() {
        val values = listOf(0.0, 10.0, 20.0, 30.0)
        val step = 90.0
        assertEquals(0.0, AngleInterpolator.linear(0.0, step, values), 1e-9)
        assertEquals(10.0, AngleInterpolator.linear(90.0, step, values), 1e-9)
        assertEquals(5.0, AngleInterpolator.linear(45.0, step, values), 1e-9)
        assertEquals(15.0, AngleInterpolator.linear(135.0, step, values), 1e-9)
    }
}
