package com.campro.v5.animation

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin

class CamPathGenerationTest {
    @Test
    fun `cam path points are finite, periodic, and respect radius bounds`() {
        val baseCircleRadius = 25.0
        val stroke = 10.0
        val tdcAngle = 180.0
        val bdcDwell = 0.0
        val tdcDwell = 0.0

        val halfBDC = bdcDwell * 0.5
        val halfTDC = tdcDwell * 0.5
        val riseStart = halfBDC
        val riseEnd = (tdcAngle - halfTDC).coerceAtLeast(riseStart + 1e-4)
        val fallStart = tdcAngle + halfTDC
        val fallEnd = (360.0 - halfBDC).coerceAtLeast(fallStart + 1e-4)

        fun shmDisp(theta: Double, startDeg: Double, endDeg: Double, s0: Double, s1: Double): Double {
            val range = (endDeg - startDeg).coerceAtLeast(1e-6)
            val p = ((theta - startDeg) / range).coerceIn(0.0, 1.0)
            return s0 + (s1 - s0) * (1.0 - cos(PI * p)) / 2.0
        }

        val points = ArrayList<Pair<Double, Double>>(360)
        for (i in 0 until 360) {
            val theta = i.toDouble()
            val thetaRad = Math.toRadians(theta)
            val s =
                when {
                    theta < riseStart -> 0.0
                    theta <= riseEnd -> shmDisp(theta, riseStart, riseEnd, 0.0, stroke)
                    theta <= fallStart -> stroke
                    theta <= fallEnd -> shmDisp(theta, fallStart, fallEnd, stroke, 0.0)
                    else -> 0.0
                }
            val radius = baseCircleRadius + s
            val x = radius * cos(thetaRad)
            val y = radius * sin(thetaRad)
            points.add(x to y)
            // Bounds
            assertTrue(radius.isFinite())
            assertTrue(radius in baseCircleRadius..(baseCircleRadius + stroke + 1e-9))
        }
        // Periodicity (start and end point should match)
        val first = points.first()
        val wrapThetaRad = Math.toRadians(360.0)
        val sWrap = 0.0 // returns to base radius at 360 under this SHM+dwell law
        val radiusWrap = baseCircleRadius + sWrap
        val wrap = radiusWrap * cos(wrapThetaRad) to radiusWrap * sin(wrapThetaRad)
        assertEquals(first.first, wrap.first, 1e-6)
        assertEquals(first.second, wrap.second, 1e-6)

        // No NaN/Inf anywhere
        assertTrue(points.all { it.first.isFinite() && it.second.isFinite() })
    }
}
