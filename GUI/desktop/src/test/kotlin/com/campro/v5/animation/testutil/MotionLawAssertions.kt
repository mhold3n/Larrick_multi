package com.campro.v5.animation.testutil

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import kotlin.math.abs
import kotlin.math.max

object MotionLawAssertions {
    data class Tolerances(
        val relX: Double,
        val absX: Double,
        val relV: Double,
        val absV: Double,
        val relA: Double,
        val absA: Double,
        val neighborhood: Int = 1,
    )

    fun assertNoNaNInf(x: DoubleArray, v: DoubleArray, a: DoubleArray) {
        fun finite(arr: DoubleArray) = arr.all { it.isFinite() }
        assertTrue(finite(x), "x contains NaN/Inf")
        assertTrue(finite(v), "v contains NaN/Inf")
        assertTrue(finite(a), "a contains NaN/Inf")
    }

    fun assertWrapContinuityExtrapolated360(thetaDeg: DoubleArray, x: DoubleArray, v: DoubleArray, a: DoubleArray, tol: Tolerances) {
        val n = thetaDeg.size
        require(n >= 2) { "Need at least 2 samples" }

        val i2 = n - 1
        val i1 = n - 2
        val step = thetaDeg[i2] - thetaDeg[i1]
        require(step > 0.0) { "Non-positive final step" }
        val ratio = (360.0 - thetaDeg[i2]) / step

        val x360 = x[i2] + ratio * (x[i2] - x[i1])
        val v360 = v[i2] + ratio * (v[i2] - v[i1])
        val a360 = a[i2] + ratio * (a[i2] - a[i1])

        val xMax = x.maxOf { abs(it) }.coerceAtLeast(1.0)
        val vMax = v.maxOf { abs(it) }.coerceAtLeast(1.0)
        val aMax = a.maxOf { abs(it) }.coerceAtLeast(1.0)

        val tx = max(tol.absX, tol.relX * xMax)
        val tv = max(tol.absV, tol.relV * vMax)
        val ta = max(tol.absA, tol.relA * aMax)

        assertEquals(x[0], x360, tx, "Wrap position mismatch (extrapolated x)")
        assertEquals(v[0], v360, tv, "Wrap velocity mismatch (extrapolated v)")
        assertEquals(a[0], a360, ta, "Wrap accel mismatch (extrapolated a)")
    }

    fun assertWrapContinuityExtrapolated180(thetaDeg: DoubleArray, x: DoubleArray, v: DoubleArray, a: DoubleArray, tol: Tolerances) {
        val n = thetaDeg.size
        require(n >= 2) { "Need at least 2 samples" }

        val i2 = n - 1
        val i1 = n - 2
        val step = thetaDeg[i2] - thetaDeg[i1]
        require(step > 0.0) { "Non-positive final step" }
        val ratio = (180.0 - thetaDeg[i2]) / step

        val x180 = x[i2] + ratio * (x[i2] - x[i1])
        val v180 = v[i2] + ratio * (v[i2] - v[i1])
        val a180 = a[i2] + ratio * (a[i2] - a[i1])

        val xMax = x.maxOf { abs(it) }.coerceAtLeast(1.0)
        val vMax = v.maxOf { abs(it) }.coerceAtLeast(1.0)
        val aMax = a.maxOf { abs(it) }.coerceAtLeast(1.0)

        val tx = max(tol.absX, tol.relX * xMax)
        val tv = max(tol.absV, tol.relV * vMax)
        val ta = max(tol.absA, tol.relA * aMax)

        assertEquals(x[0], x180, tx, "Wrap position mismatch (extrapolated x) - 180° periodicity")
        assertEquals(v[0], v180, tv, "Wrap velocity mismatch (extrapolated v) - 180° periodicity")
        assertEquals(a[0], a180, ta, "Wrap accel mismatch (extrapolated a) - 180° periodicity")
    }

    fun assertSamplingIntegrity(thetaDeg: DoubleArray, stepDeg: Double) {
        require(thetaDeg.isNotEmpty())
        var last = -1.0
        for (t in thetaDeg) {
            assertTrue(t >= last, "Theta not monotone: $t < $last")
            last = t
        }
        val k = 360.0 / stepDeg
        val kRound = kotlin.math.round(k)
        assertEquals(kRound, k, 1e-9, "360/stepDeg must be (near) integer: $k")
    }
}
