package com.campro.v5.animation

import kotlin.math.floor

object AngleInterpolator {
    fun wrapIndex(i: Int, n: Int): Int = ((i % n) + n) % n

    fun linear(theta: Double, stepDeg: Double, values: List<Double>): Double {
        val n = values.size
        if (n == 0) return 0.0
        if (n == 1) return values[0]
        val step = if (stepDeg > 0.0) stepDeg else 360.0 / n
        val t = ((theta % 360.0) + 360.0) % 360.0
        val idx = floor(t / step).toInt().coerceIn(0, n - 1)
        val next = wrapIndex(idx + 1, n)
        val t0 = idx * step
        val w = if (step > 0.0) ((t - t0 + 360.0) % 360.0) / step else 0.0
        val v0 = values[idx]
        val v1 = values[next]
        return v0 + (v1 - v0) * w
    }
}
