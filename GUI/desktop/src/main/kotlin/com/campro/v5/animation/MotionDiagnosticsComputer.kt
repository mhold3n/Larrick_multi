package com.campro.v5.animation

import com.campro.v5.data.litvin.MotionLawSamples
import kotlin.math.abs

/** Simple diagnostics from motion-law samples: peak |a| and |jerk| (finite differences). */
object MotionDiagnosticsComputer {
    data class Result(val accelMaxAbsPerOmega2: Double, val jerkMaxAbsPerOmega3: Double)

    fun compute(m: MotionLawSamples): Result {
        val n = m.samples.size
        if (n < 2) return Result(0.0, 0.0)
        var aMax = 0.0
        var jMax = 0.0
        // max |a|
        for (s in m.samples) aMax = maxOf(aMax, abs(s.aMmPerOmega2))
        // estimate jerk via centered diff on acceleration over theta (per radian step)
        val stepDeg = m.stepDeg
        val stepRad = stepDeg * Math.PI / 180.0
        if (n >= 3 && stepRad != 0.0) {
            for (i in 1 until n - 1) {
                val jm = (m.samples[i + 1].aMmPerOmega2 - m.samples[i - 1].aMmPerOmega2) / (2.0 * stepRad)
                jMax = maxOf(jMax, abs(jm))
            }
        }
        return Result(aMax, jMax)
    }
}
