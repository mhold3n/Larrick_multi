package com.campro.v5.animation

import com.campro.v5.data.litvin.RampProfile
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin

/**
 * Smooth ramp fraction utilities p(u) and its derivative dp/du for u in [0,1].
 * p(0)=0, p(1)=1, and p'(0)=p'(1)=0 to ensure acceleration continuity
 * when used for velocity ramps.
 */
object MotionProfiles {
    /** Fraction p(u) for given profile. */
    fun p(uRaw: Double, profile: RampProfile): Double {
        val u = uRaw.coerceIn(0.0, 1.0)
        return when (profile) {
            RampProfile.Cycloidal -> 0.5 * (1.0 - cos(PI * u))
            RampProfile.S5 -> {
                // 10u^3 - 15u^4 + 6u^5
                val u2 = u * u
                val u3 = u2 * u
                val u4 = u3 * u
                val u5 = u4 * u
                10.0 * u3 - 15.0 * u4 + 6.0 * u5
            }
            RampProfile.S7 -> {
                // 35u^4 - 84u^5 + 70u^6 - 20u^7
                val u2 = u * u
                val u3 = u2 * u
                val u4 = u3 * u
                val u5 = u4 * u
                val u6 = u5 * u
                val u7 = u6 * u
                35.0 * u4 - 84.0 * u5 + 70.0 * u6 - 20.0 * u7
            }
        }
    }

    /** First derivative dp/du for given profile. */
    fun dp(uRaw: Double, profile: RampProfile): Double {
        val u = uRaw.coerceIn(0.0, 1.0)
        return when (profile) {
            RampProfile.Cycloidal -> 0.5 * PI * sin(PI * u)
            RampProfile.S5 -> {
                val u2 = u * u
                val u3 = u2 * u
                val u4 = u3 * u
                30.0 * u2 - 60.0 * u3 + 30.0 * u4
            }
            RampProfile.S7 -> {
                val u2 = u * u
                val u3 = u2 * u
                val u4 = u3 * u
                val u5 = u4 * u
                val u6 = u5 * u
                140.0 * u3 - 420.0 * u4 + 420.0 * u5 - 140.0 * u6
            }
        }
    }
}
