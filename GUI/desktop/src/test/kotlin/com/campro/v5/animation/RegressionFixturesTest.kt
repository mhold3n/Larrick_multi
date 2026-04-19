package com.campro.v5.animation

import com.campro.v5.data.litvin.LitvinUserParams
import com.campro.v5.data.litvin.ProfileSolverMode
import com.campro.v5.data.litvin.RampProfile
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.Timeout
import java.util.concurrent.TimeUnit
import kotlin.math.abs

/**
 * Regression tests against known-good fixtures.
 */
class RegressionFixturesTest {
    private fun loadSmallFixture(): FixtureMotionSamples = FixtureLoader.loadMotionSamples("fixtures/motion_samples_small.json")

    @Test
    @Timeout(20, unit = TimeUnit.SECONDS)
    fun `piecewise engine matches small fixture approximately`() {
        val fixture = loadSmallFixture()
        // If generator metadata provided, reconstruct LitvinUserParams from it; otherwise fall back to amplitude-derived stroke
        val p = fixture.generator?.params ?: emptyMap()
        val strokeFromMeta = (p["strokeLengthMm"] as? Number)?.toDouble()
        val samplingFromMeta = (p["samplingStepDeg"] as? Number)?.toDouble() ?: fixture.stepDeg
        val rampProfileMeta =
            (p["rampProfile"] as? String)?.let {
                runCatching { RampProfile.valueOf(it) }.getOrNull()
            } ?: RampProfile.Cycloidal
        val dwellTdc = (p["dwellTdcDeg"] as? Number)?.toDouble() ?: 0.0
        val dwellBdc = (p["dwellBdcDeg"] as? Number)?.toDouble() ?: 0.0
        val rampAfterTdc = (p["rampAfterTdcDeg"] as? Number)?.toDouble() ?: 0.0
        val rampBeforeBdc = (p["rampBeforeBdcDeg"] as? Number)?.toDouble() ?: 0.0
        val rampAfterBdc = (p["rampAfterBdcDeg"] as? Number)?.toDouble() ?: 0.0
        val rampBeforeTdc = (p["rampBeforeTdcDeg"] as? Number)?.toDouble() ?: 0.0
        val upFraction = (p["upFraction"] as? Number)?.toDouble() ?: 0.5
        val rpm = (p["rpm"] as? Number)?.toDouble() ?: 3000.0

        val params =
            if (strokeFromMeta != null) {
                LitvinUserParams(
                    strokeLengthMm = strokeFromMeta,
                    samplingStepDeg = samplingFromMeta,
                    rampProfile = rampProfileMeta,
                    dwellTdcDeg = dwellTdc,
                    dwellBdcDeg = dwellBdc,
                    rampAfterTdcDeg = rampAfterTdc,
                    rampBeforeBdcDeg = rampBeforeBdc,
                    rampAfterBdcDeg = rampAfterBdc,
                    rampBeforeTdcDeg = rampBeforeTdc,
                    upFraction = upFraction,
                    rpm = rpm,
                    profileSolverMode = ProfileSolverMode.Piecewise,
                )
            } else {
                val fxVals = fixture.samples.map { it.xMm }
                val halfStroke =
                    fxVals.maxOrNull()?.let { max ->
                        kotlin.math.max(kotlin.math.abs(max), kotlin.math.abs(fxVals.minOrNull() ?: 0.0))
                    } ?: 7.5
                LitvinUserParams(
                    strokeLengthMm = halfStroke * 2.0,
                    samplingStepDeg = fixture.stepDeg,
                    rampProfile = RampProfile.Cycloidal,
                    profileSolverMode = ProfileSolverMode.Piecewise,
                )
            }

        val samples = MotionLawGenerator.generateMotion(params)
        assertEquals(fixture.stepDeg, samples.stepDeg, 1e-12)

        // Build value arrays for interpolation (grid-agnostic comparison)
        val fx = fixture.samples.map { it.xMm }
        val fv = fixture.samples.map { it.vMmPerOmega }
        val fa = fixture.samples.map { it.aMmPerOmega2 }
        val sx = samples.samples.map { it.xMm }
        val sv = samples.samples.map { it.vMmPerOmega }
        val sa = samples.samples.map { it.aMmPerOmega2 }

        val fxMin = fx.minOrNull() ?: 0.0
        val fxMax = fx.maxOrNull() ?: 0.0
        val xRange = (fxMax - fxMin).coerceAtLeast(1e-6)
        val tolX = maxOf(1.0, 0.12 * xRange) // 12% of range or 1mm
        val vRange = (fv.maxOrNull() ?: 0.0) - (fv.minOrNull() ?: 0.0)
        val aRange = (fa.maxOrNull() ?: 0.0) - (fa.minOrNull() ?: 0.0)
        val tolV = maxOf(0.5, 0.15 * vRange)
        val tolA = maxOf(0.3, 0.2 * aRange)

        // Compare at multiple angles via linear interpolation
        val step = fixture.stepDeg
        val thList = (0..36).map { it * 10.0 }

        // Build paired sequences for least-squares affine alignment (x only)
        val fxSeq = thList.map { AngleInterpolator.linear(it, step, fx) }
        val sxSeq = thList.map { AngleInterpolator.linear(it, samples.stepDeg, sx) }

        fun mean(xs: List<Double>) = xs.average()

        fun variance(xs: List<Double>): Double {
            val m = mean(xs)
            return xs.sumOf { (it - m) * (it - m) } / xs.size
        }

        fun covariance(xs: List<Double>, ys: List<Double>): Double {
            val mx = mean(xs)
            val my = mean(ys)
            return xs.indices.sumOf {
                (
                    xs[it] -
                        mx
                    ) *
                    (ys[it] - my)
            } /
                xs.size
        }

        val a = if (variance(sxSeq) > 1e-12) covariance(sxSeq, fxSeq) / variance(sxSeq) else 0.0
        val b = mean(fxSeq) - a * mean(sxSeq)

        // Compute MAE after alignment for x, and direct MAE for v and a
        var maeX = 0.0
        var maeV = 0.0
        var maeA = 0.0
        thList.forEach { th ->
            val fxInterp = AngleInterpolator.linear(th, step, fx)
            val fvInterp = AngleInterpolator.linear(th, step, fv)
            val faInterp = AngleInterpolator.linear(th, step, fa)
            val sxInterp = AngleInterpolator.linear(th, samples.stepDeg, sx)
            val svInterp = AngleInterpolator.linear(th, samples.stepDeg, sv)
            val saInterp = AngleInterpolator.linear(th, samples.stepDeg, sa)
            val sxAligned = a * sxInterp + b
            maeX += abs(fxInterp - sxAligned)
            maeV += abs(fvInterp - svInterp)
            maeA += abs(faInterp - saInterp)
        }
        maeX /= thList.size
        maeV /= thList.size
        maeA /= thList.size

        // Also require high shape correlation on displacement after alignment
        val alignedSxSeq = sxSeq.map { a * it + b }

        fun corr(x: List<Double>, y: List<Double>): Double {
            val mx = x.average()
            val my = y.average()
            val num = x.indices.sumOf { (x[it] - mx) * (y[it] - my) }
            val den = kotlin.math.sqrt(x.indices.sumOf { (x[it] - mx) * (x[it] - mx) } * y.indices.sumOf { (y[it] - my) * (y[it] - my) })
            return if (den > 1e-12) num / den else 0.0
        }
        val r = corr(fxSeq, alignedSxSeq)

        // Normalized RMSE for displacement after alignment
        fun rmse(x: List<Double>, y: List<Double>): Double {
            val n = x.size
            if (n == 0) return 0.0
            val sse = x.indices.sumOf { (x[it] - y[it]) * (x[it] - y[it]) }
            return kotlin.math.sqrt(sse / n)
        }
        val rangeX = xRange
        val nrmseX = if (rangeX > 1e-6) rmse(fxSeq, alignedSxSeq) / rangeX else 0.0

        // Accept if shapes correlate and error is within normalized bounds
        assertTrue(r >= 0.90, "shape correlation too low r=${"%.3f".format(r)} (<0.90)")
        assertTrue(nrmseX <= 0.20, "displacement NRMSE=${"%.3f".format(nrmseX)} (>0.20 of range)")

        // Velocity/accel can differ more across implementations; check relaxed MAE
        assertTrue(maeV <= tolV, "v MAE=${"%.3f".format(maeV)} > tol=$tolV")
        assertTrue(maeA <= tolA, "a MAE=${"%.3f".format(maeA)} > tol=$tolA")
    }

    @Test
    @Timeout(20, unit = TimeUnit.SECONDS)
    fun `engine handles fine fixture step and produces consistent grid`() {
        val fixture = FixtureLoader.loadMotionSamples("fixtures/motion_samples_fine.json")

        val fxVals = fixture.samples.map { it.xMm }
        val halfStroke =
            fxVals.maxOrNull()?.let { max ->
                kotlin.math.max(kotlin.math.abs(max), kotlin.math.abs(fxVals.minOrNull() ?: 0.0))
            } ?: 7.5
        val params =
            LitvinUserParams(
                strokeLengthMm = halfStroke * 2.0,
                samplingStepDeg = fixture.stepDeg,
                profileSolverMode = ProfileSolverMode.Piecewise,
            )

        val samples = MotionLawGenerator.generateMotion(params)
        assertEquals(fixture.stepDeg, samples.stepDeg, 1e-12)
        assertTrue(samples.samples.isNotEmpty(), "Expected non-empty samples for fine step")
        assertEquals(0.0, samples.samples.first().thetaDeg, 1e-12)
        assertTrue(samples.samples.last().thetaDeg > 350.0)
    }
}
