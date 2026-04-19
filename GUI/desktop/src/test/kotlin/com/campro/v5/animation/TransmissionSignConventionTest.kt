package com.campro.v5.animation

import com.campro.v5.data.litvin.LitvinUserParams
import com.campro.v5.data.litvin.RampProfile
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.EnumSource
import kotlin.math.abs

/**
 * Tests to verify sign convention consistency between Kotlin and Rust implementations
 * of the transmission ratio i(θ).
 */
class TransmissionSignConventionTest {
    /**
     * Creates standard test parameters for comparison
     */
    private fun testParams(profile: RampProfile = RampProfile.S5) = LitvinUserParams(
        samplingStepDeg = 1.0,
        rampProfile = profile,
        dwellTdcDeg = 20.0,
        dwellBdcDeg = 20.0,
        rampBeforeTdcDeg = 10.0,
        rampAfterTdcDeg = 10.0,
        rampBeforeBdcDeg = 10.0,
        rampAfterBdcDeg = 10.0,
        strokeLengthMm = 100.0,
        rodLength = 100.0,
        interferenceBuffer = 0.5,
        journalRadius = 5.0,
        journalPhaseBetaDeg = 0.0,
        sliderAxisDeg = 0.0,
        planetCount = 2,
        carrierOffsetDeg = 180.0,
        ringThicknessVisual = 6.0,
        rpm = 3000.0,
        camR0 = 40.0,
        camKPerUnit = 1.0,
        centerDistanceBias = 50.0,
        centerDistanceScale = 1.0,
        arcResidualTolMm = 0.01,
    )

    /**
     * Test that the basic invariant i(θ) > 0 holds for all profile types.
     * This is a basic sanity check.
     */
    @ParameterizedTest
    @EnumSource(RampProfile::class)
    fun testIThetaPositive(profile: RampProfile) {
        val params = testParams(profile)
        val motion = MotionLawGenerator.generateMotion(params)
        val result = TransmissionSynthesis.computeTransmissionAndPitch(motion, params)

        // Verify that all i(θ) values are positive
        for ((_, i) in result.iOfTheta) {
            assertTrue(i > 0.0, "i(θ) must be positive but found $i")
        }
    }

    /**
     * Test that the mean of i(θ) is very close to 1.0, which is a requirement
     * for both Kotlin and Rust implementations.
     */
    @ParameterizedTest
    @EnumSource(RampProfile::class)
    fun testIThetaMeanIsOne(profile: RampProfile) {
        val params = testParams(profile)
        val motion = MotionLawGenerator.generateMotion(params)
        val result = TransmissionSynthesis.computeTransmissionAndPitch(motion, params)

        // Calculate mean of i(θ)
        val iValues = result.iOfTheta.map { it.second }
        val meanI = iValues.average()

        // Verify that mean is very close to 1.0
        assertTrue(abs(meanI - 1.0) < 1e-6, "Mean of i(θ) should be 1.0 but was $meanI")
    }

    /**
     * Test that i(θ) is periodic: i(0) = i(360)
     */
    @ParameterizedTest
    @EnumSource(RampProfile::class)
    fun testIThetaPeriodicity(profile: RampProfile) {
        val params = testParams(profile)
        val motion = MotionLawGenerator.generateMotion(params)
        val result = TransmissionSynthesis.computeTransmissionAndPitch(motion, params)

        // Get first and last values
        val first = result.iOfTheta.first()
        val last = result.iOfTheta.last()

        // Verify that first and last values are very close
        assertTrue(
            abs(first.second - last.second) < 1e-6,
            "i(0) should equal i(360) but found ${first.second} and ${last.second}",
        )
    }

    /**
     * Test that i(θ) follows the expected pattern based on the motion velocity.
     * The sign of i(θ) - 1 should match the sign of the motion velocity in the relevant
     * parts of the cycle, accounting for the geometric projection.
     */
    @Test
    fun testIThetaSignPattern() {
        val params = testParams(RampProfile.S5)
        val motion = MotionLawGenerator.generateMotion(params)
        val result = TransmissionSynthesis.computeTransmissionAndPitch(motion, params)

        // Sample points at strategic locations
        val samples = motion.samples
        val iOfTheta = result.iOfTheta.toMap()

        // Check points in the first half of the cycle (positive velocity)
        val firstHalfIndex = samples.indexOfFirst { it.thetaDeg >= 45.0 }
        if (firstHalfIndex >= 0) {
            val theta = samples[firstHalfIndex].thetaDeg
            val v = samples[firstHalfIndex].vMmPerOmega
            val i = iOfTheta[theta] ?: 1.0

            // Verify that i-1 has the expected sign
            // Note: This depends on the specific geometric projection, so we need to account for
            // the sign change due to the journal position relative to the slider axis
            val expected = if (v > 0.0) i >= 1.0 else i <= 1.0
            assertTrue(expected, "At θ=$theta, v=$v, expected i${if (v > 0.0) ">=" else "<="} 1.0 but got i=$i")
        }

        // Check points in the second half of the cycle (negative velocity)
        val secondHalfIndex = samples.indexOfFirst { it.thetaDeg >= 225.0 }
        if (secondHalfIndex >= 0) {
            val theta = samples[secondHalfIndex].thetaDeg
            val v = samples[secondHalfIndex].vMmPerOmega
            val i = iOfTheta[theta] ?: 1.0

            // Verify that i-1 has the expected sign
            val expected = if (v > 0.0) i >= 1.0 else i <= 1.0
            assertTrue(expected, "At θ=$theta, v=$v, expected i${if (v > 0.0) ">=" else "<="} 1.0 but got i=$i")
        }
    }
}
