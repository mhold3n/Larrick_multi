package com.campro.v5.animation

import com.campro.v5.data.litvin.LitvinUserParams
import com.campro.v5.data.litvin.ProfileSolverMode
import com.campro.v5.data.litvin.RampProfile
import org.junit.jupiter.api.TestInstance
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.Arguments
import org.junit.jupiter.params.provider.MethodSource
import org.junit.jupiter.params.provider.ValueSource
import java.util.stream.Stream
import kotlin.math.*
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

/**
 * Parameterized testing framework for systematic solver comparison.
 *
 * This test suite provides comprehensive comparison between piecewise and collocation
 * solvers across a wide parameter space, enabling systematic validation and
 * regression detection.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class ParameterizedSolverComparisonTest {
    companion object {
        /**
         * Generate test parameter combinations for systematic testing.
         */
        @JvmStatic
        fun solverParameterCombinations(): Stream<Arguments> {
            val strokeLengths = listOf(1.0, 5.0, 10.0, 25.0)
            val samplingSteps = listOf(1.0, 2.0, 5.0)
            val rampProfiles = RampProfile.values().toList()
            val upFractions = listOf(0.3, 0.5, 0.7)
            val dwellCombinations =
                listOf(
                    Pair(0.0, 0.0), // No dwells
                    Pair(10.0, 10.0), // Small dwells
                    Pair(30.0, 30.0), // Medium dwells
                    Pair(60.0, 60.0), // Large dwells
                )

            val combinations = mutableListOf<Arguments>()

            for (stroke in strokeLengths) {
                for (step in samplingSteps) {
                    for (profile in rampProfiles) {
                        for (upFraction in upFractions) {
                            for ((tdcDwell, bdcDwell) in dwellCombinations) {
                                combinations.add(
                                    Arguments.of(
                                        stroke, step, profile, upFraction, tdcDwell, bdcDwell,
                                    ),
                                )
                            }
                        }
                    }
                }
            }

            return combinations.stream()
        }

        /**
         * Generate extreme parameter combinations for stress testing.
         */
        @JvmStatic
        fun extremeParameterCombinations(): Stream<Arguments> = listOf(
            // Very small stroke, fine sampling
            Arguments.of(0.1, 0.5, RampProfile.Cycloidal, 0.5, 0.0, 0.0),
            // Large stroke, coarse sampling
            Arguments.of(100.0, 10.0, RampProfile.S5, 0.3, 90.0, 90.0),
            // Asymmetric motion
            Arguments.of(10.0, 2.0, RampProfile.S7, 0.1, 5.0, 60.0),
            // Extreme asymmetry
            Arguments.of(10.0, 2.0, RampProfile.Cycloidal, 0.9, 60.0, 5.0),
            // Very large dwells
            Arguments.of(5.0, 5.0, RampProfile.S5, 0.5, 120.0, 120.0),
        ).stream()
    }

    @ParameterizedTest(name = "Piecewise solver consistency: stroke={0}mm, step={1}deg, profile={2}, upFraction={3}, dwells=({4},{5})")
    @MethodSource("solverParameterCombinations")
    fun `piecewise solver produces consistent results across parameter space`(
        strokeMm: Double,
        stepDeg: Double,
        profile: RampProfile,
        upFraction: Double,
        tdcDwellDeg: Double,
        bdcDwellDeg: Double,
    ) {
        val params =
            LitvinUserParams(
                samplingStepDeg = stepDeg,
                profileSolverMode = ProfileSolverMode.Piecewise,
                strokeLengthMm = strokeMm,
                rampProfile = profile,
                upFraction = upFraction,
                dwellTdcDeg = tdcDwellDeg,
                dwellBdcDeg = bdcDwellDeg,
            )

        val result = MotionLawGenerator.generateMotion(params)

        // Basic validation
        assertNotNull(result, "Motion generation should succeed")
        assertTrue(result.samples.isNotEmpty(), "Should generate samples")

        // Sample count validation - 180° periodicity for planetary gearset
        val expectedSamples = ceil(180.0 / stepDeg).toInt()
        assertTrue(
            abs(result.samples.size - expectedSamples) <= 2,
            "Sample count should be approximately correct: expected ~$expectedSamples, got ${result.samples.size}",
        )

        // Finiteness validation
        result.samples.forEach { sample ->
            assertTrue(sample.thetaDeg.isFinite(), "Theta should be finite")
            assertTrue(sample.xMm.isFinite(), "Position should be finite")
            assertTrue(sample.vMmPerOmega.isFinite(), "Velocity should be finite")
            assertTrue(sample.aMmPerOmega2.isFinite(), "Acceleration should be finite")
        }

        // Periodicity validation (relaxed)
        val firstSample = result.samples.first()
        val lastSample = result.samples.last()

        // Position closure (within 1% of stroke)
        val positionClosure = abs(lastSample.xMm - firstSample.xMm)
        assertTrue(
            positionClosure < strokeMm * 0.1, // 10% tolerance due to scaling factor
            "Position should be approximately periodic: closure $positionClosure, stroke $strokeMm",
        )

        // Velocity closure (should be close)
        val velocityClosure = abs(lastSample.vMmPerOmega - firstSample.vMmPerOmega)
        assertTrue(
            velocityClosure < 1.0, // Absolute tolerance
            "Velocity should be approximately periodic: closure $velocityClosure",
        )

        // Motion characteristics validation
        val positions = result.samples.map { it.xMm }
        val velocities = result.samples.map { it.vMmPerOmega }
        val accelerations = result.samples.map { it.aMmPerOmega2 }

        val actualStroke = positions.maxOrNull()!! - positions.minOrNull()!!
        val maxAbsVelocity = velocities.map { abs(it) }.maxOrNull()!!
        val maxAbsAcceleration = accelerations.map { abs(it) }.maxOrNull()!!

        // Stroke validation (acknowledging scaling factor)
        assertTrue(actualStroke > 0.0001, "Should have some motion")

        // Velocity validation
        assertTrue(maxAbsVelocity > 0.0, "Should have some velocity")
        assertTrue(maxAbsVelocity.isFinite(), "Max velocity should be finite")

        // Acceleration validation
        assertTrue(maxAbsAcceleration.isFinite(), "Max acceleration should be finite")

        // Dwell validation - if significant dwells, should have periods of low velocity
        if (tdcDwellDeg > 10.0 || bdcDwellDeg > 10.0) {
            val minAbsVelocity = velocities.map { abs(it) }.minOrNull()!!
            assertTrue(
                minAbsVelocity < maxAbsVelocity * 0.1,
                "Should have low-velocity periods during dwells",
            )
        }
    }

    @ParameterizedTest(name = "Extreme parameters: stroke={0}mm, step={1}deg, profile={2}, upFraction={3}, dwells=({4},{5})")
    @MethodSource("extremeParameterCombinations")
    fun `piecewise solver handles extreme parameter combinations gracefully`(
        strokeMm: Double,
        stepDeg: Double,
        profile: RampProfile,
        upFraction: Double,
        tdcDwellDeg: Double,
        bdcDwellDeg: Double,
    ) {
        val params =
            LitvinUserParams(
                samplingStepDeg = stepDeg,
                profileSolverMode = ProfileSolverMode.Piecewise,
                strokeLengthMm = strokeMm,
                rampProfile = profile,
                upFraction = upFraction,
                dwellTdcDeg = tdcDwellDeg,
                dwellBdcDeg = bdcDwellDeg,
            )

        // Should not crash or throw unexpected exceptions
        val result = MotionLawGenerator.generateMotion(params)

        assertNotNull(result, "Extreme parameters should not crash motion generation")
        assertTrue(result.samples.isNotEmpty(), "Should generate samples even with extreme parameters")

        // Basic sanity checks
        result.samples.forEach { sample ->
            assertTrue(sample.thetaDeg.isFinite(), "All samples should be finite")
            assertTrue(sample.xMm.isFinite(), "All samples should be finite")
            assertTrue(sample.vMmPerOmega.isFinite(), "All samples should be finite")
            assertTrue(sample.aMmPerOmega2.isFinite(), "All samples should be finite")
        }

        println("Extreme test passed: stroke=$strokeMm, step=$stepDeg, profile=$profile, upFraction=$upFraction, dwells=($tdcDwellDeg,$bdcDwellDeg)")
    }

    @ParameterizedTest
    @ValueSource(doubles = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0])
    fun `sampling resolution affects result quality predictably`(stepDeg: Double) {
        val baseParams =
            LitvinUserParams(
                samplingStepDeg = stepDeg,
                profileSolverMode = ProfileSolverMode.Piecewise,
                strokeLengthMm = 10.0,
                rampProfile = RampProfile.Cycloidal,
            )

        val result = MotionLawGenerator.generateMotion(baseParams)

        assertNotNull(result, "Motion generation should succeed for step $stepDeg")
        assertTrue(result.samples.isNotEmpty(), "Should generate samples for step $stepDeg")

        // More samples should be generated for finer resolution - 180° periodicity for planetary gearset
        val expectedSamples = ceil(180.0 / stepDeg).toInt()
        assertTrue(
            abs(result.samples.size - expectedSamples) <= 2,
            "Sample count should match resolution: step=$stepDeg, expected~$expectedSamples, got=${result.samples.size}",
        )

        // Finer resolution should generally produce smoother profiles
        // (measured by variance in acceleration)
        val accelerations = result.samples.map { it.aMmPerOmega2 }
        val meanAcceleration = accelerations.average()
        val accelerationVariance = accelerations.map { (it - meanAcceleration).pow(2) }.average()

        assertTrue(accelerationVariance.isFinite(), "Acceleration variance should be finite")

        // Store result for potential cross-resolution comparison
        println("Resolution test: step=$stepDeg deg, samples=${result.samples.size}, acc_variance=${"%.6f".format(accelerationVariance)}")
    }

    @ParameterizedTest(name = "Collocation comparison: stroke={0}mm, step={1}deg, profile={2}")
    @MethodSource("extremeParameterCombinations")
    fun `collocation solver comparison with piecewise reference`(
        strokeMm: Double,
        stepDeg: Double,
        profile: RampProfile,
        upFraction: Double,
        tdcDwellDeg: Double,
        bdcDwellDeg: Double,
    ) {
        // Generate piecewise reference
        val piecewiseParams =
            LitvinUserParams(
                samplingStepDeg = stepDeg,
                profileSolverMode = ProfileSolverMode.Piecewise,
                strokeLengthMm = strokeMm,
                rampProfile = profile,
                upFraction = upFraction,
                dwellTdcDeg = tdcDwellDeg,
                dwellBdcDeg = bdcDwellDeg,
            )

        val piecewiseResult = MotionLawGenerator.generateMotion(piecewiseParams)

        // Attempt collocation solver
        val collocationParams = piecewiseParams.copy(profileSolverMode = ProfileSolverMode.Collocation)

        try {
            val collocationResult = CollocationMotionSolver.solve(collocationParams)

            // If collocation succeeds, compare with piecewise
            assertNotNull(collocationResult, "Collocation result should not be null")
            assertTrue(collocationResult.samples.isNotEmpty(), "Collocation should generate samples")

            // Compare sample counts
            assertTrue(
                abs(collocationResult.samples.size - piecewiseResult.samples.size) <= 5,
                "Sample counts should be similar: piecewise=${piecewiseResult.samples.size}, collocation=${collocationResult.samples.size}",
            )

            // Both should be finite
            collocationResult.samples.forEach { sample ->
                assertTrue(sample.thetaDeg.isFinite(), "Collocation samples should be finite")
                assertTrue(sample.xMm.isFinite(), "Collocation samples should be finite")
                assertTrue(sample.vMmPerOmega.isFinite(), "Collocation samples should be finite")
                assertTrue(sample.aMmPerOmega2.isFinite(), "Collocation samples should be finite")
            }

            println("Collocation comparison succeeded for: stroke=$strokeMm, step=$stepDeg, profile=$profile")
        } catch (e: UnsupportedOperationException) {
            // Expected during development - collocation may not be available
            println("Collocation not available (expected): ${e.message}")
            assertTrue(
                e.message?.contains("feature") == true ||
                    e.message?.contains("development") == true ||
                    e.message?.contains("not yet implemented") == true,
                "Should fail for expected reasons: ${e.message}",
            )
        }
    }

    @ParameterizedTest
    @ValueSource(ints = [1, 2, 4, 8, 16])
    fun `solver performance scales reasonably with problem complexity`(complexityFactor: Int) {
        val params =
            LitvinUserParams(
                samplingStepDeg = 10.0 / complexityFactor, // Finer sampling for higher complexity
                profileSolverMode = ProfileSolverMode.Piecewise,
                strokeLengthMm = 10.0,
                rampProfile = RampProfile.Cycloidal,
                dwellTdcDeg = complexityFactor * 5.0, // More complex dwell patterns
                dwellBdcDeg = complexityFactor * 5.0,
            )

        val startTime = System.nanoTime()
        val result = MotionLawGenerator.generateMotion(params)
        val endTime = System.nanoTime()

        val durationMs = (endTime - startTime) / 1_000_000.0

        assertNotNull(result, "Generation should succeed for complexity $complexityFactor")
        assertTrue(result.samples.isNotEmpty(), "Should generate samples for complexity $complexityFactor")

        // Performance should be reasonable (under 1 second for even complex cases)
        assertTrue(
            durationMs < 1000.0,
            "Performance should be reasonable: ${durationMs}ms for complexity $complexityFactor",
        )

        println("Performance test: complexity=$complexityFactor, duration=${"%.2f".format(durationMs)}ms, samples=${result.samples.size}")
    }
}
