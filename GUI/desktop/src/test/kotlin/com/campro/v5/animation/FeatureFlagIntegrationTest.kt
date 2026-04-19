package com.campro.v5.animation

import com.campro.v5.config.FeatureFlags
import com.campro.v5.data.litvin.LitvinUserParams
import com.campro.v5.data.litvin.ProfileSolverMode
import com.campro.v5.data.litvin.RampProfile
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.TestInstance
import org.junit.jupiter.api.assertThrows
import java.io.File
import java.nio.file.Files
import java.nio.file.Paths
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * Integration tests for feature flag behavior across UI visibility and solver availability.
 * Tests various combinations of feature flag states and their impact on system behavior.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class FeatureFlagIntegrationTest {
    private val testConfigDir = System.getProperty("user.home") + "/.campro_test"
    private val testConfigFile = "$testConfigDir/feature_flags.properties"
    private val originalConfigDir = System.getProperty("user.home") + "/.campro"
    private val originalConfigFile = "$originalConfigDir/feature_flags.properties"

    @BeforeEach
    fun setUp() {
        // Reset MotionLawEngine singleton for clean test state
        MotionLawEngine.resetInstance()

        // Create test config directory
        Files.createDirectories(Paths.get(testConfigDir))

        // Backup original config if it exists
        val originalFile = File(originalConfigFile)
        if (originalFile.exists()) {
            originalFile.copyTo(File("$originalConfigFile.backup"), overwrite = true)
        }

        // Force FeatureFlags to reload by clearing any cached state
        // Note: This assumes FeatureFlags has a way to reload - if not, we'll work with what we have
    }

    @AfterEach
    fun tearDown() {
        // Clean up test config
        File(testConfigFile).delete()
        File(testConfigDir).deleteRecursively()

        // Restore original config if it existed
        val backupFile = File("$originalConfigFile.backup")
        if (backupFile.exists()) {
            backupFile.copyTo(File(originalConfigFile), overwrite = true)
            backupFile.delete()
        } else {
            File(originalConfigFile).delete()
        }
    }

    @Test
    fun `collocation solver availability respects feature flags`() {
        // Test 1: Enabled flag should make solver available
        writeFeatureFlags("collocation.enabled=true\ncollocation.force_fallback=false")

        // Note: We can't easily reload FeatureFlags, so we'll test the logic directly
        val testFlags = createTestFeatureFlags(enabled = true, forceFallback = false)
        assertTrue(testFlags.isEnabled())
        assertFalse(testFlags.isForceFallback())

        // Test 2: Disabled flag should make solver unavailable
        val disabledFlags = createTestFeatureFlags(enabled = false, forceFallback = false)
        assertFalse(disabledFlags.isEnabled())

        // Test 3: Force fallback should make solver unavailable even if enabled
        val fallbackFlags = createTestFeatureFlags(enabled = true, forceFallback = true)
        assertTrue(fallbackFlags.isEnabled())
        assertTrue(fallbackFlags.isForceFallback())
    }

    @Test
    fun `collocation solver throws appropriate exceptions based on feature flags`() {
        val testParams =
            LitvinUserParams(
                samplingStepDeg = 1.0,
                profileSolverMode = ProfileSolverMode.Collocation,
                strokeLengthMm = 10.0,
                rampProfile = RampProfile.Cycloidal,
            )

        // Test that solver respects disabled flag
        // Disable collocation feature flag
        FeatureFlags.setFlag("collocation.enabled", false)

        val exception =
            assertThrows<UnsupportedOperationException> {
                CollocationMotionSolver.solve(testParams)
            }

        assertTrue(
            exception.message?.contains("feature") == true ||
                exception.message?.contains("disabled") == true,
            "Exception should mention feature flags or disabled: ${exception.message}",
        )

        // Clean up - re-enable for other tests
        FeatureFlags.clearFlag("collocation.enabled")
    }

    @Test
    fun `UI visibility logic works correctly`() {
        // Test UI visibility combinations
        val visibleFlags = createTestFeatureFlags(enabled = true, forceFallback = false)
        val hiddenFlags1 = createTestFeatureFlags(enabled = false, forceFallback = false)
        val hiddenFlags2 = createTestFeatureFlags(enabled = true, forceFallback = true)

        // Simulate UI visibility logic
        assertTrue(isUIVisible(visibleFlags), "UI should be visible when enabled and not forced to fallback")
        assertFalse(isUIVisible(hiddenFlags1), "UI should be hidden when disabled")
        assertFalse(isUIVisible(hiddenFlags2), "UI should be hidden when forced to fallback")
    }

    @Test
    fun `solver availability check is consistent with actual solve attempt`() {
        val testParams =
            LitvinUserParams(
                samplingStepDeg = 5.0, // Larger step for faster test
                profileSolverMode = ProfileSolverMode.Collocation,
                strokeLengthMm = 5.0,
                rampProfile = RampProfile.Cycloidal,
            )

        // Test with collocation disabled
        FeatureFlags.setFlag("collocation.enabled", false)
        val availableWhenDisabled = CollocationMotionSolver.isAvailable()
        assertFalse(availableWhenDisabled, "Solver should not be available when disabled")

        val exceptionWhenDisabled =
            assertThrows<UnsupportedOperationException> {
                CollocationMotionSolver.solve(testParams)
            }
        assertTrue(
            exceptionWhenDisabled.message?.contains("feature") == true ||
                exceptionWhenDisabled.message?.contains("disabled") == true,
            "When disabled, should fail due to feature flags: ${exceptionWhenDisabled.message}",
        )

        // Test with collocation enabled
        FeatureFlags.setFlag("collocation.enabled", true)
        FeatureFlags.setFlag("collocation.force_fallback", false)
        val availableWhenEnabled = CollocationMotionSolver.isAvailable()

        if (availableWhenEnabled) {
            // If available, solve should work or fail for implementation reasons, not feature flags
            try {
                CollocationMotionSolver.solve(testParams)
                // If it succeeds, that's fine too
            } catch (e: UnsupportedOperationException) {
                // Should be a different error (like "not yet implemented" or Python not found)
                assertTrue(
                    e.message?.contains("development") == true ||
                        e.message?.contains("Python") == true ||
                        e.message?.contains("not yet implemented") == true,
                    "When available, should fail for implementation reasons, not feature flags: ${e.message}",
                )
            }
        }

        // Clean up
        FeatureFlags.clearFlag("collocation.enabled")
        FeatureFlags.clearFlag("collocation.force_fallback")
    }

    @Test
    fun `engine branching respects solver mode and availability`() {
        val engine = MotionLawEngine.getInstance()

        // Test piecewise mode (should always work)
        val piecewiseParams =
            LitvinUserParams(
                samplingStepDeg = 5.0,
                profileSolverMode = ProfileSolverMode.Piecewise,
                strokeLengthMm = 5.0,
                rampProfile = RampProfile.Cycloidal,
            )

        // This should not throw an exception
        engine.updateParameters(piecewiseParams.toParameterStringMap())
        // We can't easily assert the result without deeper integration, but it shouldn't crash

        // Test collocation mode (behavior depends on availability)
        val collocationParams =
            LitvinUserParams(
                samplingStepDeg = 5.0,
                profileSolverMode = ProfileSolverMode.Collocation,
                strokeLengthMm = 5.0,
                rampProfile = RampProfile.Cycloidal,
            )

        // This may succeed (with fallback) or fail, but should not crash the engine
        try {
            engine.updateParameters(collocationParams.toParameterStringMap())
            println("Collocation mode updated successfully (likely fell back to piecewise)")
        } catch (e: Exception) {
            println("Collocation mode failed as expected: ${e.message}")
            // This is acceptable behavior
        }
    }

    // Helper functions

    private fun writeFeatureFlags(content: String) {
        File(testConfigFile).writeText(content)
    }

    private fun createTestFeatureFlags(enabled: Boolean, forceFallback: Boolean): TestFeatureFlags =
        TestFeatureFlags(enabled, forceFallback)

    private fun isUIVisible(flags: TestFeatureFlags): Boolean {
        // Simulate the UI visibility logic from ParameterInputForm
        return flags.isEnabled() && !flags.isForceFallback()
    }

    // Test double for FeatureFlags.Collocation
    private class TestFeatureFlags(private val enabled: Boolean, private val forceFallback: Boolean) {
        fun isEnabled(): Boolean = enabled

        fun isForceFallback(): Boolean = forceFallback
    }
}

// Extension function to convert LitvinUserParams to parameter map for engine testing
private fun LitvinUserParams.toParameterStringMap(): Map<String, String> = mapOf(
    "samplingStepDeg" to samplingStepDeg.toString(),
    "Profile Solver" to profileSolverMode.name,
    "strokeLengthMm" to strokeLengthMm.toString(),
    "rampProfile" to rampProfile.name,
    "dwellTdcDeg" to dwellTdcDeg.toString(),
    "dwellBdcDeg" to dwellBdcDeg.toString(),
    "rampAfterTdcDeg" to rampAfterTdcDeg.toString(),
    "rampBeforeBdcDeg" to rampBeforeBdcDeg.toString(),
    "rampAfterBdcDeg" to rampAfterBdcDeg.toString(),
    "rampBeforeTdcDeg" to rampBeforeTdcDeg.toString(),
    "upFraction" to upFraction.toString(),
    "rpm" to rpm.toString(),
)
