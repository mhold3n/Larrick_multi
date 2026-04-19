package com.campro.v5

import com.campro.v5.config.FeatureFlags
import com.campro.v5.legacy.LegacyComponentWrapper
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.Assertions.*

/**
 * Simple test to verify our feature flag system and safe deactivation works.
 */
class SimpleFeatureFlagTest {

    @Test
    fun `test feature flags are properly configured`() {
        // Test that old workflow components are disabled by default
        assertFalse(FeatureFlags.ENABLE_OLD_MOTION_LAW_ENGINE)
        assertFalse(FeatureFlags.ENABLE_OLD_PERF_DIAG)
        assertFalse(FeatureFlags.ENABLE_OLD_DIAGNOSTICS_PREFLIGHT)
        assertFalse(FeatureFlags.ENABLE_OLD_COLLOCATION_MOTION_SOLVER)
        assertFalse(FeatureFlags.ENABLE_OLD_MOTION_LAW_GENERATOR)

        // Test that new workflow components are enabled by default
        assertTrue(FeatureFlags.ENABLE_NEW_UNIFIED_OPTIMIZATION)
        assertTrue(FeatureFlags.ENABLE_NEW_ADVANCED_FEATURES)
        assertTrue(FeatureFlags.ENABLE_NEW_OPTIMIZATION_STATE_MANAGER)
        assertTrue(FeatureFlags.ENABLE_NEW_OPTIMIZATION_BRIDGE)
        assertTrue(FeatureFlags.ENABLE_NEW_VISUALIZATION_COMPONENTS)
        assertTrue(FeatureFlags.ENABLE_NEW_ADVANCED_FEATURES)
    }

    @Test
    fun `test legacy component wrapper blocks access when disabled`() {
        // Test that legacy components are blocked when disabled
        val result = LegacyComponentWrapper.withMotionLawEngine {
            "This should not execute"
        }

        // Should return null because the feature is disabled
        assertNull(result)
    }

    @Test
    fun `test feature flag status summary is generated`() {
        // Test that the helper functions work correctly
        assertFalse(FeatureFlags.hasOldFeaturesEnabled())
        assertTrue(FeatureFlags.hasAllNewFeaturesEnabled())
    }

    @Test
    fun `test legacy component status logging works`() {
        // This should not throw an exception
        assertDoesNotThrow {
            LegacyComponentWrapper.logLegacyComponentStatus()
        }
    }

    @Test
    fun `test hasOldFeaturesEnabled returns false by default`() {
        assertFalse(FeatureFlags.hasOldFeaturesEnabled())
    }

    @Test
    fun `test hasAllNewFeaturesEnabled returns true by default`() {
        assertTrue(FeatureFlags.hasAllNewFeaturesEnabled())
    }
}
