package com.campro.v5.config

/**
 * Feature flags to control the activation/deactivation of workflow components.
 *
 * This allows us to safely transition from old to new workflow implementations
 * without permanently removing code until we're certain the new workflow is complete.
 */
object FeatureFlags {

    // ============================================================================
    // OLD WORKFLOW FLAGS (Currently deactivated for safety)
    // ============================================================================

    /**
     * Controls the old Motion Law Engine system.
     * Set to false to deactivate problematic motion law implementations.
     */
    const val ENABLE_OLD_MOTION_LAW_ENGINE = false

    /**
     * Controls the old Motion Law Generator.
     * Set to false to deactivate piecewise fallback with 360° periodicity issues.
     */
    const val ENABLE_OLD_MOTION_LAW_GENERATOR = false

    /**
     * Controls the old Diagnostics Preflight system.
     * Set to false to deactivate diagnostics for removed system.
     */
    const val ENABLE_OLD_DIAGNOSTICS_PREFLIGHT = false

    /**
     * Controls the old Performance Diagnostics system.
     * Set to false to deactivate PerfDiag components.
     */
    const val ENABLE_OLD_PERF_DIAG = false

    /**
     * Controls the old Collocation Motion Solver.
     * Set to false to deactivate problematic Python bridge implementation.
     */
    const val ENABLE_OLD_COLLOCATION_MOTION_SOLVER = false

    // ============================================================================
    // NEW WORKFLOW FLAGS (Currently active)
    // ============================================================================

    /**
     * Controls the new Unified Optimization Pipeline.
     * Set to true to enable the new robust workflow.
     */
    const val ENABLE_NEW_UNIFIED_OPTIMIZATION = true

    /**
     * Controls the new Optimization State Manager.
     * Set to true to enable the new state management system.
     */
    const val ENABLE_NEW_OPTIMIZATION_STATE_MANAGER = true

    /**
     * Controls the new Unified Optimization Bridge.
     * Set to true to enable the new Python bridge.
     */
    const val ENABLE_NEW_OPTIMIZATION_BRIDGE = true

    /**
     * Controls the new Visualization Components.
     * Set to true to enable the new visualization system.
     */
    const val ENABLE_NEW_VISUALIZATION_COMPONENTS = true

    /**
     * Controls the new Advanced Features (Presets, Export, Batch Processing).
     * Set to true to enable the new advanced features.
     */
    const val ENABLE_NEW_ADVANCED_FEATURES = true

    // ============================================================================
    // TRANSITION FLAGS (For gradual migration)
    // ============================================================================

    /**
     * Controls whether to show warnings about deactivated old features.
     * Set to true to inform users about the transition.
     */
    const val SHOW_OLD_FEATURE_WARNINGS = true

    /**
     * Controls whether to log old feature usage attempts.
     * Set to true to monitor usage of deactivated features.
     */
    const val LOG_OLD_FEATURE_USAGE = true

    /**
     * Check if any old workflow features are still enabled.
     */
    fun hasOldFeaturesEnabled(): Boolean = ENABLE_OLD_MOTION_LAW_ENGINE ||
        ENABLE_OLD_MOTION_LAW_GENERATOR ||
        ENABLE_OLD_DIAGNOSTICS_PREFLIGHT ||
        ENABLE_OLD_PERF_DIAG ||
        ENABLE_OLD_COLLOCATION_MOTION_SOLVER

    /**
     * Check if all new workflow features are enabled.
     */
    fun hasAllNewFeaturesEnabled(): Boolean = ENABLE_NEW_UNIFIED_OPTIMIZATION &&
        ENABLE_NEW_OPTIMIZATION_STATE_MANAGER &&
        ENABLE_NEW_OPTIMIZATION_BRIDGE &&
        ENABLE_NEW_VISUALIZATION_COMPONENTS &&
        ENABLE_NEW_ADVANCED_FEATURES
}
