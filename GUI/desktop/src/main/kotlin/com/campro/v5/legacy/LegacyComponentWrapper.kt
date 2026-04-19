package com.campro.v5.legacy

import com.campro.v5.config.FeatureFlags
import org.slf4j.LoggerFactory

/**
 * Wrapper for legacy components that provides safe access with feature flag control.
 *
 * This allows us to deactivate old workflow components without removing them,
 * providing a safe transition path while ensuring the new workflow is complete.
 */
object LegacyComponentWrapper {

    val logger: org.slf4j.Logger = LoggerFactory.getLogger(LegacyComponentWrapper::class.java)

    /**
     * Safely access Motion Law Engine with feature flag check.
     */
    inline fun <T> withMotionLawEngine(action: () -> T): T? = if (FeatureFlags.ENABLE_OLD_MOTION_LAW_ENGINE) {
        try {
            action()
        } catch (e: Exception) {
            logger.warn("Motion Law Engine access failed: ${e.message}")
            null
        }
    } else {
        if (FeatureFlags.LOG_OLD_FEATURE_USAGE) {
            logger.info("Motion Law Engine access blocked - feature deactivated")
        }
        null
    }

    /**
     * Safely access Motion Law Generator with feature flag check.
     */
    inline fun <T> withMotionLawGenerator(action: () -> T): T? = if (FeatureFlags.ENABLE_OLD_MOTION_LAW_GENERATOR) {
        try {
            action()
        } catch (e: Exception) {
            logger.warn("Motion Law Generator access failed: ${e.message}")
            null
        }
    } else {
        if (FeatureFlags.LOG_OLD_FEATURE_USAGE) {
            logger.info("Motion Law Generator access blocked - feature deactivated")
        }
        null
    }

    /**
     * Safely access Diagnostics Preflight with feature flag check.
     */
    inline fun <T> withDiagnosticsPreflight(action: () -> T): T? = if (FeatureFlags.ENABLE_OLD_DIAGNOSTICS_PREFLIGHT) {
        try {
            action()
        } catch (e: Exception) {
            logger.warn("Diagnostics Preflight access failed: ${e.message}")
            null
        }
    } else {
        if (FeatureFlags.LOG_OLD_FEATURE_USAGE) {
            logger.info("Diagnostics Preflight access blocked - feature deactivated")
        }
        null
    }

    /**
     * Safely access Performance Diagnostics with feature flag check.
     */
    inline fun <T> withPerfDiag(action: () -> T): T? = if (FeatureFlags.ENABLE_OLD_PERF_DIAG) {
        try {
            action()
        } catch (e: Exception) {
            logger.warn("Performance Diagnostics access failed: ${e.message}")
            null
        }
    } else {
        if (FeatureFlags.LOG_OLD_FEATURE_USAGE) {
            logger.info("Performance Diagnostics access blocked - feature deactivated")
        }
        null
    }

    /**
     * Safely access Collocation Motion Solver with feature flag check.
     */
    inline fun <T> withCollocationMotionSolver(action: () -> T): T? = if (FeatureFlags.ENABLE_OLD_COLLOCATION_MOTION_SOLVER) {
        try {
            action()
        } catch (e: Exception) {
            logger.warn("Collocation Motion Solver access failed: ${e.message}")
            null
        }
    } else {
        if (FeatureFlags.LOG_OLD_FEATURE_USAGE) {
            logger.info("Collocation Motion Solver access blocked - feature deactivated")
        }
        null
    }

    /**
     * Check if a legacy component is available.
     */
    fun isMotionLawEngineAvailable(): Boolean = FeatureFlags.ENABLE_OLD_MOTION_LAW_ENGINE

    fun isMotionLawGeneratorAvailable(): Boolean = FeatureFlags.ENABLE_OLD_MOTION_LAW_GENERATOR

    fun isDiagnosticsPreflightAvailable(): Boolean = FeatureFlags.ENABLE_OLD_DIAGNOSTICS_PREFLIGHT

    fun isPerfDiagAvailable(): Boolean = FeatureFlags.ENABLE_OLD_PERF_DIAG

    fun isCollocationMotionSolverAvailable(): Boolean = FeatureFlags.ENABLE_OLD_COLLOCATION_MOTION_SOLVER

    /**
     * Get a list of available legacy components.
     */
    fun getAvailableLegacyComponents(): List<String> {
        val available = mutableListOf<String>()

        if (isMotionLawEngineAvailable()) available.add("MotionLawEngine")
        if (isMotionLawGeneratorAvailable()) available.add("MotionLawGenerator")
        if (isDiagnosticsPreflightAvailable()) available.add("DiagnosticsPreflight")
        if (isPerfDiagAvailable()) available.add("PerfDiag")
        if (isCollocationMotionSolverAvailable()) available.add("CollocationMotionSolver")

        return available
    }

    /**
     * Get a list of deactivated legacy components.
     */
    fun getDeactivatedLegacyComponents(): List<String> {
        val deactivated = mutableListOf<String>()

        if (!isMotionLawEngineAvailable()) deactivated.add("MotionLawEngine")
        if (!isMotionLawGeneratorAvailable()) deactivated.add("MotionLawGenerator")
        if (!isDiagnosticsPreflightAvailable()) deactivated.add("DiagnosticsPreflight")
        if (!isPerfDiagAvailable()) deactivated.add("PerfDiag")
        if (!isCollocationMotionSolverAvailable()) deactivated.add("CollocationMotionSolver")

        return deactivated
    }

    /**
     * Log the current status of legacy components.
     */
    fun logLegacyComponentStatus() {
        val available = getAvailableLegacyComponents()
        val deactivated = getDeactivatedLegacyComponents()

        logger.info("Legacy Component Status:")
        logger.info("  Available: ${available.joinToString(", ")}")
        logger.info("  Deactivated: ${deactivated.joinToString(", ")}")

        if (deactivated.isNotEmpty() && FeatureFlags.SHOW_OLD_FEATURE_WARNINGS) {
            logger.warn("Some legacy components are deactivated. New workflow should be used instead.")
        }
    }
}
