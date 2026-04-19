package com.campro.v5.animation

import com.campro.v5.data.litvin.LitvinJsonLoader
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import java.io.File
import kotlin.system.measureTimeMillis

class PerformanceBudgetTest {
    @Test
    fun backend_build_and_json_write_within_budget_or_skip_if_native_missing() {
        val params =
            arrayOf(
                // Keep defaults close to Rust defaults for reproducible timing
                "up_fraction",
                "0.5",
                "dwell_tdc_deg",
                "20",
                "dwell_bdc_deg",
                "20",
                "ramp_before_tdc_deg",
                "10",
                "ramp_after_tdc_deg",
                "10",
                "ramp_before_bdc_deg",
                "10",
                "ramp_after_bdc_deg",
                "10",
                "ramp_profile",
                "S5",
                "stroke_length_mm",
                "100",
                "journal_radius",
                "5",
                "sampling_step_deg",
                "1.0",
                "planet_count",
                "2",
                "carrier_offset_deg",
                "180",
                // Optional tolerances — keep defaults
                "arc_residual_tol_mm",
                "0.01",
                "max_iter",
                "20",
            )

        val id =
            try {
                LitvinNative.createLitvinLawNative(params)
            } catch (e: UnsatisfiedLinkError) {
                println("[SKIP] Native library not available for PerformanceBudgetTest: ${e.message}")
                return
            }

        try {
            // Measure JSON write time (tables)
            var tablesPath: String
            val writeMs =
                measureTimeMillis {
                    tablesPath = LitvinNative.getLitvinKinematicsTablesNative(id)
                }
            val tables = LitvinJsonLoader.loadTables(File(tablesPath))

            // Extract backend build time from diagnostics (ms)
            val buildMs = tables.diagnostics?.buildMs ?: Double.NaN
            assertTrue(buildMs.isFinite(), "diagnostics.buildMs must be finite")

            // Budgets (tune for CI hardware; start lenient)
            val backendBudgetMs = 400.0 // backend build_litvin_tables (diagnostics.buildMs)
            val writeBudgetMs = 400L // JNI write_tables_json wall-clock

            assertTrue(buildMs <= backendBudgetMs, "Backend build exceeded budget: buildMs=$buildMs > $backendBudgetMs ms")
            assertTrue(writeMs <= writeBudgetMs, "JSON write exceeded budget: $writeMs > $writeBudgetMs ms")
        } finally {
            try {
                LitvinNative.disposeLitvinLawNative(id)
            } catch (_: Throwable) {
            }
        }
    }
}
