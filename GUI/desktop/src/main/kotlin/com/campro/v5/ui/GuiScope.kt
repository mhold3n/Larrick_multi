package com.campro.v5.ui

/**
 * Canonical frontend scope for the GUI-only refactor phase.
 *
 * Intended behavior:
 * - Keep this list as the contract of user-visible surfaces that must remain
 *   functional while backend internals are stubbed.
 *
 * Current behavior:
 * - This object is used as a single source of truth for migration planning,
 *   smoke testing, and future Larrak adapter integration.
 */
object GuiScope {
    val preservedScreens: List<String> =
        listOf(
            "Unified Optimization",
            "Parameters",
            "Motion Law",
            "Gear Profiles",
            "Efficiency",
            "FEA Analysis",
            "Advanced",
            "Accessibility Settings",
            "Debug Panel",
        )
}
