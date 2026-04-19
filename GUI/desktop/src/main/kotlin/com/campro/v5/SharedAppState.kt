package com.campro.v5

/**
 * Minimal shared application state used by DesktopMain for multi-window coordination.
 */
object SharedAppState {
    /** Visibility flags for named windows/panels. */
    val windowVisibility: MutableMap<String, Boolean> = mutableMapOf()

    /** Last provided parameters map for the session. */
    var parameters: Map<String, String> = emptyMap()
}
