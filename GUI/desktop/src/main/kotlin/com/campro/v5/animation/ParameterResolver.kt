package com.campro.v5.animation

/**
 * Minimal parameter resolver used by desktop animation components.
 * Provides typed accessors with sane defaults and optional display labels (ignored here).
 */
object ParameterResolver {
    fun string(parameters: Map<String, String>, key: String, default: String = "", label: String? = null): String =
        parameters[key]?.trim()?.ifBlank { default } ?: default

    fun float(parameters: Map<String, String>, key: String, default: Float = 0f, label: String? = null): Float {
        val v = parameters[key]?.trim()
        return v?.toFloatOrNull() ?: default
    }

    fun int(parameters: Map<String, String>, key: String, default: Int = 0, label: String? = null): Int {
        val v = parameters[key]?.trim()
        return v?.toIntOrNull() ?: default
    }

    fun bool(parameters: Map<String, String>, key: String, default: Boolean = false, label: String? = null): Boolean {
        val v = parameters[key]?.trim()?.lowercase()
        return when (v) {
            "true", "1", "yes", "y", "on" -> true
            "false", "0", "no", "n", "off" -> false
            else -> default
        }
    }
}
