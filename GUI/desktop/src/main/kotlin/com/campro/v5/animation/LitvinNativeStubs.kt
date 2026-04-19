package com.campro.v5.animation

/**
 * Safe fallbacks for Litvin-specific native functions.
 * These delegate to LitvinNative when available and otherwise return
 * benign defaults to avoid crashing production code paths.
 * Keep signatures aligned with JNI.
 */
object LitvinNativeStubs {
    private fun <T> tryCall(default: T, call: () -> T): T = try {
        call()
    } catch (_: UnsatisfiedLinkError) {
        default
    } catch (_: Throwable) {
        default
    }

    fun createLitvinLawNative(parameters: Array<String>): Long = tryCall(0L) { LitvinNative.createLitvinLawNative(parameters) }

    fun updateLitvinLawParametersNative(id: Long, parameters: Array<String>) {
        tryCall(Unit) {
            LitvinNative.updateLitvinLawParametersNative(id, parameters)
            Unit
        }
    }

    fun getLitvinPitchCurvesNative(id: Long): String = tryCall("") { LitvinNative.getLitvinPitchCurvesNative(id) }

    fun getLitvinSystemStateNative(id: Long, alphaDeg: Double): String =
        tryCall("") { LitvinNative.getLitvinSystemStateNative(id, alphaDeg) }

    fun getLitvinKinematicsTablesNative(id: Long): String = tryCall("") { LitvinNative.getLitvinKinematicsTablesNative(id) }

    fun getLitvinFeaBoundaryNative(id: Long): String = tryCall("") { LitvinNative.getLitvinFeaBoundaryNative(id) }

    fun disposeLitvinLawNative(id: Long) {
        tryCall(Unit) {
            LitvinNative.disposeLitvinLawNative(id)
            Unit
        }
    }

    /** Initialize Rust logger if native is available; otherwise no-op. */
    fun initRustLoggerNative(sessionId: String, level: String?, logDir: String?) {
        tryCall(Unit) {
            LitvinNative.initRustLoggerNative(sessionId, level, logDir)
            Unit
        }
    }

    /** Run diagnostics if native is available; returns empty JSON string on failure. */
    fun runDiagnosticsNative(id: Long, sessionId: String, paramHash: String?): String =
        tryCall("") { LitvinNative.runDiagnosticsNative(id, sessionId, paramHash) }

    /** Probe if the native library is actually loadable on this system. */
    fun isNativeAvailable(): Boolean = try {
        // Call a lightweight native method to verify linkage
        LitvinNative.initRustLoggerNative("probe", null, null)
        true
    } catch (_: UnsatisfiedLinkError) {
        false
    } catch (_: Throwable) {
        // Any other exception during init should still indicate native linkage exists
        true
    }
}

// Minimal DTO for per-angle system state; retained for compatibility
data class LitvinSystemStateDTO(
    val alphaDeg: Double,
    val centerX: List<Double>,
    val centerY: List<Double>,
    val spinPsiDeg: List<Double>,
    val journalX: List<Double>,
    val journalY: List<Double>,
    val pistonS: List<Double>,
)
