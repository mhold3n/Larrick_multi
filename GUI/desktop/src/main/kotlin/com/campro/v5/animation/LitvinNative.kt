package com.campro.v5.animation

/**
 * JNI shim for Litvin native integration. External declarations only; loading handled by runtime.
 */
object LitvinNative {
    @JvmStatic external fun initRustLoggerNative(sessionId: String, level: String?, logDir: String?)

    @JvmStatic external fun createLitvinLawNative(parameters: Array<String>): Long

    @JvmStatic external fun updateLitvinLawParametersNative(id: Long, parameters: Array<String>)

    @JvmStatic external fun getLitvinPitchCurvesNative(id: Long): String

    @JvmStatic external fun getLitvinKinematicsTablesNative(id: Long): String

    @JvmStatic external fun getLitvinSystemStateNative(id: Long, alphaDeg: Double): String

    @JvmStatic external fun getLitvinFeaBoundaryNative(id: Long): String

    @JvmStatic external fun runDiagnosticsNative(id: Long, sessionId: String, paramHash: String?): String

    @JvmStatic external fun disposeLitvinLawNative(id: Long)
}
