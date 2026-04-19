package com.campro.v5.animation

import com.campro.v5.SessionInfo
import java.io.File
import java.nio.file.Files
import java.nio.file.Paths
import java.time.Instant
import java.time.format.DateTimeFormatter

/**
 * Minimal diagnostics exporter used by AnimationWidget.
 * Writes a small JSON file with the run summary and returns the file.
 */
object DiagnosticsExport {
    fun write(
        sessionId: String = SessionInfo.sessionId,
        logDir: String = "logs",
        deterministic: Boolean,
        simulate: Boolean,
        preflight: DiagnosticsPreflight.Result,
        resultCode: Int,
        durationMs: Double,
    ): File {
        val dir = Paths.get(logDir)
        if (!Files.exists(dir)) Files.createDirectories(dir)
        val ts = DateTimeFormatter.ISO_INSTANT.format(Instant.now())
        val file = dir.resolve("diagnostics-$sessionId-${ts.replace(':','-')}.json").toFile()
        val itemsJson =
            preflight.items.joinToString(prefix = "[", postfix = "]") { item ->
                "{" + "\"name\":\"${item.name}\",\"ok\":${item.ok},\"detail\":\"${item.detail}\"}"
            }
        val json =
            buildString {
                append('{')
                append("\"sessionId\":\"$sessionId\",")
                append("\"deterministic\":$deterministic,")
                append("\"simulate\":$simulate,")
                append("\"preflightPassed\":${preflight.passed},")
                append("\"preflightItems\":$itemsJson,")
                append("\"resultCode\":$resultCode,")
                append("\"durationMs\":${"%.3f".format(durationMs)}")
                append('}')
            }
        file.writeText(json)
        return file
    }
}
