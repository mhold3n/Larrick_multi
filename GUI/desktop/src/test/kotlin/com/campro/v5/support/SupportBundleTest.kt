package com.campro.v5.support

import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import java.nio.file.Files
import java.nio.file.Path
import java.util.zip.ZipFile

class SupportBundleTest {
    @Test
    fun create_bundle_includes_logs_and_json_and_skips_large_files() {
        val tempRoot = Files.createTempDirectory("cpv5_test_bundle_")
        val logsDir = tempRoot.resolve("logs")
        val jsonDir = tempRoot.resolve("json")
        Files.createDirectories(logsDir)
        Files.createDirectories(jsonDir)

        val log1 = logsDir.resolve("app.log").toFile().apply { writeText("hello log") }
        val log2 = logsDir.resolve("trace.txt").toFile().apply { writeText("trace") }
        val json1 = jsonDir.resolve("out.json").toFile().apply { writeText("{\"ok\":true}") }
        // Large file over 20MB should be skipped
        val big = logsDir.resolve("big.log").toFile().apply { writeBytes(ByteArray(21 * 1024 * 1024)) }

        val sessionId = "testsession"
        val zipPath =
            SupportBundle.createSupportBundle(
                sessionId = sessionId,
                logDir = logsDir.toString(),
                jsonDirs = listOf(jsonDir.toString()),
                maxFileSizeBytes = 20L * 1024 * 1024,
            )

        assertTrue(Files.exists(Path.of(zipPath)), "Zip should exist: $zipPath")

        ZipFile(zipPath).use { zf ->
            val names =
                zf
                    .entries()
                    .asSequence()
                    .map { it.name }
                    .toList()
            assertTrue(names.any { it.endsWith("logs/app.log") }, "logs/app.log present")
            assertTrue(names.any { it.endsWith("logs/trace.txt") }, "logs/trace.txt present")
            assertTrue(names.any { it.endsWith("json/out.json") }, "json/out.json present")
            assertTrue(names.none { it.endsWith("logs/big.log") }, "big.log should be skipped")
            assertTrue(names.any { it.endsWith("manifest.txt") }, "manifest.txt present")
        }
    }
}
