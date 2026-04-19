package com.campro.v5.pipeline

import com.campro.v5.utils.SimpleJsonUtils
import org.slf4j.LoggerFactory
import java.io.File
import java.nio.file.Files
import java.nio.file.Path
import java.util.UUID
import java.util.concurrent.TimeUnit

/**
 * Shared process-boundary bridge for Larrick GUI adapters.
 *
 * Intended behavior:
 * - Keep all Python subprocess invocation logic in one place.
 * - Use a stable JSON file contract (`--input`, `--output`, `--output-dir`).
 * - Allow mode-specific calls (`optimize`, `orchestrate`, `simulate`, etc.).
 *
 * Current behavior:
 * - Resolves python executable and bridge script from properties/env with
 *   deterministic fallbacks.
 * - Returns parsed JSON response maps for adapter-level typing.
 */
class LarrickBridgeClient {
    private val logger = LoggerFactory.getLogger(LarrickBridgeClient::class.java)

    companion object {
        private const val DEFAULT_TIMEOUT_SECONDS = 45L
        private const val PYTHON_EXE_PROP = "larrick.gui.pythonExe"
        private const val PYTHON_EXE_ENV = "LARRICK_GUI_PYTHON_EXE"
        private const val BRIDGE_SCRIPT_PROP = "larrick.gui.bridgeScript"
        private const val BRIDGE_SCRIPT_ENV = "LARRICK_GUI_BRIDGE_SCRIPT"
        private const val LARRICK_ROOT_ENV = "LARRICK_MULTI_ROOT"
    }

    /**
     * Execute one bridge mode and return parsed JSON output.
     */
    fun runMode(
        mode: String,
        payload: Map<String, Any>,
        outputDir: Path,
        allowReal: Boolean = false,
    ): Map<String, Any> {
        Files.createDirectories(outputDir)
        val requestId = UUID.randomUUID().toString().take(8)
        val inputFile = outputDir.resolve("gui_bridge_input_${mode}_$requestId.json")
        val outputFile = outputDir.resolve("gui_bridge_output_${mode}_$requestId.json")

        SimpleJsonUtils.writeJsonFile(payload, inputFile)
        val command = buildCommand(mode, inputFile, outputFile, outputDir, allowReal)
        logger.debug("Running bridge command: ${command.joinToString(" ")}")

        val process = ProcessBuilder(command)
            .directory(File(System.getProperty("user.dir")))
            .start()
        val finished = process.waitFor(DEFAULT_TIMEOUT_SECONDS, TimeUnit.SECONDS)
        if (!finished) {
            process.destroyForcibly()
            return mapOf(
                "status" to "failed",
                "mode" to mode,
                "backend" to "larrick-bridge-client",
                "error" to "Bridge command timed out",
            )
        }

        val stdout = process.inputStream.bufferedReader().use { it.readText() }.trim()
        val stderr = process.errorStream.bufferedReader().use { it.readText() }.trim()
        if (stdout.isNotEmpty()) {
            logger.debug("Bridge stdout: $stdout")
        }
        if (stderr.isNotEmpty()) {
            logger.debug("Bridge stderr: $stderr")
        }

        if (!outputFile.toFile().exists()) {
            return mapOf(
                "status" to "failed",
                "mode" to mode,
                "backend" to "larrick-bridge-client",
                "error" to "Bridge output file missing",
            )
        }
        return SimpleJsonUtils.readJsonFile(outputFile)
    }

    private fun buildCommand(
        mode: String,
        inputFile: Path,
        outputFile: Path,
        outputDir: Path,
        allowReal: Boolean,
    ): List<String> {
        val pythonExe = resolvePythonExecutable()
        val scriptPath = resolveBridgeScriptPath()
        val command = mutableListOf(
            pythonExe,
            scriptPath,
            "--mode",
            mode,
            "--input",
            inputFile.toString(),
            "--output",
            outputFile.toString(),
            "--output-dir",
            outputDir.toString(),
        )
        if (allowReal) {
            command.add("--real")
        }
        return command
    }

    private fun resolvePythonExecutable(): String {
        val property = System.getProperty(PYTHON_EXE_PROP)?.trim().orEmpty()
        if (property.isNotEmpty()) return property
        val env = System.getenv(PYTHON_EXE_ENV)?.trim().orEmpty()
        if (env.isNotEmpty()) return env
        return "python3"
    }

    private fun resolveBridgeScriptPath(): String {
        val property = System.getProperty(BRIDGE_SCRIPT_PROP)?.trim().orEmpty()
        if (property.isNotEmpty()) return property
        val env = System.getenv(BRIDGE_SCRIPT_ENV)?.trim().orEmpty()
        if (env.isNotEmpty()) return env

        val userDir = Path.of(System.getProperty("user.dir")).normalize()
        val candidates = mutableListOf<Path>()

        // Preferred explicit monorepo root, then common relative fallbacks.
        System.getenv(LARRICK_ROOT_ENV)?.let { root ->
            candidates.add(Path.of(root).resolve("scripts/larrick_gui_bridge.py"))
        }
        candidates.add(userDir.resolve("../../scripts/larrick_gui_bridge.py").normalize())
        candidates.add(userDir.resolve("../scripts/larrick_gui_bridge.py").normalize())
        candidates.add(userDir.resolve("scripts/larrick_gui_bridge.py").normalize())

        val existing = candidates.firstOrNull { it.toFile().exists() }
        return (existing ?: candidates.first()).toString()
    }
}
