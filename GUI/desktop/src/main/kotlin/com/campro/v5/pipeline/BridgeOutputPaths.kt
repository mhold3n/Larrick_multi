package com.campro.v5.pipeline

import java.nio.file.Path
import java.nio.file.Paths

/**
 * Centralized output path resolution for bridge-generated artifacts.
 */
object BridgeOutputPaths {
    fun defaultOutputDir(): Path {
        val override = System.getProperty("larrick.gui.outputDir")?.trim().orEmpty()
        if (override.isNotEmpty()) {
            return Paths.get(override)
        }
        return Paths.get("./output/larrick_bridge")
    }
}
