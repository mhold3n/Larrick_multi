package com.campro.v5.animation

import java.nio.file.Files
import java.nio.file.Paths

data class FixtureMotionSample(val thetaDeg: Double, val xMm: Double, val vMmPerOmega: Double, val aMmPerOmega2: Double)

data class FixtureMotionSamples(val stepDeg: Double, val samples: List<FixtureMotionSample>, val generator: GeneratorMeta? = null)

data class GeneratorMeta(
    val version: String? = null,
    val commit: String? = null,
    val created_utc: String? = null,
    val params: Map<String, Any?> = emptyMap(),
)

object FixtureLoader {
    fun loadMotionSamples(pathStr: String): FixtureMotionSamples {
        // Resolve in this order:
        // 1) Module-relative path (when running :desktop:test)
        // 2) Repo-root relative fallback (../desktop/...)
        // 3) Absolute path as-is
        val candidates =
            listOf(
                Paths.get(pathStr),
                Paths.get("desktop").resolve(pathStr),
                Paths.get("..", "desktop").resolve(pathStr),
            )
        val existingPath =
            candidates.firstOrNull { Files.exists(it) }
                ?: error("Fixture not found via paths: ${candidates.joinToString(", ")}")
        val text = Files.readString(existingPath)
        val step =
            Regex("\"stepDeg\"\\s*:\\s*([0-9eE+.-]+)")
                .find(text)
                ?.groupValues
                ?.get(1)
                ?.toDouble()
                ?: error("stepDeg missing in $pathStr")
        val generatorBlock =
            Regex("\"generator\"\\s*:\\s*\\{(.*?)\\}", RegexOption.DOT_MATCHES_ALL)
                .find(text)
                ?.groupValues
                ?.get(1)
        val generator =
            generatorBlock?.let { gb ->
                val getStr = { key: String -> Regex("\"$key\"\\s*:\\s*\"([^\"]*)\"").find(gb)?.groupValues?.get(1) }
                val version = getStr("version")
                val commit = getStr("commit")
                val created = getStr("created_utc")
                val paramsBlock = Regex("\"params\"\\s*:\\s*\\{(.*?)\\}", RegexOption.DOT_MATCHES_ALL).find(gb)?.groupValues?.get(1)
                val kvRegex = Regex("\"([A-Za-z0-9_]+)\"\\s*:\\s*([^,\n\r\\}]*)")
                val params = mutableMapOf<String, Any?>()
                paramsBlock?.let { pb ->
                    for (m in kvRegex.findAll(pb)) {
                        val k = m.groupValues[1]
                        val raw = m.groupValues[2].trim()
                        val v: Any? =
                            when {
                                raw.startsWith("\"") && raw.endsWith("\"") -> raw.trim('"')
                                raw.equals("true", true) || raw.equals("false", true) -> raw.toBooleanStrictOrNull()
                                else -> raw.toDoubleOrNull()
                            }
                        params[k] = v
                    }
                }
                GeneratorMeta(version, commit, created, params)
            }
        val samplesArray =
            Regex("\"samples\"\\s*:\\s*\\[(.*)]", RegexOption.DOT_MATCHES_ALL)
                .find(text)
                ?.groupValues
                ?.get(1)
                ?: error("samples missing in $pathStr")
        val sampleRegex = Regex("\\{([^}]*)\\}")
        val numRegex = { key: String ->
            Regex("\"$key\"\\s*:\\s*([0-9eE+.-]+)")
        }
        val samples = mutableListOf<FixtureMotionSample>()
        for (m in sampleRegex.findAll(samplesArray)) {
            val obj = m.groupValues[1]
            val th =
                numRegex("thetaDeg")
                    .find(obj)
                    ?.groupValues
                    ?.get(1)
                    ?.toDouble() ?: continue
            val x =
                numRegex("xMm")
                    .find(obj)
                    ?.groupValues
                    ?.get(1)
                    ?.toDouble() ?: 0.0
            val v =
                numRegex("vMmPerOmega")
                    .find(obj)
                    ?.groupValues
                    ?.get(1)
                    ?.toDouble() ?: 0.0
            val a =
                numRegex("aMmPerOmega2")
                    .find(obj)
                    ?.groupValues
                    ?.get(1)
                    ?.toDouble() ?: 0.0
            samples.add(FixtureMotionSample(th, x, v, a))
        }
        return FixtureMotionSamples(step, samples, generator)
    }
}
