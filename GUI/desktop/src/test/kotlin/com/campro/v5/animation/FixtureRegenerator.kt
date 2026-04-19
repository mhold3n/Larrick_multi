package com.campro.v5.animation

import com.campro.v5.data.litvin.LitvinUserParams
import com.campro.v5.data.litvin.ProfileSolverMode
import com.campro.v5.data.litvin.RampProfile
import com.google.gson.Gson
import org.junit.jupiter.api.Assertions.assertNotNull
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.Timeout
import java.nio.file.Files
import java.nio.file.Paths
import java.util.concurrent.TimeUnit

/**
 * One-off utility test to regenerate the small motion samples fixture from embedded metadata.
 * Run this test explicitly when you intend to rebaseline the golden fixture.
 */
class FixtureRegenerator {
    private val gson = Gson()

    @Test
    @Timeout(20, unit = TimeUnit.SECONDS)
    fun regenerate_motion_samples_small() {
        // Write module-relative to desktop/fixtures
        val path = Paths.get("fixtures/motion_samples_small.json")
        val ms = FixtureLoader.loadMotionSamples("fixtures/motion_samples_small.json")
        val meta = ms.generator
        assertNotNull(meta, "Fixture must contain generator metadata")
        val p = meta!!.params

        val params =
            LitvinUserParams(
                strokeLengthMm = (p["strokeLengthMm"] as? Number)?.toDouble() ?: 15.0,
                samplingStepDeg = (p["samplingStepDeg"] as? Number)?.toDouble() ?: ms.stepDeg,
                rampProfile = (
                    (p["rampProfile"] as? String)?.let { runCatching { RampProfile.valueOf(it) }.getOrNull() }
                        ?: RampProfile.Cycloidal
                    ),
                dwellTdcDeg = (p["dwellTdcDeg"] as? Number)?.toDouble() ?: 0.0,
                dwellBdcDeg = (p["dwellBdcDeg"] as? Number)?.toDouble() ?: 0.0,
                rampAfterTdcDeg = (p["rampAfterTdcDeg"] as? Number)?.toDouble() ?: 0.0,
                rampBeforeBdcDeg = (p["rampBeforeBdcDeg"] as? Number)?.toDouble() ?: 0.0,
                rampAfterBdcDeg = (p["rampAfterBdcDeg"] as? Number)?.toDouble() ?: 0.0,
                rampBeforeTdcDeg = (p["rampBeforeTdcDeg"] as? Number)?.toDouble() ?: 0.0,
                upFraction = (p["upFraction"] as? Number)?.toDouble() ?: 0.5,
                rpm = (p["rpm"] as? Number)?.toDouble() ?: 3000.0,
                profileSolverMode = ProfileSolverMode.Piecewise,
            )

        val generated = MotionLawGenerator.generateMotion(params)

        // Compose JSON with header + samples
        val header =
            mapOf(
                "generator" to
                    mapOf(
                        "version" to (meta.version ?: "piecewise-1"),
                        "commit" to (meta.commit ?: "unknown"),
                        "created_utc" to (
                            meta.created_utc ?: java.time.Instant
                                .now()
                                .toString()
                            ),
                        "params" to meta.params,
                    ),
                "stepDeg" to generated.stepDeg,
                "samples" to
                    generated.samples.map { s ->
                        mapOf(
                            "thetaDeg" to s.thetaDeg,
                            "xMm" to s.xMm,
                            "vMmPerOmega" to s.vMmPerOmega,
                            "aMmPerOmega2" to s.aMmPerOmega2,
                        )
                    },
            )
        val json = gson.toJson(header)
        path.parent?.let { Files.createDirectories(it) }
        Files.write(path, json.toByteArray())
    }
}
