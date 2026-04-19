package com.campro.v5.animation

import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import java.nio.file.Files
import java.nio.file.Paths

class FixtureLoaderTest {
    @Test
    fun `small motion samples fixture exists and has samples`() {
        val path = Paths.get("fixtures/motion_samples_small.json")
        assertTrue(Files.exists(path), "motion_samples_small.json missing")
        val text = Files.readString(path)
        assertTrue(text.contains("\"samples\""))
        assertTrue(text.length > 100)
    }

    @Test
    fun `fine motion samples fixture exists`() {
        val path = Paths.get("fixtures/motion_samples_fine.json")
        assertTrue(Files.exists(path), "motion_samples_fine.json missing")
        val text = Files.readString(path)
        assertTrue(text.contains("\"stepDeg\""))
    }
}
