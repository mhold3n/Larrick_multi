package com.campro.v5.animation

import org.junit.jupiter.api.Assertions.assertThrows
import org.junit.jupiter.api.Test

class ErrorHandlingTest {
    @Test
    fun `missing fixture throws`() {
        assertThrows(Exception::class.java) {
            FixtureLoader.loadMotionSamples("fixtures/does_not_exist.json")
        }
    }

    @Test
    fun `malformed fixture throws`() {
        // Create a temporary malformed content in-memory via direct parse call
        val badContent = "{" // invalid JSON
        // Use reflection to call internal parser if needed; here we just expect our loader to fail when missing keys
        assertThrows(Exception::class.java) {
            // Write to a temp file to reuse loader
            val tmp =
                kotlin.io.path
                    .createTempFile(suffix = ".json")
                    .toFile()
            tmp.writeText(badContent)
            try {
                FixtureLoader.loadMotionSamples(tmp.absolutePath)
            } finally {
                tmp.delete()
            }
        }
    }
}
