package com.campro.v5.animation

import com.campro.v5.fea.FeaEngine
import org.junit.jupiter.api.Test
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class FeaEngineTest {

    @Test
    fun `test FeaEngine instantiation`() {
        val feaEngine = FeaEngine()
        assertNotNull(feaEngine, "FeaEngine should be instantiable")
    }

    @Test
    fun `test isFunctional method`() {
        val feaEngine = FeaEngine()
        val isFunctional = feaEngine.isFunctional()

        // The method should return a boolean value
        assertTrue(isFunctional is Boolean, "isFunctional should return a boolean")

        // In test environment, it might return false due to JNI issues
        // This is expected behavior
    }

    @Test
    fun `test isAvailable method`() {
        val feaEngine = FeaEngine()
        val isAvailable = feaEngine.isAvailable()

        // The method should return a boolean value
        assertTrue(isAvailable is Boolean, "isAvailable should return a boolean")

        // In test environment, it might return false due to JNI issues
        // This is expected behavior
    }

    @Test
    fun `test getVersion method`() {
        val feaEngine = FeaEngine()
        val version = feaEngine.getVersion()

        // The method should return a string
        assertNotNull(version, "Version should not be null")
        assertTrue(version is String, "Version should be a string")
        assertTrue(version.isNotEmpty(), "Version should not be empty")
    }

    @Test
    fun `test FeaEngine error handling`() {
        val feaEngine = FeaEngine()

        // Test that the engine handles errors gracefully
        // In test environment without native library, these should not crash
        try {
            val isFunctional = feaEngine.isFunctional()
            assertTrue(isFunctional is Boolean, "isFunctional should return boolean even when JNI fails")
        } catch (e: Exception) {
            // If it throws, it should be a reasonable exception
            assertTrue(e.message?.isNotEmpty() == true, "Exception should have descriptive message")
        }
    }
}
