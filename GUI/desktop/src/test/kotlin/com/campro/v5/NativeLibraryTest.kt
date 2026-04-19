package com.campro.v5

import com.campro.v5.animation.MotionLawEngine
import com.campro.v5.fea.FeaEngine
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test

/**
 * Simple test to verify native library loading.
 */
class NativeLibraryTest {
    @BeforeEach
    fun setUp() {
        println("[DEBUG_LOG] Starting native library test setup")
    }

    @Test
    fun testFeaEngineLibraryLoading() {
        println("[DEBUG_LOG] Testing FEA engine library loading...")

        try {
            val feaEngine = FeaEngine()
            println("[DEBUG_LOG] FEA engine instance created successfully")

            val available = feaEngine.isAvailable()
            println("[DEBUG_LOG] FEA engine available: $available")

            if (available) {
                val version = feaEngine.getVersion()
                println("[DEBUG_LOG] FEA engine version: $version")
            }
        } catch (e: UnsatisfiedLinkError) {
            println("[DEBUG_LOG] UnsatisfiedLinkError in FEA engine: ${e.message}")
            e.printStackTrace()
        } catch (e: Exception) {
            println("[DEBUG_LOG] Other exception in FEA engine: ${e.message}")
            e.printStackTrace()
        }
    }

    @Test
    fun testMotionLawEngineLibraryLoading() {
        println("[DEBUG_LOG] Testing Motion Law engine library loading...")

        try {
            val motionEngine = MotionLawEngine.getInstance()
            println("[DEBUG_LOG] Motion Law engine instance created successfully")

            // Try to get component positions (this should work with fallback if native fails)
            val positions = motionEngine.getComponentPositions(45.0)
            println("[DEBUG_LOG] Component positions retrieved: $positions")
        } catch (e: UnsatisfiedLinkError) {
            println("[DEBUG_LOG] UnsatisfiedLinkError in Motion Law engine: ${e.message}")
            e.printStackTrace()
        } catch (e: Exception) {
            println("[DEBUG_LOG] Other exception in Motion Law engine: ${e.message}")
            e.printStackTrace()
        }
    }

    @Test
    fun testSystemProperties() {
        println("[DEBUG_LOG] Testing system properties...")
        println("[DEBUG_LOG] OS Name: ${System.getProperty("os.name")}")
        println("[DEBUG_LOG] OS Arch: ${System.getProperty("os.arch")}")
        println("[DEBUG_LOG] Java Library Path: ${System.getProperty("java.library.path")}")

        // Check if we can find the native libraries in the classpath
        val feaResource = this::class.java.getResourceAsStream("/native/windows/x86_64/campro_fea.dll")
        val motionResource = this::class.java.getResourceAsStream("/native/windows/x86_64/campro_motion.dll")

        println("[DEBUG_LOG] FEA library resource found: ${feaResource != null}")
        println("[DEBUG_LOG] Motion library resource found: ${motionResource != null}")

        feaResource?.close()
        motionResource?.close()
    }
}
