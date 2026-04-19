package com.campro.v5.performance

import androidx.compose.ui.test.*
import androidx.compose.ui.test.junit4.createComposeRule
import com.campro.v5.models.OptimizationResult
import com.campro.v5.pipeline.UnifiedOptimizationBridge
import com.campro.v5.ui.UnifiedOptimizationTile
import kotlinx.coroutines.test.runTest
import org.junit.jupiter.api.*
import org.junit.jupiter.api.Assertions.*
import java.nio.file.Paths
import java.util.concurrent.TimeUnit
import kotlin.time.Duration.Companion.seconds

/**
 * Performance validation tests to ensure the application meets performance requirements.
 *
 * Tests performance metrics including:
 * - UI responsiveness during optimization
 * - Result display time
 * - Memory usage
 * - Startup time
 * - Animation smoothness
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class PerformanceValidationTest {

    @get:RegisterExtension
    val composeTestRule = createComposeRule()

    private lateinit var bridge: UnifiedOptimizationBridge
    private lateinit var testOutputDir: java.nio.file.Path

    @BeforeEach
    fun setup() {
        bridge = UnifiedOptimizationBridge()
        testOutputDir = Paths.get("./test_output_${System.currentTimeMillis()}")
        java.nio.file.Files.createDirectories(testOutputDir)
    }

    @AfterEach
    fun tearDown() {
        // Clean up test output directory
        try {
            java.nio.file.Files.walk(testOutputDir)
                .sorted(Comparator.reverseOrder())
                .forEach { path ->
                    java.nio.file.Files.deleteIfExists(path)
                }
        } catch (e: Exception) {
            // Ignore cleanup errors
        }
    }

    @Test
    @Timeout(value = 5, unit = TimeUnit.SECONDS)
    fun `test startup time requirement`() = runTest {
        // Given
        val startTime = System.currentTimeMillis()

        // When
        composeTestRule.setContent {
            UnifiedOptimizationTile(
                onResultsReceived = { },
            )
        }

        // Wait for UI to be fully loaded
        composeTestRule.waitUntil(timeout = 5.seconds) {
            composeTestRule.onNodeWithText("Unified Optimization").assertExists()
        }

        val endTime = System.currentTimeMillis()
        val startupTime = endTime - startTime

        // Then - Verify startup time < 3 seconds
        assertTrue(startupTime < 3000, "Startup time $startupTime ms exceeds 3 second requirement")
    }

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    fun `test UI responsiveness during optimization`() = runTest {
        // Given
        var receivedResult: OptimizationResult? = null

        composeTestRule.setContent {
            UnifiedOptimizationTile(
                onResultsReceived = { result ->
                    receivedResult = result
                },
            )
        }

        // When - Start optimization
        val optimizationStartTime = System.currentTimeMillis()
        composeTestRule.onNodeWithText("Start Optimization").performClick()

        // Then - Verify UI remains responsive during optimization
        val uiResponsivenessStartTime = System.currentTimeMillis()

        // Test UI responsiveness by checking if we can interact with UI elements
        composeTestRule.onNodeWithText("Cancel").assertIsDisplayed()
        composeTestRule.onNodeWithText("Cancel").assertIsEnabled()

        val uiResponsivenessEndTime = System.currentTimeMillis()
        val uiResponseTime = uiResponsivenessEndTime - uiResponsivenessStartTime

        // UI should respond within 100ms
        assertTrue(uiResponseTime < 100, "UI response time $uiResponseTime ms exceeds 100ms requirement")

        // Wait for optimization to complete
        composeTestRule.waitUntil(timeout = 30.seconds) {
            receivedResult != null
        }

        val optimizationEndTime = System.currentTimeMillis()
        val totalOptimizationTime = optimizationEndTime - optimizationStartTime

        // Then - Verify optimization completes within reasonable time
        assertTrue(totalOptimizationTime < 30000, "Optimization time $totalOptimizationTime ms exceeds 30 second limit")
    }

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    fun `test result display time requirement`() = runTest {
        // Given
        var receivedResult: OptimizationResult? = null

        composeTestRule.setContent {
            UnifiedOptimizationTile(
                onResultsReceived = { result ->
                    receivedResult = result
                },
            )
        }

        // When - Complete optimization
        composeTestRule.onNodeWithText("Start Optimization").performClick()

        val resultDisplayStartTime = System.currentTimeMillis()

        composeTestRule.waitUntil(timeout = 30.seconds) {
            receivedResult != null
        }

        // Then - Verify results are displayed within 1 second
        val resultDisplayEndTime = System.currentTimeMillis()
        val resultDisplayTime = resultDisplayEndTime - resultDisplayStartTime

        assertTrue(resultDisplayTime < 1000, "Result display time $resultDisplayTime ms exceeds 1 second requirement")

        // Verify results are actually displayed in UI
        composeTestRule.onNodeWithText("Motion Law").assertIsDisplayed()
        composeTestRule.onNodeWithText("Gear Profiles").assertIsDisplayed()
        composeTestRule.onNodeWithText("Efficiency").assertIsDisplayed()
        composeTestRule.onNodeWithText("FEA Analysis").assertIsDisplayed()
    }

    @Test
    fun `test memory usage requirement`() = runTest {
        // Given
        val initialMemory = getMemoryUsage()

        // When - Load the application
        composeTestRule.setContent {
            UnifiedOptimizationTile(
                onResultsReceived = { },
            )
        }

        // Wait for UI to load
        composeTestRule.waitUntil(timeout = 5.seconds) {
            composeTestRule.onNodeWithText("Unified Optimization").assertExists()
        }

        val afterLoadMemory = getMemoryUsage()
        val memoryIncrease = afterLoadMemory - initialMemory

        // Then - Verify memory usage < 200MB for typical sessions
        val memoryUsageMB = memoryIncrease / (1024 * 1024)
        assertTrue(memoryUsageMB < 200, "Memory usage $memoryUsageMB MB exceeds 200MB requirement")
    }

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    fun `test animation smoothness`() = runTest {
        // Given
        var receivedResult: OptimizationResult? = null

        composeTestRule.setContent {
            UnifiedOptimizationTile(
                onResultsReceived = { result ->
                    receivedResult = result
                },
            )
        }

        // When - Complete optimization to trigger animations
        composeTestRule.onNodeWithText("Start Optimization").performClick()

        composeTestRule.waitUntil(timeout = 30.seconds) {
            receivedResult != null
        }

        // Then - Test animation smoothness by navigating through tabs
        val animationStartTime = System.currentTimeMillis()

        composeTestRule.onNodeWithText("Motion Law").performClick()
        composeTestRule.onNodeWithText("Gear Profiles").performClick()
        composeTestRule.onNodeWithText("Efficiency").performClick()
        composeTestRule.onNodeWithText("FEA Analysis").performClick()

        val animationEndTime = System.currentTimeMillis()
        val animationTime = animationEndTime - animationStartTime

        // Animations should complete within 500ms for smooth experience
        assertTrue(animationTime < 500, "Animation time $animationTime ms exceeds 500ms requirement")

        // Verify all tabs are accessible and responsive
        composeTestRule.onNodeWithText("Motion Law").assertIsDisplayed()
        composeTestRule.onNodeWithText("Gear Profiles").assertIsDisplayed()
        composeTestRule.onNodeWithText("Efficiency").assertIsDisplayed()
        composeTestRule.onNodeWithText("FEA Analysis").assertIsDisplayed()
    }

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    fun `test concurrent operations performance`() = runTest {
        // Given
        var receivedResult: OptimizationResult? = null

        composeTestRule.setContent {
            UnifiedOptimizationTile(
                onResultsReceived = { result ->
                    receivedResult = result
                },
            )
        }

        // When - Start optimization and perform concurrent UI operations
        composeTestRule.onNodeWithText("Start Optimization").performClick()

        // Perform concurrent UI operations
        val concurrentStartTime = System.currentTimeMillis()

        // Navigate to different tabs while optimization is running
        composeTestRule.onNodeWithText("Advanced").performClick()
        composeTestRule.onNodeWithText("Presets").performClick()
        composeTestRule.onNodeWithText("Export/Import").performClick()
        composeTestRule.onNodeWithText("Batch Processing").performClick()

        val concurrentEndTime = System.currentTimeMillis()
        val concurrentOperationTime = concurrentEndTime - concurrentStartTime

        // Then - Verify concurrent operations complete within reasonable time
        assertTrue(concurrentOperationTime < 2000, "Concurrent operations time $concurrentOperationTime ms exceeds 2 second limit")

        // Wait for optimization to complete
        composeTestRule.waitUntil(timeout = 30.seconds) {
            receivedResult != null
        }

        // Verify optimization still completed successfully
        assertNotNull(receivedResult)
        assertTrue(receivedResult!!.isSuccess())
    }

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    fun `test error handling performance`() = runTest {
        // Given
        composeTestRule.setContent {
            UnifiedOptimizationTile(
                onResultsReceived = { },
            )
        }

        // When - Trigger error conditions
        val errorHandlingStartTime = System.currentTimeMillis()

        // Input invalid parameters
        composeTestRule.onNodeWithText("Sampling Step").performTextClearance()
        composeTestRule.onNodeWithText("Sampling Step").performTextInput("invalid")

        // Try to start optimization
        composeTestRule.onNodeWithText("Start Optimization").performClick()

        val errorHandlingEndTime = System.currentTimeMillis()
        val errorHandlingTime = errorHandlingEndTime - errorHandlingStartTime

        // Then - Verify error handling is fast
        assertTrue(errorHandlingTime < 500, "Error handling time $errorHandlingTime ms exceeds 500ms requirement")

        // Verify error is displayed
        composeTestRule.onNodeWithText("Invalid parameters").assertIsDisplayed()
    }

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    fun `test accessibility performance`() = runTest {
        // Given
        composeTestRule.setContent {
            UnifiedOptimizationTile(
                onResultsReceived = { },
            )
        }

        // When - Test accessibility features performance
        val accessibilityStartTime = System.currentTimeMillis()

        // Access accessibility settings
        composeTestRule.onNodeWithContentDescription("Accessibility Settings").performClick()

        // Toggle accessibility features
        composeTestRule.onNodeWithText("High Contrast Mode").performClick()
        composeTestRule.onNodeWithText("Large Text").performClick()
        composeTestRule.onNodeWithText("Screen Reader Support").performClick()

        val accessibilityEndTime = System.currentTimeMillis()
        val accessibilityTime = accessibilityEndTime - accessibilityStartTime

        // Then - Verify accessibility features are fast
        assertTrue(accessibilityTime < 1000, "Accessibility features time $accessibilityTime ms exceeds 1 second requirement")

        // Verify accessibility features are applied
        composeTestRule.onNodeWithText("High Contrast Mode").assertIsDisplayed()
        composeTestRule.onNodeWithText("Large Text").assertIsDisplayed()
        composeTestRule.onNodeWithText("Screen Reader Support").assertIsDisplayed()
    }

    /**
     * Get current memory usage in bytes.
     */
    private fun getMemoryUsage(): Long {
        val runtime = Runtime.getRuntime()
        return runtime.totalMemory() - runtime.freeMemory()
    }
}
