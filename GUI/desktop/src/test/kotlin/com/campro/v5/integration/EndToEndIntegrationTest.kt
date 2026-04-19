package com.campro.v5.integration

import androidx.compose.ui.test.*
import androidx.compose.ui.test.junit4.createComposeRule
import com.campro.v5.models.OptimizationParameters
import com.campro.v5.models.OptimizationResult
import com.campro.v5.pipeline.UnifiedOptimizationBridge
import com.campro.v5.ui.UnifiedOptimizationTile
import kotlinx.coroutines.test.runTest
import org.junit.jupiter.api.*
import org.junit.jupiter.api.Assertions.*
import org.junit.jupiter.api.extension.ExtendWith
import org.junit.jupiter.api.extension.RegisterExtension
import java.nio.file.Paths
import java.util.concurrent.TimeUnit
import kotlin.time.Duration.Companion.seconds

/**
 * Comprehensive end-to-end integration tests for the unified optimization pipeline.
 *
 * Tests the complete workflow from parameter input through optimization execution
 * to result visualization and export functionality.
 */
@ExtendWith(IntegrationTestExtension::class)
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class EndToEndIntegrationTest {

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
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    fun `test complete optimization workflow`() = runTest {
        // Given
        val parameters = OptimizationParameters.createDefault()
        var receivedResult: OptimizationResult? = null

        // When
        composeTestRule.setContent {
            UnifiedOptimizationTile(
                onResultsReceived = { result ->
                    receivedResult = result
                },
            )
        }

        // Start optimization
        composeTestRule.onNodeWithText("Start Optimization").performClick()

        // Wait for optimization to complete
        composeTestRule.waitUntil(timeout = 30.seconds) {
            receivedResult != null
        }

        // Then
        assertNotNull(receivedResult)
        assertTrue(receivedResult!!.isSuccess())
        assertTrue(receivedResult!!.executionTime > 0)
        assertNotNull(receivedResult!!.motionLaw)
        assertNotNull(receivedResult!!.optimalProfiles)
        assertNotNull(receivedResult!!.feaAnalysis)
    }

    @Test
    @Timeout(value = 60, unit = TimeUnit.SECONDS)
    fun `test parameter input and validation`() = runTest {
        // Given
        composeTestRule.setContent {
            UnifiedOptimizationTile(
                onResultsReceived = { },
            )
        }

        // When - Input parameters
        composeTestRule.onNodeWithText("Sampling Step").performTextInput("5.0")
        composeTestRule.onNodeWithText("Stroke Length").performTextInput("100.0")
        composeTestRule.onNodeWithText("Gear Ratio").performTextInput("2.5")
        composeTestRule.onNodeWithText("RPM").performTextInput("1500")

        // Then - Verify parameters are accepted
        composeTestRule.onNodeWithText("Start Optimization").assertIsEnabled()

        // When - Input invalid parameters
        composeTestRule.onNodeWithText("Sampling Step").performTextClearance()
        composeTestRule.onNodeWithText("Sampling Step").performTextInput("invalid")

        // Then - Verify validation works
        composeTestRule.onNodeWithText("Start Optimization").assertIsNotEnabled()
    }

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    fun `test preset management workflow`() = runTest {
        // Given
        composeTestRule.setContent {
            UnifiedOptimizationTile(
                onResultsReceived = { },
            )
        }

        // When - Navigate to Advanced Features
        composeTestRule.onNodeWithText("Advanced").performClick()

        // Then - Verify preset panel is accessible
        composeTestRule.onNodeWithText("Presets").assertIsDisplayed()
        composeTestRule.onNodeWithText("Save Preset").assertIsDisplayed()
        composeTestRule.onNodeWithText("Load Preset").assertIsDisplayed()

        // When - Save a preset
        composeTestRule.onNodeWithText("Save Preset").performClick()

        // Then - Verify save dialog appears
        composeTestRule.onNodeWithText("Save Preset").assertIsDisplayed()
    }

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    fun `test export functionality workflow`() = runTest {
        // Given
        var receivedResult: OptimizationResult? = null

        composeTestRule.setContent {
            UnifiedOptimizationTile(
                onResultsReceived = { result ->
                    receivedResult = result
                },
            )
        }

        // Complete an optimization first
        composeTestRule.onNodeWithText("Start Optimization").performClick()
        composeTestRule.waitUntil(timeout = 30.seconds) {
            receivedResult != null
        }

        // When - Navigate to export functionality
        composeTestRule.onNodeWithText("Advanced").performClick()
        composeTestRule.onNodeWithText("Export/Import").performClick()

        // Then - Verify export options are available
        composeTestRule.onNodeWithText("Export JSON").assertIsDisplayed()
        composeTestRule.onNodeWithText("Export CSV").assertIsDisplayed()

        // When - Perform export
        composeTestRule.onNodeWithText("Export JSON").performClick()

        // Then - Verify export completes (no error dialog)
        composeTestRule.onNodeWithText("Export completed").assertExists()
    }

    @Test
    @Timeout(value = 60, unit = TimeUnit.SECONDS)
    fun `test batch processing workflow`() = runTest {
        // Given
        composeTestRule.setContent {
            UnifiedOptimizationTile(
                onResultsReceived = { },
            )
        }

        // When - Navigate to batch processing
        composeTestRule.onNodeWithText("Advanced").performClick()
        composeTestRule.onNodeWithText("Batch Processing").performClick()

        // Then - Verify batch controls are available
        composeTestRule.onNodeWithText("Start Batch").assertIsDisplayed()

        // When - Start batch processing
        composeTestRule.onNodeWithText("Start Batch").performClick()

        // Then - Verify batch configuration dialog appears
        composeTestRule.onNodeWithText("Batch Configuration").assertIsDisplayed()
    }

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    fun `test visualization components workflow`() = runTest {
        // Given
        var receivedResult: OptimizationResult? = null

        composeTestRule.setContent {
            UnifiedOptimizationTile(
                onResultsReceived = { result ->
                    receivedResult = result
                },
            )
        }

        // Complete an optimization first
        composeTestRule.onNodeWithText("Start Optimization").performClick()
        composeTestRule.waitUntil(timeout = 30.seconds) {
            receivedResult != null
        }

        // When - Navigate through visualization tabs
        composeTestRule.onNodeWithText("Motion Law").performClick()
        composeTestRule.onNodeWithText("Gear Profiles").performClick()
        composeTestRule.onNodeWithText("Efficiency").performClick()
        composeTestRule.onNodeWithText("FEA Analysis").performClick()

        // Then - Verify all tabs are accessible and display content
        composeTestRule.onNodeWithText("Motion Law").assertIsSelected()
        composeTestRule.onNodeWithText("Gear Profiles").assertIsDisplayed()
        composeTestRule.onNodeWithText("Efficiency").assertIsDisplayed()
        composeTestRule.onNodeWithText("FEA Analysis").assertIsDisplayed()
    }

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    fun `test error handling workflow`() = runTest {
        // Given
        composeTestRule.setContent {
            UnifiedOptimizationTile(
                onResultsReceived = { },
            )
        }

        // When - Input invalid parameters to trigger error
        composeTestRule.onNodeWithText("Sampling Step").performTextClearance()
        composeTestRule.onNodeWithText("Sampling Step").performTextInput("-1.0") // Invalid negative value

        // Then - Verify error handling
        composeTestRule.onNodeWithText("Start Optimization").assertIsNotEnabled()

        // When - Try to start optimization with invalid parameters
        composeTestRule.onNodeWithText("Start Optimization").performClick()

        // Then - Verify error message is displayed
        composeTestRule.onNodeWithText("Invalid parameters").assertIsDisplayed()
    }

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    fun `test accessibility features workflow`() = runTest {
        // Given
        composeTestRule.setContent {
            UnifiedOptimizationTile(
                onResultsReceived = { },
            )
        }

        // When - Access accessibility settings
        composeTestRule.onNodeWithContentDescription("Accessibility Settings").performClick()

        // Then - Verify accessibility panel is displayed
        composeTestRule.onNodeWithText("Accessibility Settings").assertIsDisplayed()
        composeTestRule.onNodeWithText("High Contrast Mode").assertIsDisplayed()
        composeTestRule.onNodeWithText("Large Text").assertIsDisplayed()
        composeTestRule.onNodeWithText("Screen Reader Support").assertIsDisplayed()

        // When - Toggle accessibility features
        composeTestRule.onNodeWithText("High Contrast Mode").performClick()
        composeTestRule.onNodeWithText("Large Text").performClick()

        // Then - Verify toggles work
        composeTestRule.onNodeWithText("High Contrast Mode").assertIsDisplayed()
        composeTestRule.onNodeWithText("Large Text").assertIsDisplayed()
    }

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    fun `test performance monitoring workflow`() = runTest {
        // Given
        var receivedResult: OptimizationResult? = null

        composeTestRule.setContent {
            UnifiedOptimizationTile(
                onResultsReceived = { result ->
                    receivedResult = result
                },
            )
        }

        // When - Start optimization and monitor performance
        val startTime = System.currentTimeMillis()
        composeTestRule.onNodeWithText("Start Optimization").performClick()

        composeTestRule.waitUntil(timeout = 30.seconds) {
            receivedResult != null
        }

        val endTime = System.currentTimeMillis()
        val executionTime = endTime - startTime

        // Then - Verify performance requirements
        assertTrue(executionTime < 30000) // Less than 30 seconds
        assertNotNull(receivedResult)
        assertTrue(receivedResult!!.executionTime > 0)

        // Verify UI remains responsive during optimization
        composeTestRule.onNodeWithText("Cancel").assertIsDisplayed()
    }

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    fun `test state management workflow`() = runTest {
        // Given
        composeTestRule.setContent {
            UnifiedOptimizationTile(
                onResultsReceived = { },
            )
        }

        // When - Start optimization
        composeTestRule.onNodeWithText("Start Optimization").performClick()

        // Then - Verify running state
        composeTestRule.onNodeWithText("Running optimization...").assertIsDisplayed()
        composeTestRule.onNodeWithText("Cancel").assertIsDisplayed()

        // When - Cancel optimization
        composeTestRule.onNodeWithText("Cancel").performClick()

        // Then - Verify cancelled state
        composeTestRule.onNodeWithText("Optimization cancelled").assertIsDisplayed()
        composeTestRule.onNodeWithText("Start Optimization").assertIsDisplayed()
    }
}

/**
 * Integration test extension for setup and teardown.
 */
class IntegrationTestExtension :
    BeforeEachCallback,
    AfterEachCallback {

    override fun beforeEach(context: ExtensionContext) {
        // Setup integration test environment
        System.setProperty("test.mode", "integration")
    }

    override fun afterEach(context: ExtensionContext) {
        // Cleanup integration test environment
        System.clearProperty("test.mode")
    }
}
