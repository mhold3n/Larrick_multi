package com.campro.v5.acceptance

import androidx.compose.ui.test.*
import androidx.compose.ui.test.junit4.createComposeRule
import com.campro.v5.models.OptimizationResult
import com.campro.v5.ui.UnifiedOptimizationTile
import kotlinx.coroutines.test.runTest
import org.junit.jupiter.api.*
import org.junit.jupiter.api.Assertions.*
import java.util.concurrent.TimeUnit
import kotlin.time.Duration.Companion.seconds

/**
 * User acceptance tests to validate the user experience and interface usability.
 *
 * Tests user workflows, interface intuitiveness, and overall user satisfaction
 * with the optimization pipeline application.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class UserAcceptanceTest {

    @get:RegisterExtension
    val composeTestRule = createComposeRule()

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    fun `test user can complete basic optimization workflow`() = runTest {
        // Given - User opens the application
        var receivedResult: OptimizationResult? = null

        composeTestRule.setContent {
            UnifiedOptimizationTile(
                onResultsReceived = { result ->
                    receivedResult = result
                },
            )
        }

        // When - User performs basic optimization workflow
        // 1. User sees the main interface
        composeTestRule.onNodeWithText("Unified Optimization").assertIsDisplayed()
        composeTestRule.onNodeWithText("Start Optimization").assertIsDisplayed()

        // 2. User starts optimization with default parameters
        composeTestRule.onNodeWithText("Start Optimization").performClick()

        // 3. User sees progress indication
        composeTestRule.onNodeWithText("Running optimization...").assertIsDisplayed()

        // 4. User waits for completion
        composeTestRule.waitUntil(timeout = 30.seconds) {
            receivedResult != null
        }

        // Then - User sees results
        composeTestRule.onNodeWithText("Optimization Results").assertIsDisplayed()
        composeTestRule.onNodeWithText("Motion Law").assertIsDisplayed()
        composeTestRule.onNodeWithText("Gear Profiles").assertIsDisplayed()
        composeTestRule.onNodeWithText("Efficiency").assertIsDisplayed()
        composeTestRule.onNodeWithText("FEA Analysis").assertIsDisplayed()

        // Verify results are meaningful
        assertNotNull(receivedResult)
        assertTrue(receivedResult!!.isSuccess())
        assertTrue(receivedResult!!.executionTime > 0)
    }

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    fun `test user can modify parameters intuitively`() = runTest {
        // Given - User opens the application
        composeTestRule.setContent {
            UnifiedOptimizationTile(
                onResultsReceived = { },
            )
        }

        // When - User modifies parameters
        // 1. User sees parameter input fields
        composeTestRule.onNodeWithText("Sampling Step").assertIsDisplayed()
        composeTestRule.onNodeWithText("Stroke Length").assertIsDisplayed()
        composeTestRule.onNodeWithText("Gear Ratio").assertIsDisplayed()
        composeTestRule.onNodeWithText("RPM").assertIsDisplayed()

        // 2. User modifies parameters
        composeTestRule.onNodeWithText("Sampling Step").performTextInput("2.5")
        composeTestRule.onNodeWithText("Stroke Length").performTextInput("120.0")
        composeTestRule.onNodeWithText("Gear Ratio").performTextInput("3.0")
        composeTestRule.onNodeWithText("RPM").performTextInput("2000")

        // 3. User sees parameter presets
        composeTestRule.onNodeWithText("Default").assertIsDisplayed()
        composeTestRule.onNodeWithText("Quick Test").assertIsDisplayed()
        composeTestRule.onNodeWithText("High Performance").assertIsDisplayed()

        // Then - User can apply presets
        composeTestRule.onNodeWithText("Quick Test").performClick()
        composeTestRule.onNodeWithText("High Performance").performClick()

        // Verify parameters are updated
        composeTestRule.onNodeWithText("Start Optimization").assertIsEnabled()
    }

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    fun `test user can navigate results intuitively`() = runTest {
        // Given - User completes an optimization
        var receivedResult: OptimizationResult? = null

        composeTestRule.setContent {
            UnifiedOptimizationTile(
                onResultsReceived = { result ->
                    receivedResult = result
                },
            )
        }

        composeTestRule.onNodeWithText("Start Optimization").performClick()
        composeTestRule.waitUntil(timeout = 30.seconds) {
            receivedResult != null
        }

        // When - User navigates through results
        // 1. User sees tabbed interface
        composeTestRule.onNodeWithText("Motion Law").assertIsDisplayed()
        composeTestRule.onNodeWithText("Gear Profiles").assertIsDisplayed()
        composeTestRule.onNodeWithText("Efficiency").assertIsDisplayed()
        composeTestRule.onNodeWithText("FEA Analysis").assertIsDisplayed()

        // 2. User clicks through tabs
        composeTestRule.onNodeWithText("Motion Law").performClick()
        composeTestRule.onNodeWithText("Gear Profiles").performClick()
        composeTestRule.onNodeWithText("Efficiency").performClick()
        composeTestRule.onNodeWithText("FEA Analysis").performClick()

        // Then - User sees relevant content in each tab
        // Each tab should show meaningful data
        composeTestRule.onNodeWithText("Motion Law").assertIsSelected()
        composeTestRule.onNodeWithText("Gear Profiles").assertIsDisplayed()
        composeTestRule.onNodeWithText("Efficiency").assertIsDisplayed()
        composeTestRule.onNodeWithText("FEA Analysis").assertIsDisplayed()
    }

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    fun `test user can access advanced features easily`() = runTest {
        // Given - User opens the application
        composeTestRule.setContent {
            UnifiedOptimizationTile(
                onResultsReceived = { },
            )
        }

        // When - User accesses advanced features
        // 1. User sees advanced features tab
        composeTestRule.onNodeWithText("Advanced").assertIsDisplayed()

        // 2. User clicks on advanced features
        composeTestRule.onNodeWithText("Advanced").performClick()

        // Then - User sees advanced feature categories
        composeTestRule.onNodeWithText("Presets").assertIsDisplayed()
        composeTestRule.onNodeWithText("Export/Import").assertIsDisplayed()
        composeTestRule.onNodeWithText("Batch Processing").assertIsDisplayed()

        // 3. User can navigate between advanced features
        composeTestRule.onNodeWithText("Presets").performClick()
        composeTestRule.onNodeWithText("Export/Import").performClick()
        composeTestRule.onNodeWithText("Batch Processing").performClick()

        // Verify each category is accessible
        composeTestRule.onNodeWithText("Presets").assertIsDisplayed()
        composeTestRule.onNodeWithText("Export/Import").assertIsDisplayed()
        composeTestRule.onNodeWithText("Batch Processing").assertIsDisplayed()
    }

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    fun `test user can save and load presets intuitively`() = runTest {
        // Given - User opens the application
        composeTestRule.setContent {
            UnifiedOptimizationTile(
                onResultsReceived = { },
            )
        }

        // When - User works with presets
        // 1. User navigates to presets
        composeTestRule.onNodeWithText("Advanced").performClick()
        composeTestRule.onNodeWithText("Presets").performClick()

        // 2. User sees preset management options
        composeTestRule.onNodeWithText("Save Preset").assertIsDisplayed()
        composeTestRule.onNodeWithText("Load Preset").assertIsDisplayed()

        // 3. User can save a preset
        composeTestRule.onNodeWithText("Save Preset").performClick()

        // Then - User sees save dialog
        composeTestRule.onNodeWithText("Save Preset").assertIsDisplayed()

        // 4. User can load presets
        composeTestRule.onNodeWithText("Load Preset").performClick()

        // Then - User sees load dialog
        composeTestRule.onNodeWithText("Load Preset").assertIsDisplayed()
    }

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    fun `test user can export results easily`() = runTest {
        // Given - User completes an optimization
        var receivedResult: OptimizationResult? = null

        composeTestRule.setContent {
            UnifiedOptimizationTile(
                onResultsReceived = { result ->
                    receivedResult = result
                },
            )
        }

        composeTestRule.onNodeWithText("Start Optimization").performClick()
        composeTestRule.waitUntil(timeout = 30.seconds) {
            receivedResult != null
        }

        // When - User exports results
        // 1. User navigates to export functionality
        composeTestRule.onNodeWithText("Advanced").performClick()
        composeTestRule.onNodeWithText("Export/Import").performClick()

        // 2. User sees export options
        composeTestRule.onNodeWithText("Export JSON").assertIsDisplayed()
        composeTestRule.onNodeWithText("Export CSV").assertIsDisplayed()

        // 3. User can export in different formats
        composeTestRule.onNodeWithText("Export JSON").performClick()
        composeTestRule.onNodeWithText("Export CSV").performClick()

        // Then - User sees export completion feedback
        composeTestRule.onNodeWithText("Export completed").assertExists()
    }

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    fun `test user can handle errors gracefully`() = runTest {
        // Given - User opens the application
        composeTestRule.setContent {
            UnifiedOptimizationTile(
                onResultsReceived = { },
            )
        }

        // When - User encounters errors
        // 1. User inputs invalid parameters
        composeTestRule.onNodeWithText("Sampling Step").performTextClearance()
        composeTestRule.onNodeWithText("Sampling Step").performTextInput("invalid")

        // 2. User tries to start optimization
        composeTestRule.onNodeWithText("Start Optimization").performClick()

        // Then - User sees helpful error messages
        composeTestRule.onNodeWithText("Invalid parameters").assertIsDisplayed()

        // 3. User can recover from errors
        composeTestRule.onNodeWithText("Sampling Step").performTextClearance()
        composeTestRule.onNodeWithText("Sampling Step").performTextInput("5.0")

        // Then - User can proceed normally
        composeTestRule.onNodeWithText("Start Optimization").assertIsEnabled()
    }

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    fun `test user can access accessibility features`() = runTest {
        // Given - User opens the application
        composeTestRule.setContent {
            UnifiedOptimizationTile(
                onResultsReceived = { },
            )
        }

        // When - User accesses accessibility features
        // 1. User sees accessibility button
        composeTestRule.onNodeWithContentDescription("Accessibility Settings").assertIsDisplayed()

        // 2. User clicks accessibility button
        composeTestRule.onNodeWithContentDescription("Accessibility Settings").performClick()

        // Then - User sees accessibility options
        composeTestRule.onNodeWithText("Accessibility Settings").assertIsDisplayed()
        composeTestRule.onNodeWithText("High Contrast Mode").assertIsDisplayed()
        composeTestRule.onNodeWithText("Large Text").assertIsDisplayed()
        composeTestRule.onNodeWithText("Screen Reader Support").assertIsDisplayed()

        // 3. User can toggle accessibility features
        composeTestRule.onNodeWithText("High Contrast Mode").performClick()
        composeTestRule.onNodeWithText("Large Text").performClick()

        // Then - User sees changes applied
        composeTestRule.onNodeWithText("High Contrast Mode").assertIsDisplayed()
        composeTestRule.onNodeWithText("Large Text").assertIsDisplayed()
    }

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    fun `test user can cancel operations`() = runTest {
        // Given - User opens the application
        composeTestRule.setContent {
            UnifiedOptimizationTile(
                onResultsReceived = { },
            )
        }

        // When - User starts and cancels optimization
        // 1. User starts optimization
        composeTestRule.onNodeWithText("Start Optimization").performClick()

        // 2. User sees cancel option
        composeTestRule.onNodeWithText("Cancel").assertIsDisplayed()
        composeTestRule.onNodeWithText("Cancel").assertIsEnabled()

        // 3. User cancels optimization
        composeTestRule.onNodeWithText("Cancel").performClick()

        // Then - User sees cancellation feedback
        composeTestRule.onNodeWithText("Optimization cancelled").assertIsDisplayed()

        // 4. User can start again
        composeTestRule.onNodeWithText("Start Optimization").assertIsDisplayed()
        composeTestRule.onNodeWithText("Start Optimization").assertIsEnabled()
    }

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    fun `test user interface is intuitive and responsive`() = runTest {
        // Given - User opens the application
        composeTestRule.setContent {
            UnifiedOptimizationTile(
                onResultsReceived = { },
            )
        }

        // When - User interacts with the interface
        // 1. User sees clear visual hierarchy
        composeTestRule.onNodeWithText("Unified Optimization").assertIsDisplayed()
        composeTestRule.onNodeWithText("Start Optimization").assertIsDisplayed()

        // 2. User can interact with all elements
        composeTestRule.onNodeWithText("Start Optimization").assertIsEnabled()
        composeTestRule.onNodeWithText("Default").assertIsEnabled()
        composeTestRule.onNodeWithText("Quick Test").assertIsEnabled()
        composeTestRule.onNodeWithText("High Performance").assertIsEnabled()

        // 3. User sees immediate feedback
        composeTestRule.onNodeWithText("Default").performClick()
        composeTestRule.onNodeWithText("Quick Test").performClick()
        composeTestRule.onNodeWithText("High Performance").performClick()

        // Then - Interface remains responsive
        composeTestRule.onNodeWithText("Start Optimization").assertIsEnabled()
        composeTestRule.onNodeWithText("Advanced").assertIsEnabled()
        composeTestRule.onNodeWithContentDescription("Accessibility Settings").assertIsEnabled()
    }
}
