package com.campro.v5.ui

import androidx.compose.ui.test.*
import androidx.compose.ui.test.junit4.createComposeRule
import com.campro.v5.models.OptimizationResult
import com.campro.v5.models.MotionLawData
import com.campro.v5.models.GearProfileData
import com.campro.v5.models.ToothProfileData
import com.campro.v5.models.FEAAnalysisData
import com.campro.v5.pipeline.OptimizationPort
import io.mockk.*
import org.junit.Rule
import org.junit.Test
import org.junit.Assert.*

/**
 * Tests for UnifiedOptimizationTile.
 */
class UnifiedOptimizationTileTest {

    @get:Rule
    val composeTestRule = createComposeRule()

    @Test
    fun testUnifiedOptimizationTileCreation() {
        // Given
        val mockOptimizationPort = mockk<OptimizationPort>()
        val onResultsReceived = mockk<(OptimizationResult) -> Unit>()

        // When
        composeTestRule.setContent {
            UnifiedOptimizationTile(
                optimizationPort = mockOptimizationPort,
                onResultsReceived = onResultsReceived,
            )
        }

        // Then
        composeTestRule.onNodeWithText("Unified Optimization").assertIsDisplayed()
        composeTestRule.onNodeWithText("Parameters").assertIsDisplayed()
        composeTestRule.onNodeWithText("Start Optimization").assertIsDisplayed()
    }

    @Test
    fun testParameterInputFields() {
        // Given
        val mockOptimizationPort = mockk<OptimizationPort>()

        // When
        composeTestRule.setContent {
            UnifiedOptimizationTile(optimizationPort = mockOptimizationPort)
        }

        // Then
        // Check basic parameter fields are present
        composeTestRule.onNodeWithText("Sampling Step").assertIsDisplayed()
        composeTestRule.onNodeWithText("Stroke Length").assertIsDisplayed()
        composeTestRule.onNodeWithText("Gear Ratio").assertIsDisplayed()
        composeTestRule.onNodeWithText("RPM").assertIsDisplayed()
    }

    @Test
    fun testParameterTabs() {
        // Given
        val mockOptimizationPort = mockk<OptimizationPort>()

        // When
        composeTestRule.setContent {
            UnifiedOptimizationTile(optimizationPort = mockOptimizationPort)
        }

        // Then
        // Check category tabs are present
        composeTestRule.onNodeWithText("Basic").assertIsDisplayed()
        composeTestRule.onNodeWithText("Motion Law").assertIsDisplayed()
        composeTestRule.onNodeWithText("Gear Design").assertIsDisplayed()
        composeTestRule.onNodeWithText("Advanced").assertIsDisplayed()
    }

    @Test
    fun testParameterInput() {
        // Given
        val mockOptimizationPort = mockk<OptimizationPort>()

        // When
        composeTestRule.setContent {
            UnifiedOptimizationTile(optimizationPort = mockOptimizationPort)
        }

        // Then
        // Test parameter input
        composeTestRule.onNodeWithText("Sampling Step")
            .performTextInput("2.5")
            .assertTextContains("2.5")

        composeTestRule.onNodeWithText("Stroke Length")
            .performTextInput("15.0")
            .assertTextContains("15.0")
    }

    @Test
    fun testTabNavigation() {
        // Given
        val mockOptimizationPort = mockk<OptimizationPort>()

        // When
        composeTestRule.setContent {
            UnifiedOptimizationTile(optimizationPort = mockOptimizationPort)
        }

        // Then
        // Test tab navigation
        composeTestRule.onNodeWithText("Motion Law").performClick()
        composeTestRule.onNodeWithText("TDC (Top Dead Center)").assertIsDisplayed()

        composeTestRule.onNodeWithText("Gear Design").performClick()
        composeTestRule.onNodeWithText("Physical dimensions").assertIsDisplayed()

        composeTestRule.onNodeWithText("Advanced").performClick()
        composeTestRule.onNodeWithText("Planet Radius Factors").assertIsDisplayed()
    }

    @Test
    fun testActionButtons() {
        // Given
        val mockOptimizationPort = mockk<OptimizationPort>()

        // When
        composeTestRule.setContent {
            UnifiedOptimizationTile(optimizationPort = mockOptimizationPort)
        }

        // Then
        // Check action buttons are present
        composeTestRule.onNodeWithText("Show Errors").assertIsDisplayed()
        composeTestRule.onNodeWithText("Reset to Defaults").assertIsDisplayed()
        composeTestRule.onNodeWithText("Quick Test").assertIsDisplayed()
    }

    @Test
    fun testParameterValidation() {
        // Given
        val mockOptimizationPort = mockk<OptimizationPort>()

        // When
        composeTestRule.setContent {
            UnifiedOptimizationTile(optimizationPort = mockOptimizationPort)
        }

        // Then
        // Test parameter validation by entering invalid values
        composeTestRule.onNodeWithText("Sampling Step")
            .performTextInput("-1.0")

        composeTestRule.onNodeWithText("Show Errors").performClick()

        // Should show validation errors
        composeTestRule.onNodeWithText("Validation Errors").assertIsDisplayed()
    }

    @Test
    fun testPresetButtons() {
        // Given
        val mockOptimizationPort = mockk<OptimizationPort>()

        // When
        composeTestRule.setContent {
            UnifiedOptimizationTile(optimizationPort = mockOptimizationPort)
        }

        // Then
        // Test preset buttons
        composeTestRule.onNodeWithText("Reset to Defaults").performClick()
        composeTestRule.onNodeWithText("Quick Test").performClick()

        // Verify buttons are clickable
        composeTestRule.onNodeWithText("Reset to Defaults").assertIsEnabled()
        composeTestRule.onNodeWithText("Quick Test").assertIsEnabled()
    }

    private fun createTestOptimizationResult(): OptimizationResult = OptimizationResult(
        status = "success",
        motionLaw = MotionLawData(
            thetaDeg = doubleArrayOf(0.0, 90.0, 180.0),
            displacement = doubleArrayOf(0.0, 50.0, 100.0),
            velocity = doubleArrayOf(100.0, 0.0, -100.0),
            acceleration = doubleArrayOf(0.0, -1000.0, 0.0),
        ),
        optimalProfiles = GearProfileData(
            rSun = doubleArrayOf(110.0, 115.0, 120.0),
            rPlanet = doubleArrayOf(175.0, 180.0, 185.0),
            rRingInner = doubleArrayOf(460.0, 470.0, 480.0),
            gearRatio = 2.0,
            optimalMethod = "litvin",
            efficiencyAnalysis = null,
        ),
        toothProfiles = ToothProfileData(
            sunTeeth = null,
            planetTeeth = null,
            ringTeeth = null,
        ),
        feaAnalysis = FEAAnalysisData(
            maxStress = 150.0,
            naturalFrequencies = doubleArrayOf(100.0, 200.0, 300.0),
            fatigueLife = 1000000.0,
            modeShapes = arrayOf("Mode 1", "Mode 2", "Mode 3"),
            recommendations = arrayOf("Recommendation 1", "Recommendation 2"),
        ),
        executionTime = 1.5,
    )
}
