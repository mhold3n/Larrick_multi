package com.campro.v5.layout

import androidx.compose.ui.unit.dp
import com.campro.v5.EventSystem
import com.campro.v5.LayoutEvent
import com.campro.v5.waitForConditionOrFail
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertNotNull
import org.junit.jupiter.api.Assertions.assertSame
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import java.io.ByteArrayOutputStream
import java.io.PrintStream

/**
 * Tests for the LayoutManager.
 *
 * This class tests the functionality of the LayoutManager, including event emission,
 * layout template handling, and helper methods.
 */
class LayoutManagerTest {
    private lateinit var layoutManager: LayoutManager
    private val originalOut = System.out
    private val outContent = ByteArrayOutputStream()

    @BeforeEach
    fun setUp() {
        // Create a new LayoutManager for each test
        layoutManager = LayoutManager()

        // Reset the layout manager state
        layoutManager.resetState()

        // Clear the event system
        EventSystem.clear()

        // Redirect System.out to capture event logging
        System.setOut(PrintStream(outContent))

        // Set testing mode
        System.setProperty("testing.mode", "true")

        println("[DEBUG_LOG] Test setup complete")
    }

    @AfterEach
    fun tearDown() {
        // Reset System.out
        System.setOut(originalOut)

        // Clear the event system
        EventSystem.clear()

        // Reset testing mode
        System.clearProperty("testing.mode")

        println("[DEBUG_LOG] Test teardown complete")
    }

    /**
     * Test that window size changes emit events.
     */
    @Test
    fun testWindowSizeChangeEmitsEvent() = runBlocking {
        // Set up a collector for layout events
        val events = EventSystem.events("layout")

        // Update window size
        layoutManager.updateWindowSize(1000.dp, 800.dp)

        // Collect the event with a longer timeout
        val receivedEvent =
            withTimeout(3000) {
                events.first()
            }

        // Verify the event
        assertTrue(receivedEvent is LayoutEvent, "Received event should be a LayoutEvent")
        assertEquals("window_size_changed", (receivedEvent as LayoutEvent).action, "Action should be window_size_changed")
        assertEquals(1000.0f, receivedEvent.params["width"], "Width should match")
        assertEquals(800.0f, receivedEvent.params["height"], "Height should match")

        println("[DEBUG_LOG] Window size change event test complete")
    }

    /**
     * Test that density factor changes emit events.
     */
    @Test
    fun testDensityFactorChangeEmitsEvent() = runBlocking {
        // Set up a collector for layout events
        val events = EventSystem.events("layout")

        // Update density factor
        layoutManager.updateDensityFactor(1.5f)

        // Collect the event with a longer timeout
        val receivedEvent =
            withTimeout(3000) {
                events.first()
            }

        // Verify the event
        assertTrue(receivedEvent is LayoutEvent, "Received event should be a LayoutEvent")
        assertEquals("density_changed", (receivedEvent as LayoutEvent).action, "Action should be density_changed")
        assertEquals(1.5f, receivedEvent.params["factor"], "Factor should match")

        println("[DEBUG_LOG] Density factor change event test complete")
    }

    /**
     * Test that template changes emit events.
     */
    @Test
    fun testTemplateChangeEmitsEvent() = runBlocking {
        // Set up a collector for layout events
        val events = EventSystem.events("layout")

        // Set template
        layoutManager.setTemplate(LayoutManager.LayoutTemplate.ANALYSIS_WORKFLOW)

        // Collect the event with a longer timeout
        val receivedEvent =
            withTimeout(3000) {
                events.first()
            }

        // Verify the event
        assertTrue(receivedEvent is LayoutEvent, "Received event should be a LayoutEvent")
        assertEquals("template_changed", (receivedEvent as LayoutEvent).action, "Action should be template_changed")
        assertEquals("ANALYSIS_WORKFLOW", receivedEvent.params["template"], "Template should match")

        println("[DEBUG_LOG] Template change event test complete")
    }

    /**
     * Test that window size changes can trigger template changes.
     */
    @Test
    fun testWindowSizeChangesCanTriggerTemplateChanges() = runBlocking {
        // Set up a collector for layout events
        val events = mutableListOf<LayoutEvent>()

        // Set up event collection
        val job =
            launch {
                EventSystem.events("layout").collect { event ->
                    if (event is LayoutEvent) {
                        events.add(event)
                    }
                }
            }

        // Set initial window size to medium
        layoutManager.updateWindowSize(1200.dp, 900.dp)

        // Wait for the initial window size event
        waitForConditionOrFail(
            maxAttempts = 20,
            delayMs = 100,
            message = "Did not receive initial window size changed event",
        ) {
            events.any { it is LayoutEvent && it.action == "window_size_changed" }
        }

        // Clear events for the next test
        events.clear()

        // Update window size to small (should trigger template change to COMPACT)
        layoutManager.updateWindowSize(700.dp, 500.dp)

        // Wait for both events to be received with retries
        waitForConditionOrFail(
            maxAttempts = 20,
            delayMs = 100,
            message = "Did not receive both window_size_changed and template_changed events",
        ) {
            events.any { it is LayoutEvent && it.action == "window_size_changed" } &&
                events.any { it is LayoutEvent && it.action == "template_changed" }
        }

        // Get the events
        val windowSizeEvent = events.find { it.action == "window_size_changed" }
        val templateEvent = events.find { it.action == "template_changed" }

        // Cancel the event collection job
        job.cancel()

        // Verify events
        assertNotNull(windowSizeEvent, "Window size event should be emitted")
        assertNotNull(templateEvent, "Template event should be emitted")
        assertEquals("COMPACT", templateEvent?.params?.get("template"), "Template should be COMPACT")

        println("[DEBUG_LOG] Window size triggering template change test complete")
    }

    /**
     * Test helper methods for different templates.
     */
    @Test
    fun testHelperMethodsForDifferentTemplates() {
        // Test COMPACT template
        layoutManager.setTemplate(LayoutManager.LayoutTemplate.COMPACT)
        assertTrue(layoutManager.shouldUseCompactMode(), "COMPACT should use compact mode")
        assertTrue(layoutManager.shouldUseSingleColumn(), "COMPACT should use single column")

        // Test TOUCH_OPTIMIZED template
        layoutManager.setTemplate(LayoutManager.LayoutTemplate.TOUCH_OPTIMIZED)
        assertTrue(layoutManager.shouldUseCompactMode(), "TOUCH_OPTIMIZED should use compact mode")
        assertTrue(layoutManager.shouldUseSingleColumn(), "TOUCH_OPTIMIZED should use single column")

        // Test SINGLE_PANEL template
        layoutManager.setTemplate(LayoutManager.LayoutTemplate.SINGLE_PANEL)
        assertTrue(layoutManager.shouldUseSingleColumn(), "SINGLE_PANEL should use single column")

        // Test DESIGN_WORKFLOW template
        layoutManager.setTemplate(LayoutManager.LayoutTemplate.DESIGN_WORKFLOW)
        assertTrue(layoutManager.shouldEmphasizeDesign(), "DESIGN_WORKFLOW should emphasize design")
        assertFalse(layoutManager.shouldEmphasizeAnalysis(), "DESIGN_WORKFLOW should not emphasize analysis")

        // Test ANALYSIS_WORKFLOW template
        layoutManager.setTemplate(LayoutManager.LayoutTemplate.ANALYSIS_WORKFLOW)
        assertTrue(layoutManager.shouldEmphasizeAnalysis(), "ANALYSIS_WORKFLOW should emphasize analysis")
        assertFalse(layoutManager.shouldEmphasizeDesign(), "ANALYSIS_WORKFLOW should not emphasize design")

        // Test SIMULATION_WORKFLOW template
        layoutManager.setTemplate(LayoutManager.LayoutTemplate.SIMULATION_WORKFLOW)
        assertTrue(layoutManager.shouldEmphasizeSimulation(), "SIMULATION_WORKFLOW should emphasize simulation")

        // Test REPORTING template
        layoutManager.setTemplate(LayoutManager.LayoutTemplate.REPORTING)
        assertTrue(layoutManager.shouldEmphasizeReporting(), "REPORTING should emphasize reporting")

        // Test COLLABORATION template
        layoutManager.setTemplate(LayoutManager.LayoutTemplate.COLLABORATION)
        assertTrue(layoutManager.shouldEmphasizeCollaboration(), "COLLABORATION should emphasize collaboration")

        // Test DEVELOPMENT template
        layoutManager.setTemplate(LayoutManager.LayoutTemplate.DEVELOPMENT)
        assertTrue(layoutManager.shouldShowDevelopmentInfo(), "DEVELOPMENT should show development info")

        // Test PRESENTATION template
        layoutManager.setTemplate(LayoutManager.LayoutTemplate.PRESENTATION)
        assertTrue(layoutManager.shouldOptimizeForPresentation(), "PRESENTATION should optimize for presentation")

        println("[DEBUG_LOG] Helper methods test complete")
    }

    /**
     * Test appropriate spacing for different window sizes.
     */
    @Test
    fun testAppropriateSpacingForDifferentWindowSizes() {
        // Test small window
        layoutManager.updateWindowSize(700.dp, 500.dp)
        assertEquals(8.dp, layoutManager.getAppropriateSpacing(), "Small window should have 8.dp spacing")

        // Test medium window
        layoutManager.updateWindowSize(1000.dp, 800.dp)
        assertEquals(16.dp, layoutManager.getAppropriateSpacing(), "Medium window should have 16.dp spacing")

        // Test large window
        layoutManager.updateWindowSize(1600.dp, 1200.dp)
        assertEquals(24.dp, layoutManager.getAppropriateSpacing(), "Large window should have 24.dp spacing")

        // Test with different density factor
        layoutManager.updateDensityFactor(1.5f)
        assertEquals(24.dp * 1.5f, layoutManager.getAppropriateSpacing(), "Spacing should be adjusted by density factor")

        println("[DEBUG_LOG] Appropriate spacing test complete")
    }

    /**
     * Test singleton instance.
     */
    @Test
    fun testSingletonInstance() {
        val instance1 = LayoutManager.getInstance()
        val instance2 = LayoutManager.getInstance()

        assertSame(instance1, instance2, "getInstance() should return the same instance")

        println("[DEBUG_LOG] Singleton instance test complete")
    }
}
