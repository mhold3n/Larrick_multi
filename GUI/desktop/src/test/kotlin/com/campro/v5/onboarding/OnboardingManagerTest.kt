package com.campro.v5.onboarding

import com.campro.v5.EventSystem
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertNotNull
import org.junit.jupiter.api.Assertions.assertNull
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import java.io.ByteArrayOutputStream
import java.io.PrintStream

/**
 * Tests for the OnboardingManager.
 *
 * This class tests the functionality of the OnboardingManager, including event emission,
 * navigation through onboarding steps, tutorials, and sample projects.
 */
class OnboardingManagerTest {
    private lateinit var onboardingManager: OnboardingManager
    private val originalOut = System.out
    private val outContent = ByteArrayOutputStream()

    @BeforeEach
    fun setUp() {
        // Create a new OnboardingManager for each test
        onboardingManager = OnboardingManager()

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
     * Test that starting onboarding emits events.
     */
    @Test
    fun testStartOnboardingEmitsEvents() = runBlocking {
        // Set up a collector for onboarding events
        val events = EventSystem.events("onboarding_started")

        // Start onboarding
        onboardingManager.startOnboarding()

        // Collect the event with a timeout
        val receivedEvent =
            withTimeout(1000) {
                events.first()
            }

        // Verify the event
        assertTrue(receivedEvent is OnboardingEvent.OnboardingStarted, "Received event should be an OnboardingStarted event")

        // Verify that onboarding is not completed
        assertFalse(onboardingManager.isOnboardingCompleted(), "Onboarding should not be completed")

        // Verify that the current step is "welcome"
        assertEquals("welcome", onboardingManager.getCurrentStep()?.id, "Current step should be welcome")

        println("[DEBUG_LOG] Start onboarding event test complete")
    }

    /**
     * Test navigation through onboarding steps.
     */
    @Test
    fun testOnboardingStepNavigation() = runBlocking {
        // Set up a collector for step changed events
        val events = EventSystem.events("step_changed")

        // Start onboarding
        onboardingManager.startOnboarding()

        // Go to next step
        val nextStepResult = onboardingManager.nextStep()

        // Verify that next step was successful
        assertTrue(nextStepResult, "Next step should be successful")

        // Collect the event with a timeout
        val receivedEvent =
            withTimeout(1000) {
                events.first()
            }

        // Verify the event
        assertTrue(receivedEvent is OnboardingEvent.StepChanged, "Received event should be a StepChanged event")
        assertEquals("ui_overview", (receivedEvent as OnboardingEvent.StepChanged).step.id, "Step ID should be ui_overview")

        // Verify that the current step is "ui_overview"
        assertEquals("ui_overview", onboardingManager.getCurrentStep()?.id, "Current step should be ui_overview")

        // Go to previous step
        val previousStepResult = onboardingManager.previousStep()

        // Verify that previous step was successful
        assertTrue(previousStepResult, "Previous step should be successful")

        // Verify that the current step is "welcome"
        assertEquals("welcome", onboardingManager.getCurrentStep()?.id, "Current step should be welcome")

        // Go to a specific step
        val goToStepResult = onboardingManager.goToStep("parameters")

        // Verify that go to step was successful
        assertTrue(goToStepResult, "Go to step should be successful")

        // Verify that the current step is "parameters"
        assertEquals("parameters", onboardingManager.getCurrentStep()?.id, "Current step should be parameters")

        println("[DEBUG_LOG] Onboarding step navigation test complete")
    }

    /**
     * Test completing the onboarding process.
     */
    @Test
    fun testCompletingOnboarding() = runBlocking {
        // Set up a collector for onboarding completed events
        val events = EventSystem.events("onboarding_completed")

        // Start onboarding
        onboardingManager.startOnboarding()

        // Go through all steps until completion
        var result = true
        while (result) {
            result = onboardingManager.nextStep()
        }

        // Collect the event with a timeout
        val receivedEvent =
            withTimeout(1000) {
                events.first()
            }

        // Verify the event
        assertTrue(receivedEvent is OnboardingEvent.OnboardingCompleted, "Received event should be an OnboardingCompleted event")

        // Verify that onboarding is completed
        assertTrue(onboardingManager.isOnboardingCompleted(), "Onboarding should be completed")

        // Verify that the current step is null
        assertNull(onboardingManager.getCurrentStep(), "Current step should be null")

        println("[DEBUG_LOG] Completing onboarding test complete")
    }

    /**
     * Test skipping the onboarding process.
     */
    @Test
    fun testSkippingOnboarding() = runBlocking {
        // Set up a collector for onboarding skipped events
        val events = EventSystem.events("onboarding_skipped")

        // Start onboarding
        onboardingManager.startOnboarding()

        // Skip onboarding
        onboardingManager.skipOnboarding()

        // Collect the event with a timeout
        val receivedEvent =
            withTimeout(1000) {
                events.first()
            }

        // Verify the event
        assertTrue(receivedEvent is OnboardingEvent.OnboardingSkipped, "Received event should be an OnboardingSkipped event")

        // Verify that onboarding is completed
        assertTrue(onboardingManager.isOnboardingCompleted(), "Onboarding should be completed")

        // Verify that the current step is null
        assertNull(onboardingManager.getCurrentStep(), "Current step should be null")

        println("[DEBUG_LOG] Skipping onboarding test complete")
    }

    /**
     * Test resetting the onboarding process.
     */
    @Test
    fun testResettingOnboarding() = runBlocking {
        // Set up a collector for onboarding reset events
        val events = EventSystem.events("onboarding_reset")

        // Start and skip onboarding
        onboardingManager.startOnboarding()
        onboardingManager.skipOnboarding()

        // Reset onboarding
        onboardingManager.resetOnboarding()

        // Collect the event with a timeout
        val receivedEvent =
            withTimeout(1000) {
                events.first()
            }

        // Verify the event
        assertTrue(receivedEvent is OnboardingEvent.OnboardingReset, "Received event should be an OnboardingReset event")

        // Verify that onboarding is not completed
        assertFalse(onboardingManager.isOnboardingCompleted(), "Onboarding should not be completed")

        // Verify that the current step is "welcome"
        assertEquals("welcome", onboardingManager.getCurrentStep()?.id, "Current step should be welcome")

        println("[DEBUG_LOG] Resetting onboarding test complete")
    }

    /**
     * Test starting a tutorial.
     */
    @Test
    fun testStartingTutorial() = runBlocking {
        // Set up a collector for tutorial started events
        val events = EventSystem.events("tutorial_started")

        // Start a tutorial
        val result = onboardingManager.startTutorial("basic_mechanism")

        // Verify that starting the tutorial was successful
        assertTrue(result, "Starting tutorial should be successful")

        // Collect the event with a timeout
        val receivedEvent =
            withTimeout(1000) {
                events.first()
            }

        // Verify the event
        assertTrue(receivedEvent is OnboardingEvent.TutorialStarted, "Received event should be a TutorialStarted event")
        assertEquals(
            "basic_mechanism",
            (receivedEvent as OnboardingEvent.TutorialStarted).tutorial.id,
            "Tutorial ID should be basic_mechanism",
        )

        println("[DEBUG_LOG] Starting tutorial test complete")
    }

    /**
     * Test completing a tutorial.
     */
    @Test
    fun testCompletingTutorial() = runBlocking {
        // Set up a collector for tutorial completed events
        val events = EventSystem.events("tutorial_completed")

        // Start and complete a tutorial
        onboardingManager.startTutorial("basic_mechanism")
        val result = onboardingManager.completeTutorial("basic_mechanism")

        // Verify that completing the tutorial was successful
        assertTrue(result, "Completing tutorial should be successful")

        // Collect the event with a timeout
        val receivedEvent =
            withTimeout(1000) {
                events.first()
            }

        // Verify the event
        assertTrue(receivedEvent is OnboardingEvent.TutorialCompleted, "Received event should be a TutorialCompleted event")
        assertEquals(
            "basic_mechanism",
            (receivedEvent as OnboardingEvent.TutorialCompleted).tutorial.id,
            "Tutorial ID should be basic_mechanism",
        )

        println("[DEBUG_LOG] Completing tutorial test complete")
    }

    /**
     * Test loading a sample project.
     */
    @Test
    fun testLoadingSampleProject() = runBlocking {
        // Set up a collector for sample project loaded events
        val events = EventSystem.events("sample_project_loaded")

        // Load a sample project
        val parameters = onboardingManager.loadSampleProject("basic_mechanism")

        // Verify that parameters were returned
        assertNotNull(parameters, "Parameters should not be null")
        assertTrue(parameters!!.isNotEmpty(), "Parameters should not be empty")

        // Collect the event with a timeout
        val receivedEvent =
            withTimeout(1000) {
                events.first()
            }

        // Verify the event
        assertTrue(receivedEvent is OnboardingEvent.SampleProjectLoaded, "Received event should be a SampleProjectLoaded event")
        assertEquals(
            "basic_mechanism",
            (receivedEvent as OnboardingEvent.SampleProjectLoaded).project.id,
            "Project ID should be basic_mechanism",
        )

        println("[DEBUG_LOG] Loading sample project test complete")
    }

    /**
     * Test getting onboarding steps, tutorials, and sample projects.
     */
    @Test
    fun testGettingOnboardingResources() {
        // Get all onboarding steps
        val steps = onboardingManager.getAllOnboardingSteps()

        // Verify that steps were returned
        assertFalse(steps.isEmpty(), "Steps should not be empty")

        // Get all tutorials
        val tutorials = onboardingManager.getAllTutorials()

        // Verify that tutorials were returned
        assertFalse(tutorials.isEmpty(), "Tutorials should not be empty")

        // Get all sample projects
        val projects = onboardingManager.getAllSampleProjects()

        // Verify that projects were returned
        assertFalse(projects.isEmpty(), "Projects should not be empty")

        // Get tutorials for a step
        val stepTutorials = onboardingManager.getTutorialsForStep("parameters")

        // Verify that tutorials were returned
        assertFalse(stepTutorials.isEmpty(), "Step tutorials should not be empty")

        // Get sample projects for a tutorial
        val tutorialProjects = onboardingManager.getSampleProjectsForTutorial("basic_mechanism")

        // Verify that projects were returned
        assertFalse(tutorialProjects.isEmpty(), "Tutorial projects should not be empty")

        println("[DEBUG_LOG] Getting onboarding resources test complete")
    }
}
