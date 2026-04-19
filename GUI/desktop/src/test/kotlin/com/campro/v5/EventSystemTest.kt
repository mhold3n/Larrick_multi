package com.campro.v5

import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import java.io.ByteArrayOutputStream
import java.io.PrintStream

/**
 * Tests for the EventSystem.
 *
 * This class tests the functionality of the EventSystem, including event emission,
 * event reception, event filtering, and event logging.
 */
class EventSystemTest {
    private val originalOut = System.out
    private val outContent = ByteArrayOutputStream()

    @BeforeEach
    fun setUp() {
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
     * Test that events can be emitted and received.
     */
    @Test
    fun testEventEmissionAndReception() = runBlocking {
        // Create a test event
        val testEvent = ClickEvent("TestComponent")

        // Set up a collector for click events
        val events = EventSystem.events("click")

        // Emit the event
        EventSystem.emit(testEvent)

        // Collect the event with a timeout
        val receivedEvent =
            withTimeout(1000) {
                events.first()
            }

        // Verify the event
        assertTrue(receivedEvent is ClickEvent, "Received event should be a ClickEvent")
        assertEquals("TestComponent", (receivedEvent as ClickEvent).component, "Component should match")

        println("[DEBUG_LOG] Event emission and reception test complete")
    }

    /**
     * Test that events are logged in testing mode.
     */
    @Test
    fun testEventLogging() {
        // Create a test event
        val testEvent = ClickEvent("TestComponent")

        // Emit the event
        EventSystem.emit(testEvent)

        // Wait for the event to be processed
        Thread.sleep(100)

        // Get the output
        val output = outContent.toString()

        // Verify that the event was logged
        assertTrue(output.contains("EVENT:"), "Output should contain EVENT: prefix")
        assertTrue(output.contains("\"type\":\"click\""), "Output should contain event type")
        assertTrue(output.contains("\"component\":\"TestComponent\""), "Output should contain component name")

        println("[DEBUG_LOG] Event logging test complete")
    }

    /**
     * Test that events can be filtered by type.
     */
    @Test
    fun testEventFiltering() = runBlocking {
        // Create test events of different types
        val clickEvent = ClickEvent("TestComponent")
        val valueChangedEvent = ValueChangedEvent("TestComponent", "NewValue")

        // Set up collectors for different event types
        val clickEvents = EventSystem.events("click")
        val valueChangedEvents = EventSystem.events("value_changed")

        // Emit the events
        EventSystem.emit(clickEvent)
        EventSystem.emit(valueChangedEvent)

        // Collect the events with a timeout
        val receivedClickEvent =
            withTimeout(1000) {
                clickEvents.first()
            }

        val receivedValueChangedEvent =
            withTimeout(1000) {
                valueChangedEvents.first()
            }

        // Verify the events
        assertTrue(receivedClickEvent is ClickEvent, "Received click event should be a ClickEvent")
        assertEquals("TestComponent", (receivedClickEvent as ClickEvent).component, "Component should match")

        assertTrue(receivedValueChangedEvent is ValueChangedEvent, "Received value changed event should be a ValueChangedEvent")
        assertEquals("TestComponent", (receivedValueChangedEvent as ValueChangedEvent).component, "Component should match")
        assertEquals("NewValue", receivedValueChangedEvent.value, "Value should match")

        println("[DEBUG_LOG] Event filtering test complete")
    }

    /**
     * Test the extension functions for emitting events.
     */
    @Test
    fun testEventExtensionFunctions() = runBlocking {
        // Set up collectors for different event types
        val clickEvents = EventSystem.events("click")
        val valueChangedEvents = EventSystem.events("value_changed")
        val tabSelectedEvents = EventSystem.events("tab_selected")
        val gestureEvents = EventSystem.events("gesture")
        val errorEvents = EventSystem.events("error")

        // Emit events using extension functions
        emitClick("TestComponent")
        emitValueChanged("TestComponent", "NewValue")
        emitTabSelected("TestComponent", "Tab1")
        emitGesture("TestComponent", "pan", mapOf("x" to 10, "y" to 20))
        emitError("Test error", "TestComponent")

        // Collect the events with a timeout
        val receivedClickEvent =
            withTimeout(1000) {
                clickEvents.first()
            }

        val receivedValueChangedEvent =
            withTimeout(1000) {
                valueChangedEvents.first()
            }

        val receivedTabSelectedEvent =
            withTimeout(1000) {
                tabSelectedEvents.first()
            }

        val receivedGestureEvent =
            withTimeout(1000) {
                gestureEvents.first()
            }

        val receivedErrorEvent =
            withTimeout(1000) {
                errorEvents.first()
            }

        // Verify the events
        assertTrue(receivedClickEvent is ClickEvent, "Received click event should be a ClickEvent")
        assertEquals("TestComponent", (receivedClickEvent as ClickEvent).component, "Component should match")

        assertTrue(receivedValueChangedEvent is ValueChangedEvent, "Received value changed event should be a ValueChangedEvent")
        assertEquals("TestComponent", (receivedValueChangedEvent as ValueChangedEvent).component, "Component should match")
        assertEquals("NewValue", receivedValueChangedEvent.value, "Value should match")

        assertTrue(receivedTabSelectedEvent is TabSelectedEvent, "Received tab selected event should be a TabSelectedEvent")
        assertEquals("TestComponent", (receivedTabSelectedEvent as TabSelectedEvent).component, "Component should match")
        assertEquals("Tab1", receivedTabSelectedEvent.tab, "Tab should match")

        assertTrue(receivedGestureEvent is GestureEvent, "Received gesture event should be a GestureEvent")
        assertEquals("TestComponent", (receivedGestureEvent as GestureEvent).component, "Component should match")
        assertEquals("pan", receivedGestureEvent.action, "Action should match")
        assertEquals(10, receivedGestureEvent.params["x"], "X parameter should match")
        assertEquals(20, receivedGestureEvent.params["y"], "Y parameter should match")

        assertTrue(receivedErrorEvent is ErrorEvent, "Received error event should be an ErrorEvent")
        assertEquals("Test error", (receivedErrorEvent as ErrorEvent).message, "Message should match")
        assertEquals("TestComponent", receivedErrorEvent.component, "Component should match")

        println("[DEBUG_LOG] Event extension functions test complete")
    }

    /**
     * Test that the event system can handle a large number of events.
     */
    @Test
    fun testEventPerformance() = runBlocking {
        // Number of events to emit
        val numEvents = 1000

        // Count received events
        var receivedCount = 0

        // Use a latch to wait for all events to be processed
        val latch = java.util.concurrent.CountDownLatch(numEvents)

        // Start collecting events with the optimized collector
        val job =
            EventSystem.collectEvents("click") { _ ->
                receivedCount++
                latch.countDown()
            }

        // Prepare batch of events
        val events =
            (1..numEvents).map { i ->
                ClickEvent("TestComponent$i")
            }

        // Emit events as a batch
        EventSystem.emitBatch(events)

        // Wait for all events to be processed with a longer timeout
        val allEventsProcessed = latch.await(10, java.util.concurrent.TimeUnit.SECONDS)

        // Cancel the collection job
        job.cancel()

        // Verify that all events were received
        assertTrue(allEventsProcessed, "Timed out waiting for events to be processed")
        assertEquals(numEvents, receivedCount, "All events should be received")

        println("[DEBUG_LOG] Event performance test complete")
    }
}
