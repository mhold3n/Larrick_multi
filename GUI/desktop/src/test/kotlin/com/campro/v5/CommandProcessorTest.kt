package com.campro.v5

import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import java.io.BufferedWriter
import java.io.ByteArrayOutputStream
import java.io.OutputStreamWriter
import java.io.PipedInputStream
import java.io.PipedOutputStream
import java.io.PrintStream

/**
 * Tests for the CommandProcessor.
 *
 * This class tests the functionality of the CommandProcessor, including command parsing,
 * command routing, and error handling.
 */
class CommandProcessorTest {
    private lateinit var commandProcessor: CommandProcessor
    private val originalOut = System.out
    private val outContent = ByteArrayOutputStream()
    private lateinit var pipedOut: PipedOutputStream
    private lateinit var pipedIn: PipedInputStream

    @BeforeEach
    fun setUp() {
        // Redirect System.out to capture event logging
        System.setOut(PrintStream(outContent))

        // Set up piped streams for command input
        pipedOut = PipedOutputStream()
        pipedIn = PipedInputStream(pipedOut)
        System.setIn(pipedIn)

        // Create command processor
        commandProcessor = CommandProcessor()

        println("[DEBUG_LOG] Test setup complete")
    }

    @AfterEach
    fun tearDown() {
        // Reset System.out and System.in
        System.setOut(originalOut)
        System.setIn(System.`in`)

        // Stop command processor
        commandProcessor.stop()

        // Close streams
        pipedOut.close()
        pipedIn.close()

        println("[DEBUG_LOG] Test teardown complete")
    }

    /**
     * Test that the command processor can process a click command.
     */
    @Test
    fun testClickCommand() = runBlocking {
        // Start command processor in a separate coroutine
        val job =
            launch {
                commandProcessor.start()
            }

        // Wait for command processor to start
        delay(100)

        // Send a click command
        val command = """COMMAND:{"command":"click","params":{"component":"TestButton"}}"""
        sendCommand(command)

        // Wait for command to be processed
        delay(100)

        // Get the output
        val output = outContent.toString()

        // Verify that the command was processed
        assertTrue(output.contains("EVENT:"), "Output should contain EVENT: prefix")
        assertTrue(output.contains("\"type\":\"command_executed\""), "Output should contain event type")
        assertTrue(output.contains("\"command\":\"click\""), "Output should contain command type")
        assertTrue(output.contains("\"component\":\"TestButton\""), "Output should contain component name")

        // Stop command processor
        commandProcessor.stop()
        job.cancel()

        println("[DEBUG_LOG] Click command test complete")
    }

    /**
     * Test that the command processor can process a set_value command.
     */
    @Test
    fun testSetValueCommand() = runBlocking {
        // Start command processor in a separate coroutine
        val job =
            launch {
                commandProcessor.start()
            }

        // Wait for command processor to start
        delay(100)

        // Send a set_value command
        val command = """COMMAND:{"command":"set_value","params":{"component":"TestInput","value":"TestValue"}}"""
        sendCommand(command)

        // Wait for command to be processed
        delay(100)

        // Get the output
        val output = outContent.toString()

        // Verify that the command was processed
        assertTrue(output.contains("EVENT:"), "Output should contain EVENT: prefix")
        assertTrue(output.contains("\"type\":\"command_executed\""), "Output should contain event type")
        assertTrue(output.contains("\"command\":\"set_value\""), "Output should contain command type")
        assertTrue(output.contains("\"component\":\"TestInput\""), "Output should contain component name")
        assertTrue(output.contains("\"value\":\"TestValue\""), "Output should contain value")

        // Stop command processor
        commandProcessor.stop()
        job.cancel()

        println("[DEBUG_LOG] Set value command test complete")
    }

    /**
     * Test that the command processor can process a select_tab command.
     */
    @Test
    fun testSelectTabCommand() = runBlocking {
        // Start command processor in a separate coroutine
        val job =
            launch {
                commandProcessor.start()
            }

        // Wait for command processor to start
        delay(100)

        // Send a select_tab command
        val command = """COMMAND:{"command":"select_tab","params":{"component":"TestTabs","value":"Tab1"}}"""
        sendCommand(command)

        // Wait for command to be processed
        delay(100)

        // Get the output
        val output = outContent.toString()

        // Verify that the command was processed
        assertTrue(output.contains("EVENT:"), "Output should contain EVENT: prefix")
        assertTrue(output.contains("\"type\":\"command_executed\""), "Output should contain event type")
        assertTrue(output.contains("\"command\":\"select_tab\""), "Output should contain command type")
        assertTrue(output.contains("\"component\":\"TestTabs\""), "Output should contain component name")
        assertTrue(output.contains("\"value\":\"Tab1\""), "Output should contain value")

        // Stop command processor
        commandProcessor.stop()
        job.cancel()

        println("[DEBUG_LOG] Select tab command test complete")
    }

    /**
     * Test that the command processor can process a gesture command.
     */
    @Test
    fun testGestureCommand() = runBlocking {
        // Start command processor in a separate coroutine
        val job =
            launch {
                commandProcessor.start()
            }

        // Wait for command processor to start
        delay(100)

        // Send a gesture command
        val command =
            "COMMAND:{\"command\":\"gesture\",\"params\":{\"component\":\"TestCanvas\"," +
                "\"action\":\"pan\",\"offset_x\":\"10\",\"offset_y\":\"20\"}}"
        sendCommand(command)

        // Wait for command to be processed
        delay(100)

        // Get the output
        val output = outContent.toString()

        // Verify that the command was processed
        assertTrue(output.contains("EVENT:"), "Output should contain EVENT: prefix")
        assertTrue(output.contains("\"type\":\"command_executed\""), "Output should contain event type")
        assertTrue(output.contains("\"command\":\"gesture\""), "Output should contain command type")
        assertTrue(output.contains("\"component\":\"TestCanvas\""), "Output should contain component name")
        assertTrue(output.contains("\"action\":\"pan\""), "Output should contain action")
        assertTrue(output.contains("\"offset_x\":\"10\""), "Output should contain offset_x")
        assertTrue(output.contains("\"offset_y\":\"20\""), "Output should contain offset_y")

        // Stop command processor
        commandProcessor.stop()
        job.cancel()

        println("[DEBUG_LOG] Gesture command test complete")
    }

    /**
     * Test that the command processor can process a get_state command.
     */
    @Test
    fun testGetStateCommand() = runBlocking {
        // Start command processor in a separate coroutine
        val job =
            launch {
                commandProcessor.start()
            }

        // Wait for command processor to start
        delay(100)

        // Send a get_state command
        val command = """COMMAND:{"command":"get_state","params":{"component":"TestComponent"}}"""
        sendCommand(command)

        // Wait for command to be processed
        delay(100)

        // Get the output
        val output = outContent.toString()

        // Verify that the command was processed
        assertTrue(output.contains("EVENT:"), "Output should contain EVENT: prefix")
        assertTrue(output.contains("\"type\":\"command_executed\""), "Output should contain event type")
        assertTrue(output.contains("\"command\":\"get_state\""), "Output should contain command type")
        assertTrue(output.contains("\"component\":\"TestComponent\""), "Output should contain component name")

        // Stop command processor
        commandProcessor.stop()
        job.cancel()

        println("[DEBUG_LOG] Get state command test complete")
    }

    /**
     * Test that the command processor can process a reset command.
     */
    @Test
    fun testResetCommand() = runBlocking {
        // Start command processor in a separate coroutine
        val job =
            launch {
                commandProcessor.start()
            }

        // Wait for command processor to start
        delay(100)

        // Send a reset command
        val command = """COMMAND:{"command":"reset","params":{"component":"TestComponent"}}"""
        sendCommand(command)

        // Wait for command to be processed
        delay(100)

        // Get the output
        val output = outContent.toString()

        // Verify that the command was processed
        assertTrue(output.contains("EVENT:"), "Output should contain EVENT: prefix")
        assertTrue(output.contains("\"type\":\"command_executed\""), "Output should contain event type")
        assertTrue(output.contains("\"command\":\"reset\""), "Output should contain command type")
        assertTrue(output.contains("\"component\":\"TestComponent\""), "Output should contain component name")

        // Stop command processor
        commandProcessor.stop()
        job.cancel()

        println("[DEBUG_LOG] Reset command test complete")
    }

    /**
     * Test that the command processor can process an export command.
     */
    @Test
    fun testExportCommand() = runBlocking {
        // Start command processor in a separate coroutine
        val job =
            launch {
                commandProcessor.start()
            }

        // Wait for command processor to start
        delay(100)

        // Send an export command
        val command = """COMMAND:{"command":"export","params":{"component":"TestComponent","file_path":"test.json","format":"json"}}"""
        sendCommand(command)

        // Wait for command to be processed
        delay(100)

        // Get the output
        val output = outContent.toString()

        // Verify that the command was processed
        assertTrue(output.contains("EVENT:"), "Output should contain EVENT: prefix")
        assertTrue(output.contains("\"type\":\"command_executed\""), "Output should contain event type")
        assertTrue(output.contains("\"command\":\"export\""), "Output should contain command type")
        assertTrue(output.contains("\"component\":\"TestComponent\""), "Output should contain component name")
        assertTrue(output.contains("\"file_path\":\"test.json\""), "Output should contain file_path")
        assertTrue(output.contains("\"format\":\"json\""), "Output should contain format")

        // Stop command processor
        commandProcessor.stop()
        job.cancel()

        println("[DEBUG_LOG] Export command test complete")
    }

    /**
     * Test that the command processor can process an import command.
     */
    @Test
    fun testImportCommand() = runBlocking {
        // Start command processor in a separate coroutine
        val job =
            launch {
                commandProcessor.start()
            }

        // Wait for command processor to start
        delay(100)

        // Send an import command
        val command = """COMMAND:{"command":"import","params":{"component":"TestComponent","file_path":"test.json"}}"""
        sendCommand(command)

        // Wait for command to be processed
        delay(100)

        // Get the output
        val output = outContent.toString()

        // Verify that the command was processed
        assertTrue(output.contains("EVENT:"), "Output should contain EVENT: prefix")
        assertTrue(output.contains("\"type\":\"command_executed\""), "Output should contain event type")
        assertTrue(output.contains("\"command\":\"import\""), "Output should contain command type")
        assertTrue(output.contains("\"component\":\"TestComponent\""), "Output should contain component name")
        assertTrue(output.contains("\"file_path\":\"test.json\""), "Output should contain file_path")

        // Stop command processor
        commandProcessor.stop()
        job.cancel()

        println("[DEBUG_LOG] Import command test complete")
    }

    /**
     * Test that the command processor can process a generate command.
     */
    @Test
    fun testGenerateCommand() = runBlocking {
        // Start command processor in a separate coroutine
        val job =
            launch {
                commandProcessor.start()
            }

        // Wait for command processor to start
        delay(100)

        // Send a generate command
        val command = """COMMAND:{"command":"generate","params":{"component":"TestComponent","type":"report"}}"""
        sendCommand(command)

        // Wait for command to be processed
        delay(100)

        // Get the output
        val output = outContent.toString()

        // Verify that the command was processed
        assertTrue(output.contains("EVENT:"), "Output should contain EVENT: prefix")
        assertTrue(output.contains("\"type\":\"command_executed\""), "Output should contain event type")
        assertTrue(output.contains("\"command\":\"generate\""), "Output should contain command type")
        assertTrue(output.contains("\"component\":\"TestComponent\""), "Output should contain component name")
        assertTrue(output.contains("\"type\":\"report\""), "Output should contain type")

        // Stop command processor
        commandProcessor.stop()
        job.cancel()

        println("[DEBUG_LOG] Generate command test complete")
    }

    /**
     * Test that the command processor handles unknown commands.
     */
    @Test
    fun testUnknownCommand() = runBlocking {
        // Start command processor in a separate coroutine
        val job =
            launch {
                commandProcessor.start()
            }

        // Wait for command processor to start
        delay(100)

        // Send an unknown command
        val command = """COMMAND:{"command":"unknown","params":{"component":"TestComponent"}}"""
        sendCommand(command)

        // Wait for command to be processed
        delay(100)

        // Get the output
        val output = outContent.toString()

        // Verify that an error was reported
        assertTrue(output.contains("EVENT:"), "Output should contain EVENT: prefix")
        assertTrue(output.contains("\"type\":\"error\""), "Output should contain error type")
        assertTrue(output.contains("\"message\":\"Unknown command: unknown\""), "Output should contain error message")

        // Stop command processor
        commandProcessor.stop()
        job.cancel()

        println("[DEBUG_LOG] Unknown command test complete")
    }

    /**
     * Test that the command processor handles invalid JSON.
     */
    @Test
    fun testInvalidJson() = runBlocking {
        // Start command processor in a separate coroutine
        val job =
            launch {
                commandProcessor.start()
            }

        // Wait for command processor to start
        delay(100)

        // Send invalid JSON
        val command = """COMMAND:{"command":"click","params":{"component":"TestButton"} INVALID"""
        sendCommand(command)

        // Wait for command to be processed
        delay(100)

        // Get the output
        val output = outContent.toString()

        // Verify that an error was reported
        assertTrue(output.contains("EVENT:"), "Output should contain EVENT: prefix")
        assertTrue(output.contains("\"type\":\"error\""), "Output should contain error type")
        assertTrue(output.contains("\"message\":\"Error processing command:"), "Output should contain error message")

        // Stop command processor
        commandProcessor.stop()
        job.cancel()

        println("[DEBUG_LOG] Invalid JSON test complete")
    }

    /**
     * Helper method to send a command to the command processor.
     */
    private fun sendCommand(command: String) {
        val writer = BufferedWriter(OutputStreamWriter(pipedOut))
        writer.write(command)
        writer.newLine()
        writer.flush()
    }
}
