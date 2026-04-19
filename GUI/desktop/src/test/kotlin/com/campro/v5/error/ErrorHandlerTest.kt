package com.campro.v5.error

import org.junit.jupiter.api.*
import org.junit.jupiter.api.Assertions.*

/**
 * Tests for ErrorHandler.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class ErrorHandlerTest {

    private lateinit var errorHandler: ErrorHandler

    @BeforeEach
    fun setup() {
        errorHandler = ErrorHandler()
    }

    @Test
    fun `test report error`() {
        // Given
        val message = "Test error message"
        val severity = ErrorHandler.ErrorSeverity.ERROR
        val context = "Test context"

        // When
        errorHandler.reportError(
            message = message,
            severity = severity,
            context = context,
        )

        // Then
        assertEquals(1, errorHandler.errors.size)
        assertEquals(message, errorHandler.errors[0].message)
        assertEquals(severity, errorHandler.errors[0].severity)
        assertEquals(context, errorHandler.errors[0].context)
        assertNotNull(errorHandler.currentError)
    }

    @Test
    fun `test report error with recovery action`() {
        // Given
        val message = "Test error with recovery"
        val recoveryAction = ErrorHandler.RecoveryAction(
            label = "Retry",
            action = { /* Test action */ },
            canRetry = true,
        )

        // When
        errorHandler.reportError(
            message = message,
            severity = ErrorHandler.ErrorSeverity.ERROR,
            recoveryAction = recoveryAction,
        )

        // Then
        assertEquals(1, errorHandler.errors.size)
        assertEquals(message, errorHandler.errors[0].message)
        assertNotNull(errorHandler.errors[0].recoveryAction)
        assertEquals("Retry", errorHandler.errors[0].recoveryAction?.label)
        assertTrue(errorHandler.errors[0].recoveryAction?.canRetry == true)
    }

    @Test
    fun `test clear current error`() {
        // Given
        errorHandler.reportError("Test error", ErrorHandler.ErrorSeverity.ERROR)
        assertNotNull(errorHandler.currentError)

        // When
        errorHandler.clearCurrentError()

        // Then
        assertNull(errorHandler.currentError)
        assertEquals(1, errorHandler.errors.size) // Should still be in history
    }

    @Test
    fun `test clear all errors`() {
        // Given
        errorHandler.reportError("Error 1", ErrorHandler.ErrorSeverity.ERROR)
        errorHandler.reportError("Error 2", ErrorHandler.ErrorSeverity.WARNING)
        assertEquals(2, errorHandler.errors.size)

        // When
        errorHandler.clearAllErrors()

        // Then
        assertEquals(0, errorHandler.errors.size)
        assertNull(errorHandler.currentError)
    }

    @Test
    fun `test get errors by severity`() {
        // Given
        errorHandler.reportError("Error 1", ErrorHandler.ErrorSeverity.ERROR)
        errorHandler.reportError("Error 2", ErrorHandler.ErrorSeverity.WARNING)
        errorHandler.reportError("Error 3", ErrorHandler.ErrorSeverity.ERROR)

        // When
        val errorErrors = errorHandler.getErrorsBySeverity(ErrorHandler.ErrorSeverity.ERROR)
        val warningErrors = errorHandler.getErrorsBySeverity(ErrorHandler.ErrorSeverity.WARNING)

        // Then
        assertEquals(2, errorErrors.size)
        assertEquals(1, warningErrors.size)
        assertTrue(errorErrors.all { it.severity == ErrorHandler.ErrorSeverity.ERROR })
        assertTrue(warningErrors.all { it.severity == ErrorHandler.ErrorSeverity.WARNING })
    }

    @Test
    fun `test has critical errors`() {
        // Given
        errorHandler.reportError("Warning", ErrorHandler.ErrorSeverity.WARNING)
        assertFalse(errorHandler.hasCriticalErrors())

        // When
        errorHandler.reportError("Critical", ErrorHandler.ErrorSeverity.CRITICAL)

        // Then
        assertTrue(errorHandler.hasCriticalErrors())
    }

    @Test
    fun `test get error summary`() {
        // Given
        errorHandler.reportError("Info", ErrorHandler.ErrorSeverity.INFO)
        errorHandler.reportError("Warning", ErrorHandler.ErrorSeverity.WARNING)
        errorHandler.reportError("Error", ErrorHandler.ErrorSeverity.ERROR)
        errorHandler.reportError("Critical", ErrorHandler.ErrorSeverity.CRITICAL)

        // When
        val summary = errorHandler.getErrorSummary()

        // Then
        assertEquals(4, summary.totalErrors)
        assertEquals(1, summary.criticalErrors)
        assertEquals(1, summary.errorErrors)
        assertEquals(1, summary.warningErrors)
        assertEquals(1, summary.infoErrors)
        assertTrue(summary.hasErrors)
        assertTrue(summary.hasCriticalIssues)
    }

    @Test
    fun `test error info data class`() {
        // Given
        val errorInfo = ErrorHandler.ErrorInfo(
            message = "Test message",
            severity = ErrorHandler.ErrorSeverity.ERROR,
            context = "Test context",
            technicalDetails = "Technical details",
        )

        // Then
        assertEquals("Test message", errorInfo.message)
        assertEquals(ErrorHandler.ErrorSeverity.ERROR, errorInfo.severity)
        assertEquals("Test context", errorInfo.context)
        assertEquals("Technical details", errorInfo.technicalDetails)
        assertNotNull(errorInfo.timestamp)
    }

    @Test
    fun `test recovery action data class`() {
        // Given
        var actionExecuted = false
        val recoveryAction = ErrorHandler.RecoveryAction(
            label = "Test Action",
            action = { actionExecuted = true },
            canRetry = true,
        )

        // When
        recoveryAction.action()

        // Then
        assertEquals("Test Action", recoveryAction.label)
        assertTrue(recoveryAction.canRetry)
        assertTrue(actionExecuted)
    }
}
