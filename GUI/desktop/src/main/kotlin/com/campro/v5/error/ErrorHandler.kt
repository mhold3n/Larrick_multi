package com.campro.v5.error

import androidx.compose.foundation.layout.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.campro.v5.debug.DebugButton
import com.campro.v5.debug.DebugIconButton
import com.campro.v5.debug.DebugOutlinedButton
import org.slf4j.LoggerFactory
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter

/**
 * Comprehensive error handling system for the optimization pipeline.
 *
 * Provides user-friendly error display, recovery actions, and error logging
 * with Material3 design patterns.
 */
class ErrorHandler {

    private val logger = LoggerFactory.getLogger(ErrorHandler::class.java)

    /**
     * Error severity levels.
     */
    enum class ErrorSeverity {
        INFO,
        WARNING,
        ERROR,
        CRITICAL,
    }

    /**
     * Error data class with context.
     */
    data class ErrorInfo(
        val message: String,
        val severity: ErrorSeverity,
        val timestamp: String = LocalDateTime.now().format(DateTimeFormatter.ISO_LOCAL_DATE_TIME),
        val context: String? = null,
        val recoveryAction: RecoveryAction? = null,
        val technicalDetails: String? = null,
    )

    /**
     * Recovery action data class.
     */
    data class RecoveryAction(val label: String, val action: () -> Unit, val canRetry: Boolean = false)

    /**
     * Error state management.
     */
    private val _errors = mutableStateListOf<ErrorInfo>()
    val errors: List<ErrorInfo> get() = _errors.toList()

    private val _currentError = mutableStateOf<ErrorInfo?>(null)
    val currentError: ErrorInfo? get() = _currentError.value

    /**
     * Report an error.
     */
    fun reportError(
        message: String,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        context: String? = null,
        recoveryAction: RecoveryAction? = null,
        technicalDetails: String? = null,
        throwable: Throwable? = null,
    ) {
        val errorInfo = ErrorInfo(
            message = message,
            severity = severity,
            context = context,
            recoveryAction = recoveryAction,
            technicalDetails = technicalDetails,
        )

        _errors.add(errorInfo)
        _currentError.value = errorInfo

        // Log the error
        when (severity) {
            ErrorSeverity.INFO -> logger.info("$context: $message", throwable)
            ErrorSeverity.WARNING -> logger.warn("$context: $message", throwable)
            ErrorSeverity.ERROR -> logger.error("$context: $message", throwable)
            ErrorSeverity.CRITICAL -> logger.error("CRITICAL: $context: $message", throwable)
        }
    }

    /**
     * Clear current error.
     */
    fun clearCurrentError() {
        _currentError.value = null
    }

    /**
     * Clear all errors.
     */
    fun clearAllErrors() {
        _errors.clear()
        _currentError.value = null
    }

    /**
     * Get errors by severity.
     */
    fun getErrorsBySeverity(severity: ErrorSeverity): List<ErrorInfo> = _errors.filter { it.severity == severity }

    /**
     * Check if there are any critical errors.
     */
    fun hasCriticalErrors(): Boolean = _errors.any { it.severity == ErrorSeverity.CRITICAL }

    /**
     * Get error summary.
     */
    fun getErrorSummary(): ErrorSummary = ErrorSummary(
        totalErrors = _errors.size,
        criticalErrors = _errors.count { it.severity == ErrorSeverity.CRITICAL },
        errorErrors = _errors.count { it.severity == ErrorSeverity.ERROR },
        warningErrors = _errors.count { it.severity == ErrorSeverity.WARNING },
        infoErrors = _errors.count { it.severity == ErrorSeverity.INFO },
    )
}

/**
 * Error summary data class.
 */
data class ErrorSummary(val totalErrors: Int, val criticalErrors: Int, val errorErrors: Int, val warningErrors: Int, val infoErrors: Int) {
    val hasErrors: Boolean
        get() = totalErrors > 0

    val hasCriticalIssues: Boolean
        get() = criticalErrors > 0 || errorErrors > 0
}

/**
 * Error display composable with Material3 design.
 */
@Composable
fun ErrorDisplay(error: ErrorHandler.ErrorInfo, onDismiss: () -> Unit, modifier: Modifier = Modifier) {
    Card(
        modifier = modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = when (error.severity) {
                ErrorHandler.ErrorSeverity.INFO -> MaterialTheme.colorScheme.primaryContainer
                ErrorHandler.ErrorSeverity.WARNING -> MaterialTheme.colorScheme.tertiaryContainer
                ErrorHandler.ErrorSeverity.ERROR -> MaterialTheme.colorScheme.errorContainer
                ErrorHandler.ErrorSeverity.CRITICAL -> MaterialTheme.colorScheme.errorContainer
            },
        ),
        elevation = CardDefaults.cardElevation(defaultElevation = 4.dp),
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            // Error header
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    Icon(
                        imageVector = getErrorIcon(error.severity),
                        contentDescription = null,
                        tint = getErrorColor(error.severity),
                        modifier = Modifier.size(24.dp),
                    )

                    Text(
                        text = getErrorTitle(error.severity),
                        style = MaterialTheme.typography.titleMedium,
                        color = getErrorColor(error.severity),
                        fontWeight = FontWeight.Bold,
                    )
                }

                DebugIconButton(buttonId = "error-dismiss", onClick = onDismiss) {
                    Icon(
                        imageVector = Icons.Default.Close,
                        contentDescription = "Dismiss error",
                    )
                }
            }

            // Error message
            Text(
                text = error.message,
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onErrorContainer,
            )

            // Context information
            if (error.context != null) {
                Text(
                    text = "Context: ${error.context}",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onErrorContainer.copy(alpha = 0.7f),
                )
            }

            // Timestamp
            Text(
                text = "Time: ${error.timestamp}",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onErrorContainer.copy(alpha = 0.7f),
            )

            // Recovery action
            if (error.recoveryAction != null) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    DebugButton(
                        buttonId = "error-recovery",
                        onClick = {
                            error.recoveryAction.action()
                            onDismiss()
                        },
                        colors = ButtonDefaults.buttonColors(
                            containerColor = getErrorColor(error.severity),
                        ),
                    ) {
                        Icon(
                            imageVector = if (error.recoveryAction.canRetry) Icons.Default.Refresh else Icons.Default.Check,
                            contentDescription = null,
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Text(error.recoveryAction.label)
                    }

                    if (error.recoveryAction.canRetry) {
                        DebugOutlinedButton(buttonId = "error-cancel", onClick = onDismiss) {
                            Text("Cancel")
                        }
                    }
                }
            }

            // Technical details (expandable)
            if (error.technicalDetails != null) {
                var showDetails by remember { mutableStateOf(false) }

                DebugOutlinedButton(
                    buttonId = "error-tech-details-toggle",
                    onClick = { showDetails = !showDetails },
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    Icon(
                        imageVector = if (showDetails) Icons.Default.ExpandLess else Icons.Default.ExpandMore,
                        contentDescription = null,
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("Technical Details")
                }

                if (showDetails) {
                    Card(
                        colors = CardDefaults.cardColors(
                            containerColor = MaterialTheme.colorScheme.surfaceVariant,
                        ),
                    ) {
                        Text(
                            text = error.technicalDetails,
                            style = MaterialTheme.typography.bodySmall,
                            modifier = Modifier.padding(12.dp),
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                }
            }
        }
    }
}

/**
 * Error snackbar for non-critical errors.
 */
@Composable
fun ErrorSnackbar(error: ErrorHandler.ErrorInfo, onDismiss: () -> Unit, modifier: Modifier = Modifier) {
    val snackbarHostState = remember { SnackbarHostState() }

    LaunchedEffect(error) {
        snackbarHostState.showSnackbar(
            message = error.message,
            actionLabel = error.recoveryAction?.label,
            duration = when (error.severity) {
                ErrorHandler.ErrorSeverity.INFO -> SnackbarDuration.Short
                ErrorHandler.ErrorSeverity.WARNING -> SnackbarDuration.Long
                ErrorHandler.ErrorSeverity.ERROR -> SnackbarDuration.Indefinite
                ErrorHandler.ErrorSeverity.CRITICAL -> SnackbarDuration.Indefinite
            },
        )
    }

    SnackbarHost(
        hostState = snackbarHostState,
        modifier = modifier,
        snackbar = { data ->
            Snackbar(
                snackbarData = data,
            )
        },
    )
}

/**
 * Error summary card.
 */
@Composable
fun ErrorSummaryCard(summary: ErrorSummary, onViewAll: () -> Unit, modifier: Modifier = Modifier) {
    Card(
        modifier = modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = if (summary.hasCriticalIssues) {
                MaterialTheme.colorScheme.errorContainer
            } else {
                MaterialTheme.colorScheme.surfaceVariant
            },
        ),
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text(
                    text = "Error Summary",
                    style = MaterialTheme.typography.titleMedium,
                    color = if (summary.hasCriticalIssues) {
                        MaterialTheme.colorScheme.onErrorContainer
                    } else {
                        MaterialTheme.colorScheme.onSurfaceVariant
                    },
                )

                if (summary.hasErrors) {
                    DebugButton(
                        buttonId = "error-view-all",
                        onClick = onViewAll,
                        modifier = Modifier.height(32.dp),
                    ) {
                        Text("View All")
                    }
                }
            }

            if (summary.hasErrors) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                ) {
                    if (summary.criticalErrors > 0) {
                        ErrorCountChip("Critical", summary.criticalErrors, Color(0xFFD32F2F))
                    }
                    if (summary.errorErrors > 0) {
                        ErrorCountChip("Errors", summary.errorErrors, Color(0xFFF44336))
                    }
                    if (summary.warningErrors > 0) {
                        ErrorCountChip("Warnings", summary.warningErrors, Color(0xFFFF9800))
                    }
                    if (summary.infoErrors > 0) {
                        ErrorCountChip("Info", summary.infoErrors, Color(0xFF2196F3))
                    }
                }
            } else {
                Text(
                    text = "No errors",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
    }
}

/**
 * Error count chip component.
 */
@Composable
private fun ErrorCountChip(label: String, count: Int, color: Color) {
    Surface(
        shape = MaterialTheme.shapes.small,
        color = color.copy(alpha = 0.1f),
    ) {
        Row(
            modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(4.dp),
        ) {
            Text(
                text = label,
                style = MaterialTheme.typography.bodySmall,
                color = color,
            )
            Surface(
                shape = MaterialTheme.shapes.small,
                color = color,
            ) {
                Text(
                    text = count.toString(),
                    style = MaterialTheme.typography.bodySmall,
                    color = Color.White,
                    modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp),
                )
            }
        }
    }
}

/**
 * Helper functions for error display.
 */
private fun getErrorIcon(severity: ErrorHandler.ErrorSeverity): ImageVector = when (severity) {
    ErrorHandler.ErrorSeverity.INFO -> Icons.Default.Info
    ErrorHandler.ErrorSeverity.WARNING -> Icons.Default.Warning
    ErrorHandler.ErrorSeverity.ERROR -> Icons.Default.Error
    ErrorHandler.ErrorSeverity.CRITICAL -> Icons.Default.ErrorOutline
}

@Composable
private fun getErrorColor(severity: ErrorHandler.ErrorSeverity): Color = when (severity) {
    ErrorHandler.ErrorSeverity.INFO -> MaterialTheme.colorScheme.primary
    ErrorHandler.ErrorSeverity.WARNING -> MaterialTheme.colorScheme.tertiary
    ErrorHandler.ErrorSeverity.ERROR -> MaterialTheme.colorScheme.error
    ErrorHandler.ErrorSeverity.CRITICAL -> Color(0xFFD32F2F)
}

private fun getErrorTitle(severity: ErrorHandler.ErrorSeverity): String = when (severity) {
    ErrorHandler.ErrorSeverity.INFO -> "Information"
    ErrorHandler.ErrorSeverity.WARNING -> "Warning"
    ErrorHandler.ErrorSeverity.ERROR -> "Error"
    ErrorHandler.ErrorSeverity.CRITICAL -> "Critical Error"
}
