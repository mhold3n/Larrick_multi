package com.campro.v5.ui

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.unit.dp
import com.campro.v5.debug.DebugOutlinedButton

/**
 * Common UI components used across the CamProV5 application.
 *
 * This file consolidates frequently used components to eliminate duplication
 * and ensure consistent UI patterns throughout the application.
 */

// EmptyStateWidget is defined in EmptyStateWidget.kt; reuse that implementation

/**
 * A standardized parameter input field with consistent styling and validation.
 *
 * @param label The field label
 * @param value The current value
 * @param unit The unit of measurement (optional)
 * @param onValueChange Callback when value changes
 * @param modifier Modifier for customizing the component
 * @param isError Whether the field has a validation error
 * @param errorMessage Optional error message to display
 */
@Composable
fun ParameterField(
    label: String,
    value: Double,
    unit: String = "",
    onValueChange: (Double) -> Unit,
    modifier: Modifier = Modifier,
    isError: Boolean = false,
    errorMessage: String? = null,
) {
    Column(
        modifier = modifier,
        verticalArrangement = Arrangement.spacedBy(4.dp),
    ) {
        OutlinedTextField(
            value = value.toString(),
            onValueChange = {
                val newValue = it.toDoubleOrNull() ?: value
                onValueChange(newValue)
            },
            label = {
                Text("$label ${if (unit.isNotEmpty()) "($unit)" else ""}".trim())
            },
            modifier = Modifier.fillMaxWidth(),
            singleLine = true,
            isError = isError,
            supportingText = errorMessage?.let {
                { Text(it, color = MaterialTheme.colorScheme.error) }
            },
        )
    }
}

/**
 * A standardized parameter input field for string values.
 */
@Composable
fun StringParameterField(
    label: String,
    value: String,
    onValueChange: (String) -> Unit,
    modifier: Modifier = Modifier,
    isError: Boolean = false,
    errorMessage: String? = null,
    placeholder: String? = null,
) {
    Column(
        modifier = modifier,
        verticalArrangement = Arrangement.spacedBy(4.dp),
    ) {
        OutlinedTextField(
            value = value,
            onValueChange = onValueChange,
            label = { Text(label) },
            placeholder = placeholder?.let { { Text(it) } },
            modifier = Modifier.fillMaxWidth(),
            singleLine = true,
            isError = isError,
            supportingText = errorMessage?.let {
                { Text(it, color = MaterialTheme.colorScheme.error) }
            },
        )
    }
}

/**
 * A standardized section header for parameter groups.
 */
@Composable
fun ParameterSectionHeader(title: String, subtitle: String? = null, modifier: Modifier = Modifier) {
    Column(
        modifier = modifier,
        verticalArrangement = Arrangement.spacedBy(4.dp),
    ) {
        Text(
            text = title,
            style = MaterialTheme.typography.titleMedium,
            color = MaterialTheme.colorScheme.onSurface,
        )
        subtitle?.let {
            Text(
                text = it,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

/**
 * A standardized loading indicator with optional message.
 */
@Composable
fun LoadingIndicator(message: String = "Loading...", modifier: Modifier = Modifier) {
    Box(
        modifier = modifier.fillMaxSize(),
        contentAlignment = Alignment.Center,
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            CircularProgressIndicator()
            Text(
                text = message,
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

/**
 * A standardized error display component.
 */
@Composable
fun ErrorDisplay(error: Throwable, onRetry: (() -> Unit)? = null, onDismiss: (() -> Unit)? = null, modifier: Modifier = Modifier) {
    Card(
        modifier = modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.errorContainer,
        ),
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Icon(
                    imageVector = Icons.Default.Error,
                    contentDescription = null,
                    tint = MaterialTheme.colorScheme.error,
                )
                Text(
                    text = "Error",
                    style = MaterialTheme.typography.titleSmall,
                    color = MaterialTheme.colorScheme.onErrorContainer,
                )
            }

            Text(
                text = error.message ?: "An unknown error occurred",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onErrorContainer,
            )

            if (onRetry != null || onDismiss != null) {
                Row(
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    onRetry?.let {
                        DebugOutlinedButton(
                            buttonId = "error-retry",
                            onClick = it,
                            colors = ButtonDefaults.outlinedButtonColors(
                                contentColor = MaterialTheme.colorScheme.onErrorContainer,
                            ),
                        ) {
                            Icon(
                                imageVector = Icons.Default.Refresh,
                                contentDescription = null,
                                modifier = Modifier.size(16.dp),
                            )
                            Spacer(modifier = Modifier.width(4.dp))
                            Text("Retry")
                        }
                    }

                    onDismiss?.let {
                        DebugOutlinedButton(
                            buttonId = "error-dismiss",
                            onClick = it,
                            colors = ButtonDefaults.outlinedButtonColors(
                                contentColor = MaterialTheme.colorScheme.onErrorContainer,
                            ),
                        ) {
                            Icon(
                                imageVector = Icons.Default.Close,
                                contentDescription = null,
                                modifier = Modifier.size(16.dp),
                            )
                            Spacer(modifier = Modifier.width(4.dp))
                            Text("Dismiss")
                        }
                    }
                }
            }
        }
    }
}

/**
 * A standardized status indicator for different states.
 */
@Composable
fun StatusIndicator(status: StatusType, message: String, modifier: Modifier = Modifier) {
    Card(
        modifier = modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = when (status) {
                StatusType.IDLE -> MaterialTheme.colorScheme.surfaceVariant
                StatusType.RUNNING -> MaterialTheme.colorScheme.primaryContainer
                StatusType.SUCCESS -> MaterialTheme.colorScheme.tertiaryContainer
                StatusType.ERROR -> MaterialTheme.colorScheme.errorContainer
            },
        ),
    ) {
        Row(
            modifier = Modifier.padding(12.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            Icon(
                imageVector = when (status) {
                    StatusType.IDLE -> Icons.Default.Pause
                    StatusType.RUNNING -> Icons.Default.PlayArrow
                    StatusType.SUCCESS -> Icons.Default.CheckCircle
                    StatusType.ERROR -> Icons.Default.Error
                },
                contentDescription = null,
                tint = when (status) {
                    StatusType.IDLE -> MaterialTheme.colorScheme.onSurfaceVariant
                    StatusType.RUNNING -> MaterialTheme.colorScheme.primary
                    StatusType.SUCCESS -> MaterialTheme.colorScheme.tertiary
                    StatusType.ERROR -> MaterialTheme.colorScheme.error
                },
            )

            Text(
                text = message,
                style = MaterialTheme.typography.titleSmall,
                color = when (status) {
                    StatusType.IDLE -> MaterialTheme.colorScheme.onSurfaceVariant
                    StatusType.RUNNING -> MaterialTheme.colorScheme.onPrimaryContainer
                    StatusType.SUCCESS -> MaterialTheme.colorScheme.onTertiaryContainer
                    StatusType.ERROR -> MaterialTheme.colorScheme.onErrorContainer
                },
            )
        }
    }
}

/**
 * Status types for the status indicator.
 */
enum class StatusType {
    IDLE,
    RUNNING,
    SUCCESS,
    ERROR,
}

/**
 * A standardized action button with consistent styling.
 */
@Composable
fun ActionButton(
    text: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    icon: ImageVector? = null,
    enabled: Boolean = true,
    variant: ButtonVariant = ButtonVariant.PRIMARY,
) {
    when (variant) {
        ButtonVariant.PRIMARY -> {
            Button(
                onClick = onClick,
                modifier = modifier,
                enabled = enabled,
            ) {
                icon?.let {
                    Icon(
                        imageVector = it,
                        contentDescription = null,
                        modifier = Modifier.size(18.dp),
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                }
                Text(text)
            }
        }
        ButtonVariant.OUTLINED -> {
            OutlinedButton(
                onClick = onClick,
                modifier = modifier,
                enabled = enabled,
            ) {
                icon?.let {
                    Icon(
                        imageVector = it,
                        contentDescription = null,
                        modifier = Modifier.size(18.dp),
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                }
                Text(text)
            }
        }
        ButtonVariant.TEXT -> {
            TextButton(
                onClick = onClick,
                modifier = modifier,
                enabled = enabled,
            ) {
                icon?.let {
                    Icon(
                        imageVector = it,
                        contentDescription = null,
                        modifier = Modifier.size(18.dp),
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                }
                Text(text)
            }
        }
    }
}

/**
 * Button variants for consistent styling.
 */
enum class ButtonVariant {
    PRIMARY,
    OUTLINED,
    TEXT,
}

/**
 * A standardized scrollable container for parameter forms.
 */
@Composable
fun ScrollableParameterContainer(content: @Composable ColumnScope.() -> Unit, modifier: Modifier = Modifier) {
    Column(
        modifier = modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        content()
    }
}
