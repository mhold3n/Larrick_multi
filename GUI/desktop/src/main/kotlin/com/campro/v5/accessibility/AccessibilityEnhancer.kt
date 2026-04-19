package com.campro.v5.accessibility

import androidx.compose.foundation.layout.*
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.semantics.*
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import org.slf4j.LoggerFactory

/**
 * Accessibility enhancement utilities for the optimization pipeline.
 *
 * Provides accessibility features including screen reader support,
 * keyboard navigation, high contrast mode, and focus management.
 */
object AccessibilityEnhancer {

    private val logger = LoggerFactory.getLogger(AccessibilityEnhancer::class.java)

    /**
     * Accessibility settings data class.
     */
    data class AccessibilitySettings(
        val highContrast: Boolean = false,
        val largeText: Boolean = false,
        val screenReaderSupport: Boolean = true,
        val keyboardNavigation: Boolean = true,
        val focusIndicators: Boolean = true,
        val reducedMotion: Boolean = false,
    )

    /**
     * Current accessibility settings.
     */
    private val _accessibilitySettings = mutableStateOf(AccessibilitySettings())
    val accessibilitySettings: AccessibilitySettings get() = _accessibilitySettings.value

    /**
     * Update accessibility settings.
     */
    fun updateSettings(settings: AccessibilitySettings) {
        _accessibilitySettings.value = settings
        logger.info("Accessibility settings updated: $settings")
    }

    /**
     * Get high contrast colors.
     */
    fun getHighContrastColors(): HighContrastColors = HighContrastColors(
        primary = Color(0xFF000000),
        secondary = Color(0xFF0000FF),
        error = Color(0xFFFF0000),
        background = Color(0xFFFFFFFF),
        surface = Color(0xFFFFFFFF),
        onPrimary = Color(0xFFFFFFFF),
        onSecondary = Color(0xFFFFFFFF),
        onError = Color(0xFFFFFFFF),
        onBackground = Color(0xFF000000),
        onSurface = Color(0xFF000000),
    )

    /**
     * Get large text scale factor.
     */
    fun getLargeTextScale(): Float = if (_accessibilitySettings.value.largeText) 1.2f else 1.0f
}

/**
 * High contrast color scheme.
 */
data class HighContrastColors(
    val primary: Color,
    val secondary: Color,
    val error: Color,
    val background: Color,
    val surface: Color,
    val onPrimary: Color,
    val onSecondary: Color,
    val onError: Color,
    val onBackground: Color,
    val onSurface: Color,
)

/**
 * Accessible button with enhanced semantics.
 */
@Composable
fun AccessibleButton(
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    enabled: Boolean = true,
    content: @Composable RowScope.() -> Unit,
) {
    val settings = AccessibilityEnhancer.accessibilitySettings

    Button(
        onClick = onClick,
        modifier = modifier
            .semantics {
                // Enhanced semantics for screen readers
                role = Role.Button
                onClick(label = "Button action", action = {
                    onClick()
                    true
                })
                if (!enabled) {
                    disabled()
                }
            },
        enabled = enabled,
        colors = if (settings.highContrast) {
            ButtonDefaults.buttonColors(
                containerColor = AccessibilityEnhancer.getHighContrastColors().primary,
                contentColor = AccessibilityEnhancer.getHighContrastColors().onPrimary,
            )
        } else {
            ButtonDefaults.buttonColors()
        },
    ) {
        content()
    }
}

/**
 * Accessible text field with enhanced semantics.
 */
@Composable
fun AccessibleTextField(
    value: String,
    onValueChange: (String) -> Unit,
    modifier: Modifier = Modifier,
    label: String? = null,
    placeholder: String? = null,
    isError: Boolean = false,
    errorMessage: String? = null,
    singleLine: Boolean = true,
    enabled: Boolean = true,
) {
    val settings = AccessibilityEnhancer.accessibilitySettings

    OutlinedTextField(
        value = value,
        onValueChange = onValueChange,
        modifier = modifier
            .semantics {
                // Enhanced semantics for screen readers
                if (label != null) {
                    contentDescription = label
                }
                if (placeholder != null) {
                    contentDescription = placeholder
                }
                if (isError && errorMessage != null) {
                    error(errorMessage)
                }
                if (!enabled) {
                    disabled()
                }
            },
        label = if (label != null) {
            {
                Text(
                    text = label,
                    fontSize = if (settings.largeText) {
                        MaterialTheme.typography.bodyLarge.fontSize
                    } else {
                        MaterialTheme.typography.bodyMedium.fontSize
                    },
                )
            }
        } else {
            null
        },
        placeholder = if (placeholder != null) {
            {
                Text(
                    text = placeholder,
                    fontSize = if (settings.largeText) {
                        MaterialTheme.typography.bodyLarge.fontSize
                    } else {
                        MaterialTheme.typography.bodyMedium.fontSize
                    },
                )
            }
        } else {
            null
        },
        isError = isError,
        supportingText = if (isError && errorMessage != null) {
            {
                Text(
                    text = errorMessage,
                    fontSize = if (settings.largeText) {
                        MaterialTheme.typography.bodyLarge.fontSize
                    } else {
                        MaterialTheme.typography.bodyMedium.fontSize
                    },
                )
            }
        } else {
            null
        },
        singleLine = singleLine,
        enabled = enabled,
        colors = if (settings.highContrast) {
            OutlinedTextFieldDefaults.colors(
                focusedBorderColor = AccessibilityEnhancer.getHighContrastColors().primary,
                unfocusedBorderColor = AccessibilityEnhancer.getHighContrastColors().onSurface,
                errorBorderColor = AccessibilityEnhancer.getHighContrastColors().error,
                focusedTextColor = AccessibilityEnhancer.getHighContrastColors().onSurface,
                unfocusedTextColor = AccessibilityEnhancer.getHighContrastColors().onSurface,
                errorTextColor = AccessibilityEnhancer.getHighContrastColors().error,
            )
        } else {
            OutlinedTextFieldDefaults.colors()
        },
    )
}

/**
 * Accessible card with enhanced semantics.
 */
@Composable
fun AccessibleCard(
    title: String? = null,
    contentText: String? = null,
    modifier: Modifier = Modifier,
    onClick: (() -> Unit)? = null,
    content: @Composable () -> Unit,
) {
    val settings = AccessibilityEnhancer.accessibilitySettings

    Card(
        modifier = modifier
            .semantics {
                // Enhanced semantics for screen readers
                if (title != null) {
                    heading()
                    contentDescription = title
                }
                if (contentText != null) {
                    contentDescription = contentText
                }
                if (onClick != null) {
                    role = Role.Button
                    onClick(label = "Card action", action = {
                        onClick()
                        true
                    })
                }
            },
        onClick = onClick ?: {},
        colors = if (settings.highContrast) {
            CardDefaults.cardColors(
                containerColor = AccessibilityEnhancer.getHighContrastColors().surface,
                contentColor = AccessibilityEnhancer.getHighContrastColors().onSurface,
            )
        } else {
            CardDefaults.cardColors()
        },
    ) {
        content()
    }
}

/**
 * Accessible progress indicator with enhanced semantics.
 */
@Composable
fun AccessibleProgressIndicator(progress: Float, modifier: Modifier = Modifier, label: String = "Progress") {
    val settings = AccessibilityEnhancer.accessibilitySettings

    Column(
        modifier = modifier
            .semantics {
                // Enhanced semantics for screen readers
                progressBarRangeInfo = ProgressBarRangeInfo(
                    current = progress,
                    range = 0f..1f,
                    steps = 0,
                )
                contentDescription = "$label: ${(progress * 100).toInt()}%"
            },
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        Text(
            text = label,
            style = MaterialTheme.typography.bodyMedium,
            fontSize = if (settings.largeText) {
                MaterialTheme.typography.bodyLarge.fontSize
            } else {
                MaterialTheme.typography.bodyMedium.fontSize
            },
            fontWeight = FontWeight.Medium,
        )

        LinearProgressIndicator(
            progress = progress,
            modifier = Modifier.fillMaxWidth(),
            color = if (settings.highContrast) {
                AccessibilityEnhancer.getHighContrastColors().primary
            } else {
                MaterialTheme.colorScheme.primary
            },
            trackColor = if (settings.highContrast) {
                AccessibilityEnhancer.getHighContrastColors().primary.copy(alpha = 0.3f)
            } else {
                MaterialTheme.colorScheme.primary.copy(alpha = 0.3f)
            },
        )

        Text(
            text = "${(progress * 100).toInt()}%",
            style = MaterialTheme.typography.bodySmall,
            fontSize = if (settings.largeText) {
                MaterialTheme.typography.bodyMedium.fontSize
            } else {
                MaterialTheme.typography.bodySmall.fontSize
            },
            color = if (settings.highContrast) {
                AccessibilityEnhancer.getHighContrastColors().onSurface
            } else {
                MaterialTheme.colorScheme.onSurface
            },
        )
    }
}

/**
 * Accessibility settings panel.
 */
@Composable
fun AccessibilitySettingsPanel(onSettingsChanged: (AccessibilityEnhancer.AccessibilitySettings) -> Unit, modifier: Modifier = Modifier) {
    var settings by remember { mutableStateOf(AccessibilityEnhancer.accessibilitySettings) }

    Card(
        modifier = modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant,
        ),
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            Text(
                text = "Accessibility Settings",
                style = MaterialTheme.typography.titleMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                fontWeight = FontWeight.Bold,
            )

            // High contrast toggle
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Column {
                    Text(
                        text = "High Contrast Mode",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                    Text(
                        text = "Increases contrast for better visibility",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.7f),
                    )
                }

                Switch(
                    checked = settings.highContrast,
                    onCheckedChange = {
                        settings = settings.copy(highContrast = it)
                        onSettingsChanged(settings)
                    },
                )
            }

            // Large text toggle
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Column {
                    Text(
                        text = "Large Text",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                    Text(
                        text = "Increases text size for better readability",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.7f),
                    )
                }

                Switch(
                    checked = settings.largeText,
                    onCheckedChange = {
                        settings = settings.copy(largeText = it)
                        onSettingsChanged(settings)
                    },
                )
            }

            // Screen reader support toggle
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Column {
                    Text(
                        text = "Screen Reader Support",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                    Text(
                        text = "Enhances compatibility with screen readers",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.7f),
                    )
                }

                Switch(
                    checked = settings.screenReaderSupport,
                    onCheckedChange = {
                        settings = settings.copy(screenReaderSupport = it)
                        onSettingsChanged(settings)
                    },
                )
            }

            // Keyboard navigation toggle
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Column {
                    Text(
                        text = "Keyboard Navigation",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                    Text(
                        text = "Enables full keyboard navigation support",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.7f),
                    )
                }

                Switch(
                    checked = settings.keyboardNavigation,
                    onCheckedChange = {
                        settings = settings.copy(keyboardNavigation = it)
                        onSettingsChanged(settings)
                    },
                )
            }

            // Focus indicators toggle
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Column {
                    Text(
                        text = "Focus Indicators",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                    Text(
                        text = "Shows visual focus indicators",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.7f),
                    )
                }

                Switch(
                    checked = settings.focusIndicators,
                    onCheckedChange = {
                        settings = settings.copy(focusIndicators = it)
                        onSettingsChanged(settings)
                    },
                )
            }

            // Reduced motion toggle
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Column {
                    Text(
                        text = "Reduced Motion",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                    Text(
                        text = "Reduces animations for motion sensitivity",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.7f),
                    )
                }

                Switch(
                    checked = settings.reducedMotion,
                    onCheckedChange = {
                        settings = settings.copy(reducedMotion = it)
                        onSettingsChanged(settings)
                    },
                )
            }
        }
    }
}

/**
 * Focus management utilities.
 */
object FocusManager {

    /**
     * Focus order for keyboard navigation.
     */
    data class FocusOrder(val elements: List<String> = emptyList(), val currentIndex: Int = 0)

    private val _focusOrder = mutableStateOf(FocusOrder())
    val focusOrder: FocusOrder get() = _focusOrder.value

    /**
     * Add element to focus order.
     */
    fun addToFocusOrder(elementId: String) {
        val currentElements = _focusOrder.value.elements.toMutableList()
        if (!currentElements.contains(elementId)) {
            currentElements.add(elementId)
            _focusOrder.value = _focusOrder.value.copy(elements = currentElements)
        }
    }

    /**
     * Remove element from focus order.
     */
    fun removeFromFocusOrder(elementId: String) {
        val currentElements = _focusOrder.value.elements.toMutableList()
        currentElements.remove(elementId)
        _focusOrder.value = _focusOrder.value.copy(elements = currentElements)
    }

    /**
     * Move to next focusable element.
     */
    fun moveToNext() {
        val currentIndex = _focusOrder.value.currentIndex
        val nextIndex = (currentIndex + 1) % _focusOrder.value.elements.size
        _focusOrder.value = _focusOrder.value.copy(currentIndex = nextIndex)
    }

    /**
     * Move to previous focusable element.
     */
    fun moveToPrevious() {
        val currentIndex = _focusOrder.value.currentIndex
        val previousIndex = if (currentIndex > 0) currentIndex - 1 else _focusOrder.value.elements.size - 1
        _focusOrder.value = _focusOrder.value.copy(currentIndex = previousIndex)
    }
}
