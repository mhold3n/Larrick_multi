package com.campro.v5.ux

import androidx.compose.animation.*
import androidx.compose.animation.core.*
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.scale
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.campro.v5.debug.DebugIconButton
import com.campro.v5.debug.DebugOutlinedButton
import kotlinx.coroutines.delay
import org.slf4j.LoggerFactory

/**
 * User experience enhancement components for the optimization pipeline.
 *
 * Provides loading states, progress feedback, animations, and interactive
 * elements to improve user experience.
 */
object UserExperienceEnhancer {

    private val logger = LoggerFactory.getLogger(UserExperienceEnhancer::class.java)

    /**
     * Loading state data class.
     */
    data class LoadingState(
        val isLoading: Boolean = false,
        val message: String = "Loading...",
        val progress: Float = 0f,
        val indeterminate: Boolean = true,
        val canCancel: Boolean = false,
    )

    /**
     * Progress feedback data class.
     */
    data class ProgressFeedback(
        val currentStep: String = "",
        val totalSteps: Int = 0,
        val currentStepIndex: Int = 0,
        val estimatedTimeRemaining: String = "",
        val details: String = "",
    )
}

/**
 * Enhanced loading indicator with animations and feedback.
 */
@Composable
fun EnhancedLoadingIndicator(
    loadingState: UserExperienceEnhancer.LoadingState,
    onCancel: (() -> Unit)? = null,
    modifier: Modifier = Modifier,
) {
    AnimatedVisibility(
        visible = loadingState.isLoading,
        enter = fadeIn() + slideInVertically(),
        exit = fadeOut() + slideOutVertically(),
        modifier = modifier,
    ) {
        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.primaryContainer,
            ),
            elevation = CardDefaults.cardElevation(defaultElevation = 8.dp),
        ) {
            Column(
                modifier = Modifier.padding(24.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.spacedBy(16.dp),
            ) {
                // Animated loading icon
                AnimatedLoadingIcon(
                    modifier = Modifier.size(48.dp),
                )

                // Loading message
                Text(
                    text = loadingState.message,
                    style = MaterialTheme.typography.titleMedium,
                    color = MaterialTheme.colorScheme.onPrimaryContainer,
                    fontWeight = FontWeight.Medium,
                )

                // Progress indicator
                if (!loadingState.indeterminate) {
                    Column(
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.spacedBy(8.dp),
                    ) {
                        LinearProgressIndicator(
                            progress = loadingState.progress,
                            modifier = Modifier.fillMaxWidth(),
                            color = MaterialTheme.colorScheme.primary,
                            trackColor = MaterialTheme.colorScheme.primary.copy(alpha = 0.3f),
                        )

                        Text(
                            text = "${(loadingState.progress * 100).toInt()}%",
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onPrimaryContainer,
                        )
                    }
                } else {
                    LinearProgressIndicator(
                        modifier = Modifier.fillMaxWidth(),
                        color = MaterialTheme.colorScheme.primary,
                        trackColor = MaterialTheme.colorScheme.primary.copy(alpha = 0.3f),
                    )
                }

                // Cancel button
                if (loadingState.canCancel && onCancel != null) {
                    DebugOutlinedButton(
                        buttonId = "enhanced-loading-cancel",
                        onClick = onCancel,
                    ) {
                        Icon(
                            imageVector = Icons.Default.Cancel,
                            contentDescription = null,
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Text("Cancel")
                    }
                }
            }
        }
    }
}

/**
 * Animated loading icon with rotation and scaling.
 */
@Composable
private fun AnimatedLoadingIcon(modifier: Modifier = Modifier) {
    val infiniteTransition = rememberInfiniteTransition(label = "loading")

    val rotation by infiniteTransition.animateFloat(
        initialValue = 0f,
        targetValue = 360f,
        animationSpec = infiniteRepeatable(
            animation = tween(2000, easing = LinearEasing),
            repeatMode = RepeatMode.Restart,
        ),
        label = "rotation",
    )

    val scale by infiniteTransition.animateFloat(
        initialValue = 0.8f,
        targetValue = 1.2f,
        animationSpec = infiniteRepeatable(
            animation = tween(1000, easing = FastOutSlowInEasing),
            repeatMode = RepeatMode.Reverse,
        ),
        label = "scale",
    )

    Icon(
        imageVector = Icons.Default.AutoAwesome,
        contentDescription = "Loading",
        modifier = modifier
            .scale(scale)
            .alpha(0.8f),
        tint = MaterialTheme.colorScheme.primary,
    )
}

/**
 * Step-by-step progress indicator.
 */
@Composable
fun StepProgressIndicator(progress: UserExperienceEnhancer.ProgressFeedback, modifier: Modifier = Modifier) {
    Card(
        modifier = modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant,
        ),
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            // Progress header
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text(
                    text = "Progress",
                    style = MaterialTheme.typography.titleSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )

                Text(
                    text = "${progress.currentStepIndex + 1} of ${progress.totalSteps}",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }

            // Step indicators
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceEvenly,
            ) {
                repeat(progress.totalSteps) { index ->
                    StepIndicator(
                        isActive = index == progress.currentStepIndex,
                        isCompleted = index < progress.currentStepIndex,
                        stepNumber = index + 1,
                    )
                }
            }

            // Current step details
            Column(
                verticalArrangement = Arrangement.spacedBy(4.dp),
            ) {
                Text(
                    text = progress.currentStep,
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    fontWeight = FontWeight.Medium,
                )

                if (progress.details.isNotEmpty()) {
                    Text(
                        text = progress.details,
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.7f),
                    )
                }

                if (progress.estimatedTimeRemaining.isNotEmpty()) {
                    Text(
                        text = "Estimated time remaining: ${progress.estimatedTimeRemaining}",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.7f),
                    )
                }
            }
        }
    }
}

/**
 * Individual step indicator component.
 */
@Composable
private fun StepIndicator(isActive: Boolean, isCompleted: Boolean, stepNumber: Int, modifier: Modifier = Modifier) {
    val animatedColor by animateColorAsState(
        targetValue = when {
            isCompleted -> MaterialTheme.colorScheme.primary
            isActive -> MaterialTheme.colorScheme.primary
            else -> MaterialTheme.colorScheme.outline
        },
        animationSpec = tween(300),
        label = "stepColor",
    )

    val animatedScale by animateFloatAsState(
        targetValue = if (isActive) 1.2f else 1f,
        animationSpec = tween(300),
        label = "stepScale",
    )

    Box(
        modifier = modifier
            .size(32.dp)
            .scale(animatedScale),
        contentAlignment = Alignment.Center,
    ) {
        Surface(
            shape = RoundedCornerShape(16.dp),
            color = animatedColor.copy(alpha = 0.1f),
            modifier = Modifier.fillMaxSize(),
        ) {
            Box(
                contentAlignment = Alignment.Center,
            ) {
                if (isCompleted) {
                    Icon(
                        imageVector = Icons.Default.Check,
                        contentDescription = "Completed",
                        tint = animatedColor,
                        modifier = Modifier.size(20.dp),
                    )
                } else {
                    Text(
                        text = stepNumber.toString(),
                        style = MaterialTheme.typography.bodySmall,
                        color = animatedColor,
                        fontWeight = FontWeight.Bold,
                    )
                }
            }
        }
    }
}

/**
 * Success feedback with celebration animation.
 */
@Composable
fun SuccessFeedback(message: String, isVisible: Boolean, onDismiss: () -> Unit, modifier: Modifier = Modifier) {
    AnimatedVisibility(
        visible = isVisible,
        enter = scaleIn() + fadeIn(),
        exit = scaleOut() + fadeOut(),
        modifier = modifier,
    ) {
        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(
                containerColor = Color(0xFF4CAF50).copy(alpha = 0.1f),
            ),
            elevation = CardDefaults.cardElevation(defaultElevation = 4.dp),
        ) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(16.dp),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(12.dp),
                ) {
                    AnimatedIcon(
                        icon = Icons.Default.CheckCircle,
                        color = Color(0xFF4CAF50),
                        modifier = Modifier.size(24.dp),
                    )

                    Text(
                        text = message,
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurface,
                        fontWeight = FontWeight.Medium,
                    )
                }

                DebugIconButton(buttonId = "success-dismiss", onClick = onDismiss) {
                    Icon(
                        imageVector = Icons.Default.Close,
                        contentDescription = "Dismiss",
                        tint = MaterialTheme.colorScheme.onSurface,
                    )
                }
            }
        }
    }
}

/**
 * Animated icon with bounce effect.
 */
@Composable
private fun AnimatedIcon(icon: ImageVector, color: Color, modifier: Modifier = Modifier) {
    val infiniteTransition = rememberInfiniteTransition(label = "iconAnimation")

    val scale by infiniteTransition.animateFloat(
        initialValue = 1f,
        targetValue = 1.1f,
        animationSpec = infiniteRepeatable(
            animation = tween(1000, easing = FastOutSlowInEasing),
            repeatMode = RepeatMode.Reverse,
        ),
        label = "scale",
    )

    Icon(
        imageVector = icon,
        contentDescription = null,
        modifier = modifier.scale(scale),
        tint = color,
    )
}

/**
 * Interactive tooltip with hover effects.
 */
@Composable
fun InteractiveTooltip(text: String, content: @Composable () -> Unit, modifier: Modifier = Modifier) {
    var showTooltip by remember { mutableStateOf(false) }

    Box(
        modifier = modifier,
    ) {
        Box(
            modifier = Modifier,
        ) {
            content()
        }

        AnimatedVisibility(
            visible = showTooltip,
            enter = fadeIn() + slideInVertically(),
            exit = fadeOut() + slideOutVertically(),
        ) {
            Card(
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.inverseSurface,
                ),
                elevation = CardDefaults.cardElevation(defaultElevation = 8.dp),
            ) {
                Text(
                    text = text,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.inverseOnSurface,
                    modifier = Modifier.padding(8.dp),
                )
            }
        }
    }
}

/**
 * Auto-dismissing notification.
 */
@Composable
fun AutoDismissNotification(
    message: String,
    isVisible: Boolean,
    duration: Long = 3000L,
    onDismiss: () -> Unit,
    modifier: Modifier = Modifier,
) {
    LaunchedEffect(isVisible) {
        if (isVisible) {
            delay(duration)
            onDismiss()
        }
    }

    AnimatedVisibility(
        visible = isVisible,
        enter = slideInVertically() + fadeIn(),
        exit = slideOutVertically() + fadeOut(),
        modifier = modifier,
    ) {
        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.primaryContainer,
            ),
        ) {
            Text(
                text = message,
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onPrimaryContainer,
                modifier = Modifier.padding(16.dp),
            )
        }
    }
}

/**
 * Responsive layout helper.
 */
@Composable
fun ResponsiveLayout(compactContent: @Composable () -> Unit, expandedContent: @Composable () -> Unit, modifier: Modifier = Modifier) {
    val windowSizeClass = rememberWindowSizeClass()

    when (windowSizeClass) {
        WindowSizeClass.Compact -> compactContent()
        WindowSizeClass.Medium -> expandedContent()
        WindowSizeClass.Expanded -> expandedContent()
    }
}

/**
 * Window size class enumeration.
 */
enum class WindowSizeClass {
    Compact,
    Medium,
    Expanded,
}

/**
 * Remember window size class based on available width.
 */
@Composable
private fun rememberWindowSizeClass(): WindowSizeClass {
    val density = LocalDensity.current
    var windowSizeClass by remember { mutableStateOf(WindowSizeClass.Compact) }

    // This would be implemented with actual window size detection
    // For now, return a default value
    return WindowSizeClass.Medium
}
