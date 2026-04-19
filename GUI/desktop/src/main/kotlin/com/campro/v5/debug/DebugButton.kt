package com.campro.v5.debug

import androidx.compose.material3.Button
import androidx.compose.material3.ButtonColors
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.ExtendedFloatingActionButton
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.FloatingActionButton
import androidx.compose.material3.IconButton
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import org.slf4j.LoggerFactory

private val buttonLogger = LoggerFactory.getLogger("ButtonDebug")

@Composable
fun DebugButton(
    buttonId: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    colors: ButtonColors = ButtonDefaults.buttonColors(),
    content: @Composable () -> Unit,
) {
    Button(
        onClick = {
            if (DebugManager.settings.buttonDebug) {
                buttonLogger.info("[button-debug] click id={}", buttonId)
            }
            if (DebugManager.settings.interactionLogging) {
                buttonLogger.info("[interaction] button={} event=click", buttonId)
            }
            onClick()
        },
        modifier = modifier,
        colors = colors,
    ) {
        content()
    }
}

@Composable
fun DebugOutlinedButton(buttonId: String, onClick: () -> Unit, modifier: Modifier = Modifier, content: @Composable () -> Unit) {
    OutlinedButton(
        onClick = {
            if (DebugManager.settings.buttonDebug) {
                buttonLogger.info("[button-debug] click id={} type=outlined", buttonId)
            }
            if (DebugManager.settings.interactionLogging) {
                buttonLogger.info("[interaction] button={} event=click", buttonId)
            }
            onClick()
        },
        modifier = modifier,
    ) {
        content()
    }
}

@Composable
fun DebugTextButton(buttonId: String, onClick: () -> Unit, modifier: Modifier = Modifier, content: @Composable () -> Unit) {
    TextButton(
        onClick = {
            if (DebugManager.settings.buttonDebug) {
                buttonLogger.info("[button-debug] click id={} type=text", buttonId)
            }
            if (DebugManager.settings.interactionLogging) {
                buttonLogger.info("[interaction] button={} event=click", buttonId)
            }
            onClick()
        },
        modifier = modifier,
    ) { content() }
}

@Composable
fun DebugIconButton(buttonId: String, onClick: () -> Unit, modifier: Modifier = Modifier, content: @Composable () -> Unit) {
    IconButton(
        onClick = {
            if (DebugManager.settings.buttonDebug) {
                buttonLogger.info("[button-debug] click id={} type=icon", buttonId)
            }
            if (DebugManager.settings.interactionLogging) {
                buttonLogger.info("[interaction] button={} event=click", buttonId)
            }
            onClick()
        },
        modifier = modifier,
    ) { content() }
}

@Composable
fun DebugFab(buttonId: String, onClick: () -> Unit, modifier: Modifier = Modifier, content: @Composable () -> Unit) {
    FloatingActionButton(onClick = {
        if (DebugManager.settings.buttonDebug) {
            buttonLogger.info("[button-debug] click id={} type=fab", buttonId)
        }
        if (DebugManager.settings.interactionLogging) {
            buttonLogger.info("[interaction] button={} event=click", buttonId)
        }
        onClick()
    }, modifier = modifier) { content() }
}

@Composable
fun DebugExtendedFab(
    buttonId: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    icon: @Composable () -> Unit = {},
    text: @Composable () -> Unit,
) {
    ExtendedFloatingActionButton(
        onClick = {
            if (DebugManager.settings.buttonDebug) {
                buttonLogger.info("[button-debug] click id={} type=efab", buttonId)
            }
            if (DebugManager.settings.interactionLogging) {
                buttonLogger.info("[interaction] button={} event=click", buttonId)
            }
            onClick()
        },
        modifier = modifier,
        icon = icon,
        text = text,
    )
}

@Composable
fun DebugFilledTonalButton(buttonId: String, onClick: () -> Unit, modifier: Modifier = Modifier, content: @Composable () -> Unit) {
    FilledTonalButton(onClick = {
        if (DebugManager.settings.buttonDebug) {
            buttonLogger.info("[button-debug] click id={} type=filled-tonal", buttonId)
        }
        if (DebugManager.settings.interactionLogging) {
            buttonLogger.info("[interaction] button={} event=click", buttonId)
        }
        onClick()
    }, modifier = modifier) { content() }
}
