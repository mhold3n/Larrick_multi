package com.campro.v5.window

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import com.campro.v5.layout.LayoutManager

/**
 * Minimal placeholder for multi-window controller. For now, it renders a stub
 * while wiring remains to be implemented.
 */
@Composable
fun WindowsController(testingMode: Boolean, layoutManager: LayoutManager) {
    Column(Modifier.fillMaxSize()) {
        Text("WindowsController placeholder (multi-window not yet implemented)")
    }
}
