package com.campro.v5.ui

import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier

/** Placeholder panel to satisfy references; can be expanded with real plots later. */
@Composable
fun StaticProfilesPanel(parameters: Map<String, String>, testingMode: Boolean = false, modifier: Modifier = Modifier) {
    Column(modifier.fillMaxWidth()) {
        Text("Static Profiles (placeholder)")
        var count = 0
        for ((k, v) in parameters) {
            if (count >= 3) break
            Text("$k = $v")
            count++
        }
    }
}
