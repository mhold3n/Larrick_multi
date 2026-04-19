package com.campro.v5.animation

import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.drawscope.DrawScope

/**
 * Minimal placeholder renderer for Litvin visuals.
 * If Litvin mode is active but data is present, this would render curves.
 * For now, it is a no-op to keep the desktop module compiling.
 */
object LitvinRenderer {
    fun drawFrame(
        drawScope: DrawScope,
        canvasWidth: Float,
        canvasHeight: Float,
        scaleUser: Float,
        offset: Offset,
        angleDeg: Float,
        parameters: Map<String, String>,
        motion: MotionLawEngine,
    ) {
        // No-op placeholder. Full implementation can be added later.
    }
}
