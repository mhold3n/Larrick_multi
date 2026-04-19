package com.campro.v5.animation

import com.campro.v5.data.litvin.LitvinTablesDTO
import com.campro.v5.data.litvin.PlanetDTO
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import java.lang.reflect.Field

class RenderingPathSelectionTest {
    @BeforeEach
    fun setUp() {
        MotionLawEngine.resetInstance()
    }

    private fun setPrivateField(target: Any, name: String, value: Any?) {
        val f: Field = target.javaClass.getDeclaredField(name)
        f.isAccessible = true
        f.set(target, value)
    }

    @Test
    fun `litvin active when tables present`() {
        val engine = MotionLawEngine.getInstance()

        // Enable Litvin constraints feature flag
        com.campro.v5.config.FeatureFlags
            .setFlag("collocation.litvin_constraints_enabled", true)

        val planet =
            PlanetDTO(
                centerX = listOf(1.0),
                centerY = listOf(2.0),
                spinPsiDeg = listOf(0.0),
                journalX = listOf(3.0),
                journalY = listOf(4.0),
                pistonS = listOf(5.0),
            )
        val tables = LitvinTablesDTO(alphaDeg = listOf(0.0), planets = listOf(planet))
        val curves =
            com.campro.v5.data.litvin.PitchCurvesDTO(
                pitchRing = listOf(1.0, 2.0, 3.0),
                pitchPlanet = listOf(1.0, 2.0, 3.0),
            )

        setPrivateField(engine, "litvinTables", tables)
        setPrivateField(engine, "litvinCurves", curves)

        assertTrue(engine.isLitvinActive())

        // Clean up
        com.campro.v5.config.FeatureFlags
            .clearFlag("collocation.litvin_constraints_enabled")
    }

    @Test
    fun `when litvin selected but tables absent, engine should not fallback-render`() {
        val engine = MotionLawEngine.getInstance()
        // Ensure no tables present
        setPrivateField(engine, "litvinTables", null)
        // Calling getComponentPositions currently falls back; desired behavior is to avoid legacy visuals.
        val positions = engine.getComponentPositions(0.0)
        // Expect a neutral 'no data' posture (all zeros) rather than SHM-like or cam-at-origin proxy coupling rod=piston
        assertEquals(0f, positions.pistonPosition.y)
        assertEquals(0f, positions.rodPosition.y)
        assertEquals(0f, positions.camPosition.x)
        assertEquals(0f, positions.camPosition.y)
    }
}
