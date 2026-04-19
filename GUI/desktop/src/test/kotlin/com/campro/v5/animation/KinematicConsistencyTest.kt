package com.campro.v5.animation

import com.campro.v5.data.litvin.LitvinTablesDTO
import com.campro.v5.data.litvin.PlanetDTO
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test
import java.lang.reflect.Field

class KinematicConsistencyTest {
    private fun setPrivate(target: Any, name: String, value: Any?) {
        val f: Field = target.javaClass.getDeclaredField(name)
        f.isAccessible = true
        f.set(target, value)
    }

    @Test
    fun `getComponentPositions aligns with Litvin frame state`() {
        val engine = MotionLawEngine.getInstance()
        val planet =
            PlanetDTO(
                centerX = listOf(10.0),
                centerY = listOf(20.0),
                spinPsiDeg = listOf(0.0),
                journalX = listOf(30.0),
                journalY = listOf(40.0),
                pistonS = listOf(50.0),
            )
        val tables = LitvinTablesDTO(alphaDeg = listOf(0.0), planets = listOf(planet))
        setPrivate(engine, "litvinTables", tables)

        val pos = engine.getComponentPositions(0.0)
        assertEquals(50.0f, pos.pistonPosition.y)
        assertEquals(30.0f, pos.rodPosition.x)
        assertEquals(40.0f, pos.rodPosition.y)
        assertEquals(10.0f, pos.camPosition.x)
        assertEquals(20.0f, pos.camPosition.y)
    }
}
