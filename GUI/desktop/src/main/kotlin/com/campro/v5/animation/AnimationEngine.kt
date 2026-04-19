package com.campro.v5.animation

import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.drawscope.DrawScope
import com.google.gson.Gson
import com.google.gson.JsonObject
import java.io.File

/**
 * Interface for animation engines.
 * This interface defines the common functionality for different types of animations.
 */
interface AnimationEngine {
    /**
     * Get the current animation angle in degrees.
     */
    fun getCurrentAngle(): Float

    /**
     * Set the animation angle in degrees.
     *
     * @param angle The angle in degrees
     */
    fun setAngle(angle: Float)

    /**
     * Update the animation parameters.
     *
     * @param parameters The new parameters
     */
    fun updateParameters(parameters: Map<String, String>)

    /**
     * Draw the animation frame.
     *
     * @param drawScope The draw scope to use for drawing
     * @param canvasWidth The width of the canvas
     * @param canvasHeight The height of the canvas
     * @param scale The scale factor
     * @param offset The offset
     */
    fun drawFrame(drawScope: DrawScope, canvasWidth: Float, canvasHeight: Float, scale: Float, offset: Offset)

    /**
     * Get the animation parameters.
     *
     * @return The animation parameters
     */
    fun getParameters(): Map<String, String>

    /**
     * Get the animation type.
     *
     * @return The animation type
     */
    fun getType(): AnimationType

    /**
     * Clean up resources when the engine is no longer needed.
     */
    fun dispose()
}

/**
 * Types of animations.
 */
enum class AnimationType {
    COMPONENT_BASED,
    FEA_BASED,
}

/**
 * Factory for creating animation engines.
 */
object AnimationEngineFactory {
    /**
     * Create an animation engine of the specified type.
     *
     * @param type The type of animation engine to create
     * @param parameters The initial parameters for the animation
     * @return The created animation engine
     */
    fun createEngine(type: AnimationType, parameters: Map<String, String>): AnimationEngine = when (type) {
        AnimationType.COMPONENT_BASED -> ComponentBasedAnimationEngine(parameters)
        AnimationType.FEA_BASED -> FeaBasedAnimationEngine(parameters)
    }
}

/**
 * Animation engine for component-based animation.
 * This engine uses the motion law implementation from the Rust FEA engine.
 */
class ComponentBasedAnimationEngine(private var parameters: Map<String, String>) : AnimationEngine {
    private var currentAngle: Float = 0f

    // Use the global MotionLawEngine singleton that receives samples from parameter updates
    private val motionLawEngine: MotionLawEngine = MotionLawEngine.getInstance()
    private var lastDrawnAngle: Float = -1f
    private var cachedComponentPositions: ComponentPositions? = null

    // Expose MotionLawEngine for UI previews (Week 1 preview panel)
    fun motionEngine(): MotionLawEngine = motionLawEngine

    init {
        // Initialize the motion law engine with the parameters
        motionLawEngine.updateParameters(parameters)
    }

    override fun getCurrentAngle(): Float = currentAngle

    override fun setAngle(angle: Float) {
        currentAngle = angle % 360f
        // Clear cached positions when angle changes
        if (Math.abs(currentAngle - lastDrawnAngle) > 0.1f) {
            cachedComponentPositions = null
        }
    }

    override fun updateParameters(parameters: Map<String, String>) {
        this.parameters = parameters
        motionLawEngine.updateParameters(parameters)
        // Clear cached positions when parameters change
        cachedComponentPositions = null
    }

    override fun drawFrame(drawScope: DrawScope, canvasWidth: Float, canvasHeight: Float, scale: Float, offset: Offset) {
        println("DEBUG: ComponentBasedAnimationEngine.drawFrame called - Litvin active: ${motionLawEngine.isLitvinActive()}, tables: ${motionLawEngine.getLitvinTables() != null}, curves: ${motionLawEngine.getLitvinCurves() != null}")
        // Litvin rendering path (if enabled and data available)
        if (motionLawEngine.isLitvinActive() && motionLawEngine.getLitvinTables() != null && motionLawEngine.getLitvinCurves() != null) {
            LitvinRenderer.drawFrame(
                drawScope = drawScope,
                canvasWidth = canvasWidth,
                canvasHeight = canvasHeight,
                scaleUser = scale,
                offset = offset,
                angleDeg = currentAngle,
                parameters = parameters,
                motion = motionLawEngine,
            )
            return
        }
        // If Litvin mode is selected but data unavailable, block fallback rendering to avoid non-Litvin visuals
        if (motionLawEngine.isLitvinActive()) {
            return
        }

        // Determine number of assemblies and phase offset (default to 2 assemblies)
        val n = ParameterResolver.int(parameters, "assembly_count", 2, "Assembly Count").coerceAtLeast(1)
        val defaultStep = 360f / n
        val step = ParameterResolver.float(parameters, "assembly_phase_offset_deg", defaultStep)

        // Build positions list for phase-shifted assemblies
        val positionsList = ArrayList<ComponentPositions>(n)
        for (i in 0 until n) {
            val phaseAngle = (currentAngle + i * step) % 360f
            val pos =
                if (i == 0) {
                    // Cache for the base angle to avoid recomputation within small delta
                    if (cachedComponentPositions != null && Math.abs(currentAngle - lastDrawnAngle) < 0.1f) {
                        cachedComponentPositions!!
                    } else {
                        val p = motionLawEngine.getComponentPositions(phaseAngle.toDouble())
                        cachedComponentPositions = p
                        lastDrawnAngle = currentAngle
                        p
                    }
                } else {
                    motionLawEngine.getComponentPositions(phaseAngle.toDouble())
                }
            positionsList.add(pos)
        }
        println("DEBUG: ComponentBasedAnimationEngine.drawFrame - calculated ${positionsList.size} positions for angle $currentAngle")

        // Draw the components (ring + multiple assemblies)
        println("DEBUG: ComponentBasedAnimationEngine.drawFrame - positionsList size: ${positionsList.size}, currentAngle: $currentAngle")
        ComponentBasedAnimationRenderer.drawFrame(
            drawScope,
            canvasWidth,
            canvasHeight,
            scale,
            offset,
            currentAngle,
            parameters,
            positionsList,
        )
    }

    override fun getParameters(): Map<String, String> = parameters

    override fun getType(): AnimationType = AnimationType.COMPONENT_BASED

    override fun dispose() {
        motionLawEngine.dispose()
    }
}

/**
 * Animation engine for FEA-based animation.
 * This engine uses the results of the FEA engine to generate the animation.
 */
class FeaBasedAnimationEngine(private var parameters: Map<String, String>) : AnimationEngine {
    private var currentAngle: Float = 0f
    private var feaResults: File? = null
    private var analysisData: AnalysisData? = null

    override fun getCurrentAngle(): Float = currentAngle

    override fun setAngle(angle: Float) {
        currentAngle = angle % 360f
    }

    override fun updateParameters(parameters: Map<String, String>) {
        this.parameters = parameters
    }

    /**
     * Set the FEA results file.
     *
     * @param resultsFile The FEA results file
     */
    fun setFeaResults(resultsFile: File) {
        feaResults = resultsFile
        // Load the FEA results
        analysisData = FeaResultsLoader.loadResults(resultsFile)
    }

    override fun drawFrame(drawScope: DrawScope, canvasWidth: Float, canvasHeight: Float, scale: Float, offset: Offset) {
        // Draw the FEA results
        FeaBasedAnimationRenderer.drawFrame(
            drawScope,
            canvasWidth,
            canvasHeight,
            scale,
            offset,
            currentAngle,
            parameters,
            analysisData,
        )
    }

    override fun getParameters(): Map<String, String> = parameters

    override fun getType(): AnimationType = AnimationType.FEA_BASED

    override fun dispose() {
        // Clean up resources
        analysisData = null
    }
}

/**
 * Manager for animation engines.
 * This class provides a centralized way to manage animation engines.
 */
class AnimationManager {
    private var currentEngine: AnimationEngine? = null

    /**
     * Set the current animation engine.
     *
     * @param engine The animation engine to set as current
     */
    fun setCurrentEngine(engine: AnimationEngine) {
        currentEngine?.dispose()
        currentEngine = engine
    }

    /**
     * Get the current animation engine.
     *
     * @return The current animation engine, or null if none is set
     */
    fun getCurrentEngine(): AnimationEngine? = currentEngine

    /**
     * Create a new animation engine and set it as current.
     *
     * @param type The type of animation engine to create
     * @param parameters The initial parameters for the animation
     * @return The created animation engine
     */
    fun createAndSetEngine(type: AnimationType, parameters: Map<String, String>): AnimationEngine {
        val engine = AnimationEngineFactory.createEngine(type, parameters)
        setCurrentEngine(engine)
        return engine
    }

    /**
     * Clean up resources when the manager is no longer needed.
     */
    fun dispose() {
        currentEngine?.dispose()
        currentEngine = null
    }
}

// MotionLawEngine is now defined in MotionLawEngine.kt

/**
 * Data class for component positions.
 */
data class ComponentPositions(val pistonPosition: Offset, val rodPosition: Offset, val camPosition: Offset)

/**
 * Placeholder for the FEA results loader.
 */
object FeaResultsLoader {
    private val gson = Gson()

    /**
     * Load FEA results from a JSON file.
     *
     * Expected flexible schema (best-effort parsing):
     * {
     *   "displacements": [ { "node": 1, "x": 0.1, "y": -0.02 }, ... ] | { "1": {"x":..,"y":..}, ... }
     *   "stresses": [ { "element": 5, "vonMises": 120.5 }, ... ] | { "5": 120.5, ... }
     *   "timeSteps": [ 0.0, 0.01, ... ]
     * }
     */
    fun loadResults(resultsFile: File): AnalysisData {
        require(resultsFile.exists()) { "FEA results file not found: ${resultsFile.absolutePath}" }

        val root = gson.fromJson(resultsFile.readText(), JsonObject::class.java)

        val dispMap = mutableMapOf<Int, Offset>()
        val stressMap = mutableMapOf<Int, Float>()
        val timeSteps = mutableListOf<Float>()

        // Displacements
        root.get("displacements")?.let { dispNode ->
            when {
                dispNode.isJsonArray -> {
                    val arr = dispNode.asJsonArray
                    arr.forEach { item ->
                        val obj = item.asJsonObject
                        val id = obj.get("node")?.asInt ?: return@forEach
                        val x = obj.get("x")?.asFloat ?: 0f
                        val y = obj.get("y")?.asFloat ?: 0f
                        dispMap[id] = Offset(x, y)
                    }
                }
                dispNode.isJsonObject -> {
                    val obj = dispNode.asJsonObject
                    obj.entrySet().forEach { (key, value) ->
                        val id = key.toIntOrNull() ?: return@forEach
                        val vObj = value.asJsonObject
                        val x = vObj.get("x")?.asFloat ?: 0f
                        val y = vObj.get("y")?.asFloat ?: 0f
                        dispMap[id] = Offset(x, y)
                    }
                }
            }
        }

        // Stresses
        root.get("stresses")?.let { stressNode ->
            when {
                stressNode.isJsonArray -> {
                    val arr = stressNode.asJsonArray
                    arr.forEach { item ->
                        val obj = item.asJsonObject
                        val id = obj.get("element")?.asInt ?: return@forEach
                        // Handle malformed stress values gracefully
                        val vm = try {
                            obj.get("vonMises")?.asFloat
                                ?: obj.get("value")?.asFloat
                        } catch (e: NumberFormatException) {
                            // Skip malformed stress values
                            return@forEach
                        }
                        if (vm != null) {
                            stressMap[id] = vm
                        }
                    }
                }
                stressNode.isJsonObject -> {
                    val obj = stressNode.asJsonObject
                    obj.entrySet().forEach { (key, value) ->
                        val id = key.toIntOrNull() ?: return@forEach
                        // Handle malformed stress values gracefully
                        val vm = try {
                            value.asFloat
                        } catch (e: NumberFormatException) {
                            // Skip malformed stress values
                            return@forEach
                        }
                        stressMap[id] = vm
                    }
                }
            }
        }

        // timeSteps
        root.get("timeSteps")?.let { tsNode ->
            if (tsNode.isJsonArray) {
                tsNode.asJsonArray.forEach { t -> timeSteps.add(t.asFloat) }
            }
        }

        return AnalysisData(
            displacements = dispMap.toMap(),
            stresses = stressMap.toMap(),
            timeSteps = timeSteps.toList(),
        )
    }
}

/**
 * Data class for FEA analysis data.
 */
data class AnalysisData(val displacements: Map<Int, Offset>, val stresses: Map<Int, Float>, val timeSteps: List<Float>)

// Renderer classes are now defined in their own files:
// - CycloidalAnimationRenderer.kt
// - ComponentBasedAnimationRenderer.kt
// - FeaBasedAnimationRenderer.kt
