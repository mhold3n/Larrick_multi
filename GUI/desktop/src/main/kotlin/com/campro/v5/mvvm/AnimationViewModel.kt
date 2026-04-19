package com.campro.v5.mvvm

import androidx.compose.runtime.*
import com.campro.v5.models.OptimizationParameters
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import org.slf4j.LoggerFactory

/**
 * ViewModel for animation and motion law generation.
 * 
 * Handles the state and business logic for motion law generation,
 * animation playback, and visualization controls.
 */
class AnimationViewModel : BaseViewModel() {
    private val logger = LoggerFactory.getLogger(AnimationViewModel::class.java)
    
    // Animation state
    private val _isAnimating = MutableStateFlow(false)
    val isAnimating: StateFlow<Boolean> = _isAnimating.asStateFlow()
    
    private val _animationSpeed = MutableStateFlow(1.0f)
    val animationSpeed: StateFlow<Float> = _animationSpeed.asStateFlow()
    
    private val _animationProgress = MutableStateFlow(0.0f)
    val animationProgress: StateFlow<Float> = _animationProgress.asStateFlow()
    
    // Motion law data
    private val _motionLawData = MutableStateFlow<MotionLawData?>(null)
    val motionLawData: StateFlow<MotionLawData?> = _motionLawData.asStateFlow()
    
    // Generation state
    private val _isGenerating = MutableStateFlow(false)
    val isGenerating: StateFlow<Boolean> = _isGenerating.asStateFlow()
    
    // Visualization settings
    private val _showVelocity = MutableStateFlow(true)
    val showVelocity: StateFlow<Boolean> = _showVelocity.asStateFlow()
    
    private val _showAcceleration = MutableStateFlow(true)
    val showAcceleration: StateFlow<Boolean> = _showAcceleration.asStateFlow()
    
    private val _showJerk = MutableStateFlow(false)
    val showJerk: StateFlow<Boolean> = _showJerk.asStateFlow()
    
    private val _zoomLevel = MutableStateFlow(1.0f)
    val zoomLevel: StateFlow<Float> = _zoomLevel.asStateFlow()
    
    init {
        logger.info("AnimationViewModel initialized")
    }
    
    /**
     * Generate motion law from optimization parameters
     */
    fun generateMotionLaw(parameters: OptimizationParameters) {
        executeWithLoading(
            operation = {
                _isGenerating.value = true
                
                // Simulate motion law generation
                kotlinx.coroutines.delay(1000)
                
                // Create mock motion law data
                val data = MotionLawData(
                    theta = (0..360).map { it.toDouble() }.toDoubleArray(),
                    position = (0..360).map { 
                        Math.sin(Math.toRadians(it.toDouble())) * parameters.stroke / 2
                    }.toDoubleArray(),
                    velocity = (0..360).map { 
                        Math.cos(Math.toRadians(it.toDouble())) * parameters.stroke / 2 * Math.toRadians(parameters.rpm / 60.0)
                    }.toDoubleArray(),
                    acceleration = (0..360).map { 
                        -Math.sin(Math.toRadians(it.toDouble())) * parameters.stroke / 2 * Math.pow(Math.toRadians(parameters.rpm / 60.0), 2.0)
                    }.toDoubleArray(),
                    jerk = (0..360).map { 
                        -Math.cos(Math.toRadians(it.toDouble())) * parameters.stroke / 2 * Math.pow(Math.toRadians(parameters.rpm / 60.0), 3.0)
                    }.toDoubleArray()
                )
                
                _isGenerating.value = false
                data
            },
            onSuccess = { data ->
                _motionLawData.value = data
                setSuccess("Motion law generated successfully")
            }
        )
    }
    
    /**
     * Start animation playback
     */
    fun startAnimation() {
        if (_motionLawData.value != null) {
            _isAnimating.value = true
            setSuccess("Animation started")
            logger.info("Animation started")
        } else {
            setError("No motion law data available. Generate motion law first.")
        }
    }
    
    /**
     * Stop animation playback
     */
    fun stopAnimation() {
        _isAnimating.value = false
        setSuccess("Animation stopped")
        logger.info("Animation stopped")
    }
    
    /**
     * Pause animation playback
     */
    fun pauseAnimation() {
        _isAnimating.value = false
        setSuccess("Animation paused")
        logger.info("Animation paused")
    }
    
    /**
     * Reset animation to beginning
     */
    fun resetAnimation() {
        _animationProgress.value = 0.0f
        _isAnimating.value = false
        setSuccess("Animation reset")
        logger.info("Animation reset")
    }
    
    /**
     * Update animation speed
     */
    fun updateAnimationSpeed(speed: Float) {
        _animationSpeed.value = speed.coerceIn(0.1f, 5.0f)
        logger.debug("Animation speed updated to: $speed")
    }
    
    /**
     * Update animation progress
     */
    fun updateAnimationProgress(progress: Float) {
        _animationProgress.value = progress.coerceIn(0.0f, 1.0f)
    }
    
    /**
     * Toggle velocity display
     */
    fun toggleVelocityDisplay() {
        _showVelocity.value = !_showVelocity.value
        logger.debug("Velocity display toggled: ${_showVelocity.value}")
    }
    
    /**
     * Toggle acceleration display
     */
    fun toggleAccelerationDisplay() {
        _showAcceleration.value = !_showAcceleration.value
        logger.debug("Acceleration display toggled: ${_showAcceleration.value}")
    }
    
    /**
     * Toggle jerk display
     */
    fun toggleJerkDisplay() {
        _showJerk.value = !_showJerk.value
        logger.debug("Jerk display toggled: ${_showJerk.value}")
    }
    
    /**
     * Update zoom level
     */
    fun updateZoomLevel(zoom: Float) {
        _zoomLevel.value = zoom.coerceIn(0.1f, 5.0f)
        logger.debug("Zoom level updated to: $zoom")
    }
    
    /**
     * Zoom in
     */
    fun zoomIn() {
        val newZoom = _zoomLevel.value * 1.2f
        updateZoomLevel(newZoom)
    }
    
    /**
     * Zoom out
     */
    fun zoomOut() {
        val newZoom = _zoomLevel.value / 1.2f
        updateZoomLevel(newZoom)
    }
    
    /**
     * Reset zoom to default
     */
    fun resetZoom() {
        _zoomLevel.value = 1.0f
        logger.debug("Zoom reset to default")
    }
}

/**
 * Data class for motion law results
 */
data class MotionLawData(
    val theta: DoubleArray,
    val position: DoubleArray,
    val velocity: DoubleArray,
    val acceleration: DoubleArray,
    val jerk: DoubleArray
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as MotionLawData

        if (!theta.contentEquals(other.theta)) return false
        if (!position.contentEquals(other.position)) return false
        if (!velocity.contentEquals(other.velocity)) return false
        if (!acceleration.contentEquals(other.acceleration)) return false
        if (!jerk.contentEquals(other.jerk)) return false

        return true
    }

    override fun hashCode(): Int {
        var result = theta.contentHashCode()
        result = 31 * result + position.contentHashCode()
        result = 31 * result + velocity.contentHashCode()
        result = 31 * result + acceleration.contentHashCode()
        result = 31 * result + jerk.contentHashCode()
        return result
    }
}
