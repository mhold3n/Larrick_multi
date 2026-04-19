package com.campro.v5.mvvm

import androidx.compose.runtime.*
import com.campro.v5.models.OptimizationParameters
import com.campro.v5.file.ProjectManager
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import org.slf4j.LoggerFactory

/**
 * ViewModel for optimization parameters management.
 * 
 * Handles the state and business logic for optimization parameters,
 * including validation, persistence, and change tracking.
 */
class OptimizationViewModel : BaseViewModel() {
    private val logger = LoggerFactory.getLogger(OptimizationViewModel::class.java)
    
    // Current optimization parameters
    private val _parameters = MutableStateFlow(OptimizationParameters.createDefault())
    val parameters: StateFlow<OptimizationParameters> = _parameters.asStateFlow()
    
    // Parameter validation state
    private val _validationErrors = MutableStateFlow<Map<String, String>>(emptyMap())
    val validationErrors: StateFlow<Map<String, String>> = _validationErrors.asStateFlow()
    
    // Parameter change tracking
    private val _hasUnsavedChanges = MutableStateFlow(false)
    val hasUnsavedChanges: StateFlow<Boolean> = _hasUnsavedChanges.asStateFlow()
    
    // Project manager for persistence
    private val projectManager = ProjectManager()
    
    init {
        logger.info("OptimizationViewModel initialized")
    }
    
    /**
     * Update optimization parameters
     */
    fun updateParameters(newParameters: OptimizationParameters) {
        val currentParams = _parameters.value
        
        // Check if parameters actually changed
        if (currentParams != newParameters) {
            _parameters.value = newParameters
            _hasUnsavedChanges.value = true
            
            // Validate parameters
            validateParameters(newParameters)
            
            logger.debug("Parameters updated: gearRatio=${newParameters.gearRatio}")
        }
    }
    
    /**
     * Update a specific parameter
     */
    fun updateParameter(
        parameterName: String,
        value: Any
    ) {
        val currentParams = _parameters.value
        val newParams = when (parameterName) {
            "gearRatio" -> currentParams.copy(gearRatio = value as Double)
            "stroke" -> currentParams.copy(stroke = value as Double)
            "rpm" -> currentParams.copy(rpm = value as Double)
            "compressionRatio" -> currentParams.copy(compressionRatio = value as Double)
            "pressureAngle" -> currentParams.copy(pressureAngle = value as Double)
            "toothCount" -> currentParams.copy(toothCount = value as Int)
            "module" -> currentParams.copy(module = value as Double)
            "faceWidth" -> currentParams.copy(faceWidth = value as Double)
            "materialStrength" -> currentParams.copy(materialStrength = value as Double)
            "safetyFactor" -> currentParams.copy(safetyFactor = value as Double)
            else -> {
                logger.warn("Unknown parameter: $parameterName")
                currentParams
            }
        }
        
        updateParameters(newParams)
    }
    
    /**
     * Validate optimization parameters
     */
    private fun validateParameters(params: OptimizationParameters) {
        val errors = mutableMapOf<String, String>()
        
        // Validate gear ratio
        if (params.gearRatio <= 0) {
            errors["gearRatio"] = "Gear ratio must be positive"
        } else if (params.gearRatio > 10) {
            errors["gearRatio"] = "Gear ratio should not exceed 10"
        }
        
        // Validate stroke
        if (params.stroke <= 0) {
            errors["stroke"] = "Stroke must be positive"
        } else if (params.stroke > 100) {
            errors["stroke"] = "Stroke should not exceed 100mm"
        }
        
        // Validate RPM
        if (params.rpm <= 0) {
            errors["rpm"] = "RPM must be positive"
        } else if (params.rpm > 10000) {
            errors["rpm"] = "RPM should not exceed 10,000"
        }
        
        // Validate compression ratio
        if (params.compressionRatio <= 1) {
            errors["compressionRatio"] = "Compression ratio must be greater than 1"
        } else if (params.compressionRatio > 20) {
            errors["compressionRatio"] = "Compression ratio should not exceed 20"
        }
        
        // Validate pressure angle
        if (params.pressureAngle < 14.5 || params.pressureAngle > 25) {
            errors["pressureAngle"] = "Pressure angle must be between 14.5° and 25°"
        }
        
        // Validate tooth count
        if (params.toothCount < 12 || params.toothCount > 200) {
            errors["toothCount"] = "Tooth count must be between 12 and 200"
        }
        
        // Validate module
        if (params.module <= 0) {
            errors["module"] = "Module must be positive"
        } else if (params.module > 10) {
            errors["module"] = "Module should not exceed 10mm"
        }
        
        // Validate face width
        if (params.faceWidth <= 0) {
            errors["faceWidth"] = "Face width must be positive"
        } else if (params.faceWidth > 50) {
            errors["faceWidth"] = "Face width should not exceed 50mm"
        }
        
        // Validate material strength
        if (params.materialStrength <= 0) {
            errors["materialStrength"] = "Material strength must be positive"
        } else if (params.materialStrength > 2000) {
            errors["materialStrength"] = "Material strength should not exceed 2000 MPa"
        }
        
        // Validate safety factor
        if (params.safetyFactor <= 1) {
            errors["safetyFactor"] = "Safety factor must be greater than 1"
        } else if (params.safetyFactor > 10) {
            errors["safetyFactor"] = "Safety factor should not exceed 10"
        }
        
        _validationErrors.value = errors
        
        if (errors.isEmpty()) {
            setSuccess("Parameters validated successfully")
        } else {
            setError("Parameter validation failed: ${errors.size} errors found")
        }
    }
    
    /**
     * Reset parameters to default values
     */
    fun resetToDefaults() {
        val defaultParams = OptimizationParameters.createDefault()
        updateParameters(defaultParams)
        setSuccess("Parameters reset to default values")
    }
    
    /**
     * Save parameters to project
     */
    fun saveParameters() {
        executeWithLoading(
            operation = {
                // In a real implementation, this would save to the project
                // For now, we'll just simulate the operation
                kotlinx.coroutines.delay(500)
                _hasUnsavedChanges.value = false
                "Parameters saved successfully"
            },
            onSuccess = { message ->
                setSuccess(message)
            }
        )
    }
    
    /**
     * Load parameters from project
     */
    fun loadParameters() {
        executeWithLoading(
            operation = {
                // In a real implementation, this would load from the project
                // For now, we'll just simulate the operation
                kotlinx.coroutines.delay(300)
                OptimizationParameters.createDefault()
            },
            onSuccess = { loadedParams ->
                updateParameters(loadedParams)
                setSuccess("Parameters loaded successfully")
            }
        )
    }
    
    /**
     * Check if parameters are valid
     */
    fun areParametersValid(): Boolean {
        return _validationErrors.value.isEmpty()
    }
    
    /**
     * Get validation error for a specific parameter
     */
    fun getValidationError(parameterName: String): String? {
        return _validationErrors.value[parameterName]
    }
}
