package com.campro.v5.mvvm

import androidx.compose.runtime.*
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import org.slf4j.LoggerFactory

/**
 * Base ViewModel class for MVVM architecture in CamProV5.
 * 
 * Provides common functionality for all ViewModels including:
 * - State management with StateFlow
 * - Loading states
 * - Error handling
 * - Coroutine scope management
 */
abstract class BaseViewModel {
    protected val logger = LoggerFactory.getLogger(this::class.java)
    
    // Loading state
    private val _isLoading = MutableStateFlow(false)
    val isLoading: StateFlow<Boolean> = _isLoading.asStateFlow()
    
    // Error state
    private val _error = MutableStateFlow<String?>(null)
    val error: StateFlow<String?> = _error.asStateFlow()
    
    // Success state
    private val _successMessage = MutableStateFlow<String?>(null)
    val successMessage: StateFlow<String?> = _successMessage.asStateFlow()
    
    // Coroutine scope for background operations
    protected val viewModelScope = CoroutineScope(Dispatchers.Main)
    
    /**
     * Set loading state
     */
    protected fun setLoading(loading: Boolean) {
        _isLoading.value = loading
    }
    
    /**
     * Set error message
     */
    protected fun setError(error: String?) {
        _error.value = error
        if (error != null) {
            logger.error("ViewModel error: $error")
        }
    }
    
    /**
     * Set success message
     */
    protected fun setSuccess(message: String?) {
        _successMessage.value = message
        if (message != null) {
            logger.info("ViewModel success: $message")
        }
    }
    
    /**
     * Clear all messages
     */
    fun clearMessages() {
        _error.value = null
        _successMessage.value = null
    }
    
    /**
     * Execute a suspend function with loading state management
     */
    protected fun <T> executeWithLoading(
        operation: suspend () -> T,
        onSuccess: (T) -> Unit = {},
        onError: (Throwable) -> Unit = { setError(it.message) }
    ) {
        viewModelScope.launch {
            try {
                setLoading(true)
                clearMessages()
                val result = operation()
                onSuccess(result)
            } catch (e: Exception) {
                onError(e)
            } finally {
                setLoading(false)
            }
        }
    }
    
    /**
     * Clean up resources when ViewModel is destroyed
     */
    fun onCleared() {
        logger.debug("ViewModel ${this::class.simpleName} cleared")
    }
}

/**
 * Composable function to observe ViewModel state
 */
@Composable
fun <T> BaseViewModel.observeState(
    stateFlow: StateFlow<T>,
    onStateChange: (T) -> Unit
) {
    val state by stateFlow.collectAsState()
    LaunchedEffect(state) {
        onStateChange(state)
    }
}

/**
 * Composable function to observe loading state
 */
@Composable
fun BaseViewModel.observeLoading(): Boolean {
    return isLoading.collectAsState().value
}

/**
 * Composable function to observe error state
 */
@Composable
fun BaseViewModel.observeError(): String? {
    return error.collectAsState().value
}

/**
 * Composable function to observe success message
 */
@Composable
fun BaseViewModel.observeSuccess(): String? {
    return successMessage.collectAsState().value
}
