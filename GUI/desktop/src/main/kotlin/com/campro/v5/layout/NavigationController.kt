package com.campro.v5.layout

import androidx.compose.runtime.Composable
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/**
 * Controls navigation within the CamPro v5 application.
 * This class provides a tabbed interface for main application sections,
 * breadcrumb navigation for complex workflows, and keyboard shortcuts for navigation.
 */
class NavigationController {
    // Navigation destinations
    enum class Destination(val title: String, val route: String) {
        HOME("Home", "home"),
        PARAMETERS("Parameters", "parameters"),
        ANIMATION("Animation", "animation"),
        PLOTS("Plots", "plots"),
        DATA("Data", "data"),
        SETTINGS("Settings", "settings"),
        HELP("Help", "help"),
    }

    // Current navigation state
    private val _currentDestination = mutableStateOf(Destination.HOME)
    private val _navigationHistory = mutableStateOf(listOf(Destination.HOME))
    private val _breadcrumbs = mutableStateOf(listOf<String>())

    // Navigation events flow
    private val _navigationEvents = MutableStateFlow<NavigationEvent?>(null)
    val navigationEvents: StateFlow<NavigationEvent?> = _navigationEvents.asStateFlow()

    // Getters for current values
    val currentDestination: Destination
        get() = _currentDestination.value

    val navigationHistory: List<Destination>
        get() = _navigationHistory.value

    val breadcrumbs: List<String>
        get() = _breadcrumbs.value

    /**
     * Navigate to a destination.
     *
     * @param destination The destination to navigate to
     * @param addToHistory Whether to add this destination to the navigation history
     */
    fun navigateTo(destination: Destination, addToHistory: Boolean = true) {
        // Add current destination to history if requested
        if (addToHistory && _currentDestination.value != destination) {
            _navigationHistory.value = _navigationHistory.value + _currentDestination.value
        }

        // Update current destination
        _currentDestination.value = destination

        // Clear breadcrumbs when navigating to a new main destination
        _breadcrumbs.value = emptyList()

        // Emit navigation event
        _navigationEvents.value = NavigationEvent.Navigated(destination)
    }

    /**
     * Navigate back to the previous destination.
     *
     * @return True if navigation was successful, false if there's no previous destination
     */
    fun navigateBack(): Boolean {
        // Check if there's a previous destination
        if (_navigationHistory.value.isEmpty()) {
            return false
        }

        // Get the last destination from history
        val previousDestination = _navigationHistory.value.last()

        // Update history and current destination
        _navigationHistory.value = _navigationHistory.value.dropLast(1)
        _currentDestination.value = previousDestination

        // Clear breadcrumbs when navigating back
        _breadcrumbs.value = emptyList()

        // Emit navigation event
        _navigationEvents.value = NavigationEvent.NavigatedBack(previousDestination)

        return true
    }

    /**
     * Add a breadcrumb to the navigation path.
     *
     * @param breadcrumb The breadcrumb to add
     */
    fun addBreadcrumb(breadcrumb: String) {
        _breadcrumbs.value = _breadcrumbs.value + breadcrumb

        // Emit navigation event
        _navigationEvents.value = NavigationEvent.BreadcrumbAdded(breadcrumb)
    }

    /**
     * Navigate to a specific breadcrumb in the path.
     *
     * @param index The index of the breadcrumb to navigate to
     * @return True if navigation was successful, false if the index is invalid
     */
    fun navigateToBreadcrumb(index: Int): Boolean {
        // Check if the index is valid
        if (index < 0 || index >= _breadcrumbs.value.size) {
            return false
        }

        // Update breadcrumbs
        _breadcrumbs.value = _breadcrumbs.value.take(index + 1)

        // Emit navigation event
        _navigationEvents.value = NavigationEvent.NavigatedToBreadcrumb(index, _breadcrumbs.value[index])

        return true
    }

    /**
     * Clear the navigation history.
     */
    fun clearHistory() {
        _navigationHistory.value = emptyList()
    }

    /**
     * Clear the breadcrumbs.
     */
    fun clearBreadcrumbs() {
        _breadcrumbs.value = emptyList()
    }

    /**
     * Handle a keyboard shortcut for navigation.
     *
     * @param shortcut The keyboard shortcut
     * @return True if the shortcut was handled, false otherwise
     */
    fun handleShortcut(shortcut: String): Boolean = when (shortcut) {
        "ctrl+h" -> {
            navigateTo(Destination.HOME)
            true
        }
        "ctrl+p" -> {
            navigateTo(Destination.PARAMETERS)
            true
        }
        "ctrl+a" -> {
            navigateTo(Destination.ANIMATION)
            true
        }
        "ctrl+l" -> {
            navigateTo(Destination.PLOTS)
            true
        }
        "ctrl+d" -> {
            navigateTo(Destination.DATA)
            true
        }
        "ctrl+s" -> {
            navigateTo(Destination.SETTINGS)
            true
        }
        "ctrl+f1" -> {
            navigateTo(Destination.HELP)
            true
        }
        "alt+left", "backspace" -> {
            navigateBack()
        }
        else -> false
    }

    companion object {
        // Singleton instance
        private var instance: NavigationController? = null

        /**
         * Get the singleton instance of the NavigationController.
         *
         * @return The NavigationController instance
         */
        fun getInstance(): NavigationController {
            if (instance == null) {
                instance = NavigationController()
            }
            return instance!!
        }
    }
}

/**
 * Navigation events emitted by the NavigationController.
 */
sealed class NavigationEvent {
    /**
     * Event emitted when navigation to a destination occurs.
     *
     * @param destination The destination navigated to
     */
    data class Navigated(val destination: NavigationController.Destination) : NavigationEvent()

    /**
     * Event emitted when navigation back to a previous destination occurs.
     *
     * @param destination The destination navigated back to
     */
    data class NavigatedBack(val destination: NavigationController.Destination) : NavigationEvent()

    /**
     * Event emitted when a breadcrumb is added to the navigation path.
     *
     * @param breadcrumb The breadcrumb added
     */
    data class BreadcrumbAdded(val breadcrumb: String) : NavigationEvent()

    /**
     * Event emitted when navigation to a specific breadcrumb occurs.
     *
     * @param index The index of the breadcrumb
     * @param breadcrumb The breadcrumb navigated to
     */
    data class NavigatedToBreadcrumb(val index: Int, val breadcrumb: String) : NavigationEvent()
}

/**
 * Composable function to remember a NavigationController instance.
 *
 * @return The remembered NavigationController instance
 */
@Composable
fun rememberNavigationController(): NavigationController = remember { NavigationController.getInstance() }

/**
 * Composable function to create a breadcrumb navigation path.
 *
 * @param navigationController The NavigationController instance
 * @param onBreadcrumbClick Callback for when a breadcrumb is clicked
 * @return A composable function that displays the breadcrumb navigation
 */
@Composable
fun BreadcrumbNavigation(navigationController: NavigationController = rememberNavigationController(), onBreadcrumbClick: (Int) -> Unit) {
    val breadcrumbs = navigationController.breadcrumbs
    val currentDestination = navigationController.currentDestination

    // Breadcrumb navigation UI implementation would go here
    // This would typically be a Row with Text components for each breadcrumb
    // separated by a divider or arrow icon
}

/**
 * Composable function to create a tabbed navigation interface.
 *
 * @param navigationController The NavigationController instance
 * @param onTabSelected Callback for when a tab is selected
 * @return A composable function that displays the tabbed navigation
 */
@Composable
fun TabbedNavigation(
    navigationController: NavigationController = rememberNavigationController(),
    onTabSelected: (NavigationController.Destination) -> Unit,
) {
    val currentDestination = navigationController.currentDestination

    // Tabbed navigation UI implementation would go here
    // This would typically be a TabRow with Tab components for each destination
}
