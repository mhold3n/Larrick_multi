package com.campro.v5.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.graphics.Color
import com.campro.v5.layout.StateManager
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/**
 * Manages themes for the CamPro v5 application.
 * This class provides dark and light themes, custom color scheme support,
 * and theme switching functionality.
 */
class ThemeManager {
    // Theme state
    private val _currentTheme = mutableStateOf(Theme.SYSTEM)
    private val _customColorScheme = mutableStateOf<ColorScheme?>(null)
    private val _isDarkTheme = mutableStateOf(false)

    // Theme change events
    private val _themeChangeEvents = MutableStateFlow<ThemeChangeEvent?>(null)
    val themeChangeEvents: StateFlow<ThemeChangeEvent?> = _themeChangeEvents.asStateFlow()

    // State manager for persistence
    private val stateManager = StateManager.getInstance()

    init {
        // Load theme from state
        val savedTheme = stateManager.getState("theme", Theme.SYSTEM.name)
        _currentTheme.value =
            try {
                Theme.valueOf(savedTheme)
            } catch (e: IllegalArgumentException) {
                Theme.SYSTEM
            }

        // Load custom color scheme from state if available
        val customPrimary = stateManager.getState("theme.custom.primary", "")
        val customSecondary = stateManager.getState("theme.custom.secondary", "")
        val customTertiary = stateManager.getState("theme.custom.tertiary", "")

        if (customPrimary.isNotEmpty() && customSecondary.isNotEmpty() && customTertiary.isNotEmpty()) {
            try {
                val primary = parseHexColor(customPrimary)
                val secondary = parseHexColor(customSecondary)
                val tertiary = parseHexColor(customTertiary)

                _customColorScheme.value = createColorScheme(primary, secondary, tertiary)
            } catch (e: Exception) {
                // Invalid color format, ignore
            }
        }
    }

    // Getters for current values
    val currentTheme: Theme
        get() = _currentTheme.value

    val customColorScheme: ColorScheme?
        get() = _customColorScheme.value

    val isDarkTheme: Boolean
        get() = _isDarkTheme.value

    /**
     * Set the current theme.
     *
     * @param theme The theme to set
     */
    fun setTheme(theme: Theme) {
        if (_currentTheme.value != theme) {
            _currentTheme.value = theme

            // Save theme to state
            stateManager.setState("theme", theme.name)

            // Emit theme change event
            _themeChangeEvents.value = ThemeChangeEvent.ThemeChanged(theme)
        }
    }

    /**
     * Set a custom color scheme.
     *
     * @param primary The primary color
     * @param secondary The secondary color
     * @param tertiary The tertiary color
     */
    fun setCustomColorScheme(primary: Color, secondary: Color, tertiary: Color) {
        val colorScheme = createColorScheme(primary, secondary, tertiary)
        _customColorScheme.value = colorScheme

        // Save custom colors to state
        stateManager.setState("theme.custom.primary", colorToHex(primary))
        stateManager.setState("theme.custom.secondary", colorToHex(secondary))
        stateManager.setState("theme.custom.tertiary", colorToHex(tertiary))

        // Emit theme change event
        _themeChangeEvents.value = ThemeChangeEvent.CustomColorSchemeChanged(primary, secondary, tertiary)
    }

    /**
     * Clear the custom color scheme.
     */
    fun clearCustomColorScheme() {
        _customColorScheme.value = null

        // Clear custom colors from state
        stateManager.removeState("theme.custom.primary")
        stateManager.removeState("theme.custom.secondary")
        stateManager.removeState("theme.custom.tertiary")

        // Emit theme change event
        _themeChangeEvents.value = ThemeChangeEvent.CustomColorSchemeCleared
    }

    /**
     * Update the dark theme state based on the system theme.
     *
     * @param isDark Whether the system is in dark theme
     */
    fun updateSystemDarkTheme(isDark: Boolean) {
        _isDarkTheme.value =
            when (_currentTheme.value) {
                Theme.LIGHT -> false
                Theme.DARK -> true
                Theme.SYSTEM -> isDark
                Theme.CUSTOM -> _isDarkTheme.value
            }
    }

    /**
     * Set whether the custom theme is dark.
     *
     * @param isDark Whether the custom theme is dark
     */
    fun setCustomThemeDark(isDark: Boolean) {
        if (_currentTheme.value == Theme.CUSTOM) {
            _isDarkTheme.value = isDark

            // Save dark theme state to state
            stateManager.setState("theme.custom.dark", isDark)

            // Emit theme change event
            _themeChangeEvents.value = ThemeChangeEvent.DarkThemeChanged(isDark)
        }
    }

    /**
     * Get the current color scheme.
     *
     * @param isDarkTheme Whether to use dark theme colors
     * @return The color scheme
     */
    fun getColorScheme(isDarkTheme: Boolean): ColorScheme = when (_currentTheme.value) {
        Theme.LIGHT -> lightColorScheme()
        Theme.DARK -> darkColorScheme()
        Theme.SYSTEM -> if (isDarkTheme) darkColorScheme() else lightColorScheme()
        Theme.CUSTOM -> _customColorScheme.value ?: if (isDarkTheme) darkColorScheme() else lightColorScheme()
    }

    /**
     * Create a color scheme from primary, secondary, and tertiary colors.
     *
     * @param primary The primary color
     * @param secondary The secondary color
     * @param tertiary The tertiary color
     * @return The color scheme
     */
    private fun createColorScheme(primary: Color, secondary: Color, tertiary: Color): ColorScheme = if (_isDarkTheme.value) {
        darkColorScheme(
            primary = primary,
            secondary = secondary,
            tertiary = tertiary,
        )
    } else {
        lightColorScheme(
            primary = primary,
            secondary = secondary,
            tertiary = tertiary,
        )
    }

    /**
     * Convert a Color to a hex string.
     *
     * @param color The color to convert
     * @return The hex string
     */
    private fun colorToHex(color: Color): String {
        val red = (color.red * 255).toInt()
        val green = (color.green * 255).toInt()
        val blue = (color.blue * 255).toInt()
        return String.format("#%02X%02X%02X", red, green, blue)
    }

    /**
     * Parse a hex color string to a Color.
     *
     * @param hex The hex color string (e.g., "#FF0000" or "FF0000")
     * @return The Color object
     */
    private fun parseHexColor(hex: String): Color {
        val cleanHex = hex.removePrefix("#")
        val colorInt = cleanHex.toLong(16)
        val red = ((colorInt shr 16) and 0xFF) / 255f
        val green = ((colorInt shr 8) and 0xFF) / 255f
        val blue = (colorInt and 0xFF) / 255f
        return Color(red, green, blue)
    }

    companion object {
        // Singleton instance
        private var instance: ThemeManager? = null

        /**
         * Get the singleton instance of the ThemeManager.
         *
         * @return The ThemeManager instance
         */
        fun getInstance(): ThemeManager {
            if (instance == null) {
                instance = ThemeManager()
            }
            return instance!!
        }
    }
}

/**
 * Available themes.
 */
enum class Theme {
    LIGHT,
    DARK,
    SYSTEM,
    CUSTOM,
}

/**
 * Theme change events emitted by the ThemeManager.
 */
sealed class ThemeChangeEvent {
    /**
     * Event emitted when the theme changes.
     *
     * @param theme The new theme
     */
    data class ThemeChanged(val theme: Theme) : ThemeChangeEvent()

    /**
     * Event emitted when the custom color scheme changes.
     *
     * @param primary The primary color
     * @param secondary The secondary color
     * @param tertiary The tertiary color
     */
    data class CustomColorSchemeChanged(val primary: Color, val secondary: Color, val tertiary: Color) : ThemeChangeEvent()

    /**
     * Event emitted when the custom color scheme is cleared.
     */
    object CustomColorSchemeCleared : ThemeChangeEvent()

    /**
     * Event emitted when the dark theme state changes.
     *
     * @param isDark Whether dark theme is enabled
     */
    data class DarkThemeChanged(val isDark: Boolean) : ThemeChangeEvent()
}

/**
 * Composable function to remember a ThemeManager instance.
 *
 * @return The remembered ThemeManager instance
 */
@Composable
fun rememberThemeManager(): ThemeManager = remember { ThemeManager.getInstance() }

/**
 * Composable function to provide the current theme.
 *
 * @param themeManager The ThemeManager instance
 * @param content The content to display
 */
@Composable
fun CamProTheme(themeManager: ThemeManager = rememberThemeManager(), content: @Composable () -> Unit) {
    // Determine if dark theme should be used
    val systemInDarkTheme = isSystemInDarkTheme()

    // Update theme manager with system dark theme state
    themeManager.updateSystemDarkTheme(systemInDarkTheme)

    // Get the current color scheme
    val colorScheme = themeManager.getColorScheme(themeManager.isDarkTheme)

    // Apply the theme
    MaterialTheme(
        colorScheme = colorScheme,
        typography = Typography(),
        content = content,
    )
}

/**
 * Default typography for the application.
 */
@Composable
fun Typography(): Typography = Typography(
    // Use default Material 3 typography
)

/**
 * Predefined color schemes for the application.
 */
object ColorSchemes {
    // Engineering theme
    val EngineeringLight =
        lightColorScheme(
            primary = Color(0xFF0D47A1),
            secondary = Color(0xFF1976D2),
            tertiary = Color(0xFF42A5F5),
        )

    val EngineeringDark =
        darkColorScheme(
            primary = Color(0xFF90CAF9),
            secondary = Color(0xFF64B5F6),
            tertiary = Color(0xFF42A5F5),
        )

    // Manufacturing theme
    val ManufacturingLight =
        lightColorScheme(
            primary = Color(0xFF1B5E20),
            secondary = Color(0xFF2E7D32),
            tertiary = Color(0xFF4CAF50),
        )

    val ManufacturingDark =
        darkColorScheme(
            primary = Color(0xFF81C784),
            secondary = Color(0xFF66BB6A),
            tertiary = Color(0xFF4CAF50),
        )

    // Scientific theme
    val ScientificLight =
        lightColorScheme(
            primary = Color(0xFF4A148C),
            secondary = Color(0xFF6A1B9A),
            tertiary = Color(0xFF8E24AA),
        )

    val ScientificDark =
        darkColorScheme(
            primary = Color(0xFFCE93D8),
            secondary = Color(0xFFBA68C8),
            tertiary = Color(0xFFAB47BC),
        )
}
