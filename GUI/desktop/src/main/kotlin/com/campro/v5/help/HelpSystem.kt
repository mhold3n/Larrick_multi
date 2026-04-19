package com.campro.v5.help

import androidx.compose.runtime.Composable
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.util.concurrent.ConcurrentHashMap

/**
 * Provides contextual help for the CamPro v5 application.
 * This class manages help topics, tooltips, and guided tours.
 */
class HelpSystem {
    // Help topics
    private val helpTopics = ConcurrentHashMap<String, HelpTopic>()

    // Component help
    private val componentHelp = ConcurrentHashMap<String, String>()

    // Current help context
    private val _currentContext = mutableStateOf<String?>(null)

    // Help events
    private val _helpEvents = MutableStateFlow<HelpEvent?>(null)
    val helpEvents: StateFlow<HelpEvent?> = _helpEvents.asStateFlow()

    init {
        // Register default help topics
        registerDefaultHelpTopics()
    }

    /**
     * Register default help topics.
     */
    private fun registerDefaultHelpTopics() {
        // General topics
        registerHelpTopic(
            HelpTopic(
                id = "general.overview",
                title = "CamPro v5 Overview",
                content = "CamPro v5 is a comprehensive tool for designing and analyzing cycloidal mechanisms.",
                category = "General",
            ),
        )

        registerHelpTopic(
            HelpTopic(
                id = "general.ui",
                title = "User Interface",
                content = "The user interface is divided into several sections: Parameters, Animation, Plots, and Data.",
                category = "General",
            ),
        )

        // Parameter topics
        registerHelpTopic(
            HelpTopic(
                id = "parameters.geometry",
                title = "Cam Geometry Parameters",
                content = "These parameters define the basic geometry of the cycloidal mechanism.",
                category = "Parameters",
            ),
        )

        registerHelpTopic(
            HelpTopic(
                id = "parameters.materials",
                title = "Material Parameters",
                content = "These parameters define the materials used in the mechanism.",
                category = "Parameters",
            ),
        )

        // Animation topics
        registerHelpTopic(
            HelpTopic(
                id = "animation.controls",
                title = "Animation Controls",
                content = "Use these controls to play, pause, and adjust the animation speed.",
                category = "Animation",
            ),
        )

        registerHelpTopic(
            HelpTopic(
                id = "animation.view",
                title = "Animation View",
                content = "You can zoom and pan the animation view using the mouse or touchpad.",
                category = "Animation",
            ),
        )

        // Plot topics
        registerHelpTopic(
            HelpTopic(
                id = "plots.types",
                title = "Plot Types",
                content = "CamPro v5 provides several types of plots for analyzing the mechanism.",
                category = "Plots",
            ),
        )

        registerHelpTopic(
            HelpTopic(
                id = "plots.controls",
                title = "Plot Controls",
                content = "Use these controls to adjust the plot view and export plot data.",
                category = "Plots",
            ),
        )

        // Data topics
        registerHelpTopic(
            HelpTopic(
                id = "data.summary",
                title = "Data Summary",
                content = "The data summary provides key metrics about the mechanism.",
                category = "Data",
            ),
        )

        registerHelpTopic(
            HelpTopic(
                id = "data.export",
                title = "Data Export",
                content = "You can export data in various formats for further analysis.",
                category = "Data",
            ),
        )

        // Register component help
        registerComponentHelp("ParameterInputForm", "parameters.geometry")
        registerComponentHelp("CycloidalAnimationWidget", "animation.controls")
        registerComponentHelp("PlotCarouselWidget", "plots.types")
        registerComponentHelp("DataDisplayPanel", "data.summary")
    }

    /**
     * Register a help topic.
     *
     * @param topic The help topic to register
     */
    fun registerHelpTopic(topic: HelpTopic) {
        helpTopics[topic.id] = topic

        // Emit help event
        _helpEvents.value = HelpEvent.TopicRegistered(topic)
    }

    /**
     * Register help for a component.
     *
     * @param componentId The ID of the component
     * @param topicId The ID of the help topic
     */
    fun registerComponentHelp(componentId: String, topicId: String) {
        componentHelp[componentId] = topicId
    }

    /**
     * Get a help topic by ID.
     *
     * @param topicId The ID of the help topic
     * @return The help topic, or null if it wasn't found
     */
    fun getHelpTopic(topicId: String): HelpTopic? = helpTopics[topicId]

    /**
     * Get help for a component.
     *
     * @param componentId The ID of the component
     * @return The help topic, or null if it wasn't found
     */
    fun getComponentHelp(componentId: String): HelpTopic? {
        val topicId = componentHelp[componentId] ?: return null
        return helpTopics[topicId]
    }

    /**
     * Get all help topics.
     *
     * @return A list of all help topics
     */
    fun getAllHelpTopics(): List<HelpTopic> = helpTopics.values.toList()

    /**
     * Get help topics by category.
     *
     * @param category The category
     * @return A list of help topics in the category
     */
    fun getHelpTopicsByCategory(category: String): List<HelpTopic> = helpTopics.values.filter { it.category == category }

    /**
     * Set the current help context.
     *
     * @param context The context, or null to clear the context
     */
    fun setContext(context: String?) {
        _currentContext.value = context

        // Emit help event
        if (context != null) {
            _helpEvents.value = HelpEvent.ContextChanged(context)
        } else {
            _helpEvents.value = HelpEvent.ContextCleared
        }
    }

    /**
     * Get the current help context.
     *
     * @return The current context, or null if there is no context
     */
    fun getContext(): String? = _currentContext.value

    /**
     * Show help for a component.
     *
     * @param componentId The ID of the component
     * @return True if help was shown, false if no help was found
     */
    fun showComponentHelp(componentId: String): Boolean {
        val topic = getComponentHelp(componentId)
        if (topic != null) {
            // Emit help event
            _helpEvents.value = HelpEvent.HelpRequested(topic)
            return true
        }
        return false
    }

    /**
     * Show help for a topic.
     *
     * @param topicId The ID of the topic
     * @return True if help was shown, false if the topic wasn't found
     */
    fun showHelpTopic(topicId: String): Boolean {
        val topic = getHelpTopic(topicId)
        if (topic != null) {
            // Emit help event
            _helpEvents.value = HelpEvent.HelpRequested(topic)
            return true
        }
        return false
    }

    /**
     * Search for help topics.
     *
     * @param query The search query
     * @return A list of matching help topics
     */
    fun searchHelpTopics(query: String): List<HelpTopic> {
        val lowerQuery = query.lowercase()
        return helpTopics.values.filter {
            it.title.lowercase().contains(lowerQuery) ||
                it.content.lowercase().contains(lowerQuery)
        }
    }

    companion object {
        // Singleton instance
        private var instance: HelpSystem? = null

        /**
         * Get the singleton instance of the HelpSystem.
         *
         * @return The HelpSystem instance
         */
        fun getInstance(): HelpSystem {
            if (instance == null) {
                instance = HelpSystem()
            }
            return instance!!
        }
    }
}

/**
 * A help topic.
 *
 * @param id The unique ID of the topic
 * @param title The title of the topic
 * @param content The content of the topic
 * @param category The category of the topic
 */
data class HelpTopic(val id: String, val title: String, val content: String, val category: String)

/**
 * Help events emitted by the HelpSystem.
 */
sealed class HelpEvent {
    /**
     * Event emitted when a help topic is registered.
     *
     * @param topic The registered topic
     */
    data class TopicRegistered(val topic: HelpTopic) : HelpEvent()

    /**
     * Event emitted when the help context changes.
     *
     * @param context The new context
     */
    data class ContextChanged(val context: String) : HelpEvent()

    /**
     * Event emitted when the help context is cleared.
     */
    object ContextCleared : HelpEvent()

    /**
     * Event emitted when help is requested.
     *
     * @param topic The requested topic
     */
    data class HelpRequested(val topic: HelpTopic) : HelpEvent()
}

/**
 * Composable function to remember a HelpSystem instance.
 *
 * @return The remembered HelpSystem instance
 */
@Composable
fun rememberHelpSystem(): HelpSystem = remember { HelpSystem.getInstance() }
