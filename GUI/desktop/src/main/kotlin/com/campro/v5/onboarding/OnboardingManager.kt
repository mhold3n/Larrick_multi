package com.campro.v5.onboarding

import androidx.compose.runtime.Composable
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import com.campro.v5.Event
import com.campro.v5.EventSystem
import com.campro.v5.layout.StateManager
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.util.concurrent.ConcurrentHashMap

/**
 * Manages the onboarding experience for new users of the CamPro v5 application.
 * This class provides interactive tutorials, sample projects, and progress tracking.
 *
 * The OnboardingManager is responsible for:
 * - Guiding users through the onboarding process with sequential steps
 * - Providing tutorials for specific features
 * - Offering sample projects to demonstrate functionality
 * - Tracking onboarding progress and completion status
 * - Emitting events when onboarding actions occur
 *
 * Usage:
 * ```
 * // Get the singleton instance
 * val onboardingManager = OnboardingManager.getInstance()
 *
 * // Start the onboarding process
 * onboardingManager.startOnboarding()
 *
 * // Navigate through onboarding steps
 * onboardingManager.nextStep()
 * onboardingManager.previousStep()
 * onboardingManager.goToStep("parameters")
 *
 * // Start a tutorial
 * onboardingManager.startTutorial("basic_mechanism")
 *
 * // Load a sample project
 * val parameters = onboardingManager.loadSampleProject("basic_mechanism")
 *
 * // Check if onboarding is completed
 * val completed = onboardingManager.isOnboardingCompleted()
 *
 * // In Compose, use the rememberOnboardingManager() function
 * val onboardingManager = rememberOnboardingManager()
 * ```
 *
 * Events:
 * The OnboardingManager emits events through both its internal flow and the EventSystem:
 * - OnboardingStarted: When the onboarding process starts
 * - OnboardingCompleted: When the onboarding process is completed
 * - OnboardingSkipped: When the onboarding process is skipped
 * - OnboardingReset: When the onboarding process is reset
 * - StepChanged: When the onboarding step changes
 * - TutorialStarted: When a tutorial starts
 * - TutorialCompleted: When a tutorial is completed
 * - SampleProjectLoaded: When a sample project is loaded
 */
class OnboardingManager {
    // Onboarding steps
    private val onboardingSteps = ConcurrentHashMap<String, OnboardingStep>()

    // Onboarding tutorials
    private val tutorials = ConcurrentHashMap<String, OnboardingTutorial>()

    // Sample projects
    private val sampleProjects = ConcurrentHashMap<String, SampleProject>()

    // Current onboarding state
    private val _currentStep = mutableStateOf<String?>(null)
    private val _onboardingCompleted = mutableStateOf(false)
    private val _tutorialInProgress = mutableStateOf<String?>(null)

    // Onboarding events
    private val _onboardingEvents = MutableStateFlow<OnboardingEvent?>(null)
    val onboardingEvents: StateFlow<OnboardingEvent?> = _onboardingEvents.asStateFlow()

    // State manager for persistence
    private val stateManager = StateManager.getInstance()

    init {
        // Register default onboarding steps
        registerDefaultOnboardingSteps()

        // Register default tutorials
        registerDefaultTutorials()

        // Register default sample projects
        registerDefaultSampleProjects()

        // Load onboarding state
        loadOnboardingState()
    }

    /**
     * Register default onboarding steps.
     */
    private fun registerDefaultOnboardingSteps() {
        // Welcome step
        registerOnboardingStep(
            OnboardingStep(
                id = "welcome",
                title = "Welcome to CamPro v5",
                description = "Welcome to CamPro v5, a comprehensive tool for designing and analyzing cycloidal mechanisms.",
                order = 0,
            ),
        )

        // UI overview step
        registerOnboardingStep(
            OnboardingStep(
                id = "ui_overview",
                title = "User Interface Overview",
                description = "Let's take a quick tour of the user interface.",
                order = 1,
            ),
        )

        // Parameters step
        registerOnboardingStep(
            OnboardingStep(
                id = "parameters",
                title = "Setting Parameters",
                description = "Learn how to set parameters for your cycloidal mechanism.",
                order = 2,
            ),
        )

        // Animation step
        registerOnboardingStep(
            OnboardingStep(
                id = "animation",
                title = "Animation Controls",
                description = "Learn how to control the animation of your mechanism.",
                order = 3,
            ),
        )

        // Plots step
        registerOnboardingStep(
            OnboardingStep(
                id = "plots",
                title = "Analyzing Plots",
                description = "Learn how to analyze plots of your mechanism's performance.",
                order = 4,
            ),
        )

        // Data step
        registerOnboardingStep(
            OnboardingStep(
                id = "data",
                title = "Viewing Data",
                description = "Learn how to view and export data about your mechanism.",
                order = 5,
            ),
        )

        // Layout step
        registerOnboardingStep(
            OnboardingStep(
                id = "layouts",
                title = "Responsive Layouts",
                description = "Learn how to use different layout templates and adapt to different screen sizes.",
                order = 6,
            ),
        )

        // File management step
        registerOnboardingStep(
            OnboardingStep(
                id = "file_management",
                title = "File Management",
                description = "Learn how to work with files, including recent files and auto-save features.",
                order = 7,
            ),
        )

        // Collaboration step
        registerOnboardingStep(
            OnboardingStep(
                id = "collaboration",
                title = "Collaboration Features",
                description = "Learn how to export and share your designs with others.",
                order = 8,
            ),
        )

        // Completion step
        registerOnboardingStep(
            OnboardingStep(
                id = "completion",
                title = "Onboarding Complete",
                description = "Congratulations! You've completed the onboarding process.",
                order = 9,
            ),
        )
    }

    /**
     * Register default tutorials.
     */
    private fun registerDefaultTutorials() {
        // Basic mechanism tutorial
        registerTutorial(
            OnboardingTutorial(
                id = "basic_mechanism",
                title = "Creating a Basic Mechanism",
                description = "Learn how to create a basic cycloidal mechanism.",
                steps =
                listOf(
                    "Set the base circle radius",
                    "Set the rolling circle radius",
                    "Set the tracing point distance",
                    "Generate the animation",
                ),
                relatedStepId = "parameters",
            ),
        )

        // Animation tutorial
        registerTutorial(
            OnboardingTutorial(
                id = "animation_controls",
                title = "Using Animation Controls",
                description = "Learn how to use the animation controls.",
                steps =
                listOf(
                    "Play and pause the animation",
                    "Adjust the animation speed",
                    "Zoom in and out",
                    "Pan the view",
                ),
                relatedStepId = "animation",
            ),
        )

        // Plot analysis tutorial
        registerTutorial(
            OnboardingTutorial(
                id = "plot_analysis",
                title = "Analyzing Plots",
                description = "Learn how to analyze plots of your mechanism's performance.",
                steps =
                listOf(
                    "Select a plot type",
                    "Interpret the plot data",
                    "Zoom and pan the plot",
                    "Export plot data",
                ),
                relatedStepId = "plots",
            ),
        )

        // Data export tutorial
        registerTutorial(
            OnboardingTutorial(
                id = "data_export",
                title = "Exporting Data",
                description = "Learn how to export data about your mechanism.",
                steps =
                listOf(
                    "View the data summary",
                    "Export data as CSV",
                    "Generate a report",
                    "Share your results",
                ),
                relatedStepId = "data",
            ),
        )

        // Responsive layouts tutorial
        registerTutorial(
            OnboardingTutorial(
                id = "responsive_layouts",
                title = "Using Responsive Layouts",
                description = "Learn how to use different layout templates and adapt to different screen sizes.",
                steps =
                listOf(
                    "Switch between layout templates",
                    "Adapt to different screen sizes",
                    "Customize layout for your workflow",
                    "Save your layout preferences",
                ),
                relatedStepId = "layouts",
            ),
        )

        // Recent files tutorial
        registerTutorial(
            OnboardingTutorial(
                id = "recent_files",
                title = "Working with Recent Files",
                description = "Learn how to use the recent files feature to quickly access your projects.",
                steps =
                listOf(
                    "View your recent files",
                    "Pin important files",
                    "Filter and sort recent files",
                    "Preview file contents",
                ),
                relatedStepId = "file_management",
            ),
        )

        // Auto-save tutorial
        registerTutorial(
            OnboardingTutorial(
                id = "auto_save",
                title = "Using Auto-Save",
                description = "Learn how to use the auto-save feature to protect your work.",
                steps =
                listOf(
                    "Configure auto-save settings",
                    "Recover from crashes",
                    "Manage auto-save backups",
                    "Restore previous versions",
                ),
                relatedStepId = "file_management",
            ),
        )

        // Export and share tutorial
        registerTutorial(
            OnboardingTutorial(
                id = "export_share",
                title = "Exporting and Sharing",
                description = "Learn how to export and share your designs with others.",
                steps =
                listOf(
                    "Export to different formats",
                    "Generate comprehensive reports",
                    "Share designs with colleagues",
                    "Collaborate on projects",
                ),
                relatedStepId = "collaboration",
            ),
        )
    }

    /**
     * Register default sample projects.
     */
    private fun registerDefaultSampleProjects() {
        // Basic mechanism project
        registerSampleProject(
            SampleProject(
                id = "basic_mechanism",
                title = "Basic Cycloidal Mechanism",
                description = "A simple cycloidal mechanism with default parameters.",
                parameters =
                mapOf(
                    "Piston Diameter" to "70.0",
                    "Stroke" to "20.0",
                    "Rod Length" to "40.0",
                    "TDC Offset" to "40.0",
                    "Cycle Ratio" to "2.0",
                ),
                relatedTutorialId = "basic_mechanism",
            ),
        )

        // High-speed mechanism project
        registerSampleProject(
            SampleProject(
                id = "high_speed",
                title = "High-Speed Mechanism",
                description = "A cycloidal mechanism optimized for high speed.",
                parameters =
                mapOf(
                    "Piston Diameter" to "60.0",
                    "Stroke" to "15.0",
                    "Rod Length" to "45.0",
                    "TDC Offset" to "35.0",
                    "Cycle Ratio" to "3.0",
                ),
                relatedTutorialId = "animation_controls",
            ),
        )

        // High-torque mechanism project
        registerSampleProject(
            SampleProject(
                id = "high_torque",
                title = "High-Torque Mechanism",
                description = "A cycloidal mechanism optimized for high torque.",
                parameters =
                mapOf(
                    "Piston Diameter" to "80.0",
                    "Stroke" to "25.0",
                    "Rod Length" to "35.0",
                    "TDC Offset" to "45.0",
                    "Cycle Ratio" to "1.5",
                ),
                relatedTutorialId = "plot_analysis",
            ),
        )

        // Design workflow project
        registerSampleProject(
            SampleProject(
                id = "design_workflow",
                title = "Design Workflow Example",
                description = "A project demonstrating the design workflow with responsive layouts.",
                parameters =
                mapOf(
                    "Piston Diameter" to "75.0",
                    "Stroke" to "22.0",
                    "Rod Length" to "42.0",
                    "TDC Offset" to "38.0",
                    "Cycle Ratio" to "2.2",
                    "Layout Template" to "DESIGN_WORKFLOW",
                ),
                relatedTutorialId = "responsive_layouts",
            ),
        )

        // Analysis workflow project
        registerSampleProject(
            SampleProject(
                id = "analysis_workflow",
                title = "Analysis Workflow Example",
                description = "A project demonstrating the analysis workflow with responsive layouts.",
                parameters =
                mapOf(
                    "Piston Diameter" to "72.0",
                    "Stroke" to "21.0",
                    "Rod Length" to "41.0",
                    "TDC Offset" to "39.0",
                    "Cycle Ratio" to "2.1",
                    "Layout Template" to "ANALYSIS_WORKFLOW",
                ),
                relatedTutorialId = "responsive_layouts",
            ),
        )

        // Auto-save example project
        registerSampleProject(
            SampleProject(
                id = "auto_save_example",
                title = "Auto-Save Example",
                description = "A project demonstrating the auto-save feature with recovery options.",
                parameters =
                mapOf(
                    "Piston Diameter" to "68.0",
                    "Stroke" to "19.0",
                    "Rod Length" to "39.0",
                    "TDC Offset" to "41.0",
                    "Cycle Ratio" to "1.9",
                    "Auto-Save Interval" to "60",
                ),
                relatedTutorialId = "auto_save",
            ),
        )

        // Collaboration project
        registerSampleProject(
            SampleProject(
                id = "collaboration_example",
                title = "Collaboration Example",
                description = "A project demonstrating the collaboration features with export options.",
                parameters =
                mapOf(
                    "Piston Diameter" to "65.0",
                    "Stroke" to "18.0",
                    "Rod Length" to "38.0",
                    "TDC Offset" to "42.0",
                    "Cycle Ratio" to "1.8",
                    "Export Format" to "PDF",
                ),
                relatedTutorialId = "export_share",
            ),
        )
    }

    /**
     * Load onboarding state from the state manager.
     */
    private fun loadOnboardingState() {
        // Load onboarding completed state
        _onboardingCompleted.value = stateManager.getState("onboarding.completed", false)

        // Load current step
        val defaultStep = if (_onboardingCompleted.value) "" else "welcome"
        val currentStepValue = stateManager.getState("onboarding.currentStep", defaultStep)
        _currentStep.value = if (currentStepValue.isEmpty() && _onboardingCompleted.value) null else currentStepValue

        // Load tutorial progress
        val tutorialValue = stateManager.getState("onboarding.tutorialInProgress", "")
        _tutorialInProgress.value = if (tutorialValue.isEmpty()) null else tutorialValue
    }

    /**
     * Save onboarding state to the state manager.
     */
    private fun saveOnboardingState() {
        // Save onboarding completed state
        stateManager.setState("onboarding.completed", _onboardingCompleted.value)

        // Save current step
        _currentStep.value?.let { stateManager.setState("onboarding.currentStep", it) }
            ?: stateManager.removeState("onboarding.currentStep")

        // Save tutorial progress
        _tutorialInProgress.value?.let { stateManager.setState("onboarding.tutorialInProgress", it) }
            ?: stateManager.removeState("onboarding.tutorialInProgress")
    }

    /**
     * Register an onboarding step.
     *
     * @param step The onboarding step to register
     */
    fun registerOnboardingStep(step: OnboardingStep) {
        onboardingSteps[step.id] = step
    }

    /**
     * Register a tutorial.
     *
     * @param tutorial The tutorial to register
     */
    fun registerTutorial(tutorial: OnboardingTutorial) {
        tutorials[tutorial.id] = tutorial
    }

    /**
     * Register a sample project.
     *
     * @param project The sample project to register
     */
    fun registerSampleProject(project: SampleProject) {
        sampleProjects[project.id] = project
    }

    /**
     * Get an onboarding step by ID.
     *
     * @param stepId The ID of the step
     * @return The onboarding step, or null if it wasn't found
     */
    fun getOnboardingStep(stepId: String): OnboardingStep? = onboardingSteps[stepId]

    /**
     * Get a tutorial by ID.
     *
     * @param tutorialId The ID of the tutorial
     * @return The tutorial, or null if it wasn't found
     */
    fun getTutorial(tutorialId: String): OnboardingTutorial? = tutorials[tutorialId]

    /**
     * Get a sample project by ID.
     *
     * @param projectId The ID of the sample project
     * @return The sample project, or null if it wasn't found
     */
    fun getSampleProject(projectId: String): SampleProject? = sampleProjects[projectId]

    /**
     * Get all onboarding steps.
     *
     * @return A list of all onboarding steps
     */
    fun getAllOnboardingSteps(): List<OnboardingStep> = onboardingSteps.values.sortedBy { it.order }

    /**
     * Get all tutorials.
     *
     * @return A list of all tutorials
     */
    fun getAllTutorials(): List<OnboardingTutorial> = tutorials.values.toList()

    /**
     * Get all sample projects.
     *
     * @return A list of all sample projects
     */
    fun getAllSampleProjects(): List<SampleProject> = sampleProjects.values.toList()

    /**
     * Get tutorials related to an onboarding step.
     *
     * @param stepId The ID of the onboarding step
     * @return A list of tutorials related to the step
     */
    fun getTutorialsForStep(stepId: String): List<OnboardingTutorial> = tutorials.values.filter { it.relatedStepId == stepId }

    /**
     * Get sample projects related to a tutorial.
     *
     * @param tutorialId The ID of the tutorial
     * @return A list of sample projects related to the tutorial
     */
    fun getSampleProjectsForTutorial(tutorialId: String): List<SampleProject> =
        sampleProjects.values.filter { it.relatedTutorialId == tutorialId }

    /**
     * Start the onboarding process.
     */
    fun startOnboarding() {
        // Reset onboarding state
        _onboardingCompleted.value = false
        _currentStep.value = "welcome"
        _tutorialInProgress.value = null

        // Save state
        saveOnboardingState()

        // Emit onboarding event through internal flow
        _onboardingEvents.value = OnboardingEvent.OnboardingStarted

        // Emit onboarding event through EventSystem
        EventSystem.emit(OnboardingEvent.OnboardingStarted)
    }

    /**
     * Go to the next onboarding step.
     *
     * @return True if there is a next step, false if onboarding is complete
     */
    fun nextStep(): Boolean {
        val currentStepId = _currentStep.value ?: return false
        val currentStep = onboardingSteps[currentStepId] ?: return false

        // Find the next step
        val nextStep =
            onboardingSteps.values
                .filter { it.order > currentStep.order }
                .minByOrNull { it.order }

        if (nextStep != null) {
            // Go to the next step
            _currentStep.value = nextStep.id

            // Save state
            saveOnboardingState()

            // Emit onboarding event through internal flow
            _onboardingEvents.value = OnboardingEvent.StepChanged(nextStep)

            // Emit onboarding event through EventSystem
            EventSystem.emit(OnboardingEvent.StepChanged(nextStep))

            return true
        } else {
            // Onboarding is complete
            _onboardingCompleted.value = true
            _currentStep.value = null

            // Save state
            saveOnboardingState()

            // Emit onboarding event through internal flow
            _onboardingEvents.value = OnboardingEvent.OnboardingCompleted

            // Emit onboarding event through EventSystem
            EventSystem.emit(OnboardingEvent.OnboardingCompleted)

            return false
        }
    }

    /**
     * Go to the previous onboarding step.
     *
     * @return True if there is a previous step, false otherwise
     */
    fun previousStep(): Boolean {
        val currentStepId = _currentStep.value ?: return false
        val currentStep = onboardingSteps[currentStepId] ?: return false

        // Find the previous step
        val previousStep =
            onboardingSteps.values
                .filter { it.order < currentStep.order }
                .maxByOrNull { it.order }

        if (previousStep != null) {
            // Go to the previous step
            _currentStep.value = previousStep.id

            // Save state
            saveOnboardingState()

            // Emit onboarding event through internal flow
            _onboardingEvents.value = OnboardingEvent.StepChanged(previousStep)

            // Emit onboarding event through EventSystem
            EventSystem.emit(OnboardingEvent.StepChanged(previousStep))

            return true
        }

        return false
    }

    /**
     * Go to a specific onboarding step.
     *
     * @param stepId The ID of the step to go to
     * @return True if the step was found, false otherwise
     */
    fun goToStep(stepId: String): Boolean {
        val step = onboardingSteps[stepId] ?: return false

        // Go to the step
        _currentStep.value = step.id

        // Save state
        saveOnboardingState()

        // Emit onboarding event through internal flow
        _onboardingEvents.value = OnboardingEvent.StepChanged(step)

        // Emit onboarding event through EventSystem
        EventSystem.emit(OnboardingEvent.StepChanged(step))

        return true
    }

    /**
     * Skip the onboarding process.
     */
    fun skipOnboarding() {
        // Mark onboarding as completed
        _onboardingCompleted.value = true
        _currentStep.value = null
        _tutorialInProgress.value = null

        // Save state
        saveOnboardingState()

        // Emit onboarding event through internal flow
        _onboardingEvents.value = OnboardingEvent.OnboardingSkipped

        // Emit onboarding event through EventSystem
        EventSystem.emit(OnboardingEvent.OnboardingSkipped)
    }

    /**
     * Reset the onboarding process.
     */
    fun resetOnboarding() {
        // Reset onboarding state
        _onboardingCompleted.value = false
        _currentStep.value = "welcome"
        _tutorialInProgress.value = null

        // Save state
        saveOnboardingState()

        // Emit onboarding event through internal flow
        _onboardingEvents.value = OnboardingEvent.OnboardingReset

        // Emit onboarding event through EventSystem
        EventSystem.emit(OnboardingEvent.OnboardingReset)
    }

    /**
     * Start a tutorial.
     *
     * @param tutorialId The ID of the tutorial to start
     * @return True if the tutorial was found, false otherwise
     */
    fun startTutorial(tutorialId: String): Boolean {
        val tutorial = tutorials[tutorialId] ?: return false

        // Start the tutorial
        _tutorialInProgress.value = tutorial.id

        // Save state
        saveOnboardingState()

        // Emit onboarding event through internal flow
        _onboardingEvents.value = OnboardingEvent.TutorialStarted(tutorial)

        // Emit onboarding event through EventSystem
        EventSystem.emit(OnboardingEvent.TutorialStarted(tutorial))

        return true
    }

    /**
     * Complete a tutorial.
     *
     * @param tutorialId The ID of the tutorial to complete
     * @return True if the tutorial was found, false otherwise
     */
    fun completeTutorial(tutorialId: String): Boolean {
        val tutorial = tutorials[tutorialId] ?: return false

        // Complete the tutorial
        if (_tutorialInProgress.value == tutorial.id) {
            _tutorialInProgress.value = null
        }

        // Save state
        saveOnboardingState()

        // Emit onboarding event through internal flow
        _onboardingEvents.value = OnboardingEvent.TutorialCompleted(tutorial)

        // Emit onboarding event through EventSystem
        EventSystem.emit(OnboardingEvent.TutorialCompleted(tutorial))

        return true
    }

    /**
     * Load a sample project.
     *
     * @param projectId The ID of the sample project to load
     * @return The parameters of the sample project, or null if it wasn't found
     */
    fun loadSampleProject(projectId: String): Map<String, String>? {
        val project = sampleProjects[projectId] ?: return null

        // Emit onboarding event through internal flow
        _onboardingEvents.value = OnboardingEvent.SampleProjectLoaded(project)

        // Emit onboarding event through EventSystem
        EventSystem.emit(OnboardingEvent.SampleProjectLoaded(project))

        return project.parameters
    }

    /**
     * Check if onboarding is completed.
     *
     * @return True if onboarding is completed, false otherwise
     */
    fun isOnboardingCompleted(): Boolean = _onboardingCompleted.value

    /**
     * Get the current onboarding step.
     *
     * @return The current step, or null if onboarding is completed
     */
    fun getCurrentStep(): OnboardingStep? {
        val stepId = _currentStep.value ?: return null
        return onboardingSteps[stepId]
    }

    /**
     * Get the current tutorial in progress.
     *
     * @return The current tutorial, or null if no tutorial is in progress
     */
    fun getCurrentTutorial(): OnboardingTutorial? {
        val tutorialId = _tutorialInProgress.value ?: return null
        return tutorials[tutorialId]
    }

    companion object {
        // Singleton instance
        private var instance: OnboardingManager? = null

        /**
         * Get the singleton instance of the OnboardingManager.
         *
         * @return The OnboardingManager instance
         */
        fun getInstance(): OnboardingManager {
            if (instance == null) {
                instance = OnboardingManager()
            }
            return instance!!
        }
    }
}

/**
 * An onboarding step.
 *
 * @param id The unique ID of the step
 * @param title The title of the step
 * @param description The description of the step
 * @param order The order of the step in the onboarding process
 */
data class OnboardingStep(val id: String, val title: String, val description: String, val order: Int)

/**
 * An onboarding tutorial.
 *
 * @param id The unique ID of the tutorial
 * @param title The title of the tutorial
 * @param description The description of the tutorial
 * @param steps The steps of the tutorial
 * @param relatedStepId The ID of the related onboarding step
 */
data class OnboardingTutorial(
    val id: String,
    val title: String,
    val description: String,
    val steps: List<String>,
    val relatedStepId: String,
)

/**
 * A sample project.
 *
 * @param id The unique ID of the project
 * @param title The title of the project
 * @param description The description of the project
 * @param parameters The parameters of the project
 * @param relatedTutorialId The ID of the related tutorial
 */
data class SampleProject(
    val id: String,
    val title: String,
    val description: String,
    val parameters: Map<String, String>,
    val relatedTutorialId: String,
)

/**
 * Onboarding events emitted by the OnboardingManager.
 */
sealed class OnboardingEvent : Event() {
    /**
     * Event emitted when the onboarding process starts.
     */
    object OnboardingStarted : OnboardingEvent() {
        override val type: String = "onboarding_started"

        override fun toJson(): String = "{\"type\":\"$type\"}"
    }

    /**
     * Event emitted when the onboarding process is completed.
     */
    object OnboardingCompleted : OnboardingEvent() {
        override val type: String = "onboarding_completed"

        override fun toJson(): String = "{\"type\":\"$type\"}"
    }

    /**
     * Event emitted when the onboarding process is skipped.
     */
    object OnboardingSkipped : OnboardingEvent() {
        override val type: String = "onboarding_skipped"

        override fun toJson(): String = "{\"type\":\"$type\"}"
    }

    /**
     * Event emitted when the onboarding process is reset.
     */
    object OnboardingReset : OnboardingEvent() {
        override val type: String = "onboarding_reset"

        override fun toJson(): String = "{\"type\":\"$type\"}"
    }

    /**
     * Event emitted when the onboarding step changes.
     *
     * @param step The new step
     */
    data class StepChanged(val step: OnboardingStep) : OnboardingEvent() {
        override val type: String = "step_changed"

        override fun toJson(): String =
            "{\"type\":\"$type\",\"step_id\":\"${step.id}\",\"step_title\":\"${step.title}\",\"step_order\":${step.order}}"
    }

    /**
     * Event emitted when a tutorial starts.
     *
     * @param tutorial The tutorial
     */
    data class TutorialStarted(val tutorial: OnboardingTutorial) : OnboardingEvent() {
        override val type: String = "tutorial_started"

        override fun toJson(): String = "{\"type\":\"$type\",\"tutorial_id\":\"${tutorial.id}\",\"tutorial_title\":\"${tutorial.title}\"}"
    }

    /**
     * Event emitted when a tutorial is completed.
     *
     * @param tutorial The tutorial
     */
    data class TutorialCompleted(val tutorial: OnboardingTutorial) : OnboardingEvent() {
        override val type: String = "tutorial_completed"

        override fun toJson(): String = "{\"type\":\"$type\",\"tutorial_id\":\"${tutorial.id}\",\"tutorial_title\":\"${tutorial.title}\"}"
    }

    /**
     * Event emitted when a sample project is loaded.
     *
     * @param project The sample project
     */
    data class SampleProjectLoaded(val project: SampleProject) : OnboardingEvent() {
        override val type: String = "sample_project_loaded"

        override fun toJson(): String = "{\"type\":\"$type\",\"project_id\":\"${project.id}\",\"project_title\":\"${project.title}\"}"
    }
}

/**
 * Composable function to remember an OnboardingManager instance.
 *
 * @return The remembered OnboardingManager instance
 */
@Composable
fun rememberOnboardingManager(): OnboardingManager = remember { OnboardingManager.getInstance() }
