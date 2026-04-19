@file:OptIn(androidx.compose.material3.ExperimentalMaterial3Api::class)

package com.campro.v5

import androidx.compose.desktop.ui.tooling.preview.Preview
import androidx.compose.foundation.layout.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Animation
import androidx.compose.material.icons.filled.BarChart
import androidx.compose.material.icons.filled.DataArray
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.unit.DpSize
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Window
import androidx.compose.ui.window.application
import androidx.compose.ui.window.rememberWindowState
import com.campro.v5.animation.MotionLawEngine
import com.campro.v5.animation.MotionLawGenerator
import com.campro.v5.data.litvin.LitvinUserParams
import com.campro.v5.data.litvin.ProfileSolverMode
import com.campro.v5.data.litvin.RampProfile
import com.campro.v5.layout.LayoutManager
import com.campro.v5.layout.rememberLayoutManager
import com.campro.v5.ui.*
import com.campro.v5.ui.ModernTileLayout
import com.google.gson.Gson
import java.io.BufferedReader
import java.io.InputStreamReader

fun main(args: Array<String>) {
    // Handle CLI mode for motion law generation
    if (args.contains("--input") && args.contains("--output")) {
        handleCliMode(args)
        return
    }

    val testingMode = args.contains("--testing-mode")
    val enableAgent = args.contains("--enable-agent")

    // Basic CLI parser for key runtime config flags
    val argMap =
        args
            .filter { it.startsWith("--") && it.contains("=") }
            .associate {
                val idx = it.indexOf('=')
                it.substring(2, idx) to it.substring(idx + 1)
            }
    // Frontend-only runtime default: keep compute path on deterministic stubs unless overridden.
    if (System.getProperty("campro.backend.mode").isNullOrBlank()) {
        System.setProperty("campro.backend.mode", "stub")
    }
    println("[LarrickGUI] Backend mode=" + System.getProperty("campro.backend.mode"))
    val sessionId =
        argMap["session-id"] ?: java.time.format.DateTimeFormatter
            .ofPattern("yyyyMMddHHmmss")
            .format(java.time.LocalDateTime.now())
    val logLevel = argMap["log-level"] ?: System.getProperty("campro.log.level", "INFO")
    val defaultLogDir =
        java.nio.file.Paths
            .get(System.getProperty("user.home"), "CamProV5", "logs", sessionId)
            .toString()
    val logDir = argMap["log-dir"] ?: System.getProperty("campro.log.dir", defaultLogDir)
    try {
        java.nio.file.Files
            .createDirectories(
                java.nio.file.Paths
                    .get(logDir),
            )
    } catch (_: Throwable) {
    }

    // Publish to system properties for downstream components (MotionLawEngine, diagnostics, etc.)
    runCatching { System.setProperty("campro.log.level", logLevel) }
    runCatching { System.setProperty("log.level", logLevel) }
    runCatching { System.setProperty("campro.log.dir", logDir) }

    // Optional RNG seed passthrough for determinism across runs
    argMap["rng-seed"]?.let { System.setProperty("campro.rng.seed", it) }
    // Ensure app version is visible to SessionInfo and diagnostics
    runCatching { System.setProperty("campro.version", System.getProperty("campro.version") ?: "0.9.0-beta") }

    // Initialize Rust logger early (graceful no-op if native unavailable)
    try {
        com.campro.v5.animation.LitvinNativeStubs
            .initRustLoggerNative(sessionId, logLevel, logDir)
        println("[LarrickGUI] Rust logger initialized: sessionId=$sessionId level=$logLevel dir=$logDir")
    } catch (e: Throwable) {
        println("[LarrickGUI] Rust logger init failed (continuing without native logs): ${e.message}")
    }

    // One-click support bundle creation and exit
    if (args.contains("--make-support-bundle")) {
        val jsonDirs =
            (argMap["json-dirs"] ?: "")
                .split(';', ',')
                .map { it.trim() }
                .filter { it.isNotEmpty() }
        val out =
            com.campro.v5.support.SupportBundle
                .createSupportBundle(sessionId, logDir, jsonDirs)
        println("[LarrickGUI] Support bundle created at: $out")
        return
    }

    // Inform about native availability (do not crash UI)
    if (!com.campro.v5.animation.LitvinNativeStubs
            .isNativeAvailable()
    ) {
        val envDir = System.getenv("FEA_ENGINE_LIB_DIR") ?: "(unset)"
        val os = System.getProperty("os.name")
        val arch = System.getProperty("os.arch")
        println("[LarrickGUI] Native engine DLL not loaded. OS=$os ARCH=$arch FEA_ENGINE_LIB_DIR=$envDir")
        println("[LarrickGUI] The app will run with limited functionality. Ensure fea_engine.dll and dependencies are accessible.")
    }

    // Initialize command processor if in testing mode
    val commandProcessor = if (testingMode) CommandProcessor() else null
    commandProcessor?.start()

    // Decide which windows to show at startup for multi-window mode
    val windowsCsv = argMap["windows"]?.lowercase()?.trim()
    val visibleSet: Set<String> =
        when {
            !windowsCsv.isNullOrBlank() ->
                windowsCsv
                    .split(',', ';')
                    .map { it.trim() }
                    .filter { it.isNotEmpty() }
                    .toSet()
            else -> emptySet()
        }

    val initialVisibility =
        mapOf(
            "parameters" to (
                argMap["show-parameters"] == "true" ||
                    (visibleSet.isNotEmpty() && "parameters" in visibleSet) ||
                    visibleSet.isEmpty()
                ),
            "animation" to (argMap["show-animation"] == "true" || "animation" in visibleSet),
            "plots" to (argMap["show-plots"] == "true" || "plots" in visibleSet),
            "data" to (argMap["show-data"] == "true" || "data" in visibleSet),
            "static" to (argMap["show-static"] == "true" || "static" in visibleSet || argMap["static-profiles-window"] == "true"),
        )

    // Enable multi-window if explicitly requested, or if any window flags are present
    val multiWindowEnabled =
        args.contains("--multi-window") ||
            !windowsCsv.isNullOrBlank() ||
            listOf(
                "show-parameters",
                "show-animation",
                "show-plots",
                "show-data",
                "show-static",
                "static-profiles-window",
            ).any { key -> argMap[key] != null }

    if (multiWindowEnabled) {
        // Seed shared visibility and parameters before application starts
        initialVisibility.forEach { (k, v) ->
            com.campro.v5.SharedAppState.windowVisibility[k] = v
        }
        com.campro.v5.SharedAppState.parameters = emptyMap()
    }

    application {
        if (multiWindowEnabled) {
            val layoutManager = rememberLayoutManager()
            com.campro.v5.window.WindowsController(
                testingMode = testingMode,
                layoutManager = layoutManager,
            )
        }
        val windowState =
            rememberWindowState(
                size = DpSize(1400.dp, 1000.dp),
            )

        if (!multiWindowEnabled) {
            Window(
                onCloseRequest = {
                    commandProcessor?.stop()
                    exitApplication()
                },
                title = "Larrick Multi GUI",
                state = windowState,
            ) {
                // Handle window resize events to trigger responsive updates
                LaunchedEffect(windowState.size) {
                    // Force recomposition of responsive components when window size changes
                    println("DEBUG: Window resized to ${windowState.size.width}x${windowState.size.height}")
                }
                val layoutManager = rememberLayoutManager()
                val density = LocalDensity.current

                // Update layout manager when window size changes
                LaunchedEffect(windowState.size) {
                    layoutManager.updateWindowSize(
                        windowState.size.width,
                        windowState.size.height,
                    )
                    println("[UI] window size dp=${windowState.size}")
                }
                // React to per-monitor DPI changes and font scale
                LaunchedEffect(density.density, density.fontScale) {
                    layoutManager.updateDensityFactor(density.density)
                    println("[UI] density=${density.density} fontScale=${density.fontScale}")
                }

                CamProV5App(
                    testingMode = testingMode,
                    enableAgent = enableAgent,
                    layoutManager = layoutManager,
                )

                // Report UI initialization event if in testing mode
                if (testingMode) {
                    println("EVENT:{\"type\":\"ui_initialized\",\"component\":\"MainWindow\"}")
                    // Auto-exit shortly after initialization to support automated verification
                    LaunchedEffect(Unit) {
                        try {
                            kotlinx.coroutines.delay(1500)
                        } catch (_: Throwable) {
                        }
                        println("EVENT:{\"type\":\"ui_exit\",\"component\":\"MainWindow\"}")
                        exitApplication()
                    }
                }
            }
        }
    }
}

@Composable
@Preview
fun CamProV5App(testingMode: Boolean = false, enableAgent: Boolean = false, layoutManager: LayoutManager = rememberLayoutManager()) {
    MaterialTheme {
        var animationStarted by remember { mutableStateOf(false) }
        var allParameters by remember { mutableStateOf(mapOf<String, String>()) }

        // Initialize MotionLawEngine with default parameters on first composition
        LaunchedEffect(Unit) {
            if (allParameters.isEmpty()) {
                // Load default parameters from ParameterInputForm
                val defaultParams =
                    mapOf(
                        "Piston Diameter" to "70.0",
                        "Stroke" to "20",
                        "Chamber CC" to "5.0",
                        "TDC Angle" to "90",
                        "BDC Dwell" to "8",
                        "TDC Dwell" to "12",
                        "Enable Smoothing" to "1",
                        "Cam Timestep" to "1.0",
                        "Rod Length" to "40",
                        "TDC Offset" to "40.0",
                        "Cycle Ratio" to "2",
                        "Envelope Wall Thickness" to "10.0",
                        "Piston Mass" to "0.2",
                        "Manifold Pressure" to "101325.0",
                        "Ignition Timing BTDC" to "15.0",
                        "Ignition Duration" to "1.0",
                        "Equivalence Ratio (phi)" to "1.0",
                        "Gamma (Air)" to "1.4",
                        "Initial Temp BDC" to "300.0",
                        "Fuel Type" to "Diesel",
                        "IVO deg ABD" to "0.0",
                        "IVC deg ABD" to "15.0",
                        "EVO deg BBD" to "15.0",
                        "EVC deg ABD" to "0.0",
                        "Assembly RPM" to "1000",
                        "Mount Mass" to "5.0",
                        "Mount Stiffness X" to "1e6",
                        "Mount Stiffness Y" to "1e6",
                        "Mount Damping Ratio X" to "0.05",
                        "Mount Damping Ratio Y" to "0.05",
                        "Cam Material" to "Steel",
                        "Rod Material" to "Steel",
                        "Piston Material" to "Aluminum",
                        "Envelope Material" to "Steel",
                        "Profile Solver Mode" to "Piecewise",
                        "Sampling Step" to "1.0",
                        "Journal Radius" to "10.0",
                        "Journal Phase Beta" to "0.0",
                        "Up Fraction" to "0.5",
                        "Ramp Before TDC" to "5.0",
                        "Ramp After TDC" to "5.0",
                        "Ramp Before BDC" to "5.0",
                        "Ramp After BDC" to "5.0",
                        "Acceleration Limit" to "1000.0",
                        "Jerk Limit" to "10000.0",
                    )
                allParameters = defaultParams
                MotionLawEngine.getInstance().updateParameters(defaultParams)
            }
        }

        // Simplified architecture - removed complex management layers

        // Use modern tile-based environment for better UX and scaling
        ModernTileLayout(
            testingMode = testingMode,
            animationStarted = animationStarted,
            allParameters = allParameters,
            layoutManager = layoutManager,
            onParametersChanged = { parameters ->
                allParameters = parameters
                // Update the global MotionLawEngine singleton with parameter changes
                MotionLawEngine.getInstance().updateParameters(parameters)
                if (parameters.containsKey("animationStarted") && parameters["animationStarted"] == "true") {
                    animationStarted = true
                }
            },
        )
    }
}

@Composable
private fun ResizablePanelStandardLayout(
    testingMode: Boolean,
    animationStarted: Boolean,
    allParameters: Map<String, String>,
    layoutManager: LayoutManager,
    onParametersChanged: (Map<String, String>) -> Unit,
) {
    val spacing = layoutManager.getAppropriateSpacing()

    Column(
        modifier = Modifier.fillMaxSize().padding(spacing),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Text(
            "CamProV5 - Cycloidal Animation Generator",
            style = MaterialTheme.typography.headlineMedium,
        )

        Spacer(modifier = Modifier.height(spacing))

        // Parameter Input Form with resizable panel
        ResizablePanel(
            panelId = "parameter_panel",
            modifier = Modifier.fillMaxWidth(),
            initialWidth = 400.dp,
            initialHeight = if (layoutManager.shouldUseCompactMode()) 280.dp else 360.dp,
            minWidth = 280.dp,
            minHeight = 200.dp,
            maxWidth = 2000.dp,
            maxHeight = 1400.dp,
            title = "Parameters",
        ) {
            ScrollableParameterInputForm(
                testingMode = testingMode,
                onParametersChanged = onParametersChanged,
                layoutManager = layoutManager,
            )
        }

        if (animationStarted) {
            Spacer(modifier = Modifier.height(spacing))

            // Resizable panels for widgets
            Column(
                modifier = Modifier.fillMaxWidth().weight(1f),
                verticalArrangement = Arrangement.spacedBy(spacing),
            ) {
                // Top row with resizable Animation and Plot panels
                Row(
                    modifier = Modifier.fillMaxWidth().weight(2f),
                    horizontalArrangement = Arrangement.spacedBy(spacing),
                ) {
                    // Animation Widget Panel
                    ResizablePanel(
                        panelId = "animation_panel",
                        modifier = Modifier.weight(1f),
                        initialWidth = 400.dp,
                        initialHeight = if (layoutManager.shouldUseCompactMode()) 280.dp else 360.dp,
                        minWidth = 280.dp,
                        minHeight = 200.dp,
                        maxWidth = 2000.dp,
                        maxHeight = 1400.dp,
                        title = "Animation",
                    ) {
                        ScrollableAnimationWidget(
                            parameters = allParameters,
                            testingMode = testingMode,
                        )
                    }

                    // Plot Carousel Widget Panel
                    ResizablePanel(
                        panelId = "plot_panel",
                        modifier = Modifier.weight(1f),
                        initialWidth = 400.dp,
                        initialHeight = if (layoutManager.shouldUseCompactMode()) 280.dp else 360.dp,
                        minWidth = 280.dp,
                        minHeight = 200.dp,
                        maxWidth = 2000.dp,
                        maxHeight = 1400.dp,
                        title = "Plots",
                    ) {
                        ScrollablePlotCarouselWidget(
                            parameters = allParameters,
                            testingMode = testingMode,
                        )
                    }
                }

                // Bottom row with resizable Data Display panel
                ResizablePanel(
                    panelId = "data_panel",
                    modifier = Modifier.fillMaxWidth().weight(1f),
                    initialWidth = 1200.dp,
                    initialHeight = 300.dp,
                    minWidth = 600.dp,
                    minHeight = 150.dp,
                    maxWidth = 1600.dp,
                    maxHeight = 500.dp,
                    title = "Data Display",
                ) {
                    ScrollableDataDisplayPanel(
                        parameters = allParameters,
                        testingMode = testingMode,
                    )
                }
            }
        }
    }
}

@Composable
private fun StandardLayout(
    testingMode: Boolean,
    animationStarted: Boolean,
    allParameters: Map<String, String>,
    layoutManager: LayoutManager,
    onParametersChanged: (Map<String, String>) -> Unit,
) {
    val spacing = layoutManager.getAppropriateSpacing()

    Column(
        modifier = Modifier.fillMaxSize().padding(spacing),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Text(
            "CamProV5 - Cycloidal Animation Generator",
            style = MaterialTheme.typography.headlineMedium,
        )

        Spacer(modifier = Modifier.height(spacing))

        // Parameter Input Form with dynamic sizing
        Box(
            modifier =
            Modifier
                .fillMaxWidth()
                .weight(0.3f),
        ) {
            ParameterInputForm(
                testingMode = testingMode,
                onParametersChanged = onParametersChanged,
                layoutManager = layoutManager,
            )
        }

        Spacer(modifier = Modifier.height(spacing))

        Column(
            modifier = Modifier.fillMaxWidth().weight(0.7f),
            verticalArrangement = Arrangement.spacedBy(spacing),
        ) {
            // Responsive widget layout - always visible
            if (layoutManager.shouldUseCompactMode()) {
                // Stacked layout for compact mode
                if (!animationStarted) {
                    // Empty state for compact mode
                    Card(modifier = Modifier.fillMaxWidth().weight(1f)) {
                        EmptyStateWidget(
                            message = "Visualizations will appear here after parameters are set",
                            icon = Icons.Default.Animation,
                        )
                    }
                } else {
                    // Stacked layout for compact mode with data
                    CompactWidgetLayout(testingMode, allParameters, spacing, animationStarted)
                }
            } else {
                // Side-by-side layout for standard mode
                if (!animationStarted) {
                    // Empty state for standard mode
                    Card(modifier = Modifier.fillMaxWidth().weight(1f)) {
                        EmptyStateWidget(
                            message = "Visualizations will appear here after parameters are set",
                            icon = Icons.Default.Animation,
                        )
                    }
                } else {
                    // Side-by-side layout for standard mode with data
                    StandardWidgetLayout(testingMode, allParameters, spacing, animationStarted)
                }
            }
        }
    }
}

@Composable
private fun CompactWidgetLayout(
    testingMode: Boolean,
    allParameters: Map<String, String>,
    spacing: androidx.compose.ui.unit.Dp,
    animationStarted: Boolean = true,
) {
    Column(
        modifier = Modifier.fillMaxWidth(),
        verticalArrangement = Arrangement.spacedBy(spacing),
    ) {
        Card(
            modifier = Modifier.fillMaxWidth().weight(0.35f),
        ) {
            if (!animationStarted) {
                EmptyStateWidget(
                    message = "Animation will appear here after parameters are set",
                    icon = Icons.Default.Animation,
                )
            } else {
                AnimationWidget(
                    parameters = allParameters,
                    testingMode = testingMode,
                )
            }
        }

        Card(
            modifier = Modifier.fillMaxWidth().weight(0.35f),
        ) {
            if (!animationStarted) {
                EmptyStateWidget(
                    message = "Plots will appear here after parameters are set",
                    icon = Icons.Default.BarChart,
                )
            } else {
                PlotCarouselWidget(
                    parameters = allParameters,
                    testingMode = testingMode,
                )
            }
        }

        Card(
            modifier = Modifier.fillMaxWidth().weight(0.3f),
        ) {
            if (!animationStarted) {
                EmptyStateWidget(
                    message = "Data will appear here after parameters are set",
                    icon = Icons.Default.DataArray,
                )
            } else {
                DataDisplayPanel(
                    parameters = allParameters,
                    testingMode = testingMode,
                )
            }
        }
    }
}

@Composable
private fun StandardWidgetLayout(
    testingMode: Boolean,
    allParameters: Map<String, String>,
    spacing: androidx.compose.ui.unit.Dp,
    animationStarted: Boolean = true,
) {
    Column(
        modifier = Modifier.fillMaxSize(),
    ) {
        // Top row with Animation and Plot Widgets
        Row(
            modifier = Modifier.fillMaxWidth().weight(0.65f),
            horizontalArrangement = Arrangement.spacedBy(spacing),
        ) {
            // Animation Widget
            Card(
                modifier = Modifier.weight(1f).fillMaxHeight(),
            ) {
                if (!animationStarted) {
                    EmptyStateWidget(
                        message = "Animation will appear here after parameters are set",
                        icon = Icons.Default.Animation,
                    )
                } else {
                    AnimationWidget(
                        parameters = allParameters,
                        testingMode = testingMode,
                    )
                }
            }

            // Plot Carousel Widget
            Card(
                modifier = Modifier.weight(1f).fillMaxHeight(),
            ) {
                if (!animationStarted) {
                    EmptyStateWidget(
                        message = "Plots will appear here after parameters are set",
                        icon = Icons.Default.BarChart,
                    )
                } else {
                    PlotCarouselWidget(
                        parameters = allParameters,
                        testingMode = testingMode,
                    )
                }
            }
        }

        Spacer(modifier = Modifier.height(spacing))

        // Bottom row with Data Display Panel
        Card(
            modifier = Modifier.fillMaxWidth().weight(0.35f),
        ) {
            if (!animationStarted) {
                EmptyStateWidget(
                    message = "Data will appear here after parameters are set",
                    icon = Icons.Default.DataArray,
                )
            } else {
                DataDisplayPanel(
                    parameters = allParameters,
                    testingMode = testingMode,
                )
            }
        }
    }
}

// Validate input parameters
fun validateInput(baseCircleRadius: String, rollingCircleRadius: String, tracingPointDistance: String): String? {
    try {
        val baseRadius = baseCircleRadius.toDouble()
        if (baseRadius <= 0) {
            return "Base circle radius must be positive"
        }

        val rollingRadius = rollingCircleRadius.toDouble()
        if (rollingRadius <= 0) {
            return "Rolling circle radius must be positive"
        }

        val tracingDistance = tracingPointDistance.toDouble()
        if (tracingDistance < 0) {
            return "Tracing point distance must be non-negative"
        }

        return null
    } catch (e: NumberFormatException) {
        return "All parameters must be valid numbers"
    }
}

/**
 * Command processor for handling commands from the testing bridge.
 * This class processes commands sent from the KotlinUIBridge and routes them to the appropriate components.
 */
class CommandProcessor {
    private var isRunning = false
    private var inputThread: Thread? = null

    fun start() {
        if (isRunning) return

        isRunning = true
        inputThread =
            Thread {
                val reader = BufferedReader(InputStreamReader(System.`in`))

                while (isRunning) {
                    try {
                        val line = reader.readLine()
                        if (line != null) {
                            processCommand(line)
                        }
                    } catch (e: Exception) {
                        if (isRunning) {
                            println("ERROR:{\"type\":\"command_processing_error\",\"message\":\"${e.message}\"}")
                        }
                    }
                }
            }
        inputThread?.start()
    }

    fun stop() {
        isRunning = false
        inputThread?.interrupt()
    }

    private fun processCommand(command: String) {
        try {
            // Check if the command starts with COMMAND: prefix and extract the JSON part
            val jsonPart =
                if (command.startsWith("COMMAND:")) {
                    command.substring("COMMAND:".length)
                } else {
                    command
                }

            val gson = Gson()
            val commandData = gson.fromJson(jsonPart, Map::class.java)

            when (commandData["command"]) {
                "click" -> {
                    val component = commandData["params"]?.let { (it as? Map<*, *>)?.get("component") as? String } ?: ""

                    // Process click command
                    println("EVENT:{\"type\":\"command_executed\",\"command\":\"click\",\"component\":\"$component\"}")
                }
                "set_value" -> {
                    val params = commandData["params"] as? Map<*, *>
                    val component = params?.get("component") as? String ?: ""
                    val value = params?.get("value") as? String ?: ""

                    // Process set value command
                    println(
                        "EVENT:{\"type\":\"command_executed\",\"command\":\"set_value\",\"component\":\"$component\",\"value\":\"$value\"}",
                    )
                }
                "select_tab" -> {
                    val params = commandData["params"] as? Map<*, *>
                    val component = params?.get("component") as? String ?: ""
                    val value = params?.get("value") as? String ?: ""

                    // Process tab selection command
                    println(
                        "EVENT:{\"type\":\"command_executed\",\"command\":\"select_tab\",\"component\":\"$component\",\"value\":\"$value\"}",
                    )
                }
                "gesture" -> {
                    val params = commandData["params"] as? Map<*, *>
                    val component = params?.get("component") as? String ?: ""
                    val action = params?.get("action") as? String ?: ""
                    val offsetX = params?.get("offset_x") as? String ?: ""
                    val offsetY = params?.get("offset_y") as? String ?: ""

                    // Process gesture command
                    println(
                        "EVENT:{\"type\":\"command_executed\",\"command\":\"gesture\",\"component\":\"$component\",\"action\":\"$action\",\"offset_x\":\"$offsetX\",\"offset_y\":\"$offsetY\"}",
                    )
                }
                "get_state" -> {
                    val params = commandData["params"] as? Map<*, *>
                    val component = params?.get("component") as? String ?: ""

                    // Return component state
                    println("EVENT:{\"type\":\"command_executed\",\"command\":\"get_state\",\"component\":\"$component\",\"state\":{}}")
                }
                "reset" -> {
                    val params = commandData["params"] as? Map<*, *>
                    val component = params?.get("component") as? String ?: ""

                    // Process reset command
                    println("EVENT:{\"type\":\"command_executed\",\"command\":\"reset\",\"component\":\"$component\"}")
                }
                "export" -> {
                    val params = commandData["params"] as? Map<*, *>
                    val component = params?.get("component") as? String ?: ""
                    val format = params?.get("format") as? String ?: ""
                    val filePath = params?.get("file_path") as? String ?: ""

                    // Process export command
                    println(
                        "EVENT:{\"type\":\"command_executed\",\"command\":\"export\",\"component\":\"$component\",\"format\":\"$format\",\"file_path\":\"$filePath\"}",
                    )
                }
                "import" -> {
                    val params = commandData["params"] as? Map<*, *>
                    val component = params?.get("component") as? String ?: ""
                    val filePath = params?.get("file_path") as? String ?: ""

                    // Process import command
                    println(
                        "EVENT:{\"type\":\"command_executed\",\"command\":\"import\",\"component\":\"$component\",\"file_path\":\"$filePath\"}",
                    )
                }
                "generate" -> {
                    val params = commandData["params"] as? Map<*, *>
                    val component = params?.get("component") as? String ?: ""
                    val type = params?.get("type") as? String ?: ""

                    // Process generate command
                    println(
                        "EVENT:{\"type\":\"command_executed\",\"command\":\"generate\",\"component\":\"$component\",\"type\":\"$type\"}",
                    )
                }
                else -> {
                    println("EVENT:{\"type\":\"error\",\"message\":\"Unknown command: ${commandData["command"]}\"}")
                }
            }
        } catch (e: Exception) {
            println("EVENT:{\"type\":\"error\",\"message\":\"Error processing command: ${e.message}\"}")
        }
    }
}

/**
 * Handle CLI mode for motion law generation.
 * Reads input JSON file, generates motion law, and writes output JSON file.
 */
private fun handleCliMode(args: Array<String>) {
    try {
        // Parse command line arguments
        val inputIndex = args.indexOf("--input")
        val outputIndex = args.indexOf("--output")

        if (inputIndex == -1 || outputIndex == -1 || inputIndex + 1 >= args.size || outputIndex + 1 >= args.size) {
            System.err.println("Usage: --input <input.json> --output <output.json>")
            System.exit(1)
        }

        val inputFile = args[inputIndex + 1]
        val outputFile = args[outputIndex + 1]

        // Read input JSON
        val inputJson = java.io.File(inputFile).readText()
        val gson = Gson()

        // Parse input parameters (assuming it's a map of parameters)
        val inputParams = gson.fromJson(inputJson, Map::class.java) as Map<String, Any>

        // Convert to LitvinUserParams
        val litvinParams = com.campro.v5.data.litvin.LitvinUserParams(
            samplingStepDeg = (inputParams["samplingStepDeg"] as? Number)?.toDouble() ?: 1.0,
            strokeLengthMm = (inputParams["strokeLengthMm"] as? Number)?.toDouble() ?: 100.0,
            dwellTdcDeg = (inputParams["dwellTdcDeg"] as? Number)?.toDouble() ?: 4.0,
            dwellBdcDeg = (inputParams["dwellBdcDeg"] as? Number)?.toDouble() ?: 3.0,
            rampBeforeTdcDeg = (inputParams["rampBeforeTdcDeg"] as? Number)?.toDouble() ?: 6.0,
            rampAfterTdcDeg = (inputParams["rampAfterTdcDeg"] as? Number)?.toDouble() ?: 5.0,
            rampBeforeBdcDeg = (inputParams["rampBeforeBdcDeg"] as? Number)?.toDouble() ?: 7.0,
            rampAfterBdcDeg = (inputParams["rampAfterBdcDeg"] as? Number)?.toDouble() ?: 4.0,
            upFraction = (inputParams["upFraction"] as? Number)?.toDouble() ?: 110.0 / 180.0,
            rpm = (inputParams["rpm"] as? Number)?.toDouble() ?: 3000.0,
            rodLength = (inputParams["rodLength"] as? Number)?.toDouble() ?: 100.0,
            camR0 = (inputParams["camR0"] as? Number)?.toDouble() ?: 40.0,
            camKPerUnit = (inputParams["camKPerUnit"] as? Number)?.toDouble() ?: 1.0,
            centerDistanceBias = (inputParams["centerDistanceBias"] as? Number)?.toDouble() ?: 50.0,
            centerDistanceScale = (inputParams["centerDistanceScale"] as? Number)?.toDouble() ?: 1.0,
            profileSolverMode = when (inputParams["profileSolverMode"] as? String) {
                "Collocation" -> com.campro.v5.data.litvin.ProfileSolverMode.Collocation
                else -> com.campro.v5.data.litvin.ProfileSolverMode.Piecewise
            },
            rampProfile = when (inputParams["rampProfile"] as? String) {
                "Cycloidal" -> com.campro.v5.data.litvin.RampProfile.Cycloidal
                "S5" -> com.campro.v5.data.litvin.RampProfile.S5
                else -> com.campro.v5.data.litvin.RampProfile.S5
            },
        )

        // Generate motion law using MotionLawGenerator
        val motionSamples = com.campro.v5.animation.MotionLawGenerator.generateMotion(litvinParams)

        // Convert to JSON and write output
        val outputJson = gson.toJson(motionSamples)
        java.io.File(outputFile).writeText(outputJson)

        println("Motion law generated successfully: ${motionSamples.samples.size} samples")
    } catch (e: Exception) {
        System.err.println("Error in CLI mode: ${e.message}")
        e.printStackTrace()
        System.exit(1)
    }
}
