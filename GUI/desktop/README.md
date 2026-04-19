# CamProV5 Desktop Application

This module contains the desktop version of the CamProV5 application, built using Compose for Desktop.

## Overview

The desktop application is a port of the Android UI to the desktop platform, allowing the application to be run on Windows, macOS, and Linux. It uses the same UI components and logic as the Android application, but with a desktop-specific entry point and build configuration.

## Building the Application

To build the desktop application, run the following command from the project root:

```bash
./gradlew :desktop:build
```

This will compile the Kotlin code and create a JAR file in the `desktop/build/libs` directory.

## Running the Application

To run the desktop application, use the following command:

```bash
./gradlew :desktop:run
```

## Testing Mode

The desktop application supports a testing mode, which can be enabled by passing the `--testing-mode` flag:

```bash
java -jar desktop/build/libs/CamProV5-desktop.jar --testing-mode
```

In testing mode, the application will:

1. Print events to stdout in a format that can be parsed by the testing bridge
2. Accept commands from stdin in a format that can be sent by the testing bridge
3. Enable additional debugging features

## Agent Mode

The desktop application also supports an agent mode, which can be enabled by passing the `--enable-agent` flag:

```bash
java -jar desktop/build/libs/CamProV5-desktop.jar --enable-agent
```

In agent mode, the application will:

1. Connect to the agent controller
2. Allow the agent to monitor and control the UI
3. Enable additional debugging features

## Integration with Testing Framework

The desktop application is designed to be used with the testing framework, which can launch and communicate with the desktop UI. This allows for in-the-loop testing with the actual production UI instead of the PyQt5 placeholder.

To use the desktop application with the testing framework, run the following command:

```bash
python -m campro.testing.start_agent_session --use-kotlin-ui
```

This will:

1. Check if the desktop application is available
2. Launch the desktop application in testing mode
3. Connect to the desktop application via stdin/stdout
4. Allow the agent to monitor and control the UI

## Implementation Details

The desktop application is implemented using Compose for Desktop, which is a port of Jetpack Compose to the desktop platform. It uses the same UI components and logic as the Android application, but with a desktop-specific entry point and build configuration.

The main entry point is `DesktopMain.kt`, which:

1. Parses command line arguments
2. Creates a window
3. Renders the UI
4. Handles events and commands

The UI components are shared with the Android application, allowing for code reuse and consistent behavior across platforms.

## Future Improvements

Future improvements to the desktop application could include:

1. Better integration with the testing framework
2. More desktop-specific features
3. Improved performance
4. Better error handling
5. More comprehensive documentation