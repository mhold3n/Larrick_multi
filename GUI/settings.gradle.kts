import org.gradle.util.GradleVersion

// Enforce minimum Gradle version for Shadow 8.x and toolchain alignment
val requiredGradle = GradleVersion.version("8.0")
if (GradleVersion.current() < requiredGradle) {
    error(
        "This project requires Gradle ${requiredGradle.version} or newer. " +
            "Please run builds with the Gradle wrapper (./gradlew or gradlew.bat) and ensure your IDE is configured to use the wrapper and JDK 17."
    )
}

rootProject.name = "CamProV5"

include(":desktop")
include(":data-litvin")