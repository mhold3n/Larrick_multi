plugins {
    kotlin("jvm")
}

description = "CamProV5 data model and helpers for Litvin motion law"

kotlin {
    jvmToolchain(17)
}

repositories {
    mavenCentral()
    google()
}

dependencies {
    implementation("com.google.code.gson:gson:2.11.0")
    implementation("org.slf4j:slf4j-api:2.0.13")
    testImplementation(kotlin("test"))
}
