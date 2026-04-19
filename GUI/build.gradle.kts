/**
 * Aggregator build script for the parent Gradle project ':CamProV5'.
 * This exists to satisfy IDE/Gradle multi-project discovery when including ':CamProV5:desktop'.
 * No build logic is required here; all configuration lives in subprojects.
 */

plugins {
    base
    kotlin("jvm") version "1.9.21" apply false
}

description = "CamProV5 aggregator module (no build logic)"