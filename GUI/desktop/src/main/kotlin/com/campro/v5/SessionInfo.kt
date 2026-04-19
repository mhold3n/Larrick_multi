package com.campro.v5

import java.util.UUID

/**
 * Provides information about the current application session.
 * A stable sessionId is used for correlating logs/diagnostics during a run.
 */
object SessionInfo {
    /**
     * Session id is determined once per JVM run. If a system property
     * `session.id` is provided, it will be used; otherwise a random UUID is generated.
     */
    val sessionId: String by lazy {
        val prop = System.getProperty("session.id")?.trim()
        if (!prop.isNullOrBlank()) prop else UUID.randomUUID().toString()
    }
}
