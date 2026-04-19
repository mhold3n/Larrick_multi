package com.campro.v5.utils

import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.kotlin.KotlinModule
import com.fasterxml.jackson.module.kotlin.readValue
import org.slf4j.LoggerFactory

/**
 * JSON utility functions for the new workflow.
 * Simplified version without the problematic inline functions.
 */
object JsonUtils {
    private val logger = LoggerFactory.getLogger(JsonUtils::class.java)

    private val objectMapper = ObjectMapper().apply {
        registerModule(KotlinModule.Builder().build())
    }

    /**
     * Convert object to JSON string.
     */
    fun toJson(obj: Any): String = try {
        objectMapper.writeValueAsString(obj)
    } catch (e: Exception) {
        logger.error("Failed to convert object to JSON", e)
        "{}"
    }

    /**
     * Convert JSON string to object.
     */
    inline fun <reified T> fromJson(json: String): T? = try {
        objectMapper.readValue<T>(json)
    } catch (e: Exception) {
        logger.error("Failed to convert JSON to object", e)
        null
    }
}
