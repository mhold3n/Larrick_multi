package com.campro.v5.utils

import com.fasterxml.jackson.core.JsonParser
import com.fasterxml.jackson.databind.DeserializationFeature
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.kotlin.KotlinModule
import com.fasterxml.jackson.module.kotlin.readValue
import org.slf4j.LoggerFactory
import java.nio.file.Path

/**
 * Simple JSON utility functions for the new workflow.
 * This is a working version without complex features.
 */
object SimpleJsonUtils {
    val logger = LoggerFactory.getLogger(SimpleJsonUtils::class.java)

    val objectMapper = ObjectMapper().apply {
        registerModule(KotlinModule.Builder().build())
        configure(JsonParser.Feature.ALLOW_NON_NUMERIC_NUMBERS, true)
        configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false)
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

    /**
     * Write object to JSON file.
     */
    fun writeJsonFile(obj: Any, filePath: Path) {
        try {
            objectMapper.writeValue(filePath.toFile(), obj)
        } catch (e: Exception) {
            logger.error("Failed to write JSON to file: $filePath", e)
        }
    }

    /**
     * Read JSON file to Map.
     */
    fun readJsonFile(filePath: Path): Map<String, Any> = try {
        objectMapper.readValue(filePath.toFile(), Map::class.java) as Map<String, Any>
    } catch (e: Exception) {
        logger.error("Failed to read JSON from file: $filePath", e)
        emptyMap()
    }
}
