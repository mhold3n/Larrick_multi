package com.campro.v5.performance

import org.junit.jupiter.api.*
import org.junit.jupiter.api.Assertions.*

/**
 * Tests for PerformanceOptimizer.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class PerformanceOptimizerTest {

    @BeforeEach
    fun setup() {
        PerformanceOptimizer.resetCounters()
    }

    @Test
    fun `test track recomposition`() {
        // Given
        val componentName = "TestComponent"

        // When
        PerformanceOptimizer.trackRecomposition(componentName)
        PerformanceOptimizer.trackRecomposition(componentName)

        // Then
        val metrics = PerformanceOptimizer.performanceMetrics.value
        assertTrue(metrics.recompositionCount >= 2)
    }

    @Test
    fun `test track render time`() {
        // Given
        val componentName = "TestComponent"
        val renderTime = 50L

        // When
        PerformanceOptimizer.trackRenderTime(componentName, renderTime)

        // Then
        // Render time tracking is internal, but we can verify it doesn't throw
        assertDoesNotThrow { PerformanceOptimizer.trackRenderTime(componentName, renderTime) }
    }

    @Test
    fun `test get memory usage`() {
        // When
        val memoryUsage = PerformanceOptimizer.getMemoryUsage()

        // Then
        assertTrue(memoryUsage >= 0)
        assertTrue(memoryUsage < Long.MAX_VALUE)
    }

    @Test
    fun `test update metrics`() {
        // When
        PerformanceOptimizer.updateMetrics()

        // Then
        val metrics = PerformanceOptimizer.performanceMetrics.value
        assertTrue(metrics.lastUpdate > 0)
        assertTrue(metrics.memoryUsage >= 0)
    }

    @Test
    fun `test reset counters`() {
        // Given
        PerformanceOptimizer.trackRecomposition("TestComponent")
        PerformanceOptimizer.trackRenderTime("TestComponent", 100L)

        // When
        PerformanceOptimizer.resetCounters()

        // Then
        val metrics = PerformanceOptimizer.performanceMetrics.value
        assertEquals(0, metrics.recompositionCount)
    }

    @Test
    fun `test is performance acceptable`() {
        // When
        val isAcceptable = PerformanceOptimizer.isPerformanceAcceptable()

        // Then
        assertTrue(isAcceptable) // Should be acceptable with default state
    }

    @Test
    fun `test get performance recommendations`() {
        // When
        val recommendations = PerformanceOptimizer.getPerformanceRecommendations()

        // Then
        assertNotNull(recommendations)
        assertTrue(recommendations is List<String>)
    }
}

/**
 * Tests for MemoryManager.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class MemoryManagerTest {

    @Test
    fun `test optimize memory if needed`() {
        // When
        MemoryManager.optimizeMemoryIfNeeded()

        // Then
        // Should not throw and should complete successfully
        assertDoesNotThrow { MemoryManager.optimizeMemoryIfNeeded() }
    }

    @Test
    fun `test clear caches`() {
        // When
        MemoryManager.clearCaches()

        // Then
        // Should not throw and should complete successfully
        assertDoesNotThrow { MemoryManager.clearCaches() }
    }

    @Test
    fun `test get memory stats`() {
        // When
        val stats = MemoryManager.getMemoryStats()

        // Then
        assertTrue(stats.totalMemory > 0)
        assertTrue(stats.freeMemory >= 0)
        assertTrue(stats.usedMemory >= 0)
        assertTrue(stats.maxMemory > 0)
        assertTrue(stats.usedMemoryMB >= 0)
        assertTrue(stats.totalMemoryMB > 0)
        assertTrue(stats.memoryUsagePercentage >= 0)
        assertTrue(stats.memoryUsagePercentage <= 100)
    }
}
