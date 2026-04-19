package com.campro.v5.accessibility

import org.junit.jupiter.api.*
import org.junit.jupiter.api.Assertions.*

/**
 * Tests for AccessibilityEnhancer.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class AccessibilityEnhancerTest {

    @BeforeEach
    fun setup() {
        // Reset to default settings
        AccessibilityEnhancer.updateSettings(AccessibilityEnhancer.AccessibilitySettings())
    }

    @Test
    fun `test default accessibility settings`() {
        // When
        val settings = AccessibilityEnhancer.accessibilitySettings

        // Then
        assertFalse(settings.highContrast)
        assertFalse(settings.largeText)
        assertTrue(settings.screenReaderSupport)
        assertTrue(settings.keyboardNavigation)
        assertTrue(settings.focusIndicators)
        assertFalse(settings.reducedMotion)
    }

    @Test
    fun `test update accessibility settings`() {
        // Given
        val newSettings = AccessibilityEnhancer.AccessibilitySettings(
            highContrast = true,
            largeText = true,
            screenReaderSupport = false,
            keyboardNavigation = false,
            focusIndicators = false,
            reducedMotion = true,
        )

        // When
        AccessibilityEnhancer.updateSettings(newSettings)

        // Then
        val currentSettings = AccessibilityEnhancer.accessibilitySettings
        assertTrue(currentSettings.highContrast)
        assertTrue(currentSettings.largeText)
        assertFalse(currentSettings.screenReaderSupport)
        assertFalse(currentSettings.keyboardNavigation)
        assertFalse(currentSettings.focusIndicators)
        assertTrue(currentSettings.reducedMotion)
    }

    @Test
    fun `test get high contrast colors`() {
        // When
        val colors = AccessibilityEnhancer.getHighContrastColors()

        // Then
        assertNotNull(colors.primary)
        assertNotNull(colors.secondary)
        assertNotNull(colors.error)
        assertNotNull(colors.background)
        assertNotNull(colors.surface)
        assertNotNull(colors.onPrimary)
        assertNotNull(colors.onSecondary)
        assertNotNull(colors.onError)
        assertNotNull(colors.onBackground)
        assertNotNull(colors.onSurface)
    }

    @Test
    fun `test get large text scale`() {
        // Given - default settings (large text disabled)
        assertEquals(1.0f, AccessibilityEnhancer.getLargeTextScale())

        // When - enable large text
        AccessibilityEnhancer.updateSettings(
            AccessibilityEnhancer.AccessibilitySettings(largeText = true),
        )

        // Then
        assertEquals(1.2f, AccessibilityEnhancer.getLargeTextScale())
    }
}

/**
 * Tests for FocusManager.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class FocusManagerTest {

    @BeforeEach
    fun setup() {
        // Reset focus order
        FocusManager._focusOrder.value = FocusManager.FocusOrder()
    }

    @Test
    fun `test add to focus order`() {
        // Given
        val elementId = "testElement"

        // When
        FocusManager.addToFocusOrder(elementId)

        // Then
        val focusOrder = FocusManager.focusOrder
        assertTrue(focusOrder.elements.contains(elementId))
        assertEquals(1, focusOrder.elements.size)
    }

    @Test
    fun `test add duplicate to focus order`() {
        // Given
        val elementId = "testElement"
        FocusManager.addToFocusOrder(elementId)

        // When
        FocusManager.addToFocusOrder(elementId)

        // Then
        val focusOrder = FocusManager.focusOrder
        assertEquals(1, focusOrder.elements.size) // Should not add duplicate
        assertEquals(1, focusOrder.elements.count { it == elementId })
    }

    @Test
    fun `test remove from focus order`() {
        // Given
        val elementId = "testElement"
        FocusManager.addToFocusOrder(elementId)
        assertEquals(1, FocusManager.focusOrder.elements.size)

        // When
        FocusManager.removeFromFocusOrder(elementId)

        // Then
        val focusOrder = FocusManager.focusOrder
        assertFalse(focusOrder.elements.contains(elementId))
        assertEquals(0, focusOrder.elements.size)
    }

    @Test
    fun `test move to next focusable element`() {
        // Given
        FocusManager.addToFocusOrder("element1")
        FocusManager.addToFocusOrder("element2")
        FocusManager.addToFocusOrder("element3")
        assertEquals(0, FocusManager.focusOrder.currentIndex)

        // When
        FocusManager.moveToNext()

        // Then
        assertEquals(1, FocusManager.focusOrder.currentIndex)

        // When - move to last element
        FocusManager.moveToNext()

        // Then
        assertEquals(2, FocusManager.focusOrder.currentIndex)

        // When - wrap around to first element
        FocusManager.moveToNext()

        // Then
        assertEquals(0, FocusManager.focusOrder.currentIndex)
    }

    @Test
    fun `test move to previous focusable element`() {
        // Given
        FocusManager.addToFocusOrder("element1")
        FocusManager.addToFocusOrder("element2")
        FocusManager.addToFocusOrder("element3")
        assertEquals(0, FocusManager.focusOrder.currentIndex)

        // When - wrap around to last element
        FocusManager.moveToPrevious()

        // Then
        assertEquals(2, FocusManager.focusOrder.currentIndex)

        // When
        FocusManager.moveToPrevious()

        // Then
        assertEquals(1, FocusManager.focusOrder.currentIndex)

        // When
        FocusManager.moveToPrevious()

        // Then
        assertEquals(0, FocusManager.focusOrder.currentIndex)
    }

    @Test
    fun `test focus order data class`() {
        // Given
        val elements = listOf("element1", "element2", "element3")
        val currentIndex = 1

        // When
        val focusOrder = FocusManager.FocusOrder(elements, currentIndex)

        // Then
        assertEquals(elements, focusOrder.elements)
        assertEquals(currentIndex, focusOrder.currentIndex)
    }
}
