package com.campro.v5

import kotlinx.coroutines.delay

/**
 * Wait for a condition to be true with retries.
 *
 * @param maxAttempts Maximum number of attempts to check the condition
 * @param delayMs Delay between attempts in milliseconds
 * @param condition The condition to check
 * @return True if the condition was met, false if max attempts were reached
 */
suspend fun waitForCondition(maxAttempts: Int = 10, delayMs: Long = 100, condition: () -> Boolean): Boolean {
    repeat(maxAttempts) { attempt ->
        if (condition()) {
            return true
        }

        if (attempt < maxAttempts - 1) {
            delay(delayMs)
        }
    }

    return false
}

/**
 * Wait for a condition to be true with retries and throw an exception if the condition is not met.
 *
 * @param maxAttempts Maximum number of attempts to check the condition
 * @param delayMs Delay between attempts in milliseconds
 * @param message Message to include in the exception if the condition is not met
 * @param condition The condition to check
 * @throws AssertionError if the condition is not met after max attempts
 */
suspend fun waitForConditionOrFail(
    maxAttempts: Int = 10,
    delayMs: Long = 100,
    message: String = "Condition not met after $maxAttempts attempts",
    condition: () -> Boolean,
) {
    val result = waitForCondition(maxAttempts, delayMs, condition)
    if (!result) {
        throw AssertionError(message)
    }
}
