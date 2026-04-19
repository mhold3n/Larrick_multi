package com.campro.v5

import org.junit.platform.engine.discovery.DiscoverySelectors
import org.junit.platform.launcher.Launcher
import org.junit.platform.launcher.LauncherDiscoveryRequest
import org.junit.platform.launcher.core.LauncherDiscoveryRequestBuilder
import org.junit.platform.launcher.core.LauncherFactory
import org.junit.platform.launcher.listeners.SummaryGeneratingListener
import org.junit.platform.launcher.listeners.TestExecutionSummary

/**
 * Comprehensive test suite runner for the CamProV5 desktop application.
 *
 * Runs all test categories including:
 * - Unit tests
 * - Integration tests
 * - Performance tests
 * - User acceptance tests
 * - End-to-end tests
 */
object TestSuiteRunner {

    /**
     * Test categories and their descriptions.
     */
    enum class TestCategory(val displayName: String, val description: String, val packageName: String) {
        UNIT("Unit Tests", "Individual component testing", "com.campro.v5"),
        INTEGRATION("Integration Tests", "Component integration testing", "com.campro.v5.integration"),
        PERFORMANCE("Performance Tests", "Performance validation testing", "com.campro.v5.performance"),
        ACCEPTANCE("User Acceptance Tests", "User experience validation", "com.campro.v5.acceptance"),
        PIPELINE("Pipeline Tests", "Complete pipeline testing", "com.campro.v5.pipeline"),
        END_TO_END("End-to-End Tests", "Complete system testing", "com.campro.v5.integration"),
    }

    /**
     * Test execution results.
     */
    data class TestResults(
        val category: TestCategory,
        val totalTests: Long,
        val successfulTests: Long,
        val failedTests: Long,
        val skippedTests: Long,
        val executionTime: Long,
    ) {
        val successRate: Double
            get() = if (totalTests > 0) (successfulTests.toDouble() / totalTests) * 100 else 0.0

        val isSuccessful: Boolean
            get() = failedTests == 0L
    }

    /**
     * Run all test categories.
     */
    fun runAllTests(): Map<TestCategory, TestResults> {
        val results = mutableMapOf<TestCategory, TestResults>()

        println("🚀 Starting Comprehensive Test Suite for CamProV5 Desktop Application")
        println("=" * 80)

        TestCategory.values().forEach { category ->
            println("\n📋 Running ${category.displayName}")
            println("-" * 50)
            println("Description: ${category.description}")

            val result = runTestCategory(category)
            results[category] = result

            val status = if (result.isSuccessful) "✅ PASSED" else "❌ FAILED"
            println("Result: $status")
            println("Tests: ${result.successfulTests}/${result.totalTests} successful (${String.format("%.1f", result.successRate)}%)")
            println("Time: ${result.executionTime}ms")
        }

        printOverallResults(results)
        return results
    }

    /**
     * Run a specific test category.
     */
    fun runTestCategory(category: TestCategory): TestResults {
        val launcher: Launcher = LauncherFactory.create()
        val summaryListener = SummaryGeneratingListener()
        launcher.registerTestExecutionListeners(summaryListener)

        val request: LauncherDiscoveryRequest = LauncherDiscoveryRequestBuilder.request()
            .selectors(DiscoverySelectors.selectPackage(category.packageName))
            .build()

        val startTime = System.currentTimeMillis()
        launcher.execute(request)
        val endTime = System.currentTimeMillis()

        val summary: TestExecutionSummary = summaryListener.summary

        return TestResults(
            category = category,
            totalTests = summary.testsFoundCount(),
            successfulTests = summary.testsSucceededCount(),
            failedTests = summary.testsFailedCount(),
            skippedTests = summary.testsSkippedCount(),
            executionTime = endTime - startTime,
        )
    }

    /**
     * Run specific test classes.
     */
    fun runSpecificTests(testClasses: List<String>): Map<String, TestResults> {
        val results = mutableMapOf<String, TestResults>()

        println("🎯 Running Specific Test Classes")
        println("=" * 50)

        testClasses.forEach { testClass ->
            println("\n📋 Running $testClass")

            val launcher: Launcher = LauncherFactory.create()
            val summaryListener = SummaryGeneratingListener()
            launcher.registerTestExecutionListeners(summaryListener)

            val request: LauncherDiscoveryRequest = LauncherDiscoveryRequestBuilder.request()
                .selectors(DiscoverySelectors.selectClass(testClass))
                .build()

            val startTime = System.currentTimeMillis()
            launcher.execute(request)
            val endTime = System.currentTimeMillis()

            val summary: TestExecutionSummary = summaryListener.summary

            val result = TestResults(
                category = TestCategory.UNIT, // Default category
                totalTests = summary.testsFoundCount(),
                successfulTests = summary.testsSucceededCount(),
                failedTests = summary.testsFailedCount(),
                skippedTests = summary.testsSkippedCount(),
                executionTime = endTime - startTime,
            )

            results[testClass] = result

            val status = if (result.isSuccessful) "✅ PASSED" else "❌ FAILED"
            println("Result: $status")
            println("Tests: ${result.successfulTests}/${result.totalTests} successful")
            println("Time: ${result.executionTime}ms")
        }

        return results
    }

    /**
     * Print overall test results.
     */
    private fun printOverallResults(results: Map<TestCategory, TestResults>) {
        println("\n" + "=" * 80)
        println("📊 OVERALL TEST RESULTS")
        println("=" * 80)

        val totalTests = results.values.sumOf { it.totalTests }
        val totalSuccessful = results.values.sumOf { it.successfulTests }
        val totalFailed = results.values.sumOf { it.failedTests }
        val totalSkipped = results.values.sumOf { it.skippedTests }
        val totalTime = results.values.sumOf { it.executionTime }

        val overallSuccessRate = if (totalTests > 0) (totalSuccessful.toDouble() / totalTests) * 100 else 0.0
        val overallStatus = if (totalFailed == 0L) "✅ ALL TESTS PASSED" else "❌ SOME TESTS FAILED"

        println("Overall Status: $overallStatus")
        println("Total Tests: $totalTests")
        println("Successful: $totalSuccessful")
        println("Failed: $totalFailed")
        println("Skipped: $totalSkipped")
        println("Success Rate: ${String.format("%.1f", overallSuccessRate)}%")
        println("Total Time: ${totalTime}ms")

        println("\n📋 Category Breakdown:")
        results.forEach { (category, result) ->
            val status = if (result.isSuccessful) "✅" else "❌"
            println("  $status ${category.displayName}: ${result.successfulTests}/${result.totalTests} (${String.format("%.1f", result.successRate)}%)")
        }

        // Print failed tests details
        val failedCategories = results.filter { !it.value.isSuccessful }
        if (failedCategories.isNotEmpty()) {
            println("\n❌ Failed Test Categories:")
            failedCategories.forEach { (category, result) ->
                println("  - ${category.displayName}: ${result.failedTests} failed tests")
            }
        }

        // Print performance summary
        println("\n⏱️ Performance Summary:")
        results.forEach { (category, result) ->
            val avgTimePerTest = if (result.totalTests > 0) result.executionTime / result.totalTests else 0
            println("  - ${category.displayName}: ${result.executionTime}ms total, ${avgTimePerTest}ms avg per test")
        }

        println("\n" + "=" * 80)
    }

    /**
     * Generate test report.
     */
    fun generateTestReport(results: Map<TestCategory, TestResults>, outputPath: String) {
        val report = buildString {
            appendLine("# CamProV5 Desktop Application Test Report")
            appendLine("Generated: ${java.time.LocalDateTime.now()}")
            appendLine()

            appendLine("## Summary")
            val totalTests = results.values.sumOf { it.totalTests }
            val totalSuccessful = results.values.sumOf { it.successfulTests }
            val totalFailed = results.values.sumOf { it.failedTests }
            val overallSuccessRate = if (totalTests > 0) (totalSuccessful.toDouble() / totalTests) * 100 else 0.0

            appendLine("- **Total Tests**: $totalTests")
            appendLine("- **Successful**: $totalSuccessful")
            appendLine("- **Failed**: $totalFailed")
            appendLine("- **Success Rate**: ${String.format("%.1f", overallSuccessRate)}%")
            appendLine()

            appendLine("## Test Categories")
            results.forEach { (category, result) ->
                appendLine("### ${category.displayName}")
                appendLine("- **Description**: ${category.description}")
                appendLine("- **Total Tests**: ${result.totalTests}")
                appendLine("- **Successful**: ${result.successfulTests}")
                appendLine("- **Failed**: ${result.failedTests}")
                appendLine("- **Skipped**: ${result.skippedTests}")
                appendLine("- **Success Rate**: ${String.format("%.1f", result.successRate)}%")
                appendLine("- **Execution Time**: ${result.executionTime}ms")
                appendLine()
            }
        }

        java.nio.file.Files.write(
            java.nio.file.Paths.get(outputPath),
            report.toByteArray(),
        )

        println("📄 Test report generated: $outputPath")
    }

    /**
     * Main function for running tests from command line.
     */
    @JvmStatic
    fun main(args: Array<String>) {
        when {
            args.isEmpty() -> {
                // Run all tests
                val results = runAllTests()
                generateTestReport(results, "test_report.md")
            }
            args[0] == "category" && args.size > 1 -> {
                // Run specific category
                val categoryName = args[1]
                val category = TestCategory.values().find { it.name.equals(categoryName, ignoreCase = true) }
                if (category != null) {
                    val result = runTestCategory(category)
                    println("Category: ${category.displayName}")
                    println("Result: ${if (result.isSuccessful) "PASSED" else "FAILED"}")
                    println("Tests: ${result.successfulTests}/${result.totalTests}")
                } else {
                    println("Unknown category: $categoryName")
                    println("Available categories: ${TestCategory.values().joinToString { it.name }}")
                }
            }
            args[0] == "class" && args.size > 1 -> {
                // Run specific test class
                val testClass = args[1]
                val results = runSpecificTests(listOf(testClass))
                results.forEach { (className, result) ->
                    println("Class: $className")
                    println("Result: ${if (result.isSuccessful) "PASSED" else "FAILED"}")
                    println("Tests: ${result.successfulTests}/${result.totalTests}")
                }
            }
            else -> {
                println("Usage:")
                println("  java TestSuiteRunner                    # Run all tests")
                println("  java TestSuiteRunner category <name>    # Run specific category")
                println("  java TestSuiteRunner class <name>       # Run specific test class")
            }
        }
    }
}

/**
 * Extension function for string repetition.
 */
private operator fun String.times(n: Int): String = this.repeat(n)
