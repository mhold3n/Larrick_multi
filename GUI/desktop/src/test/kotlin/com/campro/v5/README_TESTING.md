# CamProV5 Desktop Application Testing Guide

## Overview

This document provides comprehensive guidance for testing the CamProV5 desktop application. The testing suite includes unit tests, integration tests, performance tests, user acceptance tests, and end-to-end tests.

## Test Structure

### Test Categories

1. **Unit Tests** (`com.campro.v5.*`)
   - Individual component testing
   - Isolated functionality validation
   - Mock dependencies

2. **Integration Tests** (`com.campro.v5.integration.*`)
   - Component integration testing
   - End-to-end workflow validation
   - Real system interactions

3. **Performance Tests** (`com.campro.v5.performance.*`)
   - Performance validation testing
   - Memory usage monitoring
   - Response time validation

4. **User Acceptance Tests** (`com.campro.v5.acceptance.*`)
   - User experience validation
   - Interface usability testing
   - Workflow completion testing

5. **Pipeline Tests** (`com.campro.v5.pipeline.*`)
   - Complete pipeline testing
   - Data flow validation
   - Error handling testing

## Running Tests

### Run All Tests

```bash
# Using Gradle
./gradlew :desktop:test

# Using TestSuiteRunner
java -cp build/classes/kotlin/test com.campro.v5.TestSuiteRunner
```

### Run Specific Test Categories

```bash
# Unit tests only
./gradlew :desktop:test --tests "com.campro.v5.*"

# Integration tests only
./gradlew :desktop:test --tests "com.campro.v5.integration.*"

# Performance tests only
./gradlew :desktop:test --tests "com.campro.v5.performance.*"

# User acceptance tests only
./gradlew :desktop:test --tests "com.campro.v5.acceptance.*"

# Pipeline tests only
./gradlew :desktop:test --tests "com.campro.v5.pipeline.*"
```

### Run Specific Test Classes

```bash
# Using TestSuiteRunner
java -cp build/classes/kotlin/test com.campro.v5.TestSuiteRunner class com.campro.v5.integration.EndToEndIntegrationTest

# Using Gradle
./gradlew :desktop:test --tests "com.campro.v5.integration.EndToEndIntegrationTest"
```

## Test Requirements

### Performance Requirements

- **Startup Time**: < 3 seconds
- **UI Responsiveness**: < 100ms response time
- **Result Display**: < 1 second
- **Memory Usage**: < 200MB for typical sessions
- **Animation Smoothness**: < 500ms for transitions

### Functional Requirements

- All optimization parameters accessible via UI
- Complete result visualization (motion law, gear profiles, efficiency, FEA)
- Parameter presets and export/import functionality
- Batch processing capabilities
- Error handling and user feedback
- Responsive design for different window sizes

### Quality Requirements

- 100% test coverage for UI components
- All tests pass consistently
- Follows Material Design guidelines
- Comprehensive error handling
- User-friendly interface

## Test Configuration

### Configuration File

Tests are configured using `test-config.properties`:

```properties
# Test Execution Settings
test.execution.timeout=30
test.execution.parallel=true
test.execution.retry.count=3

# Performance Test Settings
performance.startup.timeout=5000
performance.memory.limit.mb=200

# Integration Test Settings
integration.test.output.dir=./test_output
integration.test.cleanup=true
```

### Environment Variables

```bash
# Test environment
export TEST_ENV=integration

# Test output directory
export TEST_OUTPUT_DIR=./test_output

# Test logging level
export TEST_LOG_LEVEL=INFO
```

## Test Data

### Parameter Sets

Tests use predefined parameter sets:

- **Default Parameters**: Standard optimization parameters
- **Quick Test Parameters**: Fast execution for testing
- **High Performance Parameters**: Optimized for performance
- **Edge Case Parameters**: Boundary conditions
- **Invalid Parameters**: Error condition testing

### Test Output

Test outputs are generated in:

- `./test_output/` - Test execution outputs
- `./test_logs/` - Test execution logs
- `./test_reports/` - Test reports and summaries

## Test Reports

### Report Generation

Test reports are automatically generated after test execution:

```bash
# Generate test report
java -cp build/classes/kotlin/test com.campro.v5.TestSuiteRunner

# Report will be generated at: ./test_report.md
```

### Report Contents

- Overall test summary
- Category breakdown
- Performance metrics
- Failed test details
- Execution times
- Success rates

## Troubleshooting

### Common Issues

1. **Test Timeouts**
   - Increase timeout values in configuration
   - Check system performance
   - Verify test environment

2. **Memory Issues**
   - Monitor memory usage during tests
   - Adjust memory limits in configuration
   - Clean up test outputs

3. **UI Test Failures**
   - Verify Compose test setup
   - Check UI element selectors
   - Ensure proper test isolation

4. **Performance Test Failures**
   - Check system resources
   - Verify performance requirements
   - Monitor background processes

### Debug Mode

Enable debug mode for detailed test information:

```bash
# Enable debug logging
export TEST_LOG_LEVEL=DEBUG

# Run tests with debug output
./gradlew :desktop:test --info
```

## Continuous Integration

### CI Configuration

Tests are designed to run in CI environments:

```yaml
# GitHub Actions example
- name: Run Tests
  run: |
    ./gradlew :desktop:test
    java -cp build/classes/kotlin/test com.campro.v5.TestSuiteRunner
```

### Test Parallelization

Tests support parallel execution:

```properties
# Enable parallel execution
test.execution.parallel=true
test.execution.threads=4
```

## Best Practices

### Test Writing

1. **Use descriptive test names**
2. **Follow AAA pattern** (Arrange, Act, Assert)
3. **Test one thing at a time**
4. **Use appropriate assertions**
5. **Clean up test data**

### Test Maintenance

1. **Keep tests up to date**
2. **Remove obsolete tests**
3. **Refactor test code**
4. **Monitor test performance**
5. **Update test documentation**

### Test Coverage

1. **Aim for high coverage**
2. **Test edge cases**
3. **Test error conditions**
4. **Test user workflows**
5. **Test performance requirements**

## Support

### Getting Help

- Check test logs for error details
- Review test configuration
- Consult test documentation
- Contact development team

### Contributing

When adding new tests:

1. Follow existing test patterns
2. Update test documentation
3. Ensure proper test isolation
4. Add appropriate assertions
5. Update test configuration if needed
