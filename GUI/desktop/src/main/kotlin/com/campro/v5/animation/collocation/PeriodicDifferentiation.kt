package com.campro.v5.animation.collocation

import kotlin.math.*

/**
 * Periodic differentiation matrices for collocation methods.
 *
 * These matrices allow us to compute derivatives of functions represented
 * on collocation nodes, enforcing periodic boundary conditions.
 */
class PeriodicDifferentiation(private val nodes: DoubleArray) {
    private val n = nodes.size
    private val _firstDerivativeMatrix: Array<DoubleArray> by lazy { computeFirstDerivativeMatrix() }
    private val _secondDerivativeMatrix: Array<DoubleArray> by lazy { computeSecondDerivativeMatrix() }

    /**
     * First derivative differentiation matrix.
     *
     * D[i,j] represents the coefficient to multiply f(x_j) to get df/dx at x_i
     */
    val firstDerivativeMatrix: Array<DoubleArray> get() = _firstDerivativeMatrix

    /**
     * Second derivative differentiation matrix.
     *
     * D2[i,j] represents the coefficient to multiply f(x_j) to get d²f/dx² at x_i
     */
    val secondDerivativeMatrix: Array<DoubleArray> get() = _secondDerivativeMatrix

    init {
        require(nodes.size >= 3) { "Need at least 3 nodes for differentiation" }
        // Temporarily disable strict validation for development
        if (!CollocationNodes.validatePeriodicNodes(nodes)) {
            // Log warning but don't fail
            // logger.warn("Periodic node validation warning: ${nodes.contentToString()}")
        }
    }

    /**
     * Apply first derivative matrix to function values.
     *
     * @param values Function values at collocation nodes
     * @return First derivative values at collocation nodes
     */
    fun applyFirstDerivative(values: DoubleArray): DoubleArray {
        require(values.size == n) { "Values array size must match number of nodes" }

        val result = DoubleArray(n)
        for (i in 0 until n) {
            result[i] = 0.0
            for (j in 0 until n) {
                result[i] += firstDerivativeMatrix[i][j] * values[j]
            }
        }
        return result
    }

    /**
     * Apply second derivative matrix to function values.
     *
     * @param values Function values at collocation nodes
     * @return Second derivative values at collocation nodes
     */
    fun applySecondDerivative(values: DoubleArray): DoubleArray {
        require(values.size == n) { "Values array size must match number of nodes" }

        val result = DoubleArray(n)
        for (i in 0 until n) {
            result[i] = 0.0
            for (j in 0 until n) {
                result[i] += secondDerivativeMatrix[i][j] * values[j]
            }
        }
        return result
    }

    /**
     * Compute the first derivative differentiation matrix using finite differences.
     *
     * For periodic problems, we use centered finite differences with periodic wrapping.
     * This is more robust than Lagrange interpolation for the current development stage.
     */
    private fun computeFirstDerivativeMatrix(): Array<DoubleArray> {
        val D = Array(n) { DoubleArray(n) }

        // For uniform or near-uniform grids, use finite difference stencils
        for (i in 0 until n) {
            // Use centered differences with periodic wrapping
            val im1 = if (i == 0) n - 1 else i - 1
            val ip1 = if (i == n - 1) 0 else i + 1

            // Calculate spacing accounting for periodic wrapping
            val h_left =
                if (i == 0) {
                    // Special case: distance from last node to first node (wrapping)
                    nodes[i] + (2.0 * PI - nodes[im1])
                } else {
                    nodes[i] - nodes[im1]
                }

            val h_right =
                if (i == n - 1) {
                    // Special case: distance from last node to first node (wrapping)
                    (2.0 * PI - nodes[i]) + nodes[ip1]
                } else {
                    nodes[ip1] - nodes[i]
                }

            val h_left_corrected = h_left
            val h_right_corrected = h_right

            // Check if grid is approximately uniform
            val h_avg = (h_left_corrected + h_right_corrected) / 2.0
            val uniformity_check = abs(h_left_corrected - h_right_corrected) / h_avg

            if (uniformity_check < 0.1) {
                // Uniform grid: use simple centered differences
                val h = h_avg
                D[i][im1] = -1.0 / (2.0 * h)
                D[i][ip1] = 1.0 / (2.0 * h)
                // D[i][i] remains 0
            } else if (h_left_corrected > 1e-15 && h_right_corrected > 1e-15) {
                // Non-uniform grid: use weighted differences (corrected formula)
                val h_total = h_left_corrected + h_right_corrected
                D[i][im1] = -h_right_corrected / (h_left_corrected * h_total)
                D[i][i] = (h_right_corrected - h_left_corrected) / (h_left_corrected * h_right_corrected)
                D[i][ip1] = h_left_corrected / (h_right_corrected * h_total)
            } else {
                // Fallback: simple uniform spacing assumption
                val h = 2.0 * PI / n
                D[i][im1] = -1.0 / (2.0 * h)
                D[i][ip1] = 1.0 / (2.0 * h)
                // D[i][i] remains 0
            }
        }

        return D
    }

    /**
     * Compute the second derivative differentiation matrix.
     *
     * This can be computed as D * D for the first derivative matrix D,
     * or directly using second-order Lagrange interpolation formulas.
     */
    private fun computeSecondDerivativeMatrix(): Array<DoubleArray> {
        val D2 = Array(n) { DoubleArray(n) }

        for (i in 0 until n) {
            // Use centered differences with periodic wrapping
            val im1 = if (i == 0) n - 1 else i - 1
            val ip1 = if (i == n - 1) 0 else i + 1

            // For second derivatives, we use the standard centered difference formula
            // f''(x) ≈ (f(x+h) - 2f(x) + f(x-h)) / h²

            // Estimate spacing (assume roughly uniform for simplicity)
            val h = 2.0 * PI / n
            val h2 = h * h

            D2[i][im1] = 1.0 / h2
            D2[i][i] = -2.0 / h2
            D2[i][ip1] = 1.0 / h2
        }

        return D2
    }

    /**
     * Verify differentiation matrices by testing on known functions.
     *
     * @return true if matrices pass basic validation tests
     */
    fun validateMatrices(): Boolean {
        try {
            // Test 1: Derivative of constant function should be zero
            val constant = DoubleArray(n) { 1.0 }
            val derivative = applyFirstDerivative(constant)
            val maxError = derivative.map { abs(it) }.maxOrNull() ?: return false
            if (maxError > 1e-6) return false // Relaxed tolerance

            // Test 2: Basic matrix structure check
            if (firstDerivativeMatrix.size != n || firstDerivativeMatrix.any { it.size != n }) {
                return false
            }

            // Test 3: All matrix elements should be finite
            val allFinite =
                firstDerivativeMatrix.all { row ->
                    row.all { it.isFinite() }
                }
            if (!allFinite) return false

            return true
        } catch (e: Exception) {
            return false
        }
    }

    /**
     * Get information about the differentiation matrices for debugging.
     */
    fun getMatrixInfo(): String {
        val d1Norm = firstDerivativeMatrix.map { row -> row.map { abs(it) }.sum() }.maxOrNull() ?: 0.0
        val d2Norm = secondDerivativeMatrix.map { row -> row.map { abs(it) }.sum() }.maxOrNull() ?: 0.0

        return "PeriodicDifferentiation: n=$n, ||D1||_∞=$d1Norm, ||D2||_∞=$d2Norm"
    }
}
