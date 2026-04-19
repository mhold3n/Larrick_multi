package com.campro.v5.animation.collocation

import kotlin.math.*

/**
 * Collocation node generation for periodic domains.
 *
 * Provides LGL (Legendre-Gauss-Lobatto) and Chebyshev node distributions
 * for collocation methods on the interval [0, 2π] with periodic boundary conditions.
 */
object CollocationNodes {
    /**
     * Generate Legendre-Gauss-Lobatto (LGL) nodes for collocation.
     *
     * For periodic problems, we generate LGL nodes on [-1,1] and map them to [0,2π).
     * The last node is excluded to maintain periodicity (f(0) = f(2π)).
     *
     * @param n Number of collocation nodes
     * @return Array of nodes in [0, 2π) properly distributed for periodic LGL
     */
    fun generateLGL(n: Int): DoubleArray {
        require(n >= 3) { "Need at least 3 nodes for LGL collocation" }

        // Generate LGL nodes on [-1, 1]
        val lglNodes = computeLGLNodes(n)

        // Transform to [0, 2π) for periodic domain
        // For periodic problems, we need n evenly distributed nodes in [0, 2π)
        // Use the LGL distribution but scale to avoid the 2π endpoint
        return lglNodes
            .map { xi ->
                PI * (xi + 1.0) * (n - 1.0) / n // Maps to [0, 2π*(n-1)/n] which is [0, 2π) approximately
            }.toDoubleArray()
    }

    /**
     * Generate Chebyshev nodes for collocation.
     *
     * Chebyshev nodes provide good interpolation properties and are
     * easier to compute than LGL nodes.
     *
     * @param n Number of collocation nodes
     * @return Array of nodes in [0, 2π]
     */
    fun generateChebyshev(n: Int): DoubleArray {
        require(n >= 3) { "Need at least 3 nodes for Chebyshev collocation" }

        // Generate Chebyshev nodes on [-1, 1] first
        val chebyNodes =
            DoubleArray(n) { i ->
                cos(PI * (2 * i + 1) / (2.0 * n))
            }

        // Transform from [-1, 1] to [0, 2π] and sort
        return chebyNodes
            .map { xi ->
                PI * (xi + 1.0) // Maps [-1,1] to [0,2π]
            }.sorted()
            .toDoubleArray()
    }

    /**
     * Generate uniform nodes (for comparison and fallback).
     *
     * For periodic problems, we use n evenly spaced points in [0, 2π)
     * where the last point is NOT at 2π to avoid duplication with 0.
     *
     * @param n Number of collocation nodes
     * @return Array of uniformly spaced nodes in [0, 2π)
     */
    fun generateUniform(n: Int): DoubleArray {
        require(n >= 2) { "Need at least 2 nodes for uniform grid" }

        return DoubleArray(n) { i ->
            2.0 * PI * i / n
        }
    }

    /**
     * Compute LGL nodes on [-1, 1] using iterative method.
     *
     * This implements the algorithm for computing Legendre-Gauss-Lobatto
     * quadrature points, which are the roots of the derivative of Legendre polynomials.
     */
    private fun computeLGLNodes(n: Int): DoubleArray {
        if (n == 2) return doubleArrayOf(-1.0, 1.0)
        if (n == 3) return doubleArrayOf(-1.0, 0.0, 1.0)

        val nodes = DoubleArray(n)
        nodes[0] = -1.0
        nodes[n - 1] = 1.0

        // Interior nodes are roots of derivative of Legendre polynomial
        for (i in 1 until n - 1) {
            // Initial guess using Chebyshev nodes
            var x = cos(PI * i / (n - 1))

            // Newton-Raphson iteration to find LGL node
            for (iter in 0 until 20) {
                val (p, dp) = evaluateLegendreDerivative(n - 1, x)
                val f = dp
                val df = ((n - 1) * (x * dp - p)) / (x * x - 1.0)

                val dx = f / df
                x -= dx

                if (abs(dx) < 1e-15) break
            }

            nodes[i] = x
        }

        return nodes.sortedArray()
    }

    /**
     * Evaluate Legendre polynomial and its derivative at point x.
     *
     * @param n Degree of Legendre polynomial
     * @param x Evaluation point
     * @return Pair of (P_n(x), P_n'(x))
     */
    private fun evaluateLegendreDerivative(n: Int, x: Double): Pair<Double, Double> {
        if (n == 0) return Pair(1.0, 0.0)
        if (n == 1) return Pair(x, 1.0)

        var p0 = 1.0
        var p1 = x
        var dp0 = 0.0
        var dp1 = 1.0

        for (k in 2..n) {
            val p2 = ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
            val dp2 = ((2 * k - 1) * (x * dp1 + p1) - (k - 1) * dp0) / k

            p0 = p1
            p1 = p2
            dp0 = dp1
            dp1 = dp2
        }

        return Pair(p1, dp1)
    }

    /**
     * Validate that nodes are properly distributed for periodic problems.
     *
     * @param nodes Array of collocation nodes
     * @return true if nodes are valid for periodic collocation
     */
    fun validatePeriodicNodes(nodes: DoubleArray): Boolean {
        if (nodes.isEmpty()) return false

        // Check ordering
        for (i in 1 until nodes.size) {
            if (nodes[i] <= nodes[i - 1]) return false
        }

        // Check bounds: nodes should be in [0, 2π)
        if (nodes[0] < 0.0 || nodes.last() >= 2.0 * PI) return false

        // All nodes should be finite
        if (!nodes.all { it.isFinite() }) return false

        // For periodic problems, we need reasonable spacing
        val minSpacing = nodes.zip(nodes.drop(1)).map { it.second - it.first }.minOrNull() ?: 0.0
        return minSpacing > 1e-12 // Avoid degenerate cases
    }
}
