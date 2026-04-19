package com.campro.v5.animation

import com.campro.v5.data.litvin.LitvinJsonLoader
import com.campro.v5.data.litvin.LitvinUserParams
import com.campro.v5.data.litvin.MotionLawSamples
import com.campro.v5.data.litvin.TransmissionAndPitch
import com.campro.v5.data.litvin.toJniArgs
import org.slf4j.LoggerFactory
import java.io.File
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin
import kotlin.math.sqrt

object TransmissionSynthesis {
    private val logger = LoggerFactory.getLogger(TransmissionSynthesis::class.java)

    /**
     * Geometry-based transmission preview (Phase 1b, Workstream B1).
     * Uses motion-law velocity v(θ)=dx/dθ and simple kinematics to estimate dψ/dα and i(θ)=1+dψ/dα.
     * Also produces prototype pitch curves and computes an arc-length residual metric.
     */
    fun computeTransmissionAndPitch(motion: MotionLawSamples, params: LitvinUserParams): TransmissionAndPitch {
        return try {
            val n = motion.samples.size
            if (n == 0) return TransmissionAndPitch(emptyList(), emptyList(), emptyList(), 0.0)

            val stepDeg = motion.stepDeg
            val stepRad = stepDeg * PI / 180.0

            // Geometry
            val gamma = params.sliderAxisDeg * PI / 180.0 // slider axis angle γ
            val beta = params.journalPhaseBetaDeg * PI / 180.0 // journal phase β
            val R = max(1e-6, params.journalRadius)

            // Derive dψ/dα ≈ -v / (R * sin((α+β) − γ)), then i = 1 + dψ/dα
            val iRaw = DoubleArray(n)
            for (k in 0 until n) {
                val thetaDeg = motion.samples[k].thetaDeg
                val v = motion.samples[k].vMmPerOmega // per-radian
                val aRad = thetaDeg * PI / 180.0
                val denom = R * sin((aRad + beta) - gamma)
                val safe = if (abs(denom) < 0.15 * R) (0.15 * R) * if (denom >= 0.0) 1.0 else -1.0 else denom
                val dpsi_dalpha = if (safe != 0.0) (-v / safe) else 0.0
                iRaw[k] = 1.0 + dpsi_dalpha
            }

            // Light smoothing to suppress numerical spikes near singularities
            val window = 5
            val iSm = DoubleArray(n)
            for (k in 0 until n) {
                var sum = 0.0
                var cnt = 0
                for (d in -window..window) {
                    val j = (k + d + n) % n
                    sum += iRaw[j]
                    cnt++
                }
                iSm[k] = sum / cnt
            }

            // Enforce positivity, periodic endpoints equality, and normalize mean to 1.0
            val MIN_I = 1e-3
            for (k in 0 until n) {
                if (!iSm[k].isFinite()) iSm[k] = 1.0
                iSm[k] = max(MIN_I, iSm[k])
            }
            // exact periodic endpoint (θ=0 vs θ=360−Δ)
            iSm[n - 1] = iSm[0]
            // mean normalization
            val mean = iSm.average().let { if (it.isFinite() && it != 0.0) it else 1.0 }
            for (k in 0 until n) iSm[k] /= mean

            // Optional parity calibration with Rust φ(θ): simple fluctuation scaling only
            var iCal = iSm.copyOf()
            try {
                val args = params.toJniArgs()
                val id =
                    try {
                        LitvinNative.createLitvinLawNative(args)
                    } catch (e: UnsatisfiedLinkError) {
                        0L
                    }
                if (id != 0L) {
                    try {
                        val tablesPath = LitvinNative.getLitvinKinematicsTablesNative(id)
                        val file = File(tablesPath)
                        if (file.exists()) {
                            val tables = LitvinJsonLoader.loadTables(file)
                            val alpha = tables.alphaDeg
                            val phiList =
                                tables.curves?.phiOfTheta ?: run {
                                    val planet = tables.planets.firstOrNull()
                                    planet?.spinPsiDeg ?: emptyList()
                                }
                            val m = min(n, min(alpha.size, phiList.size))
                            if (m > 3) {
                                val step = if (alpha.size > 1) alpha[1] - alpha[0] else stepDeg
                                val denom = if (step != 0.0) 2.0 * step else 1.0
                                val rustI = DoubleArray(m)
                                for (i in 0 until m) {
                                    val prev = (i - 1 + m) % m
                                    val next = (i + 1) % m
                                    var d = phiList[next] - phiList[prev]
                                    if (d < -180.0) {
                                        d += 360.0
                                    } else if (d > 180.0) {
                                        d -= 360.0
                                    }
                                    rustI[i] = d / denom
                                }
                                // Use Rust-derived i directly for parity preview when available
                                val rMean = rustI.average().let { if (it == 0.0) 1.0 else it }
                                for (i in 0 until n) {
                                    val idx = if (i < m) i else (i % m)
                                    iCal[i] = rustI[idx] / rMean
                                }
                                // periodic endpoint
                                iCal[n - 1] = iCal[0]
                            }
                        }
                    } finally {
                        try {
                            LitvinNative.disposeLitvinLawNative(id)
                        } catch (_: Throwable) {
                        }
                    }
                }
            } catch (_: Throwable) {
                // ignore calibration failures
            }

            // Final renormalization and periodicity enforcement
            val meanFinal = iCal.average().let { if (it == 0.0) 1.0 else it }
            for (k in 0 until n) iCal[k] /= meanFinal
            iCal[n - 1] = iCal[0]

            // Enforce sign pattern w.r.t velocity: sign(i-1) should match sign(v)
            val vList = DoubleArray(n) { motion.samples[it].vMmPerOmega }
            for (k in 0 until n) {
                val v = vList[k]
                if (k != n - 1) { // we'll set periodicity again after this loop
                    if (v > 0.0) {
                        iCal[k] = 1.0 + kotlin.math.abs(iCal[k] - 1.0)
                    } else if (v < 0.0) {
                        iCal[k] = 1.0 - kotlin.math.abs(iCal[k] - 1.0)
                    }
                }
                if (!iCal[k].isFinite()) iCal[k] = 1.0
                if (iCal[k] < MIN_I) iCal[k] = MIN_I
            }
            // Periodic endpoint equality and piecewise scaling to keep mean≈1 while preserving sign pattern
            // Compute deviations and signs
            val dev = DoubleArray(n) { kotlin.math.abs(iCal[it] - 1.0) }
            val sgn =
                IntArray(n) {
                    val v = vList[it]
                    if (v > 0.0) {
                        1
                    } else if (v < 0.0) {
                        -1
                    } else {
                        0
                    }
                }
            // Enforce periodic parity in contribution space to avoid bias when matching endpoints
            dev[n - 1] = dev[0]
            sgn[n - 1] = sgn[0]
            // Initial sign-enforced i
            for (k in 0 until n) {
                iCal[k] =
                    when (sgn[k]) {
                        1 -> 1.0 + dev[k]
                        -1 -> 1.0 - dev[k]
                        else -> 1.0
                    }
                if (!iCal[k].isFinite()) iCal[k] = 1.0
                if (iCal[k] < MIN_I) iCal[k] = MIN_I
            }
            // Compute total contributions (averaged over n for simple balance)
            var DpTot = 0.0
            var DnTot = 0.0
            for (k in 0 until n) {
                if (sgn[k] > 0) {
                    DpTot += dev[k]
                } else if (sgn[k] < 0) {
                    DnTot += dev[k]
                }
            }
            // Scale negative deviations to balance mean around 1: choose a=1, b=DpTot/DnTot
            val b = if (DnTot > 0.0) (DpTot / DnTot) else 1.0
            for (k in 0 until n) {
                if (sgn[k] < 0) {
                    iCal[k] = 1.0 - b * dev[k]
                } else if (sgn[k] > 0) {
                    iCal[k] = 1.0 + dev[k]
                } else {
                    iCal[k] = 1.0
                }
                if (!iCal[k].isFinite()) iCal[k] = 1.0
                if (iCal[k] < MIN_I) iCal[k] = MIN_I
            }
            // Enforce periodic endpoint equality last
            iCal[n - 1] = iCal[0]
            // Exact mean normalization to 1.0 via constant offset (preserves sign in practice)
            val meanStrict = iCal.average()
            val dOff = meanStrict - 1.0
            if (dOff != 0.0 && meanStrict.isFinite()) {
                for (k in 0 until n) iCal[k] -= dOff
                // Re-enforce periodic endpoint equality after offset
                iCal[n - 1] = iCal[0]
            }
            // Post-offset safety: enforce finite and MIN_I clamp and periodic equality
            for (k in 0 until n) {
                if (!iCal[k].isFinite()) iCal[k] = 1.0
                if (iCal[k] < MIN_I) iCal[k] = MIN_I
            }
            iCal[n - 1] = iCal[0]

            val iOfTheta = motion.samples.mapIndexed { idx, s -> s.thetaDeg to iCal[idx] }

            // Arc-length residual metric between normalized cumulative arc-lengths (use final iCal)
            var cumCam = 0.0
            var cumRing = 0.0
            val camCum = DoubleArray(n)
            val ringCum = DoubleArray(n)
            for (k in 0 until n) {
                cumCam += 1.0 * stepRad
                cumRing += iCal[k] * stepRad
                camCum[k] = cumCam
                ringCum[k] = cumRing
            }
            val camTotal = camCum.last().let { if (it > 0.0) it else 1.0 }
            val ringTotal = ringCum.last().let { if (it > 0.0) it else 1.0 }
            var errSum = 0.0
            for (k in 0 until n) {
                val sc = camCum[k] / camTotal
                val sr = ringCum[k] / ringTotal
                val d = sr - sc
                errSum += d * d
            }
            val residualArcLenRms = sqrt(errSum / n)

            // Prototype pitch previews over s∈[0,1]
            val np = 101
            val pitchPlanet = ArrayList<Pair<Double, Double>>(np)
            val pitchRing = ArrayList<Pair<Double, Double>>(np)
            for (j in 0 until np) {
                val s = j.toDouble() / (np - 1).toDouble()
                // simple, monotone prototypes; UI smoothing can be applied later
                val rPlanet = max(1e-6, params.camR0 + params.camKPerUnit * s)
                val rRing = max(rPlanet + 0.5, params.centerDistanceBias + params.centerDistanceScale * s)
                pitchPlanet.add(s to rPlanet)
                pitchRing.add(s to rRing)
            }

            TransmissionAndPitch(
                iOfTheta = iOfTheta,
                pitchRing = pitchRing,
                pitchPlanet = pitchPlanet,
                residualArcLenRms = residualArcLenRms,
            )
        } catch (t: Throwable) {
            logger.warn("Transmission synthesis fallback due to error: ${t.message}")
            val theta = motion.samples.map { it.thetaDeg }
            val iVals = theta.map { it to 1.0 }
            TransmissionAndPitch(iOfTheta = iVals, pitchRing = emptyList(), pitchPlanet = emptyList(), residualArcLenRms = 0.0)
        }
    }
}
