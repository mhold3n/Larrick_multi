package com.campro.v5.data.litvin

data class MotionLawSample(
    val thetaDeg: Double,
    val xMm: Double,
    val vMmPerOmega: Double,
    val aMmPerOmega2: Double
)

data class MotionLawSamples(
    val stepDeg: Double,
    val samples: List<MotionLawSample>
)
