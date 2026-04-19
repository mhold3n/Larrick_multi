package com.campro.v5.data.litvin

// DTOs aligned to desktop module usages.

data class TransmissionAndPitch(
    val iOfTheta: List<Pair<Double, Double>>,
    val pitchRing: List<Pair<Double, Double>>,
    val pitchPlanet: List<Pair<Double, Double>>,
    val residualArcLenRms: Double
)

data class DiagnosticsDTO(
    val buildMs: Double? = null,
    val arcLengthResidualMax: Double? = null,
    val clearanceMin: Double? = null,
    val suggestedCenterDistanceInflation: Double? = null
)

data class PitchCurvesDTO(
    val thetaDeg: List<Double> = emptyList(),
    val phiOfTheta: List<Double> = emptyList(),
    val pitchPlanet: List<Double> = emptyList(),
    val pitchRing: List<Double> = emptyList()
)

data class PlanetDTO(
    val centerX: List<Double> = emptyList(),
    val centerY: List<Double> = emptyList(),
    val spinPsiDeg: List<Double> = emptyList(),
    val journalX: List<Double> = emptyList(),
    val journalY: List<Double> = emptyList(),
    val pistonS: List<Double> = emptyList()
)

data class CurvesDTO(
    val phiOfTheta: List<Double>? = null
)

data class LitvinTablesDTO(
    val alphaDeg: List<Double> = emptyList(),
    val planets: List<PlanetDTO> = emptyList(),
    val curves: CurvesDTO? = null,
    val diagnostics: DiagnosticsDTO? = null
)
