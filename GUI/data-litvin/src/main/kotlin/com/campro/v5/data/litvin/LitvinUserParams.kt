package com.campro.v5.data.litvin

data class LitvinUserParams(
    // Core sampling/motion controls
    val samplingStepDeg: Double = 1.0,
    val profileSolverMode: ProfileSolverMode = ProfileSolverMode.Piecewise,
    val rampProfile: RampProfile = RampProfile.S5,
    val dwellTdcDeg: Double = 4.0,
    val dwellBdcDeg: Double = 3.0,
    val rampBeforeTdcDeg: Double = 6.0,
    val rampAfterTdcDeg: Double = 5.0,
    val rampBeforeBdcDeg: Double = 7.0,
    val rampAfterBdcDeg: Double = 4.0,

    // Stroke and CV split
    val strokeLengthMm: Double = 100.0,
    val upFraction: Double = 110.0 / 180.0, // Asymmetric up/down ratio within 180° ring rotation

    // CORRECTED: Ring/Planet rotation parameters for planetary gearset
    val ringRotationDeg: Double = 180.0, // Ring rotates 180° for complete 2-stroke cycle
    val planetRotationDeg: Double = 360.0, // Planet rotates 360° for complete 2-stroke cycle
    val gearRatio: Double = 2.0, // Planet:Ring ratio = 360:180 = 2:1

    // Asymmetric stroke durations within 180° ring rotation
    val expansionDurationDeg: Double = 110.0, // 220° expansion scaled to 180° ring rotation
    val compressionDurationDeg: Double = 70.0, // 140° compression scaled to 180° ring rotation

    // Motion law phase parameters
    val linearAccelTdcDeg: Double = 8.0, // Linear acceleration near TDC
    val linearAccelBdcDeg: Double = 6.0, // Linear acceleration near BDC

    // Geometry/visualization and tuning (cover names used in tests)
    val rodLength: Double = 100.0,
    val interferenceBuffer: Double = 0.5,
    val planetCount: Int = 2,
    val carrierOffsetDeg: Double = 180.0,
    val ringThicknessVisual: Double = 6.0,
    val arcResidualTolMm: Double = 0.01,

    // CORRECTED: Planetary gearset geometry parameters
    val planetRadius: Double = 15.0, // Fixed planet radius (not max, but actual)
    val ringInnerRadiusBase: Double = 70.0, // Base ring inner radius (must be > planet radius)
    val ringInnerRadiusVariation: Double = 10.0, // Variation in ring inner radius for non-circular profile
    val ringThickness: Double = 3.0, // Ring gear thickness (outer - inner radius)
    val centerDistance: Double = 85.0, // Distance from center to planet centers

    // Specific tooth meshing parameters
    val planetTeeth: Int = 20, // Number of teeth on planet gear
    val ringTeeth: Int = 40, // Number of teeth on ring gear (2:1 ratio)
    val toothModule: Double = 2.0, // Module for gear tooth sizing

    // Existing with defaults
    val sliderAxisDeg: Double = 0.0,
    val journalPhaseBetaDeg: Double = 0.0,
    val journalRadius: Double = 5.0,
    val camR0: Double = 40.0,
    val camKPerUnit: Double = 1.0,
    val centerDistanceBias: Double = 50.0,
    val centerDistanceScale: Double = 1.0,
    val rpm: Double = 3000.0
)
