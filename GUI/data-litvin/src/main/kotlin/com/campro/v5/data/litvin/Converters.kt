package com.campro.v5.data.litvin

import com.google.gson.Gson

/** Extension for call-site convenience */
fun LitvinUserParams.validate(): List<String> = validateParams(this)

/** Return list of validation error messages; empty if ok. Enhanced with planetary gearset validation. */
fun validateParams(p: LitvinUserParams): List<String> {
    val errs = mutableListOf<String>()
    
    // Core motion law validation
    if (p.samplingStepDeg <= 0.0) errs += "samplingStepDeg must be > 0"
    if (p.upFraction !in 0.0..1.0) errs += "upFraction must be in [0,1]"
    if (p.strokeLengthMm <= 0.0) errs += "strokeLengthMm must be > 0"
    if (p.rodLength <= 0.0) errs += "rodLength must be > 0"
    
    // Ring/Planet rotation validation
    if (p.ringRotationDeg <= 0.0 || p.ringRotationDeg > 360.0) errs += "ringRotationDeg must be in (0, 360]"
    if (p.planetRotationDeg <= 0.0) errs += "planetRotationDeg must be > 0"
    if (p.gearRatio <= 0.0) errs += "gearRatio must be > 0"
    
    // Validate gear ratio consistency: planetRotationDeg / ringRotationDeg should equal gearRatio
    val expectedGearRatio = p.planetRotationDeg / p.ringRotationDeg
    if (kotlin.math.abs(expectedGearRatio - p.gearRatio) > 0.01) {
        errs += "gearRatio (${p.gearRatio}) inconsistent with ringRotationDeg (${p.ringRotationDeg}) and planetRotationDeg (${p.planetRotationDeg})"
    }
    
    // Asymmetric stroke duration validation
    if (p.expansionDurationDeg <= 0.0) errs += "expansionDurationDeg must be > 0"
    if (p.compressionDurationDeg <= 0.0) errs += "compressionDurationDeg must be > 0"
    val totalStrokeDuration = p.expansionDurationDeg + p.compressionDurationDeg
    if (kotlin.math.abs(totalStrokeDuration - p.ringRotationDeg) > 0.01) {
        errs += "expansionDurationDeg + compressionDurationDeg (${totalStrokeDuration}) must equal ringRotationDeg (${p.ringRotationDeg})"
    }
    
    // Motion law phase validation
    if (p.linearAccelTdcDeg < 0.0) errs += "linearAccelTdcDeg must be >= 0"
    if (p.linearAccelBdcDeg < 0.0) errs += "linearAccelBdcDeg must be >= 0"
    
    // Planetary gearset geometry validation
    if (p.planetRadius <= 0.0) errs += "planetRadius must be > 0"
    if (p.ringInnerRadiusBase <= 0.0) errs += "ringInnerRadiusBase must be > 0"
    if (p.ringInnerRadiusVariation < 0.0) errs += "ringInnerRadiusVariation must be >= 0"
    if (p.ringThickness <= 0.0) errs += "ringThickness must be > 0"
    if (p.centerDistance <= 0.0) errs += "centerDistance must be > 0"
    
    // UNIFIED CONSTRAINT validation: ringInnerRadiusBase must be > planetRadius
    if (p.ringInnerRadiusBase <= p.planetRadius) {
        errs += "ringInnerRadiusBase (${p.ringInnerRadiusBase}) must be > planetRadius (${p.planetRadius}) for UNIFIED CONSTRAINT"
    }
    
    // Tooth meshing validation
    if (p.planetTeeth <= 0) errs += "planetTeeth must be > 0"
    if (p.ringTeeth <= 0) errs += "ringTeeth must be > 0"
    if (p.toothModule <= 0.0) errs += "toothModule must be > 0"
    
    // Validate tooth count ratio matches gear ratio
    val expectedToothRatio = p.ringTeeth.toDouble() / p.planetTeeth.toDouble()
    if (kotlin.math.abs(expectedToothRatio - p.gearRatio) > 0.01) {
        errs += "tooth count ratio (${expectedToothRatio}) should match gearRatio (${p.gearRatio})"
    }
    
    // Interference and clearance validation
    if (p.interferenceBuffer < 0.0) errs += "interferenceBuffer must be >= 0"
    if (p.planetCount <= 0) errs += "planetCount must be > 0"
    if (p.carrierOffsetDeg <= 0.0 || p.carrierOffsetDeg > 360.0) errs += "carrierOffsetDeg must be in (0, 360]"
    
    // RPM validation
    if (p.rpm <= 0.0) errs += "rpm must be > 0"
    
    return errs
}

/** Construct params from a generic map; tolerates missing keys with defaults. Enhanced with planetary gearset parameters. */
fun litvinParamsFromMap(m: Map<String, Any?>): LitvinUserParams {
    fun num(key: String, def: Double) = when (val value = m[key]) {
        is Number -> value.toDouble()
        is String -> value.toDoubleOrNull() ?: def
        else -> def
    }
    fun int(key: String, def: Int) = when (val value = m[key]) {
        is Number -> value.toInt()
        is String -> value.toIntOrNull() ?: def
        else -> def
    }
    fun ramp(key: String, def: RampProfile) =
        (m[key] as? String)?.let {
            runCatching { RampProfile.valueOf(it) }.getOrNull()
        } ?: def
    fun solverMode(key: String, def: ProfileSolverMode) =
        (m[key] as? String)?.let {
            runCatching { ProfileSolverMode.valueOf(it) }.getOrNull()
        } ?: def
    return LitvinUserParams(
        // Core sampling/motion controls
        samplingStepDeg = num("samplingStepDeg", 1.0),
        profileSolverMode = solverMode("profileSolverMode", ProfileSolverMode.Piecewise),
        rampProfile = ramp("rampProfile", RampProfile.S5),
        dwellTdcDeg = num("dwellTdcDeg", 4.0),
        dwellBdcDeg = num("dwellBdcDeg", 3.0),
        rampBeforeTdcDeg = num("rampBeforeTdcDeg", 6.0),
        rampAfterTdcDeg = num("rampAfterTdcDeg", 5.0),
        rampBeforeBdcDeg = num("rampBeforeBdcDeg", 7.0),
        rampAfterBdcDeg = num("rampAfterBdcDeg", 4.0),

        // Stroke and CV split
        strokeLengthMm = num("strokeLengthMm", 100.0),
        upFraction = num("upFraction", 110.0 / 180.0), // Asymmetric up/down ratio within 180° ring rotation

        // CORRECTED: Ring/Planet rotation parameters for planetary gearset
        ringRotationDeg = num("ringRotationDeg", 180.0), // Ring rotates 180° for complete 2-stroke cycle
        planetRotationDeg = num("planetRotationDeg", 360.0), // Planet rotates 360° for complete 2-stroke cycle
        gearRatio = num("gearRatio", 2.0), // Planet:Ring ratio = 360:180 = 2:1

        // Asymmetric stroke durations within 180° ring rotation
        expansionDurationDeg = num("expansionDurationDeg", 110.0), // 220° expansion scaled to 180° ring rotation
        compressionDurationDeg = num("compressionDurationDeg", 70.0), // 140° compression scaled to 180° ring rotation

        // Motion law phase parameters
        linearAccelTdcDeg = num("linearAccelTdcDeg", 8.0), // Linear acceleration near TDC
        linearAccelBdcDeg = num("linearAccelBdcDeg", 6.0), // Linear acceleration near BDC

        // Geometry/visualization and tuning (cover names used in tests)
        rodLength = num("rodLength", 100.0),
        interferenceBuffer = num("interferenceBuffer", 0.5),
        planetCount = int("planetCount", 2),
        carrierOffsetDeg = num("carrierOffsetDeg", 180.0),
        ringThicknessVisual = num("ringThicknessVisual", 6.0),
        arcResidualTolMm = num("arcResidualTolMm", 0.01),

        // CORRECTED: Planetary gearset geometry parameters
        planetRadius = num("planetRadius", 15.0), // Fixed planet radius (not max, but actual)
        ringInnerRadiusBase = num("ringInnerRadiusBase", 70.0), // Base ring inner radius (must be > planet radius)
        ringInnerRadiusVariation = num("ringInnerRadiusVariation", 10.0), // Variation in ring inner radius for non-circular profile
        ringThickness = num("ringThickness", 3.0), // Ring gear thickness (outer - inner radius)
        centerDistance = num("centerDistance", 85.0), // Distance from center to planet centers

        // Specific tooth meshing parameters
        planetTeeth = int("planetTeeth", 20), // Number of teeth on planet gear
        ringTeeth = int("ringTeeth", 40), // Number of teeth on ring gear (2:1 ratio)
        toothModule = num("toothModule", 2.0), // Module for gear tooth sizing

        // Existing with defaults
        sliderAxisDeg = num("sliderAxisDeg", 0.0),
        journalPhaseBetaDeg = num("journalPhaseBetaDeg", 0.0),
        journalRadius = num("journalRadius", 5.0),
        camR0 = num("camR0", 40.0),
        camKPerUnit = num("camKPerUnit", 1.0),
        centerDistanceBias = num("centerDistanceBias", 50.0),
        centerDistanceScale = num("centerDistanceScale", 1.0),
        rpm = num("rpm", 3000.0)
    )
}

/** Extension for call-site convenience */
fun LitvinUserParams.toJniArgs(): Array<String> = toJniArgsInternal(this)

/** Serialize params for JNI argument passing (string array). Keep stable order. Enhanced with planetary gearset parameters. */
fun toJniArgsInternal(p: LitvinUserParams): Array<String> = arrayOf(
    // Core sampling/motion controls
    p.samplingStepDeg.toString(),
    p.profileSolverMode.name,
    p.rampProfile.name,
    p.dwellTdcDeg.toString(),
    p.dwellBdcDeg.toString(),
    p.rampBeforeTdcDeg.toString(),
    p.rampAfterTdcDeg.toString(),
    p.rampBeforeBdcDeg.toString(),
    p.rampAfterBdcDeg.toString(),

    // Stroke and CV split
    p.strokeLengthMm.toString(),
    p.upFraction.toString(),

    // CORRECTED: Ring/Planet rotation parameters for planetary gearset
    p.ringRotationDeg.toString(),
    p.planetRotationDeg.toString(),
    p.gearRatio.toString(),

    // Asymmetric stroke durations within 180° ring rotation
    p.expansionDurationDeg.toString(),
    p.compressionDurationDeg.toString(),

    // Motion law phase parameters
    p.linearAccelTdcDeg.toString(),
    p.linearAccelBdcDeg.toString(),

    // Geometry/visualization and tuning
    p.rodLength.toString(),
    p.interferenceBuffer.toString(),
    p.planetCount.toString(),
    p.carrierOffsetDeg.toString(),
    p.ringThicknessVisual.toString(),
    p.arcResidualTolMm.toString(),

    // CORRECTED: Planetary gearset geometry parameters
    p.planetRadius.toString(),
    p.ringInnerRadiusBase.toString(),
    p.ringInnerRadiusVariation.toString(),
    p.ringThickness.toString(),
    p.centerDistance.toString(),

    // Specific tooth meshing parameters
    p.planetTeeth.toString(),
    p.ringTeeth.toString(),
    p.toothModule.toString(),

    // Existing with defaults
    p.sliderAxisDeg.toString(),
    p.journalPhaseBetaDeg.toString(),
    p.journalRadius.toString(),
    p.camR0.toString(),
    p.camKPerUnit.toString(),
    p.centerDistanceBias.toString(),
    p.centerDistanceScale.toString(),
    p.rpm.toString()
)

/** Inverse of toJniArgs; best-effort. Enhanced with planetary gearset parameters. */
fun jniArgsToMap(args: Array<String>): Map<String, Any?> {
    val keys = listOf(
        // Core sampling/motion controls
        "samplingStepDeg", "profileSolverMode", "rampProfile", "dwellTdcDeg", "dwellBdcDeg",
        "rampBeforeTdcDeg", "rampAfterTdcDeg", "rampBeforeBdcDeg", "rampAfterBdcDeg",
        
        // Stroke and CV split
        "strokeLengthMm", "upFraction",
        
        // CORRECTED: Ring/Planet rotation parameters for planetary gearset
        "ringRotationDeg", "planetRotationDeg", "gearRatio",
        
        // Asymmetric stroke durations within 180° ring rotation
        "expansionDurationDeg", "compressionDurationDeg",
        
        // Motion law phase parameters
        "linearAccelTdcDeg", "linearAccelBdcDeg",
        
        // Geometry/visualization and tuning
        "rodLength", "interferenceBuffer", "planetCount", "carrierOffsetDeg", "ringThicknessVisual", "arcResidualTolMm",
        
        // CORRECTED: Planetary gearset geometry parameters
        "planetRadius", "ringInnerRadiusBase", "ringInnerRadiusVariation", "ringThickness", "centerDistance",
        
        // Specific tooth meshing parameters
        "planetTeeth", "ringTeeth", "toothModule",
        
        // Existing with defaults
        "sliderAxisDeg", "journalPhaseBetaDeg", "journalRadius", "camR0", "camKPerUnit", 
        "centerDistanceBias", "centerDistanceScale", "rpm"
    )
    val map = mutableMapOf<String, Any?>()
    for ((i, k) in keys.withIndex()) {
        val v = args.getOrNull(i)
        map[k] = when (k) {
            "profileSolverMode" -> v ?: ProfileSolverMode.Piecewise.name
            "rampProfile" -> v ?: RampProfile.S5.name
            "planetCount", "planetTeeth", "ringTeeth" -> v?.toIntOrNull() ?: 0
            else -> v?.toDoubleOrNull() ?: 0.0
        }
    }
    return map
}
