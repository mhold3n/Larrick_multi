package com.campro.v5.data.litvin

import com.google.gson.Gson

import java.io.File

object LitvinJsonLoader {
    private val gson = Gson()
    fun <T> fromJson(json: String, clazz: Class<T>): T = gson.fromJson(json, clazz)
    fun toJson(obj: Any): String = gson.toJson(obj)

    fun loadTables(file: File): LitvinTablesDTO = gson.fromJson(file.readText(), LitvinTablesDTO::class.java)
    fun loadPitchCurves(file: File): PitchCurvesDTO = gson.fromJson(file.readText(), PitchCurvesDTO::class.java)
    fun loadFeaBoundary(file: File): Map<String, Any?> = gson.fromJson(file.readText(), Map::class.java) as Map<String, Any?>
}

object MotionLawSerialization {
    private val gson = Gson()
    fun encode(samples: MotionLawSamples): String = gson.toJson(samples)
    fun decode(json: String): MotionLawSamples = gson.fromJson(json, MotionLawSamples::class.java)
    fun writeToFile(file: File, samples: MotionLawSamples) = file.writeText(encode(samples))
}
