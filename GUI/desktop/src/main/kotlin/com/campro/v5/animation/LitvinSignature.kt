package com.campro.v5.animation

import java.security.MessageDigest

/**
 * Deterministic signature for Litvin parameters map (order-insensitive, stable text form).
 */
object LitvinSignature {
    fun compute(params: Map<String, Any?>): String {
        val sb = StringBuilder()
        params.toSortedMap(compareBy<String> { it }).forEach { (k, v) ->
            sb
                .append(k)
                .append('=')
                .append(v?.toString() ?: "null")
                .append('\n')
        }
        val bytes = sb.toString().toByteArray(Charsets.UTF_8)
        val md = MessageDigest.getInstance("SHA-256")
        val dig = md.digest(bytes)
        return dig.joinToString("") { b -> "%02x".format(b) }
    }
}
