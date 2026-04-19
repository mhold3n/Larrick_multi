package com.campro.v5.support

import java.io.BufferedInputStream
import java.io.BufferedOutputStream
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

/**
 * Creates a diagnostic support bundle zip with logs and optional recent JSON outputs.
 *
 * - Includes only .log, .json, .txt files.
 * - Skips files larger than [maxFileSizeBytes] (default 20 MB) to avoid huge bundles.
 * - Excludes any file with name hinting PII (very light heuristic: files named like "user_*" or "secret").
 * - Returns the absolute path of the created zip.
 */
object SupportBundle {
    private val allowedExt = setOf(".log", ".json", ".txt")

    @JvmStatic
    fun createSupportBundle(
        sessionId: String,
        logDir: String?,
        jsonDirs: List<String> = emptyList(),
        maxFileSizeBytes: Long = 20L * 1024 * 1024,
    ): String {
        val logsPath: Path? = logDir?.let { Paths.get(it) }?.takeIf { Files.isDirectory(it) }
        val jsonPaths: List<Path> = jsonDirs.map { Paths.get(it) }.filter { Files.isDirectory(it) }

        val ts = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val outDir = (logsPath?.parent ?: Paths.get(System.getProperty("user.home"), "CamProV5", "support_bundles"))
        Files.createDirectories(outDir)
        val outZip = outDir.resolve("support_${ts}_$sessionId.zip").toAbsolutePath()

        ZipOutputStream(BufferedOutputStream(FileOutputStream(outZip.toFile()))).use { zos ->
            fun addDir(dir: Path, rootName: String) {
                Files.walk(dir).use { stream ->
                    stream.filter { Files.isRegularFile(it) }.forEach { p ->
                        val name = p.fileName.toString()
                        val ext = allowedExt.firstOrNull { name.lowercase(Locale.US).endsWith(it) }
                        if (ext == null) return@forEach
                        if (name.lowercase(Locale.US).startsWith("user_") || name.lowercase(Locale.US).contains("secret")) return@forEach
                        val size = Files.size(p)
                        if (size > maxFileSizeBytes) return@forEach
                        val rel = dir.relativize(p).toString().replace('\\', '/')
                        val entryName = "$rootName/$rel"
                        zos.putNextEntry(ZipEntry(entryName))
                        BufferedInputStream(FileInputStream(p.toFile())).use { bis ->
                            bis.copyTo(zos)
                        }
                        zos.closeEntry()
                    }
                }
            }
            logsPath?.let { addDir(it, "logs") }
            jsonPaths.forEach { addDir(it, "json") }

            // Write a small manifest
            val manifest =
                buildString {
                    appendLine("CamProV5 Support Bundle")
                    appendLine("sessionId=" + sessionId)
                    appendLine("created=" + ts)
                    appendLine("logDir=" + (logsPath?.toAbsolutePath()?.toString() ?: "(none)"))
                    appendLine("jsonDirs=" + jsonPaths.joinToString { it.toAbsolutePath().toString() })
                }
            val tmp = File.createTempFile("bundle_manifest", ".txt")
            tmp.writeText(manifest)
            zos.putNextEntry(ZipEntry("manifest.txt"))
            FileInputStream(tmp).use { it.copyTo(zos) }
            zos.closeEntry()
            tmp.delete()
        }
        return outZip.toString()
    }
}
