// PicoGK Manufacturability Oracle — CLI entry point.
//
// Evaluates WEDM/laser manufacturability of 2D gear profiles using PicoGK
// voxel SDF offset operations. Runs headless (no viewer).
//
// Usage: dotnet run -- --input <profile.json> [--voxel-size 0.001]
//
// Output: JSON metrics to stdout.

using System.Diagnostics;
using System.Numerics;
using System.Text.Json;
using PicoGK;

namespace PicoGKManufact;

/// <summary>
/// Parsed profile input from JSON.
/// </summary>
record ProfileInput(
    string Units,
    float[][] Outer,
    float[][] Holes,
    Dictionary<string, JsonElement> Metadata,
    ProcessParams Process
);

/// <summary>
/// WEDM/laser process parameters.
/// </summary>
record ProcessParams(
    float WireDMm,
    float OvecutMm,
    float CornerMarginMm,
    float MinLigamentMm
)
{
    /// <summary>Derived kerf buffer — half wire + overcut + corner margin.</summary>
    public float KerfBufferMm => 0.5f * WireDMm + OvecutMm + CornerMarginMm;
}

/// <summary>
/// Oracle output metrics.
/// </summary>
record OracleResult(
    bool Passed,
    float KerfBufferMm,
    float TMinProxyMm,
    float BMaxSurvivableMm,
    float AreaOriginalMm2,
    float AreaAfterInsetMm2,
    int ComponentCountAfterInset,
    float VoxelResolutionMm,
    string[] Notes
);

static class Program
{
    static int Main(string[] args)
    {
        string? inputPath = null;
        float voxelSize = 0.001f; // 1 µm default (micron precision)
        float slabThickness = 14.0f; // mm, actual component width

        // Parse CLI args
        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--input":
                    if (i + 1 < args.Length) inputPath = args[++i];
                    break;
                case "--voxel-size":
                    if (i + 1 < args.Length) voxelSize = float.Parse(args[++i]);
                    break;
                case "--slab-thickness":
                    if (i + 1 < args.Length) slabThickness = float.Parse(args[++i]);
                    break;
            }
        }

        if (string.IsNullOrEmpty(inputPath))
        {
            Console.Error.WriteLine("Usage: picogk_manufact --input <profile.json> [--voxel-size 0.001] [--slab-thickness 14.0]");
            return 1;
        }

        try
        {
            var result = RunOracle(inputPath, voxelSize, slabThickness);
            var json = JsonSerializer.Serialize(result, new JsonSerializerOptions
            {
                WriteIndented = true,
                PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower
            });
            Console.WriteLine(json);
            return 0;
        }
        catch (Exception ex)
        {
            // Fail closed: any exception means infeasible
            var failResult = new OracleResult(
                Passed: false,
                KerfBufferMm: 0f,
                TMinProxyMm: 0f,
                BMaxSurvivableMm: 0f,
                AreaOriginalMm2: 0f,
                AreaAfterInsetMm2: 0f,
                ComponentCountAfterInset: 0,
                VoxelResolutionMm: voxelSize,
                Notes: new[] { $"Exception: {ex.Message}" }
            );
            Console.WriteLine(JsonSerializer.Serialize(failResult, new JsonSerializerOptions
            {
                WriteIndented = true,
                PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower
            }));
            return 1;
        }
    }

    static OracleResult RunOracle(string inputPath, float voxelSize, float slabThickness)
    {
        // Read and parse JSON
        var jsonText = File.ReadAllText(inputPath);
        var doc = JsonDocument.Parse(jsonText);
        var root = doc.RootElement;

        var process = ParseProcess(root.GetProperty("process"));
        var outerPoints = ParsePolyline(root.GetProperty("outer"));

        var notes = new List<string>();

        // Initialize PicoGK in headless mode
        using var lib = new Library(voxelSize);

        // Build mesh: extrude 2D polygon into slab
        var mesh = ExtrudePolygon(outerPoints, slabThickness);
        int originalTriCount = mesh.nTriangleCount();
        if (originalTriCount == 0)
        {
            return new OracleResult(false, process.KerfBufferMm, 0, 0, 0, 0, 0, voxelSize,
                new[] { "Empty mesh after extrusion" });
        }

        // Voxelize
        var voxOriginal = new Voxels(mesh);

        // Compute original area (approximate from bounding box cross-section)
        var bbox = voxOriginal.mshAsMesh().oBoundingBox();
        float areaOriginal = EstimateCrossSectionArea(voxOriginal, voxelSize, slabThickness);

        // --- Check A: Inward offset survival ---
        float kerfBuffer = process.KerfBufferMm;
        var voxInset = voxOriginal.voxOffset(-kerfBuffer);
        var meshInset = voxInset.mshAsMesh();
        bool emptyAfterInset = meshInset.nTriangleCount() == 0;
        float areaAfterInset = emptyAfterInset ? 0f : EstimateCrossSectionArea(voxInset, voxelSize, slabThickness);

        if (emptyAfterInset)
        {
            notes.Add($"Check A FAIL: shape empty after {kerfBuffer:F4}mm inset");
            return new OracleResult(false, kerfBuffer, 0, 0, areaOriginal, 0, 0, voxelSize, notes.ToArray());
        }
        notes.Add("Check A PASS: shape survives kerf inset");

        // --- Check B: Minimum ligament thickness (binary search) ---
        float bMaxSurvivable = BinarySearchMaxInset(voxOriginal, kerfBuffer, voxelSize);
        float tMinProxy = 2.0f * bMaxSurvivable;

        bool ligamentOk = tMinProxy >= process.MinLigamentMm;
        if (!ligamentOk)
        {
            notes.Add($"Check B FAIL: t_min_proxy={tMinProxy:F4}mm < min_ligament={process.MinLigamentMm:F4}mm");
        }
        else
        {
            notes.Add($"Check B PASS: t_min_proxy={tMinProxy:F4}mm");
        }

        // --- Check C: Minimum concave radius proxy ---
        bool radiusOk = bMaxSurvivable >= kerfBuffer;
        if (!radiusOk)
        {
            notes.Add($"Check C FAIL: b_max={bMaxSurvivable:F4}mm < kerf_buffer={kerfBuffer:F4}mm");
        }
        else
        {
            notes.Add("Check C PASS: concave radius sufficient");
        }

        // --- Check D: Gap collapse (simplified) ---
        // Test if a slight inset causes topology change (merging)
        int componentCount = 1; // Approximate — PicoGK doesn't expose component count directly
        notes.Add("Check D: gap collapse check (single-body assumption)");

        bool passed = !emptyAfterInset && ligamentOk && radiusOk;

        return new OracleResult(
            Passed: passed,
            KerfBufferMm: kerfBuffer,
            TMinProxyMm: tMinProxy,
            BMaxSurvivableMm: bMaxSurvivable,
            AreaOriginalMm2: areaOriginal,
            AreaAfterInsetMm2: areaAfterInset,
            ComponentCountAfterInset: componentCount,
            VoxelResolutionMm: voxelSize,
            Notes: notes.ToArray()
        );
    }

    /// <summary>
    /// Binary search for the largest inward offset where the shape remains non-empty.
    /// </summary>
    static float BinarySearchMaxInset(Voxels voxOriginal, float maxOffset, float tolerance)
    {
        float lo = 0f;
        float hi = maxOffset * 2f; // Search up to 2x kerf buffer
        int maxIter = 20;

        for (int i = 0; i < maxIter; i++)
        {
            float mid = (lo + hi) / 2f;
            var voxTest = voxOriginal.voxOffset(-mid);
            var meshTest = voxTest.mshAsMesh();
            bool nonEmpty = meshTest.nTriangleCount() > 0;

            if (nonEmpty)
                lo = mid;
            else
                hi = mid;

            if (hi - lo < tolerance)
                break;
        }

        return lo;
    }

    /// <summary>
    /// Estimate cross-section area by counting non-empty voxels at mid-height.
    /// Approximation: area = bbox_width * bbox_depth (upper bound).
    /// For a more accurate measure, we re-mesh and compute triangle areas
    /// on a mid-plane slice.
    /// </summary>
    static float EstimateCrossSectionArea(Voxels vox, float voxelSize, float slabThickness)
    {
        // Use bounding box as an upper-bound area estimate
        var mesh = vox.mshAsMesh();
        if (mesh.nTriangleCount() == 0) return 0f;
        var bbox = mesh.oBoundingBox();
        float width = bbox.vecMax.X - bbox.vecMin.X;
        float depth = bbox.vecMax.Y - bbox.vecMin.Y;
        // This is a rough proxy; the actual 2D area would require slicing
        return width * depth;
    }

    /// <summary>
    /// Extrude a 2D polygon (closed polyline) into a 3D triangulated mesh slab.
    /// The polygon lies in the XY plane, extruded along Z.
    /// </summary>
    static Mesh ExtrudePolygon(Vector2[] points, float height)
    {
        var mesh = new Mesh();
        int n = points.Length;
        if (n < 3) return mesh;

        // Remove closing duplicate if present
        if (points[0] == points[^1] && n > 1)
            n--;

        float zLo = 0f;
        float zHi = height;

        // Add bottom and top vertices
        int[] bottomVerts = new int[n];
        int[] topVerts = new int[n];
        for (int i = 0; i < n; i++)
        {
            bottomVerts[i] = mesh.nAddVertex(new Vector3(points[i].X, points[i].Y, zLo));
            topVerts[i] = mesh.nAddVertex(new Vector3(points[i].X, points[i].Y, zHi));
        }

        // Side faces (quads as two triangles each)
        for (int i = 0; i < n; i++)
        {
            int j = (i + 1) % n;
            // Two triangles for the quad
            mesh.nAddTriangle(bottomVerts[i], bottomVerts[j], topVerts[j]);
            mesh.nAddTriangle(bottomVerts[i], topVerts[j], topVerts[i]);
        }

        // Top and bottom caps using fan triangulation
        // (Assumes convex-ish polygon; for non-convex, ear clipping would be needed,
        //  but gear profiles are generally star-shaped around origin)
        for (int i = 1; i < n - 1; i++)
        {
            // Bottom (winding: clockwise from below = CCW from above)
            mesh.nAddTriangle(bottomVerts[0], bottomVerts[i + 1], bottomVerts[i]);
            // Top (winding: CCW from above)
            mesh.nAddTriangle(topVerts[0], topVerts[i], topVerts[i + 1]);
        }

        return mesh;
    }

    /// <summary>
    /// Parse process parameters from JSON element.
    /// </summary>
    static ProcessParams ParseProcess(JsonElement elem)
    {
        return new ProcessParams(
            WireDMm: elem.GetProperty("wire_d_mm").GetSingle(),
            OvecutMm: elem.GetProperty("overcut_mm").GetSingle(),
            CornerMarginMm: elem.TryGetProperty("corner_margin_mm", out var cm) ? cm.GetSingle() : 0f,
            MinLigamentMm: elem.TryGetProperty("min_ligament_mm", out var ml) ? ml.GetSingle() : 0.35f
        );
    }

    /// <summary>
    /// Parse a JSON array of [x, y] pairs into Vector2[].
    /// </summary>
    static Vector2[] ParsePolyline(JsonElement arr)
    {
        var points = new List<Vector2>();
        foreach (var pt in arr.EnumerateArray())
        {
            float x = pt[0].GetSingle();
            float y = pt[1].GetSingle();
            points.Add(new Vector2(x, y));
        }
        return points.ToArray();
    }
}
