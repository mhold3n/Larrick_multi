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
            // Initialize PicoGK once for the session
            using var lib = new Library(voxelSize);

            // Read and parse JSON
            var jsonText = File.ReadAllText(inputPath);
            var doc = JsonDocument.Parse(jsonText);
            var root = doc.RootElement;

            var options = new JsonSerializerOptions
            {
                WriteIndented = true,
                PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower
            };

            if (root.ValueKind == JsonValueKind.Array)
            {
                // Batch mode
                var inputs = JsonSerializer.Deserialize<List<ProfileInput>>(jsonText, options) 
                             ?? new List<ProfileInput>();
                var results = new List<OracleResult>();

                foreach (var input in inputs)
                {
                    results.Add(EvaluateProfile(input, lib, voxelSize, slabThickness));
                }
                Console.WriteLine(JsonSerializer.Serialize(results, options));
            }
            else
            {
                // Single mode (legacy)
                var input = JsonSerializer.Deserialize<ProfileInput>(jsonText, options)
                            ?? throw new ArgumentException("Invalid profile input");
                var result = EvaluateProfile(input, lib, voxelSize, slabThickness);
                Console.WriteLine(JsonSerializer.Serialize(result, options));
            }

            return 0;
        }
        catch (Exception ex)
        {
            // Fail closed: any exception means infeasible
            Console.Error.WriteLine($"Fatal Error: {ex}");
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
            // In batch mode, we might want to return a list of fails, but for fatal crash just return one fail object
            // or let the caller handle the non-zero exit.
            // For now, print a single fail result to stdout so parsing might survive if it was single mode.
            Console.WriteLine(JsonSerializer.Serialize(failResult, new JsonSerializerOptions
            {
                WriteIndented = true,
                PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower
            }));
            return 1;
        }
    }

    static OracleResult EvaluateProfile(ProfileInput input, Library lib, float voxelSize, float slabThickness)
    {
        var process = input.Process;
        var outerPoints = ToVector2(input.Outer);
        
        var notes = new List<string>();

        // Build mesh: extrude 2D polygon into slab
        var mesh = ExtrudePolygon(outerPoints, slabThickness);
        int originalTriCount = mesh.nTriangleCount();
        if (originalTriCount == 0)
        {
            return new OracleResult(false, process.KerfBufferMm, 0, 0, 0, 0, 0, voxelSize,
                new[] { "Empty mesh after extrusion" });
        }

        // Voxelize
        // Note: Voxels constructor takes a Mesh. 
        using var voxOriginal = new Voxels(mesh);

        // Compute original area (approximate from bounding box cross-section)
        float areaOriginal = EstimateCrossSectionArea(voxOriginal, voxelSize, slabThickness);

        // --- Check A: Inward offset survival ---
        float kerfBuffer = process.KerfBufferMm;
        
        bool emptyAfterInset;
        float areaAfterInset = 0f;
        
        using (var voxInset = voxOriginal.voxOffset(-kerfBuffer))
        {
            var meshInset = voxInset.mshAsMesh();
            emptyAfterInset = meshInset.nTriangleCount() == 0;
            if (!emptyAfterInset)
            {
                areaAfterInset = EstimateCrossSectionArea(voxInset, voxelSize, slabThickness);
            }
        }

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
        int componentCount = 1; 
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

    static Vector2[] ToVector2(float[][] points)
    {
        var result = new Vector2[points.Length];
        for (int i = 0; i < points.Length; i++)
        {
            result[i] = new Vector2(points[i][0], points[i][1]);
        }
        return result;
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
            
            using var voxTest = voxOriginal.voxOffset(-mid);
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
    /// Uses center-point fan triangulation for caps, assuming star-shaped polygon (valid for gears).
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

        // Compute centroid for cap triangulation
        Vector2 center = Vector2.Zero;
        for (int i = 0; i < n; i++) center += points[i];
        center /= n;

        // Add center vertices
        int bottomCenter = mesh.nAddVertex(new Vector3(center.X, center.Y, zLo));
        int topCenter = mesh.nAddVertex(new Vector3(center.X, center.Y, zHi));

        // Add perimeter vertices
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
            // Winding: CCW from outside
            mesh.nAddTriangle(bottomVerts[i], bottomVerts[j], topVerts[j]);
            mesh.nAddTriangle(bottomVerts[i], topVerts[j], topVerts[i]);
        }

        // Top and bottom caps using center fan
        for (int i = 0; i < n; i++)
        {
            int j = (i + 1) % n;
            // Bottom (viewed from below, center is visible)
            // Winding: Clockwise around perimeter? No, standard mesh is CCW.
            // If viewed from OUTSIDE (below), normal points down.
            // Triangle: (Center, P_current, P_next) -> Normal = (P_curr - Center) x (P_next - Center)
            // If polygon is CCW in XY:
            // Cross product roughly (1,0) x (0,1) = (0,0,1) -> Points UP (inside).
            // We want Normal DOWN (0,0,-1).
            // So we need: (Center, P_next, P_current)
            mesh.nAddTriangle(bottomCenter, bottomVerts[j], bottomVerts[i]);

            // Top (viewed from above, normal points up)
            // Same polygon CCW.
            // Triangle: (Center, P_current, P_next) -> Normal UP (correct).
            mesh.nAddTriangle(topCenter, topVerts[i], topVerts[j]);
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
