using System.Collections.Generic;
using System.IO;
using System.Text.RegularExpressions;
using UnityEngine;

namespace HPBR
{
    /// <summary>
    /// Attached to a GameObject when an .hpbr file is applied via the importer window.
    /// Stores the path to the source file and the decoded haptic property arrays.
    ///
    /// The arrays are NOT serialized (too large for Unity assets) — they are
    /// reloaded at import time and can be reloaded on demand via Reload().
    /// </summary>
    [AddComponentMenu("HPBR/Haptic Surface")]
    public class HapticSurface : MonoBehaviour
    {
        // ── Serialized (persists in scene) ────────────────────────────────────────
        [Tooltip("Absolute path to the .hpbr file that was applied to this object.")]
        public string HpbrFilePath = "";

        [Tooltip("Human-readable material name read from metadata.json (e.g. 'Brown Mud Leaves 01').")]
        public string MaterialName = "";

        // ── Runtime only (populated at import / Reload time) ──────────────────────
        [System.NonSerialized] public Dictionary<string, HapticArray> MaterialProps;

        /// <summary>Per-pixel HapticNet class name strings (*_HapticNetSegmentation.npy).</summary>
        [System.NonSerialized] public StringArray SegmentationMap;

        // ── Property display order (mirrors hover.py _PROPERTY_DEFS) ─────────────
        public static readonly (string stem, string label)[] PropertyDefs =
        {
            ("youngs_modulus",       "Young's Modulus [MPa]"),
            ("poissons_ratio",       "Poisson's Ratio"),
            ("roughness_average_ra", "Roughness Average (Ra) [µm]"),
            ("static_friction",      "Static Friction Coefficient"),
            ("kinetic_friction",     "Kinetic Friction Coefficient"),
            ("thermal_conductivity", "Thermal Conductivity [W/(mK)]"),
            ("thermal_effusivity",   "Thermal Effusivity [Ws^0.5/(m²K)]"),
        };

        // ── Sampling ──────────────────────────────────────────────────────────────

        /// <summary>
        /// Sample a haptic property at UV (u, v).
        /// Returns a formatted string matching hover.py's output.
        /// </summary>
        public string Sample(string stem, float u, float v)
        {
            if (MaterialProps == null)                        return "(not loaded)";
            if (!MaterialProps.TryGetValue(stem, out var a))  return null; // missing → skip
            return a.SampleString(u, v);
        }

        // ── Reload from disk ──────────────────────────────────────────────────────

        /// <summary>
        /// Re-read haptic data, metadata, and segmentation map from HpbrFilePath.
        /// Call this after a scene load if you need runtime sampling.
        /// </summary>
        public void Reload()
        {
            if (string.IsNullOrEmpty(HpbrFilePath))
            {
                Debug.LogWarning("[hPBR] HapticSurface.Reload() called but HpbrFilePath is empty.");
                return;
            }
            try
            {
                // Haptic property arrays + embedded segmentation map
                var data = HpbrReader.Read(HpbrFilePath);
                MaterialProps = data.MaterialProps;
                Debug.Log($"[hPBR] Reloaded {MaterialProps.Count} property map(s) from {HpbrFilePath}");

                // HapticNet segmentation is now embedded in the .hpbr file as a NARR chunk
                if (data.SegmentationMap != null)
                {
                    SegmentationMap = data.SegmentationMap;
                    Debug.Log($"[hPBR] Loaded HapticNet seg {SegmentationMap.Rows}×{SegmentationMap.Cols}");
                }

                string dir = Path.GetDirectoryName(HpbrFilePath);

                // Material name from metadata.json
                string metaPath = Path.Combine(dir, "metadata.json");
                if (File.Exists(metaPath))
                    MaterialName = ParseNameFromMetadata(metaPath);
            }
            catch (System.Exception e)
            {
                Debug.LogError($"[hPBR] Reload failed: {e.Message}");
            }
        }

        // ── Helpers ───────────────────────────────────────────────────────────────

        /// <summary>Extract the "name" field from metadata.json without a full JSON library.</summary>
        public static string ParseNameFromMetadata(string metaPath)
        {
            string json  = File.ReadAllText(metaPath);
            var    match = Regex.Match(json, @"""name""\s*:\s*""([^""]+)""");
            return match.Success ? match.Groups[1].Value : "";
        }
    }
}
