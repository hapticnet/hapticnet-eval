using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEngine;
using HPBR;

namespace HPBR.Editor
{
    /// <summary>
    /// Editor window that reads a .hpbr file, wires all PBR textures onto a
    /// URP Lit material, and attaches a HapticSurface component to the selected
    /// GameObject — mirroring hpbr_importer.py in Blender.
    ///
    /// Open via:  Window > HPBR > Import hPBR Material
    /// </summary>
    public class HpbrImporterWindow : EditorWindow
    {
        // ── State ─────────────────────────────────────────────────────────────────
        private string   _hpbrPath       = "";
        private bool     _addMeshCollider = true;
        private string   _status         = "";
        private Vector2  _scroll;
        private HpbrData _previewData;   // shown after a successful import

        // ── Menu entry ────────────────────────────────────────────────────────────

        [MenuItem("Window/HPBR/Import hPBR Material", priority = 1)]
        public static void ShowWindow() =>
            GetWindow<HpbrImporterWindow>("hPBR Importer").Show();

        // ── GUI ───────────────────────────────────────────────────────────────────

        private void OnGUI()
        {
            _scroll = EditorGUILayout.BeginScrollView(_scroll);

            EditorGUILayout.Space(8);
            EditorGUILayout.LabelField("hPBR Material Importer", EditorStyles.boldLabel);
            EditorGUILayout.LabelField("Reads a .hpbr file and applies it to the selected object.",
                                       EditorStyles.miniLabel);
            EditorGUILayout.Space(10);

            // ── File picker ───────────────────────────────────────────────────────
            EditorGUILayout.BeginHorizontal();
            _hpbrPath = EditorGUILayout.TextField("hPBR File", _hpbrPath);
            if (GUILayout.Button("Browse", GUILayout.Width(70)))
            {
                string p = EditorUtility.OpenFilePanel("Select .hpbr file",
                               string.IsNullOrEmpty(_hpbrPath) ? "" : Path.GetDirectoryName(_hpbrPath),
                               "hpbr");
                if (!string.IsNullOrEmpty(p)) _hpbrPath = p;
            }
            EditorGUILayout.EndHorizontal();

            // ── Options ───────────────────────────────────────────────────────────
            _addMeshCollider = EditorGUILayout.Toggle(
                new GUIContent("Add MeshCollider",
                    "Required for per-pixel UV sampling in the Haptic Hover Inspector."),
                _addMeshCollider);

            EditorGUILayout.Space(8);

            // ── Selected object ───────────────────────────────────────────────────
            GameObject sel      = Selection.activeGameObject;
            string     selLabel = sel != null ? sel.name : "(none — select an object in the scene)";
            EditorGUILayout.LabelField("Selected Object", selLabel);

            EditorGUILayout.Space(8);

            // ── Apply button ──────────────────────────────────────────────────────
            bool canApply = sel != null
                         && !string.IsNullOrEmpty(_hpbrPath)
                         && File.Exists(_hpbrPath);

            using (new EditorGUI.DisabledScope(!canApply))
            {
                if (GUILayout.Button("Apply hPBR to Selected Object", GUILayout.Height(32)))
                    ApplyToObject(sel);
            }

            // ── Status ────────────────────────────────────────────────────────────
            if (!string.IsNullOrEmpty(_status))
            {
                EditorGUILayout.Space(4);
                bool isError = _status.StartsWith("Error");
                EditorGUILayout.HelpBox(_status, isError ? MessageType.Error : MessageType.Info);
            }

            // ── Preview of loaded data ────────────────────────────────────────────
            if (_previewData != null)
            {
                EditorGUILayout.Space(8);
                EditorGUILayout.LabelField("Loaded PBR Textures", EditorStyles.boldLabel);
                foreach (var kv in _previewData.PbrTiles)
                    EditorGUILayout.LabelField($"  • {kv.Key}", $"{kv.Value.width} × {kv.Value.height}");

                EditorGUILayout.Space(4);
                EditorGUILayout.LabelField("Loaded Haptic Properties", EditorStyles.boldLabel);
                foreach (var kv in _previewData.MaterialProps)
                    EditorGUILayout.LabelField($"  • {kv.Key}", $"{kv.Value.Rows} × {kv.Value.Cols}");
            }

            EditorGUILayout.EndScrollView();
        }

        // ── Import logic ──────────────────────────────────────────────────────────

        private void ApplyToObject(GameObject obj)
        {
            try
            {
                // 1 ── Parse the file
                HpbrData data = HpbrReader.Read(_hpbrPath);
                _previewData  = data;

                // 2 ── Save textures + material as Unity assets
                string stem    = Path.GetFileNameWithoutExtension(_hpbrPath);
                string assetDir = EnsureAssetDir($"Assets/HPBR_Imported/{stem}");
                SaveTextures(data.PbrTiles, assetDir);
                Material mat = BuildAndSaveMaterial(data, stem, assetDir);

                // 3 ── Assign material to renderer
                var rend = obj.GetComponent<Renderer>()
                        ?? obj.GetComponentInChildren<Renderer>();
                if (rend != null)
                {
                    Undo.RecordObject(rend, "Apply hPBR Material");
                    rend.sharedMaterial = mat;
                }
                else
                {
                    Debug.LogWarning($"[hPBR] No Renderer on '{obj.name}' — material created but not assigned.");
                }

                // 4 ── Attach / update HapticSurface component
                var haptic = obj.GetComponent<HapticSurface>()
                          ?? Undo.AddComponent<HapticSurface>(obj);
                Undo.RecordObject(haptic, "Apply hPBR Material");
                haptic.HpbrFilePath  = _hpbrPath;
                haptic.MaterialProps = data.MaterialProps;

                // 4a ── Load metadata.json (material name) from same directory
                string hpbrDir  = Path.GetDirectoryName(_hpbrPath);
                string metaPath = Path.Combine(hpbrDir, "metadata.json");
                if (File.Exists(metaPath))
                {
                    haptic.MaterialName = HapticSurface.ParseNameFromMetadata(metaPath);
                    Debug.Log($"[hPBR] Material name: {haptic.MaterialName}");
                }
                else
                {
                    Debug.LogWarning($"[hPBR] metadata.json not found next to .hpbr file.");
                }

                // 4b ── Segmentation map is now embedded in the .hpbr file as a NARR chunk
                if (data.SegmentationMap != null)
                {
                    haptic.SegmentationMap = data.SegmentationMap;
                    Debug.Log($"[hPBR] HapticNet seg: {haptic.SegmentationMap.Rows}×{haptic.SegmentationMap.Cols}");
                }
                else
                {
                    Debug.LogWarning("[hPBR] No segmentation map found in .hpbr file.");
                }

                // 5 ── Ensure MeshCollider (required for UV raycasting)
                if (_addMeshCollider && obj.GetComponent<MeshCollider>() == null)
                    Undo.AddComponent<MeshCollider>(obj);

                AssetDatabase.SaveAssets();
                AssetDatabase.Refresh();

                string segText = data.SegmentationMap != null ? " + segmentation" : "";
                _status = $"Applied {data.PbrTiles.Count} texture(s) and " +
                          $"{data.MaterialProps.Count} haptic property map(s){segText} to '{obj.name}'.";
                Debug.Log($"[hPBR] {_status}");
            }
            catch (System.Exception e)
            {
                _status = $"Error: {e.Message}";
                Debug.LogError($"[hPBR] Import failed: {e}");
            }
        }

        // ── Material builder ──────────────────────────────────────────────────────

        private static Material BuildAndSaveMaterial(HpbrData data, string stem, string assetDir)
        {
            // Prefer the custom HPBR shader (has displacement); fall back to URP Lit
            var shader = Shader.Find("HPBR/Surface") ?? Shader.Find("Universal Render Pipeline/Lit");
            var mat    = new Material(shader) { name = stem };

            // Collect what we have
            Texture2D basecolor   = FindTile(data, "basecolor", "albedo", "diffuse");
            Texture2D normalMap   = FindTile(data, "normal");
            Texture2D metallicTex = FindTile(data, "metallic", "metalness");
            Texture2D roughness   = FindTile(data, "roughness");
            Texture2D occlusion   = FindTile(data, "occlusion");
            Texture2D emission    = FindTile(data, "emission");
            Texture2D heightMap   = FindTile(data, "height", "displacement");

            // Base colour
            if (basecolor != null)
            {
                mat.SetTexture("_BaseMap", basecolor);
                mat.SetColor("_BaseColor", Color.white);
            }

            // Normal map
            if (normalMap != null)
            {
                mat.SetTexture("_BumpMap", normalMap);
                mat.SetFloat("_BumpScale", 1f);
                mat.EnableKeyword("_NORMALMAP");
            }

            // Metallic + smoothness
            // URP Lit reads smoothness from the alpha channel of _MetallicGlossMap.
            // We pack:  R = metallic (0 if absent),  A = 1 - roughness (0.5 if absent).
            Texture2D metalSmooth = BuildMetallicSmoothness(metallicTex, roughness,
                                        $"{stem}_metallic_smooth", assetDir);
            mat.SetTexture("_MetallicGlossMap", metalSmooth);
            mat.EnableKeyword("_METALLICSPECGLOSSMAP");
            mat.SetFloat("_SmoothnessTextureChannel", 0f); // read from _MetallicGlossMap alpha

            // Occlusion
            if (occlusion != null)
            {
                mat.SetTexture("_OcclusionMap", occlusion);
                mat.SetFloat("_OcclusionStrength", 1f);
            }

            // Emission
            if (emission != null)
            {
                mat.SetTexture("_EmissionMap", emission);
                mat.SetColor("_EmissionColor", Color.white);
                mat.EnableKeyword("_EMISSION");
                mat.globalIlluminationFlags = MaterialGlobalIlluminationFlags.BakedEmissive;
            }

            // Height / displacement (used by HPBR/Surface shader for parallax mapping)
            if (heightMap != null)
            {
                mat.SetTexture("_HeightMap", heightMap);
                mat.SetFloat("_HeightScale", 0.02f);   // adjust in Inspector
                mat.SetFloat("_HeightSteps", 24f);
            }

            // Save material as asset
            string matPath = $"{assetDir}/{stem}.mat";
            AssetDatabase.CreateAsset(mat, matPath);
            return mat;
        }

        /// <summary>
        /// Creates a metallic-smoothness packed texture:  R = metallic, A = 1 - roughness.
        /// Both inputs are optional; missing channels fall back to sensible defaults.
        /// </summary>
        private static Texture2D BuildMetallicSmoothness(Texture2D metallic, Texture2D roughness,
                                                         string name, string assetDir)
        {
            int w = 4, h = 4;
            if (metallic  != null) { w = metallic.width;  h = metallic.height; }
            if (roughness != null) { w = roughness.width; h = roughness.height; }

            Color[] mPix = metallic  != null ? metallic.GetPixels()  : null;
            Color[] rPix = roughness != null ? roughness.GetPixels() : null;

            int total  = w * h;
            var pixels = new Color[total];
            for (int i = 0; i < total; i++)
            {
                float m = mPix != null ? mPix[i % mPix.Length].r : 0f;
                float s = rPix != null ? 1f - rPix[i % rPix.Length].r : 0.5f;
                pixels[i] = new Color(m, 0f, 0f, s);
            }

            var tex = new Texture2D(w, h, TextureFormat.RGBA32, mipChain: true, linear: true);
            tex.name = name;
            tex.SetPixels(pixels);
            tex.Apply();

            string path = $"{assetDir}/{name}.png";
            File.WriteAllBytes(Path.GetFullPath(path), tex.EncodeToPNG());
            UnityEngine.Object.DestroyImmediate(tex);

            AssetDatabase.ImportAsset(path);
            var imported = AssetDatabase.LoadAssetAtPath<Texture2D>(path);
            // Mark as linear (non-sRGB)
            var imp = AssetImporter.GetAtPath(path) as TextureImporter;
            if (imp != null) { imp.sRGBTexture = false; imp.SaveAndReimport(); }
            return imported;
        }

        // ── Asset helpers ─────────────────────────────────────────────────────────

        private static void SaveTextures(Dictionary<string, Texture2D> tiles, string assetDir)
        {
            // Snapshot keys first — modifying the dict while iterating throws InvalidOperationException
            var keys = new List<string>(tiles.Keys);
            foreach (string key in keys)
            {
                Texture2D tex  = tiles[key];
                string    path = $"{assetDir}/{key}.png";
                File.WriteAllBytes(Path.GetFullPath(path), tex.EncodeToPNG());
                AssetDatabase.ImportAsset(path);

                // Colour-data textures (basecolor, diffuse, emission) need sRGB.
                // All data/mask textures must be linear so shader values are correct.
                string keyLo    = key.ToLower();
                bool isNormal   = keyLo.Contains("normal");
                bool isLinear   = isNormal
                               || keyLo.Contains("height")
                               || keyLo.Contains("displacement")
                               || keyLo.Contains("roughness")
                               || keyLo.Contains("metallic")
                               || keyLo.Contains("specular")
                               || keyLo.Contains("opacity")
                               || keyLo.Contains("occlusion")
                               || keyLo.Contains("ao");
                var  imp        = AssetImporter.GetAtPath(path) as TextureImporter;
                if (imp != null)
                {
                    imp.sRGBTexture = !isLinear;
                    imp.textureType = isNormal ? TextureImporterType.NormalMap
                                               : TextureImporterType.Default;
                    imp.isReadable  = true;
                    imp.SaveAndReimport();
                }

                // Replace in-memory Texture2D with the saved asset so the material refs it
                UnityEngine.Object.DestroyImmediate(tex);
                tiles[key] = AssetDatabase.LoadAssetAtPath<Texture2D>(path);
            }
        }

        private static string EnsureAssetDir(string assetPath)
        {
            string[] parts  = assetPath.Split('/');
            string   current = parts[0];
            for (int i = 1; i < parts.Length; i++)
            {
                string next = current + "/" + parts[i];
                if (!AssetDatabase.IsValidFolder(next))
                    AssetDatabase.CreateFolder(current, parts[i]);
                current = next;
            }
            return current;
        }

        private static Texture2D FindTile(HpbrData data, params string[] keys)
        {
            foreach (string k in keys)
                if (data.PbrTiles.TryGetValue(k, out var t)) return t;
            return null;
        }
    }
}
