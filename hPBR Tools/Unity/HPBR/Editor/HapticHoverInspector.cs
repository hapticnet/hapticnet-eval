using System;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;
using HPBR;

namespace HPBR.Editor
{
    /// <summary>
    /// SceneView overlay that shows per-pixel haptic properties when the mouse
    /// hovers over any GameObject that has a HapticSurface component.
    ///
    /// Mirrors hover.py from the Blender toolchain.
    ///
    /// Requirements on the target object:
    ///   • A MeshCollider (NOT BoxCollider/SphereCollider) — needed for UV raycasting.
    ///   • A HapticSurface component (added automatically by the importer).
    ///
    /// Toggle via:  Window > HPBR > Haptic Hover Inspector  (shows a check mark when active)
    /// </summary>
    [InitializeOnLoad]
    public static class HapticHoverInspector
    {
        // ── State ─────────────────────────────────────────────────────────────────
        private static bool     _enabled = false;
        private static string[] _lines   = Array.Empty<string>();
        private static Vector2  _mousePos;

        // GUI styles (lazily initialised inside an OnGUI call)
        private static GUIStyle _boxStyle;
        private static GUIStyle _lineStyle;

        // ── Initialise on domain reload ───────────────────────────────────────────

        static HapticHoverInspector()
        {
            SceneView.duringSceneGui -= OnSceneGUI;
            SceneView.duringSceneGui += OnSceneGUI;
        }

        // ── Menu entry ────────────────────────────────────────────────────────────

        [MenuItem("Window/HPBR/Haptic Hover Inspector", priority = 2)]
        private static void Toggle()
        {
            _enabled = !_enabled;
            _lines   = Array.Empty<string>();
            SceneView.RepaintAll();
            Debug.Log($"[hPBR] Haptic hover inspector {(_enabled ? "enabled" : "disabled")}.");
        }

        [MenuItem("Window/HPBR/Haptic Hover Inspector", validate = true)]
        private static bool ToggleValidate()
        {
            Menu.SetChecked("Window/HPBR/Haptic Hover Inspector", _enabled);
            return true;
        }

        // ── SceneView callback ────────────────────────────────────────────────────

        private static void OnSceneGUI(SceneView sv)
        {
            if (!_enabled) return;

            Event e = Event.current;

            if (e.type == EventType.MouseMove)
            {
                _mousePos = e.mousePosition;
                _lines    = BuildLines(_mousePos);
                sv.Repaint();
                e.Use();    // prevent other handlers from consuming the move event
            }

            if (e.type == EventType.Repaint && _lines.Length > 0)
                DrawTooltip(sv, _mousePos, _lines);
        }

        // ── Raycast + tooltip content ─────────────────────────────────────────────

        private static string[] BuildLines(Vector2 mousePos)
        {
            Ray ray = HandleUtility.GUIPointToWorldRay(mousePos);

            if (!Physics.Raycast(ray, out RaycastHit hit))
                return new[] { "No surface under cursor" };

            // Check for a HapticSurface up the hierarchy
            var haptic = hit.collider.GetComponent<HapticSurface>()
                      ?? hit.collider.GetComponentInParent<HapticSurface>();

            if (haptic == null)
                return new[]
                {
                    $"Object: {hit.collider.gameObject.name}",
                    "(no HapticSurface component)"
                };

            // Lazy-load haptic data
            if (haptic.MaterialProps == null)
                haptic.Reload();

            if (haptic.MaterialProps == null)
                return new[] { "Object: " + haptic.name, "(haptic data failed to load)" };

            float u = hit.textureCoord.x;
            float v = hit.textureCoord.y;

            var lines = new List<string>();

            // 0 ── Per-pixel HapticNet class name (changes as you hover — mirrors hover.py line 1)
            string matClass = "Unknown";
            if (haptic.SegmentationMap != null)
            {
                string raw = haptic.SegmentationMap.Sample(u, v);
                if (!string.IsNullOrEmpty(raw))
                {
                    raw      = raw.Replace("_", " ");
                    matClass = char.ToUpper(raw[0]) + raw.Substring(1);
                }
            }
            lines.Add($"Haptic PBR Material: {matClass}");

            // separator
            lines.Add("──────────────────────────");

            // 2 ── Known properties in defined order (matches _PROPERTY_DEFS in hover.py)
            foreach (var (stem, label) in HapticSurface.PropertyDefs)
            {
                string val = haptic.Sample(stem, u, v);
                if (val != null)                       // null means the property is absent
                    lines.Add($"{label}: {val}");
            }

            // 3 ── Any additional properties not in PropertyDefs
            var defined = new HashSet<string>();
            foreach (var (stem, _) in HapticSurface.PropertyDefs) defined.Add(stem);

            foreach (var kv in haptic.MaterialProps)
                if (!defined.Contains(kv.Key))
                    lines.Add($"{kv.Key}: {kv.Value.SampleString(u, v)}");

            // 4 ── Footer: object and file info
            lines.Add("──────────────────────────");
            lines.Add($"Object: {haptic.gameObject.name}");
            lines.Add($"File:   {System.IO.Path.GetFileName(haptic.HpbrFilePath)}");

            return lines.ToArray();
        }

        // ── Draw tooltip ──────────────────────────────────────────────────────────

        private static void DrawTooltip(SceneView sv, Vector2 mouse, string[] lines)
        {
            EnsureStyles();

            const float pad = 8f;
            float lineH = _lineStyle.lineHeight + 3f;

            // Measure widest line
            float maxW = 0f;
            foreach (var l in lines)
            {
                float w = _lineStyle.CalcSize(new GUIContent(l)).x;
                if (w > maxW) maxW = w;
            }

            float bw = maxW + pad * 2f;
            float bh = lineH * lines.Length + pad * 2f;

            // Anchor to cursor; nudge up if it would clip below the SceneView
            float sx = mouse.x + 20f;
            float sy = mouse.y + 10f;
            if (sy + bh > sv.position.height - 20f)
                sy = mouse.y - bh - 10f;

            Handles.BeginGUI();

            var bgRect = new Rect(sx, sy, bw, bh);

            // Solid dark background — EditorGUI.DrawRect ignores GUI alpha blending
            EditorGUI.DrawRect(bgRect, new Color(0.08f, 0.08f, 0.08f, 1f));

            // 1-pixel border
            EditorGUI.DrawRect(new Rect(sx,          sy,          bw, 1f), new Color(0.4f, 0.4f, 0.4f, 1f));
            EditorGUI.DrawRect(new Rect(sx,          sy + bh - 1, bw, 1f), new Color(0.4f, 0.4f, 0.4f, 1f));
            EditorGUI.DrawRect(new Rect(sx,          sy,          1f, bh), new Color(0.4f, 0.4f, 0.4f, 1f));
            EditorGUI.DrawRect(new Rect(sx + bw - 1, sy,          1f, bh), new Color(0.4f, 0.4f, 0.4f, 1f));

            // Text lines (top to bottom)
            for (int i = 0; i < lines.Length; i++)
            {
                var rect = new Rect(sx + pad, sy + pad + i * lineH,
                                    bw - pad * 2f, lineH);
                GUI.Label(rect, lines[i], _lineStyle);
            }

            Handles.EndGUI();
        }

        // ── Style helpers ─────────────────────────────────────────────────────────

        private static void EnsureStyles()
        {
            if (_boxStyle != null) return;

            _boxStyle = new GUIStyle(GUI.skin.box)
            {
                normal = { background = MakeTex(4, 4, new Color(0.08f, 0.08f, 0.08f, 1.00f)) },
                border  = new RectOffset(2, 2, 2, 2),
                padding = new RectOffset(0, 0, 0, 0),
            };

            _lineStyle = new GUIStyle(EditorStyles.label)
            {
                normal   = { textColor = Color.white },
                fontSize = 13,
                wordWrap = false,
            };
        }

        private static Texture2D MakeTex(int w, int h, Color col)
        {
            var pix = new Color[w * h];
            for (int i = 0; i < pix.Length; i++) pix[i] = col;
            var t = new Texture2D(w, h);
            t.SetPixels(pix);
            t.Apply();
            return t;
        }
    }
}
