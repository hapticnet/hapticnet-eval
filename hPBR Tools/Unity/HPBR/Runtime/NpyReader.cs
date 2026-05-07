using System;
using System.IO;
using System.Text;
using System.Text.RegularExpressions;
using UnityEngine;

namespace HPBR
{
    /// <summary>
    /// Minimal reader for NumPy .npy files (format v1.0 and v2.0).
    /// Supports dtypes: float32, float64, int8, int16, int32, int64 — C (row-major) order only.
    /// Matches the arrays written by numpy.save() in the hPBR Python toolchain.
    /// </summary>
    public static class NpyReader
    {
        // Magic bytes: \x93NUMPY
        private static readonly byte[] Magic =
            { 0x93, (byte)'N', (byte)'U', (byte)'M', (byte)'P', (byte)'Y' };

        // ── Public API ────────────────────────────────────────────────────────────

        /// <summary>Load a .npy file and return it as a HapticArray (float storage).</summary>
        public static HapticArray Read(string path)
        {
            using (var fs     = new FileStream(path, FileMode.Open, FileAccess.Read))
            using (var reader = new BinaryReader(fs, Encoding.UTF8, leaveOpen: false))
                return Parse(reader, path);
        }

        /// <summary>
        /// Parse .npy bytes already in memory (e.g. extracted from a NARR chunk payload).
        /// </summary>
        public static HapticArray ReadBytes(byte[] data)
        {
            using (var ms     = new MemoryStream(data))
            using (var reader = new BinaryReader(ms, Encoding.UTF8, leaveOpen: false))
                return Parse(reader, "(embedded)");
        }

        // ── Core parser ───────────────────────────────────────────────────────────

        private static HapticArray Parse(BinaryReader reader, string label)
        {
            // ── Magic + version ───────────────────────────────────────────────
            byte[] magic = reader.ReadBytes(6);
            for (int i = 0; i < 6; i++)
                if (magic[i] != Magic[i])
                    throw new Exception($"[hPBR] Not a .npy file: {label}");

            byte major = reader.ReadByte();
            byte minor = reader.ReadByte();

            // ── Header length ─────────────────────────────────────────────────
            int headerLen;
            if (major == 1)
                headerLen = reader.ReadUInt16();        // 2-byte little-endian
            else if (major == 2)
                headerLen = (int)reader.ReadUInt32();   // 4-byte little-endian
            else
                throw new Exception($"[hPBR] Unsupported .npy version {major}.{minor}: {label}");

            // ── Header dict string ────────────────────────────────────────────
            string header = Encoding.ASCII.GetString(reader.ReadBytes(headerLen)).Trim();

            // ── Parse shape ───────────────────────────────────────────────────
            var shapeMatch = Regex.Match(header, @"'shape'\s*:\s*\(([^)]*)\)");
            if (!shapeMatch.Success)
                throw new Exception($"[hPBR] Cannot parse shape from .npy header in {label}");

            string[] parts = shapeMatch.Groups[1].Value
                .Split(new[] { ',' }, StringSplitOptions.RemoveEmptyEntries);

            int rows = 1, cols = 1;
            if (parts.Length >= 1 && int.TryParse(parts[0].Trim(), out int r)) rows = r;
            if (parts.Length >= 2 && int.TryParse(parts[1].Trim(), out int c)) cols = c;

            // ── Parse dtype descriptor ────────────────────────────────────────
            var descrMatch = Regex.Match(header, @"'descr'\s*:\s*'([^']+)'");
            if (!descrMatch.Success)
                throw new Exception($"[hPBR] Cannot parse descr from .npy header in {label}");
            string descr = descrMatch.Groups[1].Value;

            // ── Read data ─────────────────────────────────────────────────────
            int     total = rows * cols;
            float[] data  = new float[total];

            // Normalise descriptor: strip byte-order prefix and lowercase
            string d = descr.TrimStart('<', '>', '=', '|').ToLower();
            switch (d)
            {
                case "f4":
                    for (int i = 0; i < total; i++) data[i] = reader.ReadSingle();          break;
                case "f8":
                    for (int i = 0; i < total; i++) data[i] = (float)reader.ReadDouble();   break;
                case "i1": case "u1":
                    for (int i = 0; i < total; i++) data[i] = reader.ReadByte();            break;
                case "i2":
                    for (int i = 0; i < total; i++) data[i] = reader.ReadInt16();           break;
                case "u2":
                    for (int i = 0; i < total; i++) data[i] = reader.ReadUInt16();          break;
                case "i4":
                    for (int i = 0; i < total; i++) data[i] = reader.ReadInt32();           break;
                case "u4":
                    for (int i = 0; i < total; i++) data[i] = (float)reader.ReadUInt32();   break;
                case "i8":
                    for (int i = 0; i < total; i++) data[i] = (float)reader.ReadInt64();    break;
                case "u8":
                    for (int i = 0; i < total; i++) data[i] = (float)reader.ReadUInt64();   break;
                default:
                    throw new Exception($"[hPBR] Unsupported .npy dtype '{descr}' in {label}");
            }

            return new HapticArray(data, rows, cols);
        }
    }
}
