using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Text.RegularExpressions;
using UnityEngine;

namespace HPBR
{
    /// <summary>
    /// A 2-D array of strings — used to store per-pixel material class names
    /// from the HapticNet segmentation map.
    /// </summary>
    public class StringArray
    {
        public readonly string[] Data;
        public readonly int      Rows;
        public readonly int      Cols;

        public StringArray(string[] data, int rows, int cols)
        {
            Data = data;
            Rows = rows;
            Cols = cols;
        }

        /// <summary>Sample at normalised UV (wrapping, matches hover.py indexing).</summary>
        public string Sample(float u, float v)
        {
            int x = ((int)(u * Cols) % Cols + Cols) % Cols;
            int y = ((int)(v * Rows) % Rows + Rows) % Rows;
            return Data[y * Cols + x];
        }
    }

    /// <summary>
    /// Reads *_HapticNetSegmentation.npy — a numpy object array where every pixel
    /// stores the material class name as a Python string (e.g. "mud", "leaf").
    ///
    /// Internally the .npy data section is a Python pickle stream.  This reader
    /// implements just enough pickle opcodes to reconstruct a flat string array in
    /// row-major order, matching numpy's C-order layout.
    /// </summary>
    public static class HapticNetSegReader
    {
        // ── Strings that are numpy / pickle internals, not material classes ────────
        // Class names can be multi-word with spaces, mixed case, and parentheses
        // (e.g. "Terracotta Ceramics", "Resin Birch (Betula glandulosa) Wood").
        // We filter by known numpy/pickle internal identifiers instead.
        private static readonly HashSet<string> Skip = new HashSet<string>(
            StringComparer.OrdinalIgnoreCase)
        {
            "numpy","ndarray","dtype","reconstruct","multiarray","b","r","c","f",
            "version","typecode","shape","rawdata","data","strides","descr","order",
            "is_f_order","allow_pickle","umath","core",
            // numpy dtype descriptor codes (1–2 char):
            "O8","O4","U","S","V","i1","i2","i4","i8","u1","u2","u4","u8",
            "f4","f8","f2","c8","c16","m8","M8",
        };

        /// <summary>
        /// Returns true when <paramref name="s"/> is a material class name.
        /// Class names may contain spaces, uppercase letters, digits, and parentheses
        /// (e.g. "Terracotta Ceramics", "Mud shale Stone").
        /// We reject numpy internals: strings containing '.', starting with '_',
        /// single-character strings, and known numpy/pickle keyword strings.
        /// </summary>
        private static bool IsClass(string s)
        {
            if (s == null || s.Length < 2) return false;
            if (s.Contains('.') || s.StartsWith("_", StringComparison.Ordinal)) return false;
            return !Skip.Contains(s);
        }

        // ── Public entry points ────────────────────────────────────────────────────

        /// <summary>Parse a *_HapticNetSegmentation.npy file from disk.</summary>
        public static StringArray Read(string path) => Read(File.ReadAllBytes(path));

        /// <summary>
        /// Parse .npy bytes already in memory (e.g. extracted from an embedded NARR chunk).
        /// </summary>
        public static StringArray Read(byte[] all)
        {

            // ── Parse .npy header ─────────────────────────────────────────────────
            int  pos    = 6;                      // skip 6-byte magic
            byte major  = all[pos++];
            /* minor */ pos++;

            int headerLen = (major == 1)
                ? all[pos] | (all[pos + 1] << 8)
                : BitConverter.ToInt32(all, pos);
            pos += (major == 1) ? 2 : 4;

            string header = Encoding.ASCII.GetString(all, pos, headerLen).Trim();
            pos += headerLen;

            var shapeM = Regex.Match(header, @"'shape'\s*:\s*\(([^)]*)\)");
            string[] sp = shapeM.Groups[1].Value.Split(',');
            int rows  = int.Parse(sp[0].Trim());
            int cols  = (sp.Length > 1 && int.TryParse(sp[1].Trim(), out int c)) ? c : 1;
            int total = rows * cols;

            // ── Scan pickle stream ─────────────────────────────────────────────────
            // Strategy:
            //   • Read every string literal (SHORT_BINUNICODE / BINUNICODE)
            //   • Track memo assignments (MEMOIZE / BINPUT / LONG_BINPUT)
            //     — memo stores class name or null for non-class strings
            //   • On BINGET / LONG_BINGET recall the memo entry
            //   • Class names flow into `flat` in row-major pixel order
            //
            // Memory note: some files store every pixel as a fresh string+MEMOIZE
            // (no BINGET reuse), which would otherwise create millions of memo entries.
            // We cap the memo at MemoSize — class-name IDs from BINGET-style files
            // are always small (< 50), well within the cap.

            const int MemoSize = 1024;
            var    memo        = new string[MemoSize]; // memo-id → class or null (capped)
            int    memoCounter = 0;     // MEMOIZE uses sequential ids
            string lastStr     = null;  // last string pushed onto stack
            bool   lastIsStr   = false; // TOS is a string we just read

            var flat = new string[total];
            int fill = 0;
            int i    = pos;
            int end  = all.Length;

            while (i < end && fill < total)
            {
                byte op = all[i++];

                switch (op)
                {
                    // ── Framing ───────────────────────────────────────────────
                    case 0x80: i++;      break; // PROTO — skip version
                    case 0x95: i += 8;   break; // FRAME — skip length

                    // ── String literals ───────────────────────────────────────
                    // string.Intern() ensures repeated class names share one heap object,
                    // avoiding millions of duplicate allocations in MEMOIZE-heavy files.
                    case 0x8c: // SHORT_BINUNICODE (1-byte length)
                    {
                        int len = all[i++];
                        lastStr    = string.Intern(Encoding.UTF8.GetString(all, i, len)); i += len;
                        lastIsStr  = true;
                        break;
                    }
                    case 0x58: // BINUNICODE (4-byte length)
                    {
                        int len = BitConverter.ToInt32(all, i); i += 4;
                        if (i + len > end) goto done;
                        lastStr    = string.Intern(Encoding.UTF8.GetString(all, i, len)); i += len;
                        lastIsStr  = true;
                        break;
                    }
                    case 0x55: // SHORT_BINSTRING (protocol 2, 1-byte len, latin-1)
                    {
                        int len = all[i++];
                        lastStr    = string.Intern(Encoding.GetEncoding(28591).GetString(all, i, len)); i += len;
                        lastIsStr  = true;
                        break;
                    }

                    // ── Memo store ────────────────────────────────────────────
                    case 0x94: // MEMOIZE — sequential id
                    {
                        string v = (lastIsStr && IsClass(lastStr)) ? lastStr : null;
                        if (memoCounter < MemoSize) memo[memoCounter] = v;
                        memoCounter++;
                        if (v != null && fill < total) flat[fill++] = v;
                        lastIsStr = false;
                        break;
                    }
                    case 0x71: // BINPUT — explicit 1-byte id
                    {
                        int id = all[i++];
                        string v = (lastIsStr && IsClass(lastStr)) ? lastStr : null;
                        if (id < MemoSize) memo[id] = v;
                        if (v != null && fill < total) flat[fill++] = v;
                        lastIsStr = false;
                        break;
                    }
                    case 0x72: // LONG_BINPUT — explicit 4-byte id
                    {
                        int id = BitConverter.ToInt32(all, i); i += 4;
                        string v = (lastIsStr && IsClass(lastStr)) ? lastStr : null;
                        if (id >= 0 && id < MemoSize) memo[id] = v;
                        if (v != null && fill < total) flat[fill++] = v;
                        lastIsStr = false;
                        break;
                    }

                    // ── Memo recall ───────────────────────────────────────────
                    case 0x68: // BINGET — 1-byte id
                    {
                        int    id = all[i++];
                        string v  = (id < MemoSize) ? memo[id] : null;
                        if (v != null)
                        {
                            if (fill < total) flat[fill++] = v;
                            lastStr = v; lastIsStr = true;
                        }
                        else lastIsStr = false;
                        break;
                    }
                    case 0x6a: // LONG_BINGET — 4-byte id
                    {
                        int    id = BitConverter.ToInt32(all, i); i += 4;
                        string v  = (id >= 0 && id < MemoSize) ? memo[id] : null;
                        if (v != null)
                        {
                            if (fill < total) flat[fill++] = v;
                            lastStr = v; lastIsStr = true;
                        }
                        else lastIsStr = false;
                        break;
                    }

                    // ── Stack-changing ops — invalidate lastStr ───────────────
                    case 0x93: case 0x81: case 0x52: case 0x62:
                    case 0x85: case 0x86: case 0x87: case 0x74:
                    case 0x65: case 0x75: case 0x7d: case 0x64:
                    case 0x28: case 0x61: case 0x92:
                        lastIsStr = false;
                        break;

                    case 0x2e: goto done; // STOP

                    // ── Skip fixed-width payloads ─────────────────────────────
                    case 0x4b: i++;      lastIsStr = false; break; // BININT1
                    case 0x4d: i += 2;   lastIsStr = false; break; // BININT2
                    case 0x4a: i += 4;   lastIsStr = false; break; // BININT
                    case 0x43: i += 1 + all[i]; lastIsStr = false; break; // SHORT_BINBYTES
                    case 0x42: { int l = BitConverter.ToInt32(all, i); i += 4 + l; lastIsStr = false; break; }
                    case 0x63: // GLOBAL — two newline-terminated strings
                        while (i < end && all[i] != '\n') i++; i++;
                        while (i < end && all[i] != '\n') i++; i++;
                        lastIsStr = false;
                        break;
                    case 0x8a: i += 1 + all[i]; lastIsStr = false; break; // LONG1
                    case 0x4c: while (i < end && all[i] != '\n') i++; i++; lastIsStr = false; break; // INT

                    default:
                        lastIsStr = false;
                        break;
                }
            }
            done:

            if (fill < total)
                Debug.LogWarning($"[hPBR] HapticNetSeg: expected {total} entries, got {fill} — partial segmentation loaded");

            return new StringArray(flat, rows, cols);
        }
    }
}
