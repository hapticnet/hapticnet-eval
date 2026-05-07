using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEngine;

namespace HPBR
{
    // ── Data containers ───────────────────────────────────────────────────────────

    /// <summary>All data decoded from a single .hpbr v2 file.</summary>
    public class HpbrData
    {
        /// <summary>Haptic property arrays (NARR chunks). Key = stem, e.g. "youngs_modulus".</summary>
        public Dictionary<string, HapticArray> MaterialProps = new Dictionary<string, HapticArray>();

        /// <summary>PBR tile textures (IMAG chunks). Key = tile name, e.g. "basecolor", "normal".</summary>
        public Dictionary<string, Texture2D> PbrTiles = new Dictionary<string, Texture2D>();

        /// <summary>
        /// Per-pixel HapticNet class names decoded from an embedded *_HapticNetSegmentation NARR chunk.
        /// Null when the .hpbr file does not contain a segmentation chunk.
        /// </summary>
        public StringArray SegmentationMap;
    }

    /// <summary>A 2-D float array decoded from a NARR chunk.</summary>
    public class HapticArray
    {
        public readonly float[] Data;
        public readonly int    Rows;
        public readonly int    Cols;

        public HapticArray(float[] data, int rows, int cols)
        {
            Data = data;
            Rows = rows;
            Cols = cols;
        }

        /// <summary>
        /// Sample at normalised UV coordinates (wrapping applied).
        /// Matches hover.py's sample_npy: no V-flip, same modular indexing.
        /// </summary>
        public float Sample(float u, float v)
        {
            int x = ((int)(u * Cols) % Cols + Cols) % Cols;
            int y = ((int)(v * Rows) % Rows + Rows) % Rows;
            return Data[y * Cols + x];
        }

        public string SampleString(float u, float v)
        {
            float val = Sample(u, v);
            return float.IsNaN(val) ? "N/A" : val.ToString("F4");
        }
    }

    // ── Parser ────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Reads hPBR v2 binary files.
    ///
    /// File layout (from hPBR.py):
    ///   [MAGIC 4B] [VERSION 1B]
    ///   ( [TAG 4B] [CHUNK_LEN 4B] [NAME_LEN 1B] [NAME var] [PAYLOAD var] [CRC32 4B] )*
    ///   [IEND sentinel]
    ///
    /// NARR payload:  rows(uint32) + cols(uint32) + pickle_flag(int8) + .npy-format bytes
    ///   pickle_flag 0 = numeric array, 1 = object array (e.g. *_HapticNetSegmentation strings)
    /// IMAG payload:  format_code(uint8=1 means PNG) + PNG bytes
    /// CRC32 covers:  TAG + NAME_LEN_BYTE + NAME + PAYLOAD
    /// </summary>
    public static class HpbrReader
    {
        private static readonly byte[] Magic   = { 0x68, 0x50, 0x42, 0x52 }; // "hPBR"
        private const           byte   Version = 2;

        // ── Public API ────────────────────────────────────────────────────────────

        public static HpbrData Read(string filePath)
        {
            var result = new HpbrData();

            using (var stream = new FileStream(filePath, FileMode.Open, FileAccess.Read))
            using (var reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: false))
            {
                ValidateHeader(reader, filePath);

                long fileSize = stream.Length;
                while (stream.Position < fileSize)
                {
                    ReadChunk(reader, out string tag, out string name, out byte[] payload);

                    if (tag == "IEND") break;

                    if (tag == "NARR")
                    {
                        if (name.EndsWith("_HapticNetSegmentation", StringComparison.Ordinal))
                        {
                            try   { result.SegmentationMap = ParseNarrSeg(payload); }
                            catch (Exception ex)
                            { Debug.LogWarning($"[hPBR] Could not decode segmentation NARR '{name}': {ex.Message}"); }
                        }
                        else
                        {
                            try   { result.MaterialProps[name] = ParseNarr(payload); }
                            catch (Exception ex)
                            { Debug.LogWarning($"[hPBR] Could not decode NARR '{name}': {ex.Message}"); }
                        }
                    }
                    else if (tag == "IMAG")
                    {
                        Texture2D tex = ParseImag(payload, name);
                        if (tex != null) result.PbrTiles[name] = tex;
                    }
                    else
                    {
                        // Log unrecognised chunks so we can see what's in the file
                        string preview = payload.Length > 0
                            ? Encoding.UTF8.GetString(payload, 0,
                                                      Math.Min(payload.Length, 128))
                            : "(empty)";
                        Debug.Log($"[hPBR] Unknown chunk  tag='{tag}'  name='{name}'  " +
                                  $"bytes={payload.Length}  preview: {preview}");
                    }
                }
            }

            return result;
        }

        // ── Header ────────────────────────────────────────────────────────────────

        private static void ValidateHeader(BinaryReader reader, string filePath)
        {
            byte[] magic   = reader.ReadBytes(4);
            byte   version = reader.ReadByte();

            if (!BytesEqual(magic, Magic))
                throw new Exception($"[hPBR] Not an hPBR file: {filePath}");
            if (version != Version)
                throw new Exception($"[hPBR] Unsupported version {version} (expected {Version}): {filePath}");
        }

        // ── Chunk reader ──────────────────────────────────────────────────────────

        private static void ReadChunk(BinaryReader reader,
                                      out string tag, out string name, out byte[] payload)
        {
            byte[] tagBytes  = reader.ReadBytes(4);
            tag              = Encoding.ASCII.GetString(tagBytes);

            uint   chunkLen  = reader.ReadUInt32();   // payload bytes only
            byte   nameLen   = reader.ReadByte();
            byte[] nameBytes = reader.ReadBytes(nameLen);
            name             = Encoding.UTF8.GetString(nameBytes);

            payload = reader.ReadBytes((int)chunkLen);

            uint storedCrc = reader.ReadUInt32();

            if (tag != "IEND")
            {
                // CRC32 over: TAG(4) + NAME_LEN(1) + NAME(nameLen) + PAYLOAD(chunkLen)
                byte[] crcBuf = new byte[4 + 1 + nameLen + chunkLen];
                int    off    = 0;
                Buffer.BlockCopy(tagBytes,   0, crcBuf, off, 4);                          off += 4;
                crcBuf[off] = nameLen;                                                    off += 1;
                Buffer.BlockCopy(nameBytes,  0, crcBuf, off, nameLen);                   off += nameLen;
                Buffer.BlockCopy(payload,    0, crcBuf, off, (int)chunkLen);

                uint computed = Crc32(crcBuf);
                if (computed != storedCrc)
                    throw new Exception(
                        $"[hPBR] CRC mismatch for chunk '{name}' " +
                        $"(stored=0x{storedCrc:X8}, computed=0x{computed:X8})");
            }
        }

        // ── NARR decoders ─────────────────────────────────────────────────────────

        /// <summary>
        /// Decode a numeric NARR payload (new format).
        /// Header: rows(4B) + cols(4B) + pickle_flag(1B) — followed by .npy-format bytes.
        /// The shape is also encoded inside the .npy bytes; we skip our 9-byte header and
        /// hand the rest straight to NpyReader.
        /// </summary>
        private static HapticArray ParseNarr(byte[] payload)
        {
            byte[] npyBytes = new byte[payload.Length - 9];
            Buffer.BlockCopy(payload, 9, npyBytes, 0, npyBytes.Length);
            return NpyReader.ReadBytes(npyBytes);
        }

        /// <summary>
        /// Decode a segmentation NARR payload (pickle_flag = 1, object array of strings).
        /// Same 9-byte header as ParseNarr; the body is a pickle stream wrapped in .npy format.
        /// </summary>
        private static StringArray ParseNarrSeg(byte[] payload)
        {
            byte[] npyBytes = new byte[payload.Length - 9];
            Buffer.BlockCopy(payload, 9, npyBytes, 0, npyBytes.Length);
            return HapticNetSegReader.Read(npyBytes);
        }

        // ── IMAG decoder ──────────────────────────────────────────────────────────

        private static Texture2D ParseImag(byte[] payload, string name)
        {
            if (payload.Length < 2) { Debug.LogWarning($"[hPBR] Empty IMAG payload '{name}'"); return null; }

            byte format = payload[0];
            if (format != 1) { Debug.LogWarning($"[hPBR] Unsupported image format {format} for '{name}'"); return null; }

            byte[] png = new byte[payload.Length - 1];
            Buffer.BlockCopy(payload, 1, png, 0, png.Length);

            var tex = new Texture2D(2, 2, TextureFormat.RGBA32, mipChain: true, linear: false);
            if (!ImageConversion.LoadImage(tex, png, markNonReadable: false))
            {
                Debug.LogWarning($"[hPBR] Failed to decode PNG for tile '{name}'");
                UnityEngine.Object.DestroyImmediate(tex);
                return null;
            }
            tex.name = name;
            return tex;
        }

        // ── CRC-32 (IEEE 802.3 — matches Python zlib.crc32) ──────────────────────

        private static readonly uint[] CrcTable = BuildCrcTable();

        private static uint[] BuildCrcTable()
        {
            uint[] t = new uint[256];
            for (uint n = 0; n < 256; n++)
            {
                uint c = n;
                for (int k = 0; k < 8; k++)
                    c = (c & 1u) != 0u ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
                t[n] = c;
            }
            return t;
        }

        private static uint Crc32(byte[] data)
        {
            uint crc = 0xFFFFFFFFu;
            foreach (byte b in data)
                crc = CrcTable[(crc ^ b) & 0xFFu] ^ (crc >> 8);
            return crc ^ 0xFFFFFFFFu;
        }

        // ── Utility ───────────────────────────────────────────────────────────────

        private static bool BytesEqual(byte[] a, byte[] b)
        {
            if (a.Length != b.Length) return false;
            for (int i = 0; i < a.Length; i++) if (a[i] != b[i]) return false;
            return true;
        }
    }
}
